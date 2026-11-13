"""Controlled A/B/C relation losses in SDC-Net's probability cosine space.

A: no relation objective. B: selected-pair DSCL BCE without self-pairs.
C: independent DANML Eq. (22) adaptation with fixed virtual radii:
relu(margin + b_gamma_pos(D_pos union {radius_pos})
            - b_gamma_neg(D_neg union {radius_neg})).
b_gamma(a) = -log(mean(exp(-gamma*a))) / gamma.

Reference: Song et al., Adaptive Neighborhood Metric Learning,
https://arxiv.org/html/2201.08314v1 , VI-A, Eq. (22).
This is not the authors' code or a claim of exact paper reproduction.
No new reliability mechanism is implemented. Target truth is never accepted.
"""
from dataclasses import dataclass
import math
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class Relations:
    positive: Tensor
    negative: Tensor

    @property
    def selected(self):
        return self.positive | self.negative

    @property
    def valid_anchors(self):
        return self.positive.any(dim=1) & self.negative.any(dim=1)


def _similarity_matrix(matrix, name):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("{} must be square".format(name))
    if matrix.shape[0] == 0 or not torch.is_floating_point(matrix):
        raise ValueError("{} must be a nonempty floating-point matrix".format(name))
    values = matrix.detach()
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("{} contains non-finite values".format(name))
    if bool(((values < -1e-5) | (values > 1.00001)).any().item()):
        raise ValueError("{} must be probability-vector cosine in [0,1]".format(name))
    return matrix.float().clamp(0.0, 1.0)


@torch.no_grad()
def source_relations(similarity, labels):
    labels = labels.detach().to(device=similarity.device)
    if labels.ndim == 2 and labels.shape[1] > 0:
        classes = labels.argmax(dim=1)
    elif labels.ndim == 1:
        classes = labels.long()
        if not bool((labels == classes.to(labels.dtype)).all().item()):
            raise ValueError("Source class indices must be integers")
    else:
        raise ValueError("Source labels must be [N] or [N,C]")
    n = similarity.shape[0]
    if classes.shape[0] != n:
        raise ValueError("Source labels and similarities disagree")
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
    same = classes[:, None] == classes[None, :]
    return Relations(same & off_diagonal, ~same & off_diagonal)


@torch.no_grad()
def target_relations(similarity, lower_threshold, upper_threshold, pair_threshold):
    """Original strict selection/truth thresholds, detached, without labels."""
    if not all(math.isfinite(float(v)) for v in
               (lower_threshold, upper_threshold, pair_threshold)):
        raise ValueError("Relation thresholds must be finite")
    if not 0 <= lower_threshold <= pair_threshold <= upper_threshold <= 1:
        raise ValueError("Require 0 <= lower <= pair_threshold <= upper <= 1")
    values = similarity.detach()
    off_diagonal = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    selected = ((values > upper_threshold) | (values < lower_threshold)) & off_diagonal
    positive = values > pair_threshold
    return Relations(selected & positive, selected & ~positive)


def _stats(similarity, relations):
    """All pair counts are directed; target statistics are label-free."""
    with torch.no_grad():
        n = similarity.shape[0]
        possible = n * (n - 1)
        positive = int(relations.positive.sum().item())
        negative = int(relations.negative.sum().item())
        selected = positive + negative
        valid = relations.valid_anchors
        supported = int((relations.selected & valid[:, None]).sum().item())
        values = similarity.detach()
        return {
            "anchors": float(n), "possible_pairs": float(possible),
            "positive_pairs": float(positive), "negative_pairs": float(negative),
            "selected_pairs": float(selected),
            "selected_coverage": selected / max(possible, 1),
            "positive_fraction": positive / max(selected, 1),
            "anchors_with_positive": float(relations.positive.any(dim=1).sum().item()),
            "anchors_with_negative": float(relations.negative.any(dim=1).sum().item()),
            "valid_anchors": float(valid.sum().item()),
            "valid_anchor_fraction": float(valid.float().mean().item()),
            "anml_supported_pairs": float(supported),
            "anml_pair_coverage": supported / max(possible, 1),
            "anml_selected_fraction": supported / max(selected, 1),
            "mean_positive_similarity": (
                float(values[relations.positive].mean().item()) if positive else 0.0
            ),
            "mean_negative_similarity": (
                float(values[relations.negative].mean().item()) if negative else 0.0
            ),
        }


def _bce(similarity, relations, eps):
    selected = relations.selected
    if not bool(selected.any().item()):
        return similarity.sum() * 0.0
    predictions = similarity[selected].clamp(eps, 1.0 - eps)
    targets = relations.positive[selected].to(predictions.dtype)
    return F.binary_cross_entropy(predictions, targets, reduction="mean")


def _radius(distances, mask, gamma, fixed_radius):
    """One virtual radius plus real neighbors; includes the n+1 divisor."""
    logits = (-gamma * distances).masked_fill(~mask, float("-inf"))
    virtual = distances.new_full((distances.shape[0], 1), -gamma * fixed_radius)
    log_total = torch.logsumexp(torch.cat((logits, virtual), dim=1), dim=1)
    log_count = (mask.sum(dim=1).to(distances.dtype) + 1.0).log()
    return -(log_total - log_count) / gamma


def _danml(similarity, relations, gamma_pos, gamma_neg,
           radius_pos, radius_neg, margin):
    valid = relations.valid_anchors
    if not bool(valid.any().item()):
        return similarity.sum() * 0.0, {
            "anml_mean_positive_radius": 0.0,
            "anml_mean_negative_radius": 0.0,
            "anml_hinge_active_anchors": 0.0,
        }
    distances = 1.0 - similarity[valid]
    rp = _radius(distances, relations.positive[valid], gamma_pos, radius_pos)
    rn = _radius(distances, relations.negative[valid], gamma_neg, radius_neg)
    violation = margin + rp - rn
    return F.relu(violation).mean(), {
        "anml_mean_positive_radius": float(rp.detach().mean().item()),
        "anml_mean_negative_radius": float(rn.detach().mean().item()),
        "anml_hinge_active_anchors": float((violation.detach() > 0).sum().item()),
    }


def relation_losses(
    variant, source_similarity, target_similarity, source_labels, *,
    lower_threshold, upper_threshold, pair_threshold,
    gamma_pos=-2.0, gamma_neg=20.0, radius_pos=0.3, radius_neg=0.5,
    margin=0.5, eps=1e-6,
):
    """Unweighted source loss, target loss, numeric diagnostics.

    B averages selected pairs; C averages anchors with both neighbor types.
    External source/target coefficients are shared across A/B/C.
    Missing neighbor types are not invented using the virtual radii.
    """
    mode = variant.upper()
    if mode not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B, or C")
    parameters = (gamma_pos, gamma_neg, radius_pos, radius_neg, margin, eps)
    if not all(math.isfinite(float(v)) for v in parameters):
        raise ValueError("Relation parameters must be finite")
    if not gamma_pos < -1e-6 or not gamma_neg > 1e-6:
        raise ValueError("Require gamma_pos < 0 < gamma_neg")
    if not 0 <= radius_pos <= 1 or not 0 <= radius_neg <= 1:
        raise ValueError("Radii must lie in the [0,1] distance range")
    if margin < 0 or not 0 < eps < 0.5:
        raise ValueError("Require margin >= 0 and 0 < eps < 0.5")
    source = _similarity_matrix(source_similarity, "source_similarity")
    target = _similarity_matrix(target_similarity, "target_similarity")
    source_candidates = source_relations(source, source_labels)
    target_candidates = target_relations(
        target, lower_threshold, upper_threshold, pair_threshold
    )
    losses, output_stats = [], {}
    for prefix, similarity, candidates in (
        ("source", source, source_candidates), ("target", target, target_candidates)
    ):
        stats = _stats(similarity, candidates)
        if mode == "A":
            loss = similarity.sum() * 0.0
            eligible_anchors, eligible_pairs = 0.0, 0.0
        elif mode == "B":
            loss = _bce(similarity, candidates, eps)
            eligible_anchors = float(candidates.selected.any(dim=1).sum().item())
            eligible_pairs = stats["selected_pairs"]
        else:
            loss, extra = _danml(
                similarity, candidates, gamma_pos, gamma_neg,
                radius_pos, radius_neg, margin,
            )
            stats.update(extra)
            eligible_anchors = stats["valid_anchors"]
            eligible_pairs = stats["anml_supported_pairs"]
        stats["objective_eligible_anchors"] = eligible_anchors
        stats["objective_eligible_pairs"] = eligible_pairs
        stats["raw_relation_loss"] = float(loss.detach().item())
        output_stats.update({prefix + "_" + k: v for k, v in stats.items()})
        losses.append(loss)
    return losses[0], losses[1], output_stats


def selected_relation_loss(
    variant, similarity, relations, *, gamma_pos=-2.0, gamma_neg=20.0,
    radius_pos=0.3, radius_neg=0.5, margin=0.5, eps=1e-6,
):
    """Raw loss for post-training diagnostics on already constructed relations."""
    mode = variant.upper()
    if mode == "A":
        return similarity.sum() * 0.0
    if mode == "B":
        return _bce(similarity, relations, eps)
    if mode == "C":
        return _danml(similarity, relations, gamma_pos, gamma_neg,
                      radius_pos, radius_neg, margin)[0]
    raise ValueError("variant must be A, B, or C")
