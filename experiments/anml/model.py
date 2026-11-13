"""Common SDC architecture and legacy alignment schedules for A/B/C.

Epoch arguments are zero-based. Target ground-truth labels are not model inputs.
conditional_mmd accepts model-generated target pseudo-labels only.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
import backbone


class SDCModel(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.backbone = backbone.network_dict["simple_net"]()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return features, self.classifier(features)


def initialize_weights(model):
    """Match the original Linear initialization, including the classifier."""
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.03)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    with torch.no_grad():
        model.apply(initialize)


def probability_similarity(logits):
    """Cosine of softmax probabilities; losses exclude self-pairs."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes].")
    normalized = F.normalize(F.softmax(logits, dim=1), p=2, dim=1, eps=1e-12)
    return (normalized @ normalized.T).clamp(0.0, 1.0)


def _check_features(source, target):
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Features must have shape [batch, dimensions].")
    if source.shape[1] != target.shape[1] or source.shape[0] == 0:
        raise ValueError("Feature widths must match; source cannot be empty.")
    if source.device != target.device or source.dtype != target.dtype:
        raise ValueError("Source and target must share device and dtype.")
    if not source.is_floating_point():
        raise ValueError("Features must have a floating-point dtype.")


def _gaussian_kernels(source, target):
    """Original five-scale RBF sum with detached, guarded bandwidth."""
    total = torch.cat((source, target), dim=0)
    n = total.shape[0]
    if n < 2:
        raise ValueError("Kernel estimation requires at least two samples.")
    epsilon = 1e-5
    distances = (total[:, None, :] - total[None, :, :]).square().sum(-1)
    distances = distances + epsilon
    bandwidth = (distances.detach().sum() / (n * (n - 1))).clamp_min(epsilon)
    return sum(
        torch.exp(-distances / (bandwidth * (2.0 ** exponent)))
        for exponent in (-2, -1, 0, 1, 2)
    )


def marginal_mmd(source, target):
    """Original cyclic estimator; negative estimates are allowed."""
    _check_features(source, target)
    batch = source.shape[0]
    if target.shape[0] != batch or batch < 2:
        raise ValueError("Marginal MMD requires equal batch sizes >= 2.")
    kernels = _gaussian_kernels(source, target)
    index = torch.arange(batch, device=source.device)
    following = (index + 1) % batch
    return (
        kernels[index, following] + kernels[index + batch, following + batch]
        - kernels[index, following + batch] - kernels[following, index + batch]
    ).mean()


def conditional_mmd(source, target, s_labels, t_pseudo, n_classes):
    """cmmd_1's nonempty formula; normalize by total batch, not class counts.

    Empty target selection contributes a differentiable zero.
    No extra division by n_classes is applied.
    """
    _check_features(source, target)
    ns, nt = source.shape[0], target.shape[0]
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")
    if s_labels.ndim != 1 or s_labels.shape[0] != ns:
        raise ValueError("s_labels must have shape [source_batch].")
    if t_pseudo.ndim != 1 or t_pseudo.shape[0] != nt:
        raise ValueError("t_pseudo must have shape [selected_target_batch].")
    if nt == 0:
        return source.sum() * 0.0 + target.sum() * 0.0
    source_onehot = F.one_hot(s_labels.to(source.device).long(), n_classes).to(source.dtype)
    target_onehot = F.one_hot(t_pseudo.to(target.device).long(), n_classes).to(target.dtype)
    kernels = _gaussian_kernels(source, target)
    ss = source_onehot @ source_onehot.T
    tt = target_onehot @ target_onehot.T
    st = source_onehot @ target_onehot.T
    return (
        (ss * kernels[:ns, :ns]).mean()
        + (tt * kernels[ns:, ns:]).mean()
        - (st * kernels[:ns, ns:]).mean()
        - (st.T * kernels[ns:, :ns]).mean()
    )


def legacy_mmd_weight(epoch):
    if epoch < 0:
        raise ValueError("epoch must be nonnegative.")
    if epoch <= 10:
        return 1.0
    if epoch <= 40:
        return 2.0 / (1.0 + math.exp(-10.0)) - 1.0
    if epoch <= 85:
        return math.exp(-0.6 * epoch)
    return 0.01


def legacy_confidence_threshold(epoch):
    if epoch < 0:
        raise ValueError("epoch must be nonnegative.")
    if epoch < 10:
        return 0.0
    if epoch < 40:
        return 0.5
    if epoch <= 85:
        return 0.75
    return 1.0


def legacy_cmmd_weight(clf_loss_float):
    if not math.isfinite(clf_loss_float):
        raise ValueError("Classification loss must be finite.")
    if clf_loss_float <= 0.1:
        return 1.0
    if clf_loss_float < 0.15:
        return 0.5
    return 0.0


class ThresholdSchedule:
    """Call update once at each epoch end, including epoch zero."""
    def __init__(self, total_epochs, upper=0.9, lower=0.5, pair_threshold=0.9):
        if not isinstance(total_epochs, int) or total_epochs < 3:
            raise ValueError("total_epochs must be an integer >= 3.")
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("Require 0 <= lower < upper <= 1.")
        if not lower <= pair_threshold <= upper:
            raise ValueError("pair_threshold must lie between lower and upper.")
        self.total_epochs = total_epochs
        self.upper = float(upper)
        self.lower = float(lower)
        self.pair_threshold = float(pair_threshold)

    def update(self, epoch):
        if not 0 <= epoch < self.total_epochs:
            raise ValueError("epoch is outside the configured training range.")
        if epoch != 0:
            step = (self.upper - self.lower) / self.total_epochs
            self.upper -= step
            self.lower += step
        self.pair_threshold = (self.upper + self.lower) / 2.0
