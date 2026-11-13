"""Audit data or run a fixed-epoch, transductive SDC-Net A/B/C pilot.

The optimizer function receives source labels and target FEATURES only.
Target ground truth is used after the final checkpoint is saved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_subjects, make_loaders
from .losses import relation_losses, target_relations, selected_relation_loss
from .model import (
    SDCModel, initialize_weights, probability_similarity, marginal_mmd,
    conditional_mmd, legacy_mmd_weight, legacy_confidence_threshold,
    legacy_cmmd_weight, ThresholdSchedule,
)

ROOT = Path(__file__).resolve().parents[2]
LOSS_KEYS = ("gamma_pos", "gamma_neg", "radius_pos", "radius_neg", "margin", "eps")


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(request):
    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, cuda or cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose cpu/auto or install a CUDA PyTorch build.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index is unavailable.")
    return device


def snapshot_version():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True,
            text=True, check=False, timeout=5,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {
        "git_commit": commit, "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def validate_config(config):
    for section in ("data", "experiment", "relation", "protocol"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration object: {section}")
    data, exp, rel = config["data"], config["experiment"], config["relation"]
    if int(data["expected_features"]) != 310:
        raise ValueError("This pilot preserves the original 310-input backbone.")
    for key, minimum in (("epochs", 3), ("batch_size", 2),
                         ("log_every_batches", 1), ("diagnostic_samples", 2)):
        if not isinstance(exp[key], int) or exp[key] < minimum:
            raise ValueError(f"experiment.{key} must be an integer >= {minimum}.")
    if not math.isfinite(exp["lr"]) or exp["lr"] <= 0:
        raise ValueError("lr must be finite and positive.")
    if not isinstance(exp["ss_mix"], bool):
        raise ValueError("ss_mix must be a JSON boolean.")
    for name in ("source_scale", "target_scale"):
        if not math.isfinite(rel[name]) or rel[name] < 0:
            raise ValueError(f"relation.{name} must be finite and nonnegative.")
    for name in LOSS_KEYS:
        if not math.isfinite(rel[name]):
            raise ValueError(f"relation.{name} must be finite.")
    if not rel["gamma_pos"] < -1e-6 or not rel["gamma_neg"] > 1e-6:
        raise ValueError("Require gamma_pos < 0 < gamma_neg.")
    if not (0 <= rel["radius_pos"] <= 1 and 0 <= rel["radius_neg"] <= 1):
        raise ValueError("DANML radii must be in [0,1].")
    if rel["margin"] < 0 or not 0 < rel["eps"] < 0.5:
        raise ValueError("Require margin >= 0 and 0 < eps < 0.5.")
    ThresholdSchedule(
        exp["epochs"], upper=rel["upper_threshold"], lower=rel["lower_threshold"],
        pair_threshold=rel["pair_threshold"],
    )
    if config["protocol"]["model_selection"] != "fixed_final_epoch":
        raise ValueError("Only fixed_final_epoch is implemented; target-best selection is forbidden.")
    if config["protocol"]["target_access"] != "transductive":
        raise ValueError("This loader uses full unlabeled target features: transductive only.")
    for key in ("variants", "subjects", "seeds"):
        if not isinstance(exp[key], list) or not exp[key]:
            raise ValueError(f"experiment.{key} must be a nonempty list.")
        if len(set(map(str, exp[key]))) != len(exp[key]):
            raise ValueError(f"Duplicate entries in experiment.{key}.")
    if any(v not in ("A", "B", "C") for v in exp["variants"]):
        raise ValueError("variants must contain A, B and/or C.")
    if any(not isinstance(s, int) or not 0 <= s < 2**32 - 1 for s in exp["seeds"]):
        raise ValueError("seeds must be integers in [0, 2**32-2].")


def train_fixed(model, source_loader, target_loader, exp, rel, device, folder, variant):
    """No eval loader, target labels, or test metrics enter this function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=exp["lr"])
    schedule = ThresholdSchedule(
        exp["epochs"], upper=rel["upper_threshold"], lower=rel["lower_threshold"],
        pair_threshold=rel["pair_threshold"],
    )
    loss_args = {key: rel[key] for key in LOSS_KEYS}
    n_batches = min(len(source_loader), len(target_loader))
    if n_batches == 0:
        raise ValueError("No complete training batches. Reduce batch_size.")
    final_thresholds = None
    with (folder / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = None
        for epoch in range(exp["epochs"]):
            model.train()
            started = time.perf_counter()
            totals = {}
            threshold = legacy_confidence_threshold(epoch)
            beta = legacy_mmd_weight(epoch)
            target_weight = 2.0 * epoch / exp["epochs"] * rel["target_scale"]
            final_thresholds = {
                "lower_threshold": schedule.lower,
                "upper_threshold": schedule.upper,
                "pair_threshold": schedule.pair_threshold,
            }
            # Identically seeded generators and shared step budget across variants.
            for step, ((xs, ys), (xt, _target_rows)) in enumerate(
                zip(source_loader, target_loader), start=1
            ):
                xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
                optimizer.zero_grad(set_to_none=True)
                fs, logits_s = model(xs)
                ft, logits_t = model(xt)
                ce = F.cross_entropy(logits_s, ys)
                mmd = marginal_mmd(fs, ft)
                confidence, pseudo = F.softmax(logits_t.detach(), dim=1).max(dim=1)
                selected = confidence > threshold
                cmmd = conditional_mmd(
                    fs, ft[selected], ys, pseudo[selected], logits_s.shape[1]
                )
                sim_s = probability_similarity(logits_s)
                sim_t = probability_similarity(logits_t)
                rs, rt, stats = relation_losses(
                    variant, sim_s, sim_t, ys, **final_thresholds, **loss_args
                )
                cmmd_weight = legacy_cmmd_weight(float(ce.detach().item()))
                source_weight = cmmd_weight * rel["source_scale"]
                loss = ce + beta * mmd + cmmd_weight * cmmd + source_weight * rs + target_weight * rt
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}, step {step}.")
                # Sensitivity to pair similarities, NOT parameter-gradient norms.
                gs, gt = torch.autograd.grad(rs + rt, (sim_s, sim_t), retain_graph=True)
                sensitivity_s = float(gs.detach().abs().sum().item())
                sensitivity_t = float(gt.detach().abs().sum().item())
                loss.backward()
                squared_norm = sum(
                    float(p.grad.detach().square().sum().item())
                    for p in model.parameters() if p.grad is not None
                )
                if not math.isfinite(squared_norm):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch+1}, step {step}.")
                optimizer.step()
                values = {
                    "loss": float(loss.detach().item()), "ce": float(ce.detach().item()),
                    "mmd": float(mmd.detach().item()), "cmmd": float(cmmd.detach().item()),
                    "relation_source": float(rs.detach().item()),
                    "relation_target": float(rt.detach().item()),
                    "cmmd_selected_fraction": float(selected.float().mean().item()),
                    "cmmd_weight": cmmd_weight, "source_relation_weight": source_weight,
                    "source_similarity_sensitivity_l1": sensitivity_s,
                    "target_similarity_sensitivity_l1": sensitivity_t,
                    "weighted_target_similarity_sensitivity_l1": target_weight * sensitivity_t,
                    "parameter_gradient_norm": math.sqrt(squared_norm),
                }
                values.update(stats)
                for key, value in values.items():
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Nonfinite diagnostic {key}.")
                    totals[key] = totals.get(key, 0.0) + value
                if step == 1 or step % exp["log_every_batches"] == 0:
                    print(
                        f"  epoch {epoch+1:03d}/{exp['epochs']} batch {step}/{n_batches} "
                        f"loss={values['loss']:.5f} CE={values['ce']:.5f} "
                        f"target_pairs={int(stats['target_selected_pairs'])} "
                        f"ANML_valid={int(stats['target_valid_anchors'])}", flush=True,
                    )
            row = {
                "epoch": epoch + 1, "batches": n_batches,
                "seconds": time.perf_counter() - started,
                "mmd_weight": beta, "confidence_threshold": threshold,
                "target_relation_weight": target_weight, **final_thresholds,
            }
            row.update({key: value / n_batches for key, value in totals.items()})
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(
                f"  finished epoch {epoch+1:03d}: loss={row['loss']:.5f}, "
                f"pair_coverage={row['target_selected_coverage']:.3f}, "
                f"{row['seconds']:.1f}s; target accuracy reserved for final evaluation",
                flush=True,
            )
            schedule.update(epoch)
    return final_thresholds


@torch.no_grad()
def evaluate_final(model, loader, device, n_classes):
    model.eval()
    truths, predictions, probabilities = [], [], []
    for x, y in loader:
        _, logits = model(x.to(device))
        prob = F.softmax(logits, dim=1).cpu().numpy()
        truths.append(y.numpy())
        probabilities.append(prob)
        predictions.append(prob.argmax(1))
    truth, pred, prob = np.concatenate(truths), np.concatenate(predictions), np.concatenate(probabilities)
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    tp = np.diag(matrix).astype(float)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    metrics = {
        "accuracy": float((truth == pred).mean()), "macro_f1": float(f1.mean()),
        "per_class_f1": f1.tolist(), "confusion_matrix": matrix.tolist(),
        "n_target_samples": int(len(truth)),
        "model_selection": "fixed_final_epoch", "target_access": "transductive",
    }
    return metrics, truth, pred, prob


def diagnose_frozen_pairs(probabilities, truth, variant, rel, thresholds, exp, seed, folder):
    """Post hoc only. Labels cannot feed back to training or checkpoint choice.

    Keep diagnostic batches at the training batch size. Sensitivity is
    |d L_relation / d similarity_ij| treating the matrix entries independently;
    it is NOT a model-parameter gradient attribution or causal mechanism proof.
    """
    rng = np.random.default_rng(seed + 100003)
    rows = rng.permutation(len(truth))[:min(exp["diagnostic_samples"], len(truth))]
    loss_args = {key: rel[key] for key in LOSS_KEYS}
    counts = {"possible": 0, "selected": 0, "correct": 0, "wrong_positive": 0,
              "wrong_negative": 0, "valid_anchors": 0, "anchors": 0}
    sensitivity_total, sensitivity_wrong = 0.0, 0.0
    sensitivity_positive, sensitivity_negative = 0.0, 0.0
    records = []
    for start in range(0, len(rows), exp["batch_size"]):
        indices = rows[start:start + exp["batch_size"]]
        if len(indices) < 2:
            continue
        p = torch.as_tensor(probabilities[indices], dtype=torch.float32)
        p = F.normalize(p, p=2, dim=1, eps=1e-12)
        similarity = (p @ p.T).clamp(0, 1).detach().requires_grad_(True)
        candidates = target_relations(similarity, **thresholds)
        loss = selected_relation_loss(variant, similarity, candidates, **loss_args)
        gradient = torch.autograd.grad(loss, similarity)[0].detach().abs().numpy()
        predicted_same = candidates.positive.cpu().numpy()
        selected = candidates.selected.cpu().numpy()
        actual_same = truth[indices, None] == truth[indices][None, :]
        wrong = selected & (predicted_same != actual_same)
        valid = candidates.valid_anchors.cpu().numpy()
        counts["possible"] += len(indices) * (len(indices) - 1)
        counts["selected"] += int(selected.sum())
        counts["correct"] += int((selected & ~wrong).sum())
        counts["wrong_positive"] += int((wrong & predicted_same).sum())
        counts["wrong_negative"] += int((wrong & ~predicted_same).sum())
        counts["valid_anchors"] += int(valid.sum())
        counts["anchors"] += len(indices)
        sensitivity_total += float(gradient.sum())
        sensitivity_wrong += float(gradient[wrong].sum())
        sensitivity_positive += float(gradient[wrong & predicted_same].sum())
        sensitivity_negative += float(gradient[wrong & ~predicted_same].sum())
        sim = similarity.detach().numpy()
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    records.append({
                        "row_i": int(indices[i]), "row_j": int(indices[j]),
                        "similarity": float(sim[i, j]), "selected": int(selected[i, j]),
                        "inferred_same": int(predicted_same[i, j]) if selected[i, j] else "",
                        "actual_same": int(actual_same[i, j]),
                        "anchor_has_both_neighbor_types": int(valid[i]),
                        "abs_dloss_dsimilarity": float(gradient[i, j]),
                    })
    write_csv(folder / "final_pair_samples.csv", records)
    report = {
        "analysis_stage": "after_final_checkpoint", "directed_pairs": True,
        "diagnostic_batch_size": exp["batch_size"], "seed": seed + 100003,
        "selected_pair_precision": counts["correct"] / counts["selected"] if counts["selected"] else None,
        "selected_pair_coverage": counts["selected"] / counts["possible"] if counts["possible"] else None,
        "wrong_similarity_sensitivity_share": (
            sensitivity_wrong / sensitivity_total if sensitivity_total > 0 else None
        ),
        "similarity_sensitivity_l1": sensitivity_total,
        "wrong_positive_similarity_sensitivity_l1": sensitivity_positive,
        "wrong_negative_similarity_sensitivity_l1": sensitivity_negative,
        "counts": counts,
        "interpretation": "Partial derivatives of raw relation loss with respect to pair similarities; not parameter-gradient attribution.",
    }
    write_json(folder / "final_pair_diagnostics.json", report)


def save_summary(run_dir, results):
    write_csv(run_dir / "results.csv", results)
    summary = {}
    for variant in sorted({row["variant"] for row in results}):
        subset = [r for r in results if r["variant"] == variant]
        by_subject = {}
        for row in subset:
            by_subject.setdefault(row["subject"], []).append(row)
        subject_values = [
            {"subject": sid, "accuracy": float(np.mean([r["accuracy"] for r in group])),
             "macro_f1": float(np.mean([r["macro_f1"] for r in group])),
             "completed_seeds": len(group)}
            for sid, group in by_subject.items()
        ]
        summary[variant] = {
            "completed_runs": len(subset), "subjects": subject_values,
            "accuracy_mean_over_subjects": float(np.mean([r["accuracy"] for r in subject_values])),
            "accuracy_sd_over_subjects": (
                float(np.std([r["accuracy"] for r in subject_values], ddof=1))
                if len(subject_values) > 1 else None
            ),
            "macro_f1_mean_over_subjects": float(np.mean([r["macro_f1"] for r in subject_values])),
        }
    write_json(run_dir / "summary.json", {
        "aggregation": "Average available seeds within each subject, then summarize subjects. Check completed_seeds before comparison.",
        "stage": "exploratory_pilot", "variants": summary,
    })
    lookup = {(r["variant"], r["subject"], r["seed"]): r for r in results}
    paired = []
    for base, candidate in (("A", "B"), ("B", "C")):
        for (variant, subject, seed), row in lookup.items():
            ref = lookup.get((base, subject, seed))
            if variant == candidate and ref is not None:
                paired.append({
                    "comparison": f"{candidate}-{base}", "subject": subject, "seed": seed,
                    "accuracy_delta": row["accuracy"] - ref["accuracy"],
                    "macro_f1_delta": row["macro_f1"] - ref["macro_f1"],
                })
    write_csv(run_dir / "paired_comparison.csv", paired)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "anml_pilot.json"))
    parser.add_argument("--mode", choices=["audit", "train"], default="audit")
    parser.add_argument("--data-root")
    parser.add_argument("--variants", nargs="+", choices=["A", "B", "C"])
    parser.add_argument("--subjects", nargs="+", help="File subject IDs, or all.")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.data_root:
        config["data"]["data_root"] = args.data_root
    exp = config["experiment"]
    for key in ("variants", "subjects", "seeds", "epochs", "batch_size", "device"):
        value = getattr(args, key)
        if value is not None:
            exp[key] = value
    # Relative paths are relative to the project, independent of PyCharm's cwd.
    data_root = Path(config["data"]["data_root"]).expanduser()
    if not data_root.is_absolute():
        config["data"]["data_root"] = str((ROOT / data_root).resolve())
    validate_config(config)
    print("Loading and validating MAT files...", flush=True)
    subjects = load_subjects(config["data"])
    available = [s.subject_id for s in subjects]
    if exp["subjects"] == ["all"]:
        exp["subjects"] = available
    else:
        exp["subjects"] = [str(int(s)) if str(s).isdigit() else str(s) for s in exp["subjects"]]
    if len(set(exp["subjects"])) != len(exp["subjects"]) or any(s not in available for s in exp["subjects"]):
        raise ValueError(f"Invalid/duplicate subject IDs. Available: {available}")
    device = resolve_device(exp["device"])
    output = Path(exp["output_dir"]).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output / (args.mode + "_" + stamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    canonical_config = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    write_json(run_dir / "environment.json", {
        **snapshot_version(), "device": str(device),
        "config_sha256": hashlib.sha256(canonical_config).hexdigest(),
        "base_commit": config.get("provenance", {}).get("base_commit"),
    })
    print(f"Subjects: {available}; device: {device}; output: {run_dir}", flush=True)
    manifests = []
    for subject in exp["subjects"]:
        loaders = make_loaders(
            subjects, subject, exp["batch_size"], exp["seeds"][0],
            ss_mix=exp["ss_mix"], mixup_alpha=exp["mixup_alpha"],
        )
        manifests.append(loaders["manifest"])
    write_json(run_dir / "data_audit.json", manifests)
    print("DATA AUDIT PASSED. Target training loader contains features + row indices only.", flush=True)
    if args.mode == "audit":
        print("Audit finished. Set MODE='train' in run_anml_pycharm.py to train manually.", flush=True)
        return
    results = []
    try:
        for subject in exp["subjects"]:
            for seed in exp["seeds"]:
                for variant in exp["variants"]:
                    setup_seed(seed)
                    loaders = make_loaders(
                        subjects, subject, exp["batch_size"], seed,
                        ss_mix=exp["ss_mix"], mixup_alpha=exp["mixup_alpha"],
                    )
                    folder = run_dir / f"{variant}_subject_{subject}_seed_{seed}"
                    folder.mkdir()
                    write_json(folder / "manifest.json", loaders["manifest"])
                    model = SDCModel(n_classes=config["data"]["num_classes"]).to(device)
                    initialize_weights(model)
                    print(f"\nSTART variant={variant} subject={subject} seed={seed}", flush=True)
                    thresholds = train_fixed(
                        model, loaders["source"], loaders["target"], exp,
                        config["relation"], device, folder, variant,
                    )
                    # Save the fixed final model BEFORE any target-label evaluation.
                    torch.save({
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "epoch": exp["epochs"], "variant": variant, "subject": subject,
                        "seed": seed, "config": config, "final_training_thresholds": thresholds,
                    }, folder / "final.pt")
                    metrics, truth, pred, prob = evaluate_final(
                        model, loaders["eval"], device, config["data"]["num_classes"]
                    )
                    write_json(folder / "metrics.json", metrics)
                    prediction_rows = [
                        {"row": i, "true": int(truth[i]), "predicted": int(pred[i]),
                         "confidence": float(prob[i].max()),
                         **{f"prob_{c}": float(prob[i, c]) for c in range(prob.shape[1])}}
                        for i in range(len(truth))
                    ]
                    write_csv(folder / "predictions.csv", prediction_rows)
                    results.append({
                        "variant": variant, "subject": subject, "seed": seed,
                        "epoch": exp["epochs"], "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"], "output_folder": folder.name,
                    })
                    save_summary(run_dir, results)
                    diagnose_frozen_pairs(
                        prob, truth, variant, config["relation"], thresholds, exp, seed, folder
                    )
                    print(f"FINAL {variant} subject={subject} seed={seed}: "
                          f"accuracy={100*metrics['accuracy']:.2f}% "
                          f"macro-F1={metrics['macro_f1']:.4f}", flush=True)
                    del model, loaders
        write_json(run_dir / "completion.json", {"status": "completed", "runs": len(results)})
    except BaseException as exc:
        write_json(run_dir / "completion.json", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "error_type": type(exc).__name__, "message": str(exc),
            "completed_runs": len(results),
        })
        raise
    print(f"\nCompleted. Results: {run_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
