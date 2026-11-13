"""Strict MAT loading and transductive LOSO loaders.

Processed: dataset_sessionN.feature [N,310], .label [N,1].
Raw: de_LDS1..15 (SEED) or ..24 (SEED-IV), each [62,time,5].
SS-Mix needs explicit trial IDs; no boundaries are inferred from labels.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SubjectData:
    subject_id: str
    features: np.ndarray
    labels: np.ndarray
    trial_ids: object
    path: str
    label_encoding: str
    normalization: str
    num_classes: int


def _integer_vector(values, description):
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description}: expected numeric integers.") from exc
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{description}: empty values, NaN or infinity.")
    if not np.equal(array, np.rint(array)).all():
        raise ValueError(f"{description}: contains non-integer values.")
    return array.astype(np.int64)


def _read_mat(path):
    try:
        return loadmat(str(path), struct_as_record=False, squeeze_me=False)
    except NotImplementedError as exc:
        raise ValueError(f"{path}: MATLAB v7.3 unsupported; re-save as MAT v7.") from exc
    except Exception as exc:
        raise ValueError(f"Cannot read MAT file {path}: {exc}") from exc


def _resolve(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _subject_id(stem):
    match = re.match(r"^(\d+)(?:_|$)", stem)
    return str(int(match.group(1))) if match else stem


def _sort_key(path):
    sid = _subject_id(path.stem)
    return (0, int(sid), path.name) if sid.isdigit() else (1, sid, path.name)


def _labels(values, n_samples, n_classes, encoding, path):
    array = np.asarray(values)
    if array.ndim == 2 and array.shape == (n_samples, n_classes):
        if not np.isin(array, [0, 1]).all() or not (array.sum(1) == 1).all():
            raise ValueError(f"{path}: label matrix is not strict one-hot.")
        return array.argmax(1).astype(np.int64), "one_hot_zero_based"
    labels = _integer_vector(array, f"{path}: labels")
    if len(labels) != n_samples:
        raise ValueError(f"{path}: {n_samples} feature rows but {len(labels)} labels.")
    schemes = {
        "zero_based": (set(range(n_classes)), 0),
        "one_based": (set(range(1, n_classes + 1)), -1),
    }
    if n_classes == 3:
        schemes["seed_signed"] = ({-1, 0, 1}, 1)
    observed = set(labels.tolist())
    if encoding == "auto":
        matches = [name for name, (allowed, _) in schemes.items() if observed == allowed]
        if len(matches) != 1:
            raise ValueError(
                f"{path}: ambiguous/incomplete label set {sorted(observed)}. "
                "Set label_encoding explicitly; auto requires all classes."
            )
        encoding = matches[0]
    if encoding not in schemes:
        raise ValueError(f"Unsupported label_encoding: {encoding}")
    allowed, offset = schemes[encoding]
    if not observed.issubset(allowed):
        raise ValueError(f"{path}: labels {sorted(observed)} conflict with {encoding}.")
    return labels + offset, encoding


def _normalize(features, method, path):
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: features must be numeric.") from exc
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError(f"{path}: expected nonempty [samples, features].")
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: features contain NaN or infinity.")
    if method == "minmax":
        low, high = x.min(0), x.max(0)
        span = high - low
        x = 2.0 * (x - low) / np.where(span > 0, span, 1.0) - 1.0
    elif method == "zscore":
        mean, std = x.mean(0), x.std(0)
        x = (x - mean) / np.where(std > 0, std, 1.0)
    elif method != "none":
        raise ValueError(f"Unsupported normalization: {method}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: features overflow float32.")
    return x


def _trial_metadata(config, root):
    if not config.get("trial_metadata"):
        return {}
    path = _resolve(root, config["trial_metadata"])
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("confirmed") is not True or not metadata.get("source"):
        raise ValueError(f"{path}: require confirmed=true and a source description.")
    if not isinstance(metadata.get("subjects"), dict):
        raise ValueError(f"{path}: require a subjects object.")
    return metadata["subjects"]


def _trials(explicit_ids, entry, n_samples, description):
    ids = None
    if explicit_ids is not None:
        ids = _integer_vector(explicit_ids, f"{description}: trial IDs")
    if entry is not None:
        if not isinstance(entry, dict):
            raise ValueError(f"{description}: trial metadata must be an object.")
        if ("trial_ids" in entry) == ("trial_lengths" in entry):
            raise ValueError(f"{description}: provide exactly one trial_ids/trial_lengths.")
        if "trial_ids" in entry:
            sidecar_ids = _integer_vector(entry["trial_ids"], description)
        else:
            lengths = _integer_vector(entry["trial_lengths"], description)
            if (lengths <= 0).any() or int(lengths.sum()) != n_samples:
                raise ValueError(f"{description}: positive lengths must sum to {n_samples}.")
            sidecar_ids = np.repeat(np.arange(len(lengths)), lengths)
        if ids is not None and not np.array_equal(ids, sidecar_ids):
            raise ValueError(f"{description}: MAT and JSON trial IDs disagree.")
        ids = sidecar_ids
    if ids is not None and (len(ids) != n_samples or (ids < 0).any()):
        raise ValueError(f"{description}: require {n_samples} nonnegative trial IDs.")
    return ids


def load_subjects(config):
    """One session; file IDs are used instead of os.listdir positions."""
    if not config.get("data_root"):
        raise ValueError("Set data_root to a single-session MAT folder.")
    root = Path(config["data_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    session = int(config.get("session", 1))
    n_classes = int(config.get("num_classes", 3))
    if session not in (1, 2, 3) or n_classes not in (3, 4):
        raise ValueError("Expected session 1/2/3 and num_classes 3/4.")
    layout = config.get("data_format", "processed")
    if layout not in ("processed", "raw_de_LDS"):
        raise ValueError("data_format must be processed or raw_de_LDS.")
    encoding = config.get("label_encoding", "auto")
    normalization = config.get("normalization", "minmax")
    expected_dim = int(config.get("expected_features", 310))
    metadata = _trial_metadata(config, root)
    label_path, raw_labels = None, None
    if layout == "raw_de_LDS":
        if not config.get("label_file"):
            raise ValueError("raw_de_LDS requires explicit label_file.")
        label_path = _resolve(root, config["label_file"])
        label_mat = _read_mat(label_path)
        label_key = config.get("label_key", "label")
        if label_key not in label_mat:
            raise ValueError(f"{label_path}: missing label key {label_key!r}.")
        raw_labels = _integer_vector(label_mat[label_key], str(label_path))
    paths = sorted(
        [p for p in root.glob("*.mat") if p.resolve() != label_path], key=_sort_key
    )
    if len(paths) < 2:
        raise ValueError(f"{root}: LOSO requires at least two subject MAT files.")
    expected_count = config.get("expected_subjects")
    if expected_count is not None and len(paths) != int(expected_count):
        raise ValueError(f"{root}: expected {expected_count} subjects, found {len(paths)}.")
    subjects, seen = [], set()
    for path in paths:
        sid = _subject_id(path.stem)
        if sid in seen:
            raise ValueError(f"Duplicate subject ID {sid}; use one session only.")
        seen.add(sid)
        mat = _read_mat(path)
        if layout == "processed":
            key = f"dataset_session{session}"
            if key not in mat or np.asarray(mat[key]).size != 1:
                raise ValueError(f"{path}: missing scalar MAT struct {key}.")
            record = np.asarray(mat[key]).reshape(-1)[0]
            if not hasattr(record, "feature") or not hasattr(record, "label"):
                raise ValueError(f"{path}: {key} needs feature and label fields.")
            features, labels = record.feature, record.label
            explicit_ids = getattr(record, "trial_ids", getattr(record, "trial_id", None))
        else:
            indices = sorted(int(k[6:]) for k in mat if re.fullmatch(r"de_LDS\d+", k))
            n_trials = 15 if n_classes == 3 else 24
            if indices != list(range(1, n_trials + 1)):
                raise ValueError(f"{path}: require de_LDS1 ... de_LDS{n_trials}.")
            if len(raw_labels) != n_trials:
                raise ValueError(f"{label_path}: expected {n_trials} trial labels.")
            blocks, label_blocks, trial_blocks = [], [], []
            for index in indices:
                array = np.asarray(mat[f"de_LDS{index}"])
                if array.ndim != 3 or array.shape[0] != 62 or array.shape[2] != 5:
                    raise ValueError(f"{path}: de_LDS{index} must be [62,T,5].")
                count = array.shape[1]
                if count == 0:
                    raise ValueError(f"{path}: de_LDS{index} is empty.")
                blocks.append(array.transpose(1, 2, 0).reshape(count, 310))
                label_blocks.append(np.full(count, raw_labels[index - 1]))
                trial_blocks.append(np.full(count, index - 1))
            features, labels = np.concatenate(blocks), np.concatenate(label_blocks)
            explicit_ids = np.concatenate(trial_blocks)
        x = _normalize(features, normalization, path)
        if x.shape[1] != expected_dim:
            raise ValueError(f"{path}: expected {expected_dim} features, got {x.shape[1]}.")
        y, used_encoding = _labels(labels, len(x), n_classes, encoding, path)
        ids = _trials(explicit_ids, metadata.get(sid), len(x), str(path))
        subjects.append(SubjectData(sid, x, y, ids, str(path),
                                    used_encoding, normalization, n_classes))
    return subjects


def make_loaders(subjects, target_subject, batch_size, seed,
                 ss_mix=False, mixup_alpha=0.5):
    """Target training TensorDataset contains only features and row indices."""
    target_id = _subject_id(str(target_subject))
    matches = [s for s in subjects if s.subject_id == target_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown subject {target_id!r}; available: "
                         f"{[s.subject_id for s in subjects]}")
    target = matches[0]
    sources = [s for s in subjects if s.subject_id != target_id]
    if not sources or batch_size < 2:
        raise ValueError("Require a source and batch_size >= 2.")
    if ss_mix and (not np.isfinite(mixup_alpha) or mixup_alpha <= 0):
        raise ValueError("SS-Mix requires finite mixup_alpha > 0.")
    rng = np.random.default_rng(int(seed))
    feature_parts, label_parts = [], []
    for subject in sources:
        x, y = subject.features, subject.labels
        feature_parts.append(x)
        label_parts.append(y)
        if ss_mix:
            if subject.trial_ids is None:
                raise ValueError(
                    f"{subject.path}: SS-Mix requires confirmed trial metadata. "
                    "Provide trial_metadata/raw_de_LDS, or set ss_mix=False."
                )
            augmented = np.empty_like(x)
            for trial in np.unique(subject.trial_ids):
                rows = np.flatnonzero(subject.trial_ids == trial)
                if len(np.unique(y[rows])) != 1:
                    raise ValueError(f"{subject.path}: trial {trial} has multiple labels.")
                a = rng.choice(rows, size=len(rows), replace=True)
                b = rng.choice(rows, size=len(rows), replace=True)
                lam = rng.beta(mixup_alpha, mixup_alpha, size=(len(rows), 1))
                augmented[rows] = lam * x[a] + (1.0 - lam) * x[b]
            feature_parts.append(augmented)
            label_parts.append(y.copy())
    source_x, source_y = np.concatenate(feature_parts), np.concatenate(label_parts)
    if min(len(source_x), len(target.features)) < batch_size:
        raise ValueError("batch_size exceeds source/target rows with drop_last=True.")
    source_ds = TensorDataset(torch.from_numpy(source_x), torch.from_numpy(source_y))
    target_x = torch.from_numpy(target.features)
    target_ds = TensorDataset(target_x, torch.arange(len(target_x), dtype=torch.long))
    eval_ds = TensorDataset(target_x, torch.from_numpy(target.labels))
    source_loader = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
        generator=torch.Generator().manual_seed(int(seed) + 1),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    manifest = {
        "protocol": "transductive_loso", "target_subject": target_id,
        "source_subjects": [s.subject_id for s in sources],
        "input_dim": int(target.features.shape[1]), "num_classes": target.num_classes,
        "normalization": target.normalization,
        "normalization_scope": "each subject independently; all unlabeled target rows",
        "ss_mix": bool(ss_mix), "mixup_alpha": float(mixup_alpha) if ss_mix else None,
        "source_rows_before_augmentation": sum(len(s.features) for s in sources),
        "source_rows_after_augmentation": len(source_x),
        "target_rows": len(target.features),
        "subjects": [
            {"subject_id": s.subject_id, "path": s.path, "rows": len(s.features),
             "label_encoding": s.label_encoding,
             "trial_metadata_available": s.trial_ids is not None}
            for s in subjects
        ],
    }
    return {"source": source_loader, "target": target_loader,
            "eval": eval_loader, "manifest": manifest}
