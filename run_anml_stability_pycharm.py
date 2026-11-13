"""Run a fixed A/B/source-relation-only seed check manually in PyCharm.

Reuses DATA_ROOT and DEVICE from run_anml_pycharm.py.
Defaults: subject 1, seeds 1 and 2, three cases, six fresh 200-epoch runs.
See docs/ANML_STABILITY_CHECK.md.
"""
from datetime import datetime, timezone
from pathlib import Path
import uuid

from run_anml_pycharm import DATA_ROOT, DEVICE

# ======================== EDIT THESE SETTINGS ========================
CASES = ["A", "B", "source_relation_only"]
SUBJECTS = ["1"]
SEEDS = [1, 2]                  # Keep the existing seed-0 runs for comparison.
EPOCHS = 200
BATCH_SIZE = 32
# ====================================================================

ROOT = Path(__file__).resolve().parent
# B must use the full baseline config, including its target DSCL term.
CASE_CONFIGS = {
    "A": (ROOT / "configs" / "anml_pilot.json", "A"),
    "B": (ROOT / "configs" / "anml_pilot.json", "B"),
    "source_relation_only": (
        ROOT / "configs" / "anml_source_relation_only.json", "C"
    ),
}


def run_stability():
    if not CASES or any(case not in CASE_CONFIGS for case in CASES):
        raise ValueError("CASES must contain A, B and/or source_relation_only.")
    if len(CASES) != len(set(CASES)):
        raise ValueError("Duplicate entries in CASES.")
    if not SUBJECTS or not SEEDS:
        raise ValueError("SUBJECTS and SEEDS must be nonempty.")
    missing = [
        str(CASE_CONFIGS[case][0])
        for case in CASES
        if not CASE_CONFIGS[case][0].is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing config files: " + ", ".join(missing))

    from experiments.anml.train import main

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_root = ROOT / "outputs" / "anml_stability" / ("batch_" + stamp)
    total_runs = len(CASES) * len(SUBJECTS) * len(SEEDS)
    print(f"SEED CHECK: {total_runs} fresh runs; fixed final epoch {EPOCHS}.", flush=True)
    print("Batch output: " + str(batch_root), flush=True)
    for index, case in enumerate(CASES, start=1):
        config_path, variant = CASE_CONFIGS[case]
        print(f"\nCASE {index}/{len(CASES)}: {case}; seeds={SEEDS}", flush=True)
        main([
            "--config", str(config_path),
            "--data-root", str(DATA_ROOT),
            "--mode", "train",
            "--variants", variant,
            "--subjects", *[str(x) for x in SUBJECTS],
            "--seeds", *[str(x) for x in SEEDS],
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--device", str(DEVICE),
            "--output-dir", str(batch_root / case),
        ])
    print("\nRequested seed check completed. Zip this batch folder: "
          + str(batch_root), flush=True)


if __name__ == "__main__":
    run_stability()
