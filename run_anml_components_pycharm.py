"""Run source-only and target-only RELATION diagnostics manually in PyCharm.

DATA_ROOT and DEVICE are reused from your existing run_anml_pycharm.py settings.
Both cases keep the common MMD/CMMD framework and start from fresh seed-0 models.
See docs/ANML_COMPONENT_DIAGNOSTICS.md.
"""
from pathlib import Path

from run_anml_pycharm import DATA_ROOT, DEVICE

# ======================== EDIT THESE SETTINGS ========================
CASES = ["source_relation_only", "target_relation_only"]
SUBJECTS = ["1"]
SEEDS = [0]
EPOCHS = 200
BATCH_SIZE = 32
# ====================================================================

ROOT = Path(__file__).resolve().parent
CASE_CONFIGS = {
    "source_relation_only": ROOT / "configs" / "anml_source_relation_only.json",
    "target_relation_only": ROOT / "configs" / "anml_target_relation_only.json",
}


def run_components():
    if not CASES or any(case not in CASE_CONFIGS for case in CASES):
        raise ValueError("CASES must contain source_relation_only and/or target_relation_only.")
    if len(CASES) != len(set(CASES)):
        raise ValueError("Duplicate entries in CASES.")
    missing = [str(CASE_CONFIGS[case]) for case in CASES if not CASE_CONFIGS[case].is_file()]
    if missing:
        raise FileNotFoundError("Missing component config files: " + ", ".join(missing))

    from experiments.anml.train import main

    for case in CASES:
        print("\nCOMPONENT DIAGNOSTIC: " + case, flush=True)
        main([
            "--config", str(CASE_CONFIGS[case]),
            "--data-root", str(DATA_ROOT),
            "--mode", "train",
            "--variants", "C",
            "--subjects", *[str(x) for x in SUBJECTS],
            "--seeds", *[str(x) for x in SEEDS],
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--device", str(DEVICE),
        ])
    print("\nRequested component diagnostics completed. Results: "
          + str(ROOT / "outputs" / "anml_components"), flush=True)


if __name__ == "__main__":
    run_components()
