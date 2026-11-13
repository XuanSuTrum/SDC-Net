"""Open this file in PyCharm, edit the settings below, then click Run.

First run MODE="audit". After DATA AUDIT PASSED, use MODE="train".
Command-line arguments, when provided, override these launcher settings.
"""
from pathlib import Path
import sys

# ======================== EDIT THESE SETTINGS ========================
DATA_ROOT = r"F:\Emotion_datasets\SEED\feature_for_net_session1_LDS_de"
MODE = "audit"                  # "audit" first; then "train"
VARIANTS = ["B"]                 # A: no DSCL; B: revised DSCL; C: DANML
SUBJECTS = ["1"]                 # File ID 1.mat / 1_date.mat, not zero-based
SEEDS = [0]                     # Later use [0, 1, 2] for the same variants
EPOCHS = 200                     # Fixed final epoch; no target-best selection
BATCH_SIZE = 32
DEVICE = "auto"                  # "auto", "cpu", "cuda:0"
CONFIG = Path(__file__).resolve().parent / "configs" / "anml_pilot.json"
# Other options (raw MAT, labels, SS-Mix, loss parameters): edit CONFIG.
# ====================================================================


if __name__ == "__main__":
    from experiments.anml.train import main

    if len(sys.argv) > 1:
        main()
    else:
        main([
            "--config", str(CONFIG),
            "--data-root", DATA_ROOT,
            "--mode", MODE,
            "--variants", *VARIANTS,
            "--subjects", *[str(x) for x in SUBJECTS],
            "--seeds", *[str(x) for x in SEEDS],
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--device", DEVICE,
        ])
