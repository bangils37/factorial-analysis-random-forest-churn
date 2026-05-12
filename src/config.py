"""
config.py
=========
Project-wide constants and paths.

All experimental constraints are enforced here as single sources of truth:
  - RANDOM_SEED   : 1234 (used in every random call)
  - K_VALUES      : cross-validation fold counts to test
  - MAX_DEPTH_VALUES : RF max_depth levels to test
  - REPEATS       : number of repetitions per CV strategy
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
PLOTS_DIR: Path = PROJECT_ROOT / "plots"

# Default dataset filename (can be overridden at runtime)
RAW_DATA_FILENAME: str = "mlc_churn.csv"
RAW_DATA_PATH: Path = DATA_RAW_DIR / RAW_DATA_FILENAME

# ---------------------------------------------------------------------------
# Reproducibility seed — MUST be used everywhere (model, CV split, etc.)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 1234

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL: str = "churn"
POSITIVE_CLASS: str = "yes"

# ---------------------------------------------------------------------------
# Experimental Design factors
# ---------------------------------------------------------------------------
# CRD factor: number of folds k ∈ {3, 5, 10}
K_VALUES: list[int] = [3, 5, 10]

# CRFD additional factor: max_depth ∈ {3, 5, None}
MAX_DEPTH_VALUES: list[int | None] = [3, 5, None]

# Number of repetitions for RepeatedStratifiedKFold
REPEATS: int = 10

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
SCORING_METRIC: str = "f1"          # sklearn scorer key
F1_POS_LABEL: str = POSITIVE_CLASS  # pos_label for f1_score()
