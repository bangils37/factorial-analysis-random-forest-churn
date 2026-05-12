"""
run_crd.py
==========
Completely Randomized Design (CRD) experiment.

Factor
------
    k (number of CV folds) ∈ {3, 5, 10}

Fixed settings
--------------
    max_depth = None   (unlimited — not a factor in CRD)
    n_repeats = 10
    random_state = 1234 (everywhere)

Output
------
    results/crd_results.csv

Columns
-------
    repeat_id  : 1-indexed repetition number (1..n_repeats)
    fold_id    : 1-indexed fold within a repetition (1..k)
    k          : number of splits used for this run
    f1_score   : F1-score (pos_label='yes') on the validation fold
"""

import sys
from pathlib import Path

# Allow running as a script from the project root without installing src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import (
    K_VALUES,
    MAX_DEPTH_VALUES,
    RANDOM_SEED,
    REPEATS,
    RESULTS_DIR,
    F1_POS_LABEL,
)
from src.data_loader import load_data
from src.model import build_rf
from src.utils import runtime_tracker


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_crd() -> pd.DataFrame:
    """Execute the CRD experiment and return a tidy DataFrame of results.

    Returns
    -------
    pd.DataFrame
        Columns: ``repeat_id``, ``fold_id``, ``k``, ``f1_score``.
    """
    print("=" * 60)
    print("CRD Experiment — Factor: k (CV folds)")
    print(f"  k values    : {K_VALUES}")
    print(f"  max_depth   : None  (fixed, not a factor)")
    print(f"  n_repeats   : {REPEATS}")
    print(f"  random_state: {RANDOM_SEED}")
    print("=" * 60)

    X, y = load_data()
    print(f"\nDataset loaded: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"Class distribution:\n{y.value_counts().to_string()}\n")

    records: list[dict] = []

    with runtime_tracker("CRD Full Execution"):
        for k in K_VALUES:
        print(f"Running k={k} ...")
        cv = RepeatedStratifiedKFold(
            n_splits=k,
            n_repeats=REPEATS,
            random_state=RANDOM_SEED,
        )

        # RepeatedStratifiedKFold generates n_repeats × k splits in order
        split_idx = 0
        for repeat in range(1, REPEATS + 1):
            for fold in range(1, k + 1):
                # Manually iterate so we can track repeat_id and fold_id
                pass

        # Re-iterate properly: sklearn yields splits sequentially
        for split_no, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            repeat_id = (split_no // k) + 1          # 1-indexed repetition
            fold_id = (split_no % k) + 1             # 1-indexed fold

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = build_rf(max_depth=None)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            score = f1_score(y_val, y_pred, pos_label=F1_POS_LABEL)

            records.append(
                {
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "k": k,
                    "f1_score": round(score, 6),
                }
            )

        total_splits = REPEATS * k
        print(f"  -> {total_splits} splits completed.")

    results_df = pd.DataFrame(records)
    return results_df


def save_results(df: pd.DataFrame) -> Path:
    """Save *df* to ``results/crd_results.csv``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "crd_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")
    print(f"Total rows: {len(df)}")
    print(df.groupby("k")["f1_score"].agg(["mean", "std"]).round(4))
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_crd()
    save_results(results)
