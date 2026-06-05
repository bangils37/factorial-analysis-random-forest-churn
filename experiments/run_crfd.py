"""
run_crfd.py
===========
Completely Randomized Factorial Design (CRFD) experiment.

Factors
-------
    Factor A — k (number of CV folds)    ∈ {3, 5, 10}
    Factor B — max_depth (RF tree depth) ∈ {3, 5, None}

Fixed settings
--------------
    n_repeats    = 10
    random_state = 1234 (everywhere)

Output
------
    results/crfd_results.csv

Columns
-------
    repeat_id  : 1-indexed repetition number (1..n_repeats)
    fold_id    : 1-indexed fold within a repetition (1..k)
    k          : number of splits used for this combination
    max_depth  : tree depth used for this combination  (integer or 'None')
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
    F1_POS_LABEL,
    K_VALUES,
    MAX_DEPTH_VALUES,
    RANDOM_SEED,
    REPEATS,
    RESULTS_DIR,
)
from src.data_loader import load_data
from src.model import build_rf
from src.utils import runtime_tracker


from src.feature_analysis import remove_highly_correlated_features


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_crfd() -> pd.DataFrame:
    """Execute the CRFD experiment and return a tidy DataFrame of results.

    Returns
    -------
    pd.DataFrame
        Columns: ``repeat_id``, ``fold_id``, ``k``, ``max_depth``,
        ``f1_score``.
    """
    print("=" * 60)
    print("CRFD Experiment — Factors: k × max_depth")
    print(f"  k values        : {K_VALUES}")
    print(f"  max_depth values: {MAX_DEPTH_VALUES}")
    print(f"  n_repeats       : {REPEATS}")
    print(f"  random_state    : {RANDOM_SEED}")
    print("=" * 60)

    X, y = load_data()
    X, dropped_cols = remove_highly_correlated_features(X, threshold=0.95)
    print(f"\nHighly correlated columns dropped: {dropped_cols}")
    print(f"Dataset loaded: {X.shape[0]} rows × {X.shape[1]} features (after dropping)")
    print(f"Class distribution:\n{y.value_counts().to_string()}\n")

    records: list[dict] = []

    with runtime_tracker("CRFD Full Execution"):
        for k in K_VALUES:
            for max_depth in MAX_DEPTH_VALUES:
                depth_label = str(max_depth) if max_depth is not None else "None"
                print(f"Running k={k}, max_depth={depth_label} ...")

                cv = RepeatedStratifiedKFold(
                    n_splits=k,
                    n_repeats=REPEATS,
                    random_state=RANDOM_SEED,
                )

                for split_no, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    repeat_id = (split_no // k) + 1      # 1-indexed repetition
                    fold_id = (split_no % k) + 1         # 1-indexed fold

                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = build_rf(max_depth=max_depth)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    score = f1_score(y_val, y_pred, pos_label=F1_POS_LABEL)

                    records.append(
                        {
                            "repeat_id": repeat_id,
                            "fold_id": fold_id,
                            "k": k,
                            "max_depth": depth_label,   # 'None' as string for CSV readability
                            "f1_score": round(score, 6),
                        }
                    )

                total_splits = REPEATS * k
                print(f"  -> {total_splits} splits completed.")

    results_df = pd.DataFrame(records)
    return results_df


def save_results(df: pd.DataFrame) -> Path:
    """Save *df* to ``results/crfd_results.csv``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "crfd_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved -> {out_path}")
    print(f"Total rows: {len(df)}")
    print(
        df.groupby(["k", "max_depth"])["f1_score"]
        .agg(["mean", "std"])
        .round(4)
    )
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_crfd()
    save_results(results)
