"""
data_loader.py
==============
Responsible for loading and splitting the raw Churn dataset.

Usage
-----
    from src.data_loader import load_data

    X, y = load_data()          # uses default path from config
    X, y = load_data("other.csv")
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PROCESSED_DIR,
    F1_POS_LABEL,
    RAW_DATA_PATH,
    TARGET_COL,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    csv_path: str | Path | None = None,
    *,
    drop_non_numeric_ids: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the churn dataset, encode categorical features, and return X, y.

    Parameters
    ----------
    csv_path : str | Path | None
        Path to the raw CSV file. Defaults to ``config.RAW_DATA_PATH``.
    drop_non_numeric_ids : bool
        When *True* (default), the ``state`` and ``area_code`` columns are
        dropped because they are high-cardinality nominal identifiers that
        add noise without contributing predictive value for this study.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except TARGET_COL), ready for sklearn.
    y : pd.Series
        Target vector with string labels ``'yes'`` / ``'no'``.
    """
    path = Path(csv_path) if csv_path is not None else RAW_DATA_PATH
    df = pd.read_csv(path)

    # --- optional: drop high-cardinality nominal IDs ---
    if drop_non_numeric_ids:
        id_cols = [c for c in ("state", "area_code") if c in df.columns]
        df = df.drop(columns=id_cols)

    # --- encode binary yes/no columns (everything except target) ---
    # Note: pandas 2.x may infer string columns as StringDtype (dtype.name == 'str')
    # rather than classic 'object', so we check for both.
    binary_cols = [
        c for c in df.columns
        if c != TARGET_COL and (
            df[c].dtype == object or df[c].dtype.name in ("string", "str")
        )
    ]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # --- split X / y ---
    y: pd.Series = df[TARGET_COL].copy()           # keep 'yes'/'no' strings
    X: pd.DataFrame = df.drop(columns=[TARGET_COL])

    return X, y


def save_processed(X: pd.DataFrame, y: pd.Series, filename: str = "churn_processed.csv") -> Path:
    """Persist the processed dataset to ``data/processed/``.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target series.
    filename : str
        Output filename (saved inside ``config.DATA_PROCESSED_DIR``).

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / filename
    df_out = X.copy()
    df_out[TARGET_COL] = y.values
    df_out.to_csv(out_path, index=False)
    return out_path
