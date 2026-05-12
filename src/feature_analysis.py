"""
feature_analysis.py
===================
Tools for exploring inter-feature correlations and removing redundant features
before training.

Public functions
----------------
compute_correlation_matrix(X)
    Return the Pearson correlation matrix of X.

plot_correlation_heatmap(X, *, save_path, title, figsize)
    Draw and optionally save a styled heatmap.

remove_highly_correlated_features(df, threshold=0.85)
    Drop one column from each pair that exceeds the correlation threshold.

check_data_leakage(df, target_col)
    Perform a heuristic check for potential data leakage features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Return the absolute Pearson correlation matrix of numeric columns in X.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric columns only; non-numeric are silently ignored).

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix with values in [0, 1].
    """
    numeric_X = X.select_dtypes(include="number")
    corr = numeric_X.corr(method="pearson").abs()
    return corr


# ---------------------------------------------------------------------------
# Heatmap visualisation
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    X: pd.DataFrame,
    *,
    save_path: Optional[str | Path] = None,
    title: str = "Feature Correlation Heatmap",
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
) -> plt.Figure:
    """Draw a Pearson correlation heatmap for the features in *X*.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    save_path : str | Path | None
        If provided, the figure is saved to this path (PNG).
        If *None*, the figure is saved to ``plots/correlation_heatmap.png``.
    title : str
        Figure title.
    figsize : tuple[int, int]
        Matplotlib figure size ``(width, height)`` in inches.
    cmap : str
        Seaborn / Matplotlib colour map name.
    annot : bool
        Whether to annotate each cell with its value.
    fmt : str
        String format for annotation values.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object (caller may call ``plt.show()`` if
        running in an interactive environment).
    """
    corr = compute_correlation_matrix(X)

    # Build mask for upper triangle (avoids redundant information)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    fig.tight_layout()

    # Resolve save path
    out_path = (
        Path(save_path)
        if save_path is not None
        else PLOTS_DIR / "correlation_heatmap.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Feature selection: remove highly correlated features
# ---------------------------------------------------------------------------

def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove one feature from every highly correlated pair.

    Strategy
    --------
    For each pair (i, j) where ``|corr(i, j)| >= threshold``, the feature
    with the *higher mean absolute correlation* with all other features is
    dropped.  This greedy approach is deterministic because columns are
    evaluated in their original order.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix **without** the target column.
    threshold : float
        Absolute Pearson correlation above which a feature is considered
        redundant.  Default is ``0.85``.

    Returns
    -------
    df_reduced : pd.DataFrame
        Copy of *df* with redundant columns removed.
    dropped_cols : list[str]
        Names of the dropped columns (useful for reporting).

    Examples
    --------
    >>> X_reduced, dropped = remove_highly_correlated_features(X, threshold=0.85)
    >>> print(f"Dropped {len(dropped)} features: {dropped}")
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1]. Got {threshold!r}.")

    corr_matrix = compute_correlation_matrix(df)
    dropped_cols: list[str] = []
    columns = list(corr_matrix.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_i, col_j = columns[i], columns[j]
            # Skip if one of the columns was already dropped
            if col_i in dropped_cols or col_j in dropped_cols:
                continue

            if corr_matrix.loc[col_i, col_j] >= threshold:
                # Drop the column with higher mean absolute correlation
                mean_i = corr_matrix[col_i].mean()
                mean_j = corr_matrix[col_j].mean()
                to_drop = col_i if mean_i >= mean_j else col_j
                dropped_cols.append(to_drop)

    df_reduced = df.drop(columns=dropped_cols)
    return df_reduced, dropped_cols


# ---------------------------------------------------------------------------
# Leakage Check
# ---------------------------------------------------------------------------

def check_data_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.95) -> list[str]:
    """Perform a heuristic check for potential data leakage features.
    
    A feature is considered a potential leak if it has an extremely high
    correlation with the target variable, indicating it might be a proxy
    for the target or recorded after the target event occurred.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full dataset including the target column.
    target_col : str
        The name of the target column.
    threshold : float
        The absolute correlation threshold above which a feature is flagged.
        
    Returns
    -------
    list[str]
        A list of column names flagged as potential leakage.
    """
    if target_col not in df.columns:
        return []
        
    # Create a temporary numeric version of the target if it's categorical
    temp_df = df.copy()
    if temp_df[target_col].dtype == object or temp_df[target_col].dtype.name in ("string", "str"):
        from sklearn.preprocessing import LabelEncoder
        temp_df[target_col] = LabelEncoder().fit_transform(temp_df[target_col])
        
    numeric_df = temp_df.select_dtypes(include="number")
    if target_col not in numeric_df.columns:
        return []
        
    correlations = numeric_df.corr(method="pearson")[target_col].abs()
    # Drop the target itself
    correlations = correlations.drop(index=target_col)
    
    leaky_features = correlations[correlations >= threshold].index.tolist()
    return leaky_features
