"""
generate_feature_importance.py
==============================
Trains a Random Forest classifier with max_depth=None on the reduced Churn dataset
to extract and plot Gini feature importances.
"""

import sys
import shutil
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import RANDOM_SEED, PLOTS_DIR
from src.data_loader import load_data
from src.feature_analysis import (
    remove_highly_correlated_features,
    compute_feature_importance,
    plot_feature_importance,
)
from src.model import build_rf


def main():
    print("=" * 60)
    print("GENERATE FEATURE IMPORTANCE")
    print(f"  random_state: {RANDOM_SEED}")
    print("=" * 60)

    # 1. Load data
    X, y = load_data()
    print(f"Original dataset: {X.shape[0]} rows × {X.shape[1]} features")

    # 2. Preprocess: drop highly correlated features
    X_red, dropped_cols = remove_highly_correlated_features(X, threshold=0.95)
    print(f"Dropped highly correlated features (threshold=0.95): {dropped_cols}")
    print(f"Reduced dataset: {X_red.shape[0]} rows × {X_red.shape[1]} features")

    # 3. Build & train Random Forest (best configuration: max_depth=None)
    model = build_rf(max_depth=None)
    print("Training Random Forest model ...")
    model.fit(X_red, y)

    # 4. Compute feature importances
    print("Computing Gini feature importances ...")
    importances = compute_feature_importance(model, X_red.columns.tolist())
    print("\nFeature Importances (Sorted):")
    for feat, imp in importances.items():
        print(f"  {feat:<30}: {imp:.4f}")

    # 5. Plot and save to plots/
    print("\nPlotting feature importances ...")
    plot_path = PLOTS_DIR / "feature_importance.png"
    plot_feature_importance(importances, save_path=plot_path)
    print(f"Saved plot -> {plot_path}")

    # 6. Copy to reports/latex/images/
    latex_img_dir = Path(__file__).resolve().parent.parent / "reports" / "latex" / "images"
    if latex_img_dir.exists():
        latex_dest = latex_img_dir / "feature_importance.png"
        shutil.copy(plot_path, latex_dest)
        print(f"Copied plot to LaTeX assets -> {latex_dest}")

    print("\nFeature Importance Generation Completed Successfully!")


if __name__ == "__main__":
    main()
