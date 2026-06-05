import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from src.config import RESULTS_DIR, PLOTS_DIR, PROJECT_ROOT

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

def run_crd_analysis():
    file_path = RESULTS_DIR / "crd_results.csv"
    if not file_path.exists(): return
    
    df = pd.read_csv(file_path)
    print("\n" + "="*50)
    print("ANALYSIS: CRD (One-way ANOVA on k)")
    print("="*50)

    # 1. 95% Confidence Intervals of the Mean (t-distribution)
    print("\n95% Confidence Intervals of the Mean:")
    ci_stats = df.groupby('k')['f1_score'].agg(['mean', 'std', 'count']).round(4)
    for k, row in ci_stats.iterrows():
        mean, std, n = row['mean'], row['std'], int(row['count'])
        if n > 1 and std > 0:
            h = stats.t.ppf(0.975, df=n-1) * (std / np.sqrt(n))
            ci_lower = mean - h
            ci_upper = mean + h
        else:
            ci_lower, ci_upper = mean, mean
        print(f"  k={k:2d}: Mean={mean:.4f}, Std={std:.4f}, N={n} | 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 2. Assumption Checks
    print("\nAssumption Checks:")
    model = ols('f1_score ~ C(k)', data=df).fit()
    
    # - Normality: Shapiro-Wilk on residuals
    stat_sw, p_sw = stats.shapiro(model.resid)
    print(f"  Shapiro-Wilk (Normality): Statistic={stat_sw:.4f}, p-value={p_sw:.4f}")
    if p_sw < 0.05:
        print("    -> Warning: Residuals may not be normally distributed.")
        
    # - Homogeneity: Levene Test
    groups = [group["f1_score"].values for name, group in df.groupby("k")]
    stat_lev, p_lev = stats.levene(*groups)
    print(f"  Levene Test (Homogeneity): Statistic={stat_lev:.4f}, p-value={p_lev:.4f}")
    if p_lev < 0.05:
        print("    -> Warning: Variances may not be equal.")

    # 3. One-way ANOVA & Effect Size (eta_squared)
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Calculate eta squared: SS_between / SS_total
    ss_between = anova_table.loc['C(k)', 'sum_sq']
    ss_total = anova_table['sum_sq'].sum()
    eta_sq = ss_between / ss_total
    anova_table['eta_sq'] = [eta_sq, np.nan]
    
    print("\nOne-way ANOVA Table (with Effect Size):")
    print(anova_table)

    # Welch's ANOVA (Robust check since Levene test is rejected and sizes are unbalanced)
    welch_res = anova_oneway(df['f1_score'], df['k'], use_var='unequal')
    print("\nWelch's ANOVA (Robust to Heteroscedasticity & Unbalanced Groups):")
    print(f"  F-statistic = {welch_res.statistic:.4f}")
    print(f"  p-value     = {welch_res.pvalue:.4f}")
    print(f"  df (num, denom) = ({welch_res.df_num:.1f}, {welch_res.df_denom:.2f})")

    # 4. Tukey HSD & Tukey Plot
    tukey = pairwise_tukeyhsd(endog=df['f1_score'], groups=df['k'], alpha=0.05)
    print("\nTukey HSD Post-hoc Test:")
    print(tukey)
    
    # Tukey Plot
    fig = tukey.plot_simultaneous(xlabel='F1-score', ylabel='k-folds')
    plt.title("Tukey HSD Plot: Pairwise Comparisons (CRD)", fontsize=14)
    plt.savefig(PLOTS_DIR / "tukey_plot.png")
    plt.close(fig)

    # 5. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='k', y='f1_score', data=df, hue='k', palette="Set2", showmeans=True, legend=False)
    plt.title("F1-score distribution by k (CRD)", fontsize=14, fontweight='bold')
    plt.savefig(PLOTS_DIR / "crd_boxplot.png")
    plt.close()

def run_crfd_analysis():
    file_path = RESULTS_DIR / "crfd_results.csv"
    if not file_path.exists(): return
    
    # keep_default_na=False ensures the string "None" isn't parsed as NaN
    df = pd.read_csv(file_path, keep_default_na=False)
    df['max_depth'] = df['max_depth'].astype(str)
    
    print("\n" + "="*50)
    print("ANALYSIS: CRFD (Two-way ANOVA: k * max_depth)")
    print("="*50)

    # 1. 95% Confidence Intervals of the Mean (t-distribution)
    print("\n95% Confidence Intervals of the Mean:")
    ci_stats = df.groupby(['k', 'max_depth'])['f1_score'].agg(['mean', 'std', 'count']).round(4)
    for index, row in ci_stats.iterrows():
        mean, std, n = row['mean'], row['std'], int(row['count'])
        if n > 1 and std > 0:
            h = stats.t.ppf(0.975, df=n-1) * (std / np.sqrt(n))
            ci_lower = mean - h
            ci_upper = mean + h
        else:
            ci_lower, ci_upper = mean, mean
        print(f"  k={index[0]:2d}, max_depth={index[1]:<4}: Mean={mean:.4f}, Std={std:.4f}, N={n} | 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 2. Assumption Check
    model = ols('f1_score ~ C(k) * C(max_depth)', data=df).fit()
    stat_sw, p_sw = stats.shapiro(model.resid)
    print(f"\nAssumption Checks:")
    print(f"  Shapiro-Wilk (Normality): Statistic={stat_sw:.4f}, p-value={p_sw:.4f}")

    # 3. Two-way ANOVA & Partial Effect Size (partial_eta_squared)
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Calculate partial eta squared: SS_effect / (SS_effect + SS_residual)
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_residual)
    anova_table.loc['Residual', 'partial_eta_sq'] = np.nan
    
    print("\nTwo-way ANOVA Table (with Partial Effect Size):")
    print(anova_table)

    # 4. Interaction Plot
    plt.figure(figsize=(12, 7))
    sns.pointplot(x='k', y='f1_score', hue='max_depth', data=df, capsize=.1, errorbar=('ci', 95))
    plt.title("Interaction: k vs max_depth on F1-score", fontsize=14, fontweight='bold')
    plt.savefig(PLOTS_DIR / "interaction_plot.png")
    plt.close()

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    original_stdout = sys.stdout
    with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        try:
            run_crd_analysis()
            run_crfd_analysis()
        finally:
            sys.stdout = original_stdout
    
    print("Analysis completed. Results saved in analysis_summary.txt")
    
    # Automatically copy plots to LaTeX images directory
    import shutil
    latex_img_dir = PROJECT_ROOT / "reports" / "latex" / "images"
    if latex_img_dir.exists():
        copied_files = []
        for plot_name in ["crd_boxplot.png", "tukey_plot.png", "interaction_plot.png"]:
            src = PLOTS_DIR / plot_name
            if src.exists():
                shutil.copy(src, latex_img_dir / plot_name)
                copied_files.append(plot_name)
        if copied_files:
            print(f"Automatically copied plots to LaTeX images: {copied_files}")
