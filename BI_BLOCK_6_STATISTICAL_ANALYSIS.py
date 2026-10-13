#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_6_STATISTICAL_ANALYSIS.py
# ================================================================
# BLOCK 6 — STATISTICAL ANALYSIS
# ================================================================
# TESTS: Wilcoxon (one-sided), Friedman, Confidence Intervals, Cohen's d
# FIGURES: ST1 (Boxplot), ST2 (Effect Size)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
from pathlib import Path

RESULTS_DIR = Path("./Results")
LOSO_DIR = RESULTS_DIR / "BI_LOSO_Results"
FIG_DIR = Path("./Figures/BI_Block6_Statistics")
TAB_DIR = Path("./Tables/BI_Block6_Statistics")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BI_BLOCK_6: STATISTICAL ANALYSIS")
print("="*80)

shots = [0, 1, 3, 5, 7, 10]
subject_accs = {}
for s in shots:
    f = LOSO_DIR / f"BI_LOSO_SubjectWise_{s}shot.csv"
    if f.exists():
        subject_accs[s] = pd.read_csv(f)['Accuracy'].values
        print(f"✅ Loaded {s}-shot: {len(subject_accs[s])} subjects")

# Wilcoxon signed-rank tests (one-sided: calibration > baseline)
print("\n📊 Wilcoxon Signed-Rank Tests (one-sided, calibration > 0-shot):")
wilcoxon_results = []
baseline = subject_accs[0]
for s in [1, 3, 5, 7, 10]:
    if s in subject_accs:
        stat, p = wilcoxon(subject_accs[s], baseline, alternative='greater')
        sig = 'Yes' if p < 0.05 else 'No'
        if p < 0.001:
            sig_marker = '***'
        elif p < 0.01:
            sig_marker = '**'
        elif p < 0.05:
            sig_marker = '*'
        else:
            sig_marker = 'ns'
        wilcoxon_results.append({
            'Comparison': f'0 vs {s}-shot',
            'Statistic': f'{stat:.3f}',
            'p-value': f'{p:.6f}',
            'Significant (α=0.05)': sig,
            'Significance': sig_marker
        })
        print(f"  0 vs {s}-shot: stat={stat:.3f}, p={p:.6f} "
              f"{'★' + sig_marker if p < 0.05 else '(not significant)'}")

# Friedman test
print("\n📊 Friedman Test:")
all_accs = [subject_accs[s] for s in shots if s in subject_accs]
friedman_result = []
if len(all_accs) >= 3:
    min_len = min(len(a) for a in all_accs)
    aligned = [a[:min_len] for a in all_accs]
    stat_f, p_f = friedmanchisquare(*aligned)
    sig_f = 'Yes' if p_f < 0.05 else 'No'
    print(f"  χ²={stat_f:.3f}, p={p_f:.6f} {'★' + ('***' if p_f < 0.001 else '**' if p_f < 0.01 else '*') if p_f < 0.05 else '(not significant)'}")
    friedman_result = [{
        'Test': 'Friedman',
        'Chi-square': f'{stat_f:.3f}',
        'p-value': f'{p_f:.6f}',
        'Degrees of Freedom': len(all_accs) - 1,
        'Significant (α=0.05)': sig_f
    }]

# 95% Confidence Intervals
print("\n📊 95% Confidence Intervals:")
ci_results = []
for s in shots:
    if s in subject_accs:
        mean_v = np.mean(subject_accs[s])
        sem = np.std(subject_accs[s], ddof=1) / np.sqrt(len(subject_accs[s]))
        ci95 = 1.96 * sem
        ci_results.append({
            'Shots': s,
            'Mean Accuracy (%)': f'{mean_v:.2f}',
            'Standard Error': f'{sem:.2f}',
            '95% CI': f'±{ci95:.2f}',
            'CI Lower': f'{mean_v - ci95:.2f}',
            'CI Upper': f'{mean_v + ci95:.2f}'
        })
        print(f"  {s}-shot: {mean_v:.2f}% ± {ci95:.2f}%")

# Paired Cohen's d (effect size)
print("\n📊 Effect Sizes (Paired Cohen's d):")
effect_rows = []
for s in [1, 3, 5, 7, 10]:
    if s in subject_accs:
        diff = subject_accs[s] - baseline
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)
        if abs(d) > 0.8:
            interp = 'Large'
        elif abs(d) > 0.5:
            interp = 'Medium'
        elif abs(d) > 0.2:
            interp = 'Small'
        else:
            interp = 'Very Small'
        effect_rows.append({
            'Comparison': f'0 vs {s}-shot',
            "Cohen's d (paired)": f'{d:.3f}',
            'Interpretation': interp
        })
        print(f"  0 vs {s}-shot: d={d:.3f} ({interp})")

# Figure ST1: Statistical Boxplot
print("\n📊 Figure BI_ST1: Statistical Boxplot")
data = [subject_accs[s] for s in shots if s in subject_accs]
labels = [f'{s}-shot' for s in shots if s in subject_accs]
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_ST1: Statistical Distribution of LOSO Accuracy Across Calibration Levels', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_ST1_Statistical_Boxplot.png", dpi=300)
plt.savefig(FIG_DIR / "BI_ST1_Statistical_Boxplot.pdf")
plt.close()
print("  ✅ Saved")

# Figure ST2: Effect Size Bar Chart
if effect_rows:
    print("\n📊 Figure BI_ST2: Effect Size Chart")
    eff_df = pd.DataFrame(effect_rows)
    eff_d = eff_df["Cohen's d (paired)"].astype(float).values
    eff_lbl = eff_df['Comparison'].values
    colors_eff = ['#2ca02c' if abs(d) > 0.8 else '#ff7f0e' if abs(d) > 0.5 else '#1f77b4' for d in eff_d]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(eff_lbl, eff_d, color=colors_eff, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, label='Large (d=0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Medium (d=0.5)')
    ax.axhline(y=0.2, color='blue', linestyle=':', linewidth=1.5, label='Small (d=0.2)')
    ax.set_ylabel("Cohen's d (paired)", fontsize=12)
    ax.set_xlabel("Comparison", fontsize=12)
    ax.set_title("BI_ST2: Effect Size — Each Calibration Level vs 0-Shot Baseline", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d_val in zip(bars, eff_d):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{d_val:.2f}', ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "BI_ST2_Effect_Size.png", dpi=300)
    plt.savefig(FIG_DIR / "BI_ST2_Effect_Size.pdf")
    plt.close()
    print("  ✅ Saved")

# Save tables
pd.DataFrame(wilcoxon_results).to_csv(TAB_DIR / "BI_Table_Wilcoxon.csv", index=False)
if friedman_result:
    pd.DataFrame(friedman_result).to_csv(TAB_DIR / "BI_Table_Friedman.csv", index=False)
pd.DataFrame(ci_results).to_csv(TAB_DIR / "BI_Table_Confidence_Intervals.csv", index=False)
if effect_rows:
    pd.DataFrame(effect_rows).to_csv(TAB_DIR / "BI_Table_Effect_Sizes.csv", index=False)

print("\n✅ BI_BLOCK_6 COMPLETED")