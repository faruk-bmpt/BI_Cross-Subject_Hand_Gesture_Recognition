#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_2_SUBJECT_INDEPENDENT.py
# ================================================================
# BLOCK 2 — SUBJECT-INDEPENDENT (SI) FIGURES
# ================================================================
# PURPOSE: Generate publication-ready figures for LOSO evaluation
#
# IMPORTANT: All confusion matrices are derived from TRUE LOSO
#            cross-subject evaluation (Block 0 results), NOT
#            from final deployment models.
#
# FIGURES (SI1-SI11):
#   SI1: Calibration Curve (Accuracy vs Shots)
#   SI2: LOSO Distribution Boxplot
#   SI3: Confusion Matrix — 0-Shot (Ensemble, LOSO)
#   SI4: Confusion Matrix — 10-Shot (Ensemble, LOSO)
#   SI5: Calibration Gain Bar Chart
#   SI6: Per-Subject LOSO Accuracy (10-shot)
#   SI7: Accuracy vs Shots with 95% CI
#   SI8: Radar Chart — Gesture-wise Recall (LOSO)
#   SI9: Model Comparison Under LOSO (XGB vs RF vs Ensemble)
#
# TABLES (SI1-SI4):
#   SI1: LOSO Results Summary
#   SI2: Metrics Summary (Accuracy, F1, MCC)
#   SI3: Per-Gesture Recall (LOSO)
#   SI4: Model Comparison Summary (LOSO)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================================================================
# CONFIGURATION
# ================================================================
RESULTS_DIR = Path("./Results")
LOSO_DIR = RESULTS_DIR / "BI_LOSO_Results"
SD_DIR = RESULTS_DIR / "BI_SD_Results"
FIG_DIR = Path("./Figures/BI_Block2_SI")
TAB_DIR = Path("./Tables/BI_Block2_SI")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ['HOC', 'TFM', 'IFM', 'MFM', 'RFM', 'LFM', 'PH']
RANDOM_CHANCE = 100 / len(GESTURES)  # 14.2857%

print("="*80)
print("BI_BLOCK_2: SUBJECT-INDEPENDENT (SI) FIGURES")
print("LOSO Evaluation Results — 405-Feature Bioimpedance Pipeline")
print("="*80)

# ================================================================
# PART 2.0 — DATA LOADING WITH VALIDATION
# ================================================================

# Load LOSO results
loso_results = pd.read_csv(LOSO_DIR / "BI_LOSO_Results.csv")
shots = loso_results['Shots'].values
accuracies = loso_results['Accuracy'].astype(float).values
stds = loso_results['Std'].astype(float).values
f1_scores = loso_results['F1'].astype(float).values
mcc_scores = loso_results['MCC'].astype(float).values

# FIX 2: Validate per-model columns exist - NO FALLBACK
if 'XGB_Accuracy' not in loso_results.columns:
    raise ValueError("XGB_Accuracy column missing from LOSO results. "
                     "Please re-run Block 0 with per-model accuracy saving enabled.")
if 'RF_Accuracy' not in loso_results.columns:
    raise ValueError("RF_Accuracy column missing from LOSO results. "
                     "Please re-run Block 0 with per-model accuracy saving enabled.")

xgb_accs = loso_results['XGB_Accuracy'].astype(float).values
rf_accs = loso_results['RF_Accuracy'].astype(float).values

# Load SD upper bound
sd_results = pd.read_csv(SD_DIR / "BI_SD_Results.csv")
avg_ens_sd = sd_results['Ensemble_Acc'].astype(float).mean()

# Number of subjects
n_subj_file = LOSO_DIR / "BI_LOSO_SubjectWise_0shot.csv"
n_subjects = len(pd.read_csv(n_subj_file)) if n_subj_file.exists() else 18

print(f"\n✅ Loaded LOSO results:")
print(f"   Calibration levels: {len(shots)} (0 to {max(shots)}-shot)")
print(f"   Subjects: {n_subjects}")
print(f"   SD upper bound: {avg_ens_sd:.1f}%")

# ================================================================
# PART 2.1 — LOAD LOSO CONFUSION MATRICES (TRUE CROSS-SUBJECT)
# ================================================================

# Load ensemble confusion matrices from Block 0 LOSO evaluation
cm0_ensemble = pd.read_csv(LOSO_DIR / "BI_CM_0shot.csv", index_col=0)
cm10_ensemble = pd.read_csv(LOSO_DIR / "BI_CM_10shot.csv", index_col=0)

# Normalize (safe division)
cm0_ens_norm = cm0_ensemble.values.astype('float') / (cm0_ensemble.values.sum(axis=1)[:, np.newaxis] + 1e-9)
cm10_ens_norm = cm10_ensemble.values.astype('float') / (cm10_ensemble.values.sum(axis=1)[:, np.newaxis] + 1e-9)

print(f"✅ Loaded TRUE LOSO confusion matrices (0-shot, 10-shot)")

# ================================================================
# FIGURE SI1: Calibration Curve
# ================================================================
print("\n📊 BI_SI1: Calibration Curve")
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(shots, accuracies, yerr=stds, fmt='o-', color='#1f77b4', 
            linewidth=2.5, markersize=10, capsize=5, capthick=2)
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle='--', 
           linewidth=2, label=f'Random Chance ({RANDOM_CHANCE:.1f}%)')
ax.axhline(y=avg_ens_sd, color='green', linestyle='--', 
           linewidth=2, label=f'SD Upper Bound: {avg_ens_sd:.1f}%')
ax.set_xlabel('Calibration Shots Per Gesture', fontsize=12)
ax.set_ylabel('Cross-Subject Accuracy (%)', fontsize=12)
ax.set_title('BI_SI1: Multi-Shot Calibration Performance — LOSO Evaluation', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI1_Calibration_Curve.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI1_Calibration_Curve.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI2: LOSO Distribution Boxplot
# ================================================================
print("\n📊 BI_SI2: LOSO Distribution")
data_to_plot = []
valid_shots = []
for s in shots:
    f = LOSO_DIR / f"BI_LOSO_SubjectWise_{int(s)}shot.csv"
    if f.exists():
        data_to_plot.append(pd.read_csv(f)['Accuracy'].values)
        valid_shots.append(s)
fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(data_to_plot, labels=[f'{int(s)}-shot' for s in valid_shots], 
                patch_artist=True, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Calibration Shots Per Gesture', fontsize=12)
ax.set_title('BI_SI2: LOSO Accuracy Distribution Across Subjects', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI2_LOSO_Distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI2_LOSO_Distribution.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI3: Confusion Matrix — 0-Shot (Ensemble, TRUE LOSO)
# ================================================================
print("\n📊 BI_SI3: Confusion Matrix — 0-Shot (Ensemble, LOSO)")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm0_ens_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=GESTURES, yticklabels=GESTURES, ax=ax,
            cbar_kws={'label': 'Normalized Proportion'})
ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)
ax.set_title(f'BI_SI3: Confusion Matrix — 0-Shot (Ensemble, LOSO)\nAccuracy: {accuracies[0]:.1f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI3_Confusion_0shot.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI3_Confusion_0shot.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI4: Confusion Matrix — 10-Shot (Ensemble, TRUE LOSO)
# ================================================================
print("\n📊 BI_SI4: Confusion Matrix — 10-Shot (Ensemble, LOSO)")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm10_ens_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=GESTURES, yticklabels=GESTURES, ax=ax,
            cbar_kws={'label': 'Normalized Proportion'})
ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)
ax.set_title(f'BI_SI4: Confusion Matrix — 10-Shot (Ensemble, LOSO)\nAccuracy: {accuracies[-1]:.1f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI4_Confusion_10shot.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI4_Confusion_10shot.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI5: Calibration Gain Bar Chart
# ================================================================
print("\n📊 BI_SI5: Calibration Gain")
baseline = accuracies[0]
improvements = accuracies - baseline
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar([f'{int(s)}-shot' for s in shots[1:]], improvements[1:],
              color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='black', linewidth=1.5)
ax.set_ylabel('Improvement Over 0-shot (%)', fontsize=12)
ax.set_xlabel('Calibration Shots Per Gesture', fontsize=12)
ax.set_title('BI_SI5: Calibration Gain Relative to 0-Shot Baseline — LOSO', 
             fontsize=14, fontweight='bold')
for bar, imp in zip(bars, improvements[1:]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'+{imp:.1f}%', ha='center', fontweight='bold', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI5_Calibration_Gain.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI5_Calibration_Gain.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI6: Per-Subject Accuracy (10-shot)
# ================================================================
print("\n📊 BI_SI6: Per-Subject Accuracy (10-shot)")
df = pd.read_csv(LOSO_DIR / "BI_LOSO_SubjectWise_10shot.csv")
df_sorted = df.sort_values('Accuracy', ascending=False)
mean_acc = df_sorted['Accuracy'].mean()
colors_subj = ['green' if a > mean_acc else 'red' for a in df_sorted['Accuracy']]
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(df_sorted['Subject'].astype(str), df_sorted['Accuracy'],
              color=colors_subj, alpha=0.7, edgecolor='black')
ax.axhline(y=mean_acc, color='blue', linestyle='--', linewidth=2,
           label=f'Mean: {mean_acc:.1f}%')
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle=':', linewidth=1.5,
           label=f'Random Chance: {RANDOM_CHANCE:.1f}%')
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_ylabel('10-Shot Cross-Subject Accuracy (%)', fontsize=12)
ax.set_title('BI_SI6: Per-Subject LOSO Accuracy — 10-Shot Calibration', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, df_sorted['Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.0f}%', ha='center', fontsize=8, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI6_Per_Subject_Accuracy.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI6_Per_Subject_Accuracy.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI7: Accuracy vs Shots with 95% CI
# ================================================================
print("\n📊 BI_SI7: Accuracy vs Shots with 95% CI")
ci95 = [1.96 * s / np.sqrt(n_subjects) for s in stds]
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(shots, accuracies, yerr=ci95, fmt='o-', color='#1f77b4',
            linewidth=2.5, markersize=10, capsize=5)
ax.fill_between(shots, [m - c for m, c in zip(accuracies, ci95)],
                [m + c for m, c in zip(accuracies, ci95)], alpha=0.2)
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle='--', 
           linewidth=2, label=f'Random Chance ({RANDOM_CHANCE:.1f}%)')
ax.set_xlabel('Calibration Shots Per Gesture', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_SI7: Accuracy Improvement with 95% Confidence Interval — LOSO', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI7_Accuracy_vs_Shots.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI7_Accuracy_vs_Shots.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI8: Radar Chart — Gesture-wise Recall (LOSO)
# ================================================================
print("\n📊 BI_SI8: Radar Chart — Gesture-wise Recall (LOSO)")
recall_0 = [cm0_ensemble.iloc[i, i] / (cm0_ensemble.iloc[i, :].sum() + 1e-9) * 100 
            for i in range(len(GESTURES))]
recall_10 = [cm10_ensemble.iloc[i, i] / (cm10_ensemble.iloc[i, :].sum() + 1e-9) * 100 
             for i in range(len(GESTURES))]
recall_0 += recall_0[:1]
recall_10 += recall_10[:1]
angles = np.linspace(0, 2*np.pi, len(GESTURES), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.plot(angles, recall_0, 'o-', linewidth=2, label='0-shot', color='#d62728', markersize=8)
ax.fill(angles, recall_0, alpha=0.25, color='#d62728')
ax.plot(angles, recall_10, 'o-', linewidth=2, label='10-shot', color='#2ca02c', markersize=8)
ax.fill(angles, recall_10, alpha=0.25, color='#2ca02c')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(GESTURES, fontsize=10, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
ax.set_title('BI_SI8: Gesture-wise Recall Comparison — 0-Shot vs 10-Shot (LOSO)', 
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI8_Radar_Chart.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI8_Radar_Chart.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# FIGURE SI9: Model Comparison Under LOSO (XGB vs RF vs Ensemble)
# ================================================================
print("\n📊 BI_SI9: Model Comparison Under LOSO")
shot_mask = np.isin(shots, [0, 1, 3, 5, 7, 10])
shots_plot = shots[shot_mask]
xgb_plot = xgb_accs[shot_mask]
rf_plot = rf_accs[shot_mask]
ens_plot = accuracies[shot_mask]
x_idx = np.arange(len(shots_plot))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x_idx - width, xgb_plot, width, label='XGBoost', 
            color='#1f77b4', alpha=0.75, edgecolor='black')
b2 = ax.bar(x_idx, rf_plot, width, label='Random Forest', 
            color='#ff7f0e', alpha=0.75, edgecolor='black')
b3 = ax.bar(x_idx + width, ens_plot, width, label='Ensemble', 
            color='#2ca02c', alpha=0.75, edgecolor='black')
ax.set_ylabel('Cross-Subject Accuracy (%)', fontsize=12)
ax.set_xlabel('Calibration Shots Per Gesture', fontsize=12)
ax.set_title('BI_SI9: Model Comparison Under LOSO — XGBoost vs RF vs Ensemble', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x_idx)
ax.set_xticklabels([f'{int(s)}-shot' for s in shots_plot])
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3, axis='y')
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f'{h:.1f}', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SI9_Model_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SI9_Model_Comparison.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved")

# ================================================================
# TABLES
# ================================================================

# Table SI1: LOSO Results Summary
print("\n📊 BI_Table_SI1: LOSO Results Summary")
loso_results.to_csv(TAB_DIR / "BI_Table_SI1_LOSO_Results.csv", index=False)
print("  ✅ Saved")

# Table SI2: Metrics Summary
print("\n📊 BI_Table_SI2: Metrics Summary")
metrics_df = pd.DataFrame({
    'Shots': shots,
    'Accuracy (%)': [f'{a:.2f}' for a in accuracies],
    'Std (%)': [f'{s:.2f}' for s in stds],
    'F1 (%)': [f'{f1:.2f}' for f1 in f1_scores],
    'MCC': [f'{mcc:.3f}' for mcc in mcc_scores]
})
metrics_df.to_csv(TAB_DIR / "BI_Table_SI2_Metrics.csv", index=False)
print(metrics_df.to_string(index=False))
print("  ✅ Saved")

# Table SI3: Per-Gesture Recall (LOSO)
print("\n📊 BI_Table_SI3: Per-Gesture Recall (LOSO)")
gesture_recall = []
for i, g in enumerate(GESTURES):
    r0 = cm0_ensemble.iloc[i, i] / (cm0_ensemble.iloc[i, :].sum() + 1e-9) * 100
    r10 = cm10_ensemble.iloc[i, i] / (cm10_ensemble.iloc[i, :].sum() + 1e-9) * 100
    gesture_recall.append({
        'Gesture': g,
        '0-shot Recall (%)': f'{r0:.1f}',
        '10-shot Recall (%)': f'{r10:.1f}',
        'Improvement (%)': f'{r10 - r0:.1f}'
    })
recall_df = pd.DataFrame(gesture_recall)
recall_df.to_csv(TAB_DIR / "BI_Table_SI3_PerGesture_Recall.csv", index=False)
print(recall_df.to_string(index=False))
print("  ✅ Saved")

# Table SI4: Model Comparison Summary (LOSO)
print("\n📊 BI_Table_SI4: Model Comparison Summary")
model_comp_df = pd.DataFrame({
    'Shots': shots_plot,
    'XGBoost Acc (%)': [f'{v:.2f}' for v in xgb_plot],
    'Random Forest Acc (%)': [f'{v:.2f}' for v in rf_plot],
    'Ensemble Acc (%)': [f'{v:.2f}' for v in ens_plot],
    'Ensemble Gain vs XGB (pp)': [f'{e - x:.2f}' for e, x in zip(ens_plot, xgb_plot)],
    'Ensemble Gain vs RF (pp)': [f'{e - r:.2f}' for e, r in zip(ens_plot, rf_plot)]
})
model_comp_df.to_csv(TAB_DIR / "BI_Table_SI4_Model_Comparison.csv", index=False)
print(model_comp_df.to_string(index=False))
print("  ✅ Saved")

# ================================================================
# COMPLETION SUMMARY
# ================================================================
print("\n" + "="*80)
print("BI_BLOCK_2 COMPLETED SUCCESSFULLY")
print("="*80)
print("\nGenerated Outputs (TRUE LOSO Cross-Subject Evaluation):")
print(f"\n  Figures (SI1-SI9):")
print(f"    → SI1: Calibration Curve")
print(f"    → SI2: LOSO Distribution Boxplot")
print(f"    → SI3: Confusion Matrix 0-shot (Ensemble, LOSO)")
print(f"    → SI4: Confusion Matrix 10-shot (Ensemble, LOSO)")
print(f"    → SI5: Calibration Gain")
print(f"    → SI6: Per-Subject Accuracy (10-shot)")
print(f"    → SI7: Accuracy vs Shots with 95% CI")
print(f"    → SI8: Radar Chart (Gesture-wise Recall)")
print(f"    → SI9: Model Comparison (XGB vs RF vs Ensemble)")
print(f"\n  Tables (SI1-SI4):")
print(f"    → {TAB_DIR}/BI_Table_SI1_LOSO_Results.csv")
print(f"    → {TAB_DIR}/BI_Table_SI2_Metrics.csv")
print(f"    → {TAB_DIR}/BI_Table_SI3_PerGesture_Recall.csv")
print(f"    → {TAB_DIR}/BI_Table_SI4_Model_Comparison.csv")
print("\n" + "="*80)
print("SUBJECT-INDEPENDENT ANALYSIS COMPLETE")
print("All figures use TRUE LOSO cross-subject evaluation")
print("="*80)