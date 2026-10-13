#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_3_SUBJECT_DEPENDENT.py
# ================================================================
# BLOCK 3 — SUBJECT-DEPENDENT (SD) FIGURES
# ================================================================
# FIGURES (SD1-SD5, sequential, for PUBLICATION):
#   SD1: Model Comparison (Bar Chart) — XGBoost vs RF vs Ensemble
#   SD2: Per-Subject Accuracy — Ensemble performance per subject
#   SD3: Protocol Comparison — Subject-Dependent vs LOSO 10-shot
#   SD4: Cross-Validated Ensemble Confusion Matrix (PUBLICATION)
#   SD5: Model Comparison with 95% CI
#
# INTERNAL STUDY FIGURES (saved separately in Internal_Study/):
#   - XGBoost Deployment Confusion Matrix (Internal Study)
#   - Random Forest Deployment Confusion Matrix (Internal Study)
#   - Ensemble Cross-Validated Confusion Matrix (Internal Study copy)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import joblib
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)

# ================================================================
# CONFIGURATION
# ================================================================
RESULTS_DIR = Path("./Results")
SD_DIR = RESULTS_DIR / "BI_SD_Results"
LOSO_DIR = RESULTS_DIR / "BI_LOSO_Results"
PREDS_DIR = RESULTS_DIR / "BI_Predictions"
MODELS_DIR = RESULTS_DIR / "BI_Models"
CACHE_DIR = Path("./Cached_Data")

# Publication directories
FIG_DIR = Path("./Figures/BI_Block3_SD")
TAB_DIR = Path("./Tables/BI_Block3_SD")

# Internal study directory
INTERNAL_DIR = Path("./Internal_Study/Confusion_Matrices")

# Create directories
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
INTERNAL_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ['HOC', 'TFM', 'IFM', 'MFM', 'RFM', 'LFM', 'PH']
RANDOM_CHANCE = 100 / len(GESTURES)  # 14.3%

print("="*80)
print("BI_BLOCK_3: SUBJECT-DEPENDENT (SD) FIGURES")
print("Publication figures → Figures/BI_Block3_SD/")
print("Internal study figures → Internal_Study/Confusion_Matrices/")
print("="*80)

# ================================================================
# LOAD SD RESULTS
# ================================================================
sd_results = pd.read_csv(SD_DIR / "BI_SD_Results.csv")
avg_xgb = sd_results['XGBoost_Acc'].astype(float).mean()
avg_rf = sd_results['RF_Acc'].astype(float).mean()
avg_ens = sd_results['Ensemble_Acc'].astype(float).mean()

loso_results = pd.read_csv(LOSO_DIR / "BI_LOSO_Results.csv")
loso_10shot = float(loso_results[loso_results['Shots'] == 10]['Accuracy'].values[0])

print(f"✅ SD: XGB={avg_xgb:.1f}%, RF={avg_rf:.1f}%, Ens={avg_ens:.1f}%")
print(f"✅ LOSO 10-shot: {loso_10shot:.1f}%")

# ================================================================
# LOAD ENSEMBLE PREDICTIONS (CROSS-VALIDATED SD) - FOR PUBLICATION
# ================================================================
with open(PREDS_DIR / "BI_SD_predictions.pkl", 'rb') as f:
    ens_preds = pickle.load(f)

# Reconstruct ensemble predictions from cross-validated folds
ens_all_true = []
ens_all_pred = []
for sub_id, folds in ens_preds.items():
    for fold in folds:
        ens_all_true.extend(fold['true'])
        ens_all_pred.extend(fold['pred'])

# Compute confusion matrix from cross-validated predictions
ens_cm = confusion_matrix(ens_all_true, ens_all_pred, labels=range(len(GESTURES)))
ens_cm_norm = ens_cm.astype('float') / (ens_cm.sum(axis=1)[:, np.newaxis] + 1e-9)
ens_acc = ens_cm.trace() / ens_cm.sum() * 100

print(f"\n✅ Cross-validated Ensemble confusion matrix: {ens_acc:.1f}%")

# ================================================================
# GENERATE DEPLOYMENT MODEL CONFUSION MATRICES (FOR INTERNAL STUDY)
# ================================================================
print("\n📊 Generating deployment model confusion matrices (Internal Study)...")

# Load final models
xgb_model = joblib.load(MODELS_DIR / "BI_xgb_final.pkl")
rf_model = joblib.load(MODELS_DIR / "BI_rf_final.pkl")
scaler = joblib.load(MODELS_DIR / "BI_scaler.pkl")

# Load data
X = np.load(CACHE_DIR / "BI_X.npy")
y_enc = np.load(CACHE_DIR / "BI_y_enc.npy")

# Scale data
X_scaled = scaler.transform(X)

# Get predictions from final models
xgb_pred_all = xgb_model.predict(X_scaled)
rf_pred_all = rf_model.predict(X_scaled)

# Compute confusion matrices
xgb_cm = confusion_matrix(y_enc, xgb_pred_all, labels=range(len(GESTURES)))
xgb_cm_norm = xgb_cm.astype('float') / (xgb_cm.sum(axis=1)[:, np.newaxis] + 1e-9)
xgb_acc = accuracy_score(y_enc, xgb_pred_all) * 100

rf_cm = confusion_matrix(y_enc, rf_pred_all, labels=range(len(GESTURES)))
rf_cm_norm = rf_cm.astype('float') / (rf_cm.sum(axis=1)[:, np.newaxis] + 1e-9)
rf_acc = accuracy_score(y_enc, rf_pred_all) * 100

print(f"   XGBoost (deployment model): {xgb_acc:.1f}%")
print(f"   Random Forest (deployment model): {rf_acc:.1f}%")
# ================================================================
# INTERNAL STUDY: Save deployment confusion matrices
# ================================================================

# ------------------------------------------------------------------
# Random Forest Deployment Confusion Matrix
# ------------------------------------------------------------------
print("\n📊 Internal Study: Random Forest Deployment Confusion Matrix")

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    rf_cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=GESTURES,
    yticklabels=GESTURES,
    ax=ax,
    cbar_kws={'label': 'Normalized Proportion'}
)

ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)

ax.set_title(
    'Random Forest Deployment Confusion Matrix (Internal Study)\n'
    f'Accuracy: {rf_acc:.1f}% (trained on full dataset)',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    INTERNAL_DIR / "RF_CM.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    INTERNAL_DIR / "RF_CM.pdf",
    bbox_inches='tight'
)

plt.close()

print("  ✅ Saved to Internal_Study/Confusion_Matrices/RF_CM.png/pdf")

# ------------------------------------------------------------------
# XGBoost Deployment Confusion Matrix
# ------------------------------------------------------------------
print("\n📊 Internal Study: XGBoost Deployment Confusion Matrix")

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    xgb_cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=GESTURES,
    yticklabels=GESTURES,
    ax=ax,
    cbar_kws={'label': 'Normalized Proportion'}
)

ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)

ax.set_title(
    'XGBoost Deployment Confusion Matrix (Internal Study)\n'
    f'Accuracy: {xgb_acc:.1f}% (trained on full dataset)',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    INTERNAL_DIR / "XGB_CM.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    INTERNAL_DIR / "XGB_CM.pdf",
    bbox_inches='tight'
)

plt.close()

print("  ✅ Saved to Internal_Study/Confusion_Matrices/XGB_CM.png/pdf")

# ------------------------------------------------------------------
# Ensemble Cross-Validated Confusion Matrix
# ------------------------------------------------------------------
print("\n📊 Internal Study: Ensemble Cross-Validated Confusion Matrix")

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    ens_cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Blues',
    xticklabels=GESTURES,
    yticklabels=GESTURES,
    ax=ax,
    cbar_kws={'label': 'Normalized Proportion'}
)

ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)

ax.set_title(
    'Ensemble Cross-Validated Confusion Matrix (Internal Study)\n'
    f'Accuracy: {ens_acc:.1f}% (from 5-fold subject-dependent CV)',
    fontsize=13,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    INTERNAL_DIR / "Ensemble_CM.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    INTERNAL_DIR / "Ensemble_CM.pdf",
    bbox_inches='tight'
)

plt.close()

print("  ✅ Saved to Internal_Study/Confusion_Matrices/Ensemble_CM.png/pdf")

# ================================================================
# PUBLICATION FIGURES (SD1-SD5)
# ================================================================

# ----------------------------------------------------------------------
# FIGURE SD1: Model Comparison (Bar Chart)
# ----------------------------------------------------------------------
print("\n📊 BI_SD1: Model Comparison (Publication)")
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['XGBoost', 'Random Forest', 'Ensemble'], [avg_xgb, avg_rf, avg_ens],
              color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_SD1: Model Comparison — Subject-Dependent Evaluation', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle='--', 
           linewidth=1.5, label=f'Random Chance ({RANDOM_CHANCE:.1f}%)')
for bar, acc in zip(bars, [avg_xgb, avg_rf, avg_ens]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SD1_Model_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SD1_Model_Comparison.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved to Figures/BI_Block3_SD/BI_SD1_Model_Comparison.png/pdf")

# ----------------------------------------------------------------------
# FIGURE SD2: Per-Subject Accuracy
# ----------------------------------------------------------------------
print("\n📊 BI_SD2: Per-Subject Accuracy (Publication)")
subjects_sd = sd_results['subject'].values
ens_accs = sd_results['Ensemble_Acc'].astype(float).values
colors = ['#2ca02c' if a > avg_ens else '#d62728' for a in ens_accs]
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(subjects_sd, ens_accs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=avg_ens, color='blue', linestyle='--', linewidth=2,
           label=f'Mean: {avg_ens:.1f}%')
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle=':', linewidth=1.5,
           label=f'Random Chance: {RANDOM_CHANCE:.1f}%')
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_ylabel('Ensemble Accuracy (%)', fontsize=12)
ax.set_title('BI_SD2: Per-Subject Accuracy — Subject-Dependent Ensemble', 
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, ens_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.0f}%', ha='center', fontsize=8, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SD2_Per_Subject_Accuracy.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SD2_Per_Subject_Accuracy.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved to Figures/BI_Block3_SD/BI_SD2_Per_Subject_Accuracy.png/pdf")

# ----------------------------------------------------------------------
# FIGURE SD3: Protocol Comparison (SD vs LOSO)
# ----------------------------------------------------------------------
print("\n📊 BI_SD3: Protocol Comparison (Publication)")
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(['Subject-Dependent\n(Upper Bound)', 'LOSO 10-shot\n(Cross-Subject)'],
              [avg_ens, loso_10shot], color=['#2ca02c', '#1f77b4'], 
              alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_SD3: Protocol Comparison — Subject-Dependent vs LOSO', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=RANDOM_CHANCE, color='red', linestyle='--', linewidth=1.5,
           label=f'Random Chance: {RANDOM_CHANCE:.1f}%')
for bar, acc in zip(bars, [avg_ens, loso_10shot]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{acc:.1f}%', ha='center', fontweight='bold')
gap = avg_ens - loso_10shot
ax.text(0.5, 45, f'Gap: {gap:.1f}%', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SD3_Protocol_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SD3_Protocol_Comparison.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved to Figures/BI_Block3_SD/BI_SD3_Protocol_Comparison.png/pdf")

# ----------------------------------------------------------------------
# FIGURE SD4: Cross-Validated Ensemble Confusion Matrix (PUBLICATION)
# ----------------------------------------------------------------------
print("\n📊 BI_SD4: Cross-Validated Ensemble Confusion Matrix (Publication)")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(ens_cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=GESTURES, yticklabels=GESTURES, ax=ax,
            cbar_kws={'label': 'Normalized Proportion'})
ax.set_xlabel('Predicted Gesture', fontsize=12)
ax.set_ylabel('True Gesture', fontsize=12)
ax.set_title(f'BI_SD4: Cross-Validated Ensemble Confusion Matrix\n'
             f'Accuracy: {ens_acc:.1f}% (from 5-fold subject-dependent CV)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SD4_Confusion_Matrix_Ensemble.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SD4_Confusion_Matrix_Ensemble.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved to Figures/BI_Block3_SD/BI_SD4_Confusion_Matrix_Ensemble.png/pdf")

# ----------------------------------------------------------------------
# FIGURE SD5: Model Comparison with 95% CI
# ----------------------------------------------------------------------
print("\n📊 BI_SD5: Model Comparison with 95% CI (Publication)")
stds_model = [sd_results['XGBoost_Acc'].astype(float).std(ddof=1),
              sd_results['RF_Acc'].astype(float).std(ddof=1),
              sd_results['Ensemble_Acc'].astype(float).std(ddof=1)]
n = len(sd_results)
ci95s = [1.96 * s / np.sqrt(n) for s in stds_model]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(['XGBoost', 'Random Forest', 'Ensemble'], [avg_xgb, avg_rf, avg_ens],
              yerr=ci95s, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
              alpha=0.7, edgecolor='black', capsize=5,
              error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_SD5: Model Comparison with 95% CI — Subject-Dependent', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
for bar, mean_v, ci in zip(bars, [avg_xgb, avg_rf, avg_ens], ci95s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{mean_v:.1f}% ± {ci:.1f}', ha='center', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_SD5_CI95_Comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / "BI_SD5_CI95_Comparison.pdf", bbox_inches='tight')
plt.close()
print("  ✅ Saved to Figures/BI_Block3_SD/BI_SD5_CI95_Comparison.png/pdf")

# ================================================================
# SAVE PUBLICATION TABLES
# ================================================================

# Table SD1: Detailed SD Results (per-subject)
print("\n📊 BI_Table_SD1: Subject-Dependent Results")
sd_results.to_csv(TAB_DIR / "BI_Table_SD1_Results.csv", index=False)
print("  ✅ Saved to Tables/BI_Block3_SD/BI_Table_SD1_Results.csv")

# Table SD2: Summary Statistics
summary_df = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest', 'Ensemble'],
    'Accuracy (%)': [f'{avg_xgb:.2f}', f'{avg_rf:.2f}', f'{avg_ens:.2f}'],
    '95% CI': [f'±{ci95s[0]:.2f}', f'±{ci95s[1]:.2f}', f'±{ci95s[2]:.2f}']
})
summary_df.to_csv(TAB_DIR / "BI_Table_SD2_Summary.csv", index=False)
print(summary_df.to_string(index=False))
print("  ✅ Saved to Tables/BI_Block3_SD/BI_Table_SD2_Summary.csv")

# Table SD3: Cross-Validated Ensemble Confusion Matrix
pd.DataFrame(ens_cm_norm, index=GESTURES, columns=GESTURES).to_csv(
    TAB_DIR / "BI_Table_CM_Ensemble.csv")
print("  ✅ Saved to Tables/BI_Block3_SD/BI_Table_CM_Ensemble.csv")

# ================================================================
# COMPLETION SUMMARY
# ================================================================
print("\n" + "="*80)
print("BI_BLOCK_3 COMPLETED SUCCESSFULLY")
print("="*80)
print("\n📁 PUBLICATION FIGURES (Figures/BI_Block3_SD/):")
print("   SD1: BI_SD1_Model_Comparison.png/pdf")
print("   SD2: BI_SD2_Per_Subject_Accuracy.png/pdf")
print("   SD3: BI_SD3_Protocol_Comparison.png/pdf")
print("   SD4: BI_SD4_Confusion_Matrix_Ensemble.png/pdf (Cross-Validated)")
print("   SD5: BI_SD5_CI95_Comparison.png/pdf")
print("\n📁 PUBLICATION TABLES (Tables/BI_Block3_SD/):")
print("   SD1: BI_Table_SD1_Results.csv")
print("   SD2: BI_Table_SD2_Summary.csv")
print("   CM: BI_Table_CM_Ensemble.csv")
print("\n📁 INTERNAL STUDY FIGURES (Internal_Study/Confusion_Matrices/):")
print("   → RF_CM.png/pdf (Random Forest Deployment)")
print("   → XGB_CM.png/pdf (XGBoost Deployment)")
print("   → Ensemble_CM.png/pdf (Cross-Validated)")
print("\n" + "="*80)
print("SUBJECT-DEPENDENT ANALYSIS COMPLETE")
print("Publication figures use cross-validated SD predictions (5-fold CV)")
print("Internal figures for debugging, interpretation, and presentations")
print("="*80)