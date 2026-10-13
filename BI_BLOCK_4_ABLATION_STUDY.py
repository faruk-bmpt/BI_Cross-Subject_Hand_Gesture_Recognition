#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_4_ABLATION_STUDY.py
# ================================================================
# BLOCK 4 — ABLATION STUDY (405-FEATURE)
# ================================================================
# PURPOSE: Evaluate contribution of each feature group
# FIGURES: A1 (Ablation Bar Chart)
# TABLES: A1 (Ablation Results)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import gc

RESULTS_DIR = Path("./Results")
CACHE_DIR = Path("./Cached_Data")
FIG_DIR = Path("./Figures/BI_Block4_Ablation")
TAB_DIR = Path("./Tables/BI_Block4_Ablation")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("BI_BLOCK_4: ABLATION STUDY — 405-Feature Analysis")
print("="*80)

X = np.load(CACHE_DIR / "BI_X.npy")
y_enc = np.load(CACHE_DIR / "BI_y_enc.npy")
subjects = np.load(CACHE_DIR / "BI_subjects.npy")

print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"✅ Subjects: {len(np.unique(subjects))}")

# Feature groups (405 features)
all_features = set(range(405))
feature_groups = {
    'Magnitude': set(range(0, 45)) | set(range(90, 135)),
    'Phase': set(range(45, 90)) | set(range(135, 180)),
    'Gradients': set(range(180, 270)),
    'Ratio': set(range(270, 315)),
    'Cumulative': set(range(315, 405))
}

ablation_configs = {'Full Features': all_features}
for name, indices in feature_groups.items():
    ablation_configs[f'Without {name}'] = all_features - indices

print("\n📊 Running Ablation Study...")
ablation_results = {}

RF_ABLATION = {'n_estimators': 100, 'max_depth': 6, 'random_state': RANDOM_STATE, 'n_jobs': 1}

for config_name, feat_indices in ablation_configs.items():
    print(f"\n  Testing {config_name}...")
    X_sub = X[:, list(feat_indices)]
    accs = []
    for sub_id in np.unique(subjects): 
        mask = (subjects == sub_id)
        X_s = X_sub[mask]
        y_s = y_enc[mask]
        if len(np.unique(y_s)) < 2:
            continue
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in skf.split(X_s, y_s):
            X_tr, X_te = X_s[train_idx], X_s[test_idx]
            y_tr, y_te = y_s[train_idx], y_s[test_idx]
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            rf = RandomForestClassifier(**RF_ABLATION)
            rf.fit(X_tr, y_tr)
            accs.append(accuracy_score(y_te, rf.predict(X_te)) * 100)
            del rf, scaler
            gc.collect()
    ablation_results[config_name] = np.mean(accs) if accs else 0
    print(f"    Accuracy: {ablation_results[config_name]:.1f}%")

# Figure BI_A1: Ablation Bar Chart
print("\n📊 Figure BI_A1: Ablation Results")
fig, ax = plt.subplots(figsize=(12, 6))
configs = list(ablation_results.keys())
accs = list(ablation_results.values())
colors_abl = plt.cm.Set3(np.linspace(0, 1, len(configs)))
bars = ax.bar(configs, accs, color=colors_abl, alpha=0.7, edgecolor='black')
ax.axhline(y=ablation_results['Full Features'], color='red', linestyle='--', linewidth=2,
           label=f'Full Features: {ablation_results["Full Features"]:.1f}%')
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('BI_A1: Ablation Study — Feature Group Contribution (405 Features)', 
             fontsize=14, fontweight='bold')
ax.set_ylim(0, max(accs) + 10)
ax.legend()
plt.xticks(rotation=45, ha='right')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{acc:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_A1_Ablation_Results.png", dpi=300)
plt.savefig(FIG_DIR / "BI_A1_Ablation_Results.pdf")
plt.close()
print("  ✅ Saved")

# Save table
full_acc = ablation_results['Full Features']

ablation_table = pd.DataFrame([
    {
        'Configuration': k,
        'Accuracy (%)': round(v, 2),
        'Drop (%)': round(full_acc - v, 2)
    }
    for k, v in ablation_results.items()
])

ablation_table.to_csv(
    TAB_DIR / "BI_Table_A1_Ablation.csv",
    index=False
)

print("\n✅ BI_BLOCK_4 COMPLETED")