#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_5_EXPLAINABILITY.py
# ================================================================
# BLOCK 5 — EXPLAINABILITY ANALYSIS
# ================================================================
# FIGURES: FE1 (Feature Importance), FE2 (SHAP Summary), FE3 (Frequency Regions)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

RESULTS_DIR = Path("./Results")
CACHE_DIR = Path("./Cached_Data")
FIG_DIR = Path("./Figures/BI_Block5_Explainability")
TAB_DIR = Path("./Tables/BI_Block5_Explainability")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("BI_BLOCK_5: EXPLAINABILITY ANALYSIS")
print("="*80)

X = np.load(CACHE_DIR / "BI_X.npy")
y_enc = np.load(CACHE_DIR / "BI_y_enc.npy")
feature_names = np.load(CACHE_DIR / "BI_feature_names.npy", allow_pickle=True)

print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Train XGBoost on full data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

xgb = XGBClassifier(n_estimators=600, max_depth=7, learning_rate=0.04,
                    subsample=0.85, colsample_bytree=0.80, random_state=RANDOM_STATE,
                    n_jobs=1, eval_metric='mlogloss', tree_method='hist', 
                    device='cpu', verbosity=0)
xgb.fit(X_scaled, y_enc)
importances = xgb.feature_importances_

top_n = 15
indices = np.argsort(importances)[::-1][:top_n]
top_imp = importances[indices]
top_labels = [feature_names[i] for i in indices]

# Figure BI_FE1: Feature Importance
print("\n📊 Figure BI_FE1: Feature Importance")
fig, ax = plt.subplots(figsize=(12, 8))
colors_feat = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
ax.barh(range(top_n), top_imp, color=colors_feat, alpha=0.7, edgecolor='black')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_labels)
ax.set_xlabel('Importance Score (XGBoost Gain)', fontsize=12)
ax.set_title('BI_FE1: Top 15 Feature Importances — XGBoost (405 Features)', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, imp in zip(ax.patches, top_imp):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{imp:.4f}', va='center', fontsize=8)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_FE1_Feature_Importance.png", dpi=300)
plt.savefig(FIG_DIR / "BI_FE1_Feature_Importance.pdf")
plt.close()
print("  ✅ Saved")

# ================================================================
# Figure BI_FE2: SHAP Summary
# ================================================================
print("\n📊 Figure BI_FE2: SHAP Summary")

try:
    import shap

    # ------------------------------------------------------------
    # Memory-safe sampling
    # ------------------------------------------------------------
    sample_size = min(20, len(X_scaled))
    background_size = min(50, len(X_scaled))

    sample_idx = np.random.choice(
        len(X_scaled),
        sample_size,
        replace=False
    )

    background_idx = np.random.choice(
        len(X_scaled),
        background_size,
        replace=False
    )

    X_shap = X_scaled[sample_idx]
    background = X_scaled[background_idx]

    print(f"  Using {sample_size} samples for SHAP explanation")
    print(f"  Using {background_size} background samples")

    # ------------------------------------------------------------
    # SHAP Explainer
    # ------------------------------------------------------------
    explainer = shap.TreeExplainer(
        xgb,
        background
    )

    shap_values = explainer.shap_values(X_shap)

    # ------------------------------------------------------------
    # Create SHAP Summary Figure
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 8))

    if isinstance(shap_values, list):
        # Older SHAP multiclass format
        shap.summary_plot(
            shap_values[0],
            X_shap,
            feature_names=feature_names,
            show=False
        )

    elif len(np.array(shap_values).shape) == 3:
        # New SHAP multiclass format:
        # (samples, features, classes)
        shap.summary_plot(
            shap_values[:, :, 0],
            X_shap,
            feature_names=feature_names,
            show=False
        )

    else:
        shap.summary_plot(
            shap_values,
            X_shap,
            feature_names=feature_names,
            show=False
        )

    plt.title(
        'BI_FE2: SHAP Feature Importance — Aggregated (405 Features)',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "BI_FE2_SHAP_Summary.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        FIG_DIR / "BI_FE2_SHAP_Summary.pdf",
        bbox_inches='tight'
    )

    plt.close()

    print("  ✅ Saved")

except ImportError:

    print("  ⚠️ SHAP not installed")
    print("     Install with: pip install shap")

except Exception as e:

    print(f"  ⚠️ SHAP error: {e}")

    import traceback
    traceback.print_exc()
# ================================================================
# Figure BI_FE3: Feature Type Distribution Among Top Features
# ================================================================
print("\n📊 Figure BI_FE3: Most Discriminative Feature Groups")

feature_types = []

for f in top_labels:

    parts = str(f).split('_')

    if len(parts) > 1:
        feature_types.append(parts[0])
    else:
        feature_types.append(str(f))

# Count occurrences
feature_counts = pd.Series(feature_types).value_counts()

fig, ax = plt.subplots(figsize=(10, 6))

colors_feat = plt.cm.tab20(
    np.linspace(0, 1, len(feature_counts))
)

bars = ax.bar(
    feature_counts.index,
    feature_counts.values,
    color=colors_feat,
    edgecolor='black',
    alpha=0.8
)

ax.set_xlabel(
    'Feature Group',
    fontsize=12
)

ax.set_ylabel(
    'Count within Top Features',
    fontsize=12
)

ax.set_title(
    f'BI_FE3: Distribution of Top {top_n} Most Important Features',
    fontsize=14,
    fontweight='bold'
)

ax.grid(
    True,
    axis='y',
    alpha=0.3
)

for bar in bars:

    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.1,
        f'{int(height)}',
        ha='center',
        fontweight='bold'
    )

plt.tight_layout()

plt.savefig(
    FIG_DIR / "BI_FE3_Feature_Group_Distribution.png",
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    FIG_DIR / "BI_FE3_Feature_Group_Distribution.pdf",
    bbox_inches='tight'
)

plt.close()

print("  ✅ Saved")

# Save supporting table
feature_count_table = pd.DataFrame({
    'Feature Group': feature_counts.index,
    'Count': feature_counts.values
})

feature_count_table.to_csv(
    TAB_DIR / "BI_Table_FE2_Feature_Group_Distribution.csv",
    index=False
)

# Save top-feature table
pd.DataFrame([
    {
        'Rank': i + 1,
        'Feature': top_labels[i],
        'Feature Type': feature_types[i],
        'Importance Score': round(float(top_imp[i]), 6)
    }
    for i in range(top_n)
]).to_csv(
    TAB_DIR / "BI_Table_FE1_Top_Features.csv",
    index=False
)
print("\n✅ BI_BLOCK_5 COMPLETED")