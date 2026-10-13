#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_8_INTERNAL_VISUALIZATION.py
# ================================================================

# Suppress TensorFlow warnings BEFORE importing anything
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("./Results")
CACHE_DIR = Path("./Cached_Data")
FIG_DIR = Path("./Figures/BI_Block8_InternalViz")
FIG_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ['HOC', 'TFM', 'IFM', 'MFM', 'RFM', 'LFM', 'PH']
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("BI_BLOCK_8: INTERNAL VISUALIZATION")
print("NOTE: StandardScaler used ONLY for visualization — not in model evaluation")
print("="*80)

X = np.load(CACHE_DIR / "BI_X.npy")
y = np.load(CACHE_DIR / "BI_y.npy")

print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Visualization-only normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sub-sample for faster visualization
N_VIZ = min(1000, len(X_scaled))
idx_viz = np.random.choice(len(X_scaled), N_VIZ, replace=False)
X_viz = X_scaled[idx_viz]
y_viz = y[idx_viz]

colors = plt.cm.Set3(np.linspace(0, 1, len(GESTURES)))

# Figure BI_IV1: PCA
print("\n📊 Figure BI_IV1: PCA Visualization")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_viz)
fig, ax = plt.subplots(figsize=(10, 8))
for i, g in enumerate(GESTURES):
    mask = (y_viz == g)
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i]], label=g, alpha=0.5, s=15)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('BI_IV1: PCA Visualization (405 Features)\n'
             'Note: Visualization-only normalization; partial separability trends shown',
             fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_IV1_PCA.png", dpi=300)
plt.savefig(FIG_DIR / "BI_IV1_PCA.pdf")
plt.close()
print("  ✅ Saved")

# Figure BI_IV2: t-SNE 
print("\n📊 Figure BI_IV2: t-SNE Visualization")
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=500)
X_tsne = tsne.fit_transform(X_viz)
fig, ax = plt.subplots(figsize=(10, 8))
for i, g in enumerate(GESTURES):
    mask = (y_viz == g)
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[colors[i]], label=g, alpha=0.5, s=15)
ax.set_xlabel('t-SNE Component 1', fontsize=12)
ax.set_ylabel('t-SNE Component 2', fontsize=12)
ax.set_title('BI_IV2: t-SNE Visualization (405 Features)\n'
             'Note: Visualization-only normalization; local cluster structure shown',
             fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_IV2_tSNE.png", dpi=300)
plt.savefig(FIG_DIR / "BI_IV2_tSNE.pdf")
plt.close()
print("  ✅ Saved")

# Figure BI_IV3: UMAP (optional)
print("\n📊 Figure BI_IV3: UMAP Visualization")
try:
    import umap
    reducer = umap.UMAP(random_state=RANDOM_STATE, n_components=2)
    X_umap = reducer.fit_transform(X_viz)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, g in enumerate(GESTURES):
        mask = (y_viz == g)
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[colors[i]], label=g, alpha=0.5, s=15)
    ax.set_xlabel('UMAP Component 1', fontsize=12)
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    ax.set_title('BI_IV3: UMAP Visualization (405 Features)\n'
                 'Note: Visualization-only normalization; manifold structure shown',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "BI_IV3_UMAP.png", dpi=300)
    plt.savefig(FIG_DIR / "BI_IV3_UMAP.pdf")
    plt.close()
    print("  ✅ Saved")
except ImportError:
    print("  ⚠️ UMAP not installed — skipping")

# Figure BI_IV4: Correlation Matrix
print("\n📊 Figure BI_IV4: Feature Correlation Matrix")
corr_idx = np.random.choice(
    len(X_scaled),
    min(500, len(X_scaled)),
    replace=False
)

corr_data = X_scaled[corr_idx, :50]
corr_matrix = np.corrcoef(corr_data.T)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Pearson Correlation')
ax.set_xlabel('Feature Index (1-50)', fontsize=12)
ax.set_ylabel('Feature Index (1-50)', fontsize=12)
ax.set_title('BI_IV4: Feature Correlation Matrix — First 50 Features (405 Features)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_IV4_Correlation_Matrix.png", dpi=300)
plt.savefig(FIG_DIR / "BI_IV4_Correlation_Matrix.pdf")
plt.close()
print("  ✅ Saved")

# Figure BI_IV5: Calibration Sensitivity
print("\n📊 Figure BI_IV5: Calibration Sensitivity")
LOSO_DIR = RESULTS_DIR / "BI_LOSO_Results"
loso_df = pd.read_csv(LOSO_DIR / "BI_LOSO_Results.csv")
loso_df = loso_df.sort_values('Shots')
shots_arr = loso_df['Shots'].values
accs_arr = loso_df['Accuracy'].astype(float).values
derivatives = np.gradient(
    accs_arr,
    shots_arr
)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(shots_arr, accs_arr, 'o-', color='#1f77b4', linewidth=2.5, markersize=10, label='Accuracy')
ax2 = ax.twinx()
ax2.bar(shots_arr, derivatives, width=0.8, alpha=0.3, color='orange', label='Marginal Gain/shot')
ax2.set_ylabel('Marginal Gain (% per shot)', fontsize=12, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax.set_xlabel('Calibration Shots', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12, color='#1f77b4')
ax.set_title('BI_IV5: Calibration Sensitivity — Marginal Return per Shot (405 Features)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_IV5_Calibration_Sensitivity.png", dpi=300)
plt.savefig(FIG_DIR / "BI_IV5_Calibration_Sensitivity.pdf")
plt.close()
print("  ✅ Saved")

print("\n✅ BI_BLOCK_8 COMPLETED")