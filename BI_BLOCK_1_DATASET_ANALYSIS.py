#!/usr/bin/env python3
# ================================================================
# BI_BLOCK_1_DATASET_ANALYSIS.py
# ================================================================
# BLOCK 1 — DATASET ANALYSIS
# ================================================================
# PURPOSE: Generate descriptive statistics and visualizations
# FIGURES: D1 (Gesture Distribution), D2 (Subject Distribution),
#          D3 (Subject-Gesture Heatmap), D4 (Raw Signal Examples)
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import savgol_filter

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_DIR = Path("./Results")
CACHE_DIR = Path("./Cached_Data")
FIG_DIR = Path("./Figures/BI_Block1_Dataset")
TAB_DIR = Path("./Tables/BI_Block1_Dataset")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

GESTURES = ['HOC', 'TFM', 'IFM', 'MFM', 'RFM', 'LFM', 'PH']
DATASET_ROOT = Path("/mnt/f/Faruk/Bio-impedance_Gesture_Project/Bioimpedance_7Gesture_Final_Dataset")

print("="*80)
print("BI_BLOCK_1: DATASET ANALYSIS")
print("="*80)

# Load data
X = np.load(CACHE_DIR / "BI_X.npy")
y = np.load(CACHE_DIR / "BI_y.npy")
subjects = np.load(CACHE_DIR / "BI_subjects.npy")

print(f"✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Subjects: {len(np.unique(subjects))}, Gestures: {len(np.unique(y))}")

# ----------------------------------------------------------------------
# Figure BI_D1: Gesture Distribution
# ----------------------------------------------------------------------
print("\n📊 Figure BI_D1: Gesture Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
gesture_counts = pd.Series(y).value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(GESTURES)))
bars = ax.bar(gesture_counts.index, gesture_counts.values, color=colors, edgecolor='black')
ax.set_xlabel('Gesture', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('BI_D1: Distribution of Samples per Gesture', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, gesture_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{val}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_D1_Gesture_Distribution.png", dpi=300)
plt.savefig(FIG_DIR / "BI_D1_Gesture_Distribution.pdf")
plt.close()
print("  ✅ Saved")

# ----------------------------------------------------------------------
# Figure BI_D2: Subject Distribution
# ----------------------------------------------------------------------
print("\n📊 Figure BI_D2: Subject Distribution")
fig, ax = plt.subplots(figsize=(14, 6))
subject_counts = pd.Series(subjects).value_counts().sort_index()
colors_subj = plt.cm.viridis(np.linspace(0.2, 0.8, len(subject_counts)))
bars = ax.bar(subject_counts.index.astype(str), subject_counts.values, 
              color=colors_subj, edgecolor='black')
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('BI_D2: Distribution of Samples per Subject', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, subject_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_D2_Subject_Distribution.png", dpi=300)
plt.savefig(FIG_DIR / "BI_D2_Subject_Distribution.pdf")
plt.close()
print("  ✅ Saved")

# ----------------------------------------------------------------------
# Figure BI_D3: Subject-Gesture Heatmap
# ----------------------------------------------------------------------
print("\n📊 Figure BI_D3: Subject-Gesture Heatmap")
unique_subjects = np.unique(subjects)
heatmap_data = []
for sub in unique_subjects:
    sub_mask = (subjects == sub)
    row = [np.sum(sub_mask & (y == gesture)) for gesture in GESTURES]
    heatmap_data.append(row)

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(GESTURES)))
ax.set_xticklabels(GESTURES, rotation=45, ha='right')
ax.set_yticks(range(len(unique_subjects)))
ax.set_yticklabels(unique_subjects)
ax.set_xlabel('Gesture', fontsize=12)
ax.set_ylabel('Subject ID', fontsize=12)
ax.set_title('BI_D3: Subject-Gesture Distribution Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Number of Samples')
for i in range(len(unique_subjects)):
    for j in range(len(GESTURES)):
        ax.text(j, i, heatmap_data[i][j], ha="center", va="center", 
                color="black", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_D3_Subject_Gesture_Heatmap.png", dpi=300)
plt.savefig(FIG_DIR / "BI_D3_Subject_Gesture_Heatmap.pdf")
plt.close()
print("  ✅ Saved")

# ----------------------------------------------------------------------
# Figure BI_D4: Raw Signal Example
# ----------------------------------------------------------------------
print("\n📊 Figure BI_D4: Raw Signal Example")
sample_subject = unique_subjects[0]
sample_gestures = [GESTURES[0], GESTURES[1]]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, gesture in enumerate(sample_gestures):
    for subject_folder in sorted(DATASET_ROOT.iterdir()):
        if subject_folder.name != sample_subject:
            continue
        for gesture_folder in sorted(subject_folder.iterdir()):
            if gesture_folder.name.split('_')[-1] != gesture:
                continue
            ch1_path = next((gesture_folder / p for p in ["Channel_1", "ch1"] 
                           if (gesture_folder / p).exists()), None)
            if ch1_path:
                f1 = sorted(ch1_path.glob("*.spec"))[0]
                d1 = pd.read_csv(f1, skiprows=6, header=None).apply(
                    pd.to_numeric, errors='coerce').dropna().values
                c1 = d1[:,1] + 1j * d1[:,2]
                m1 = np.abs(c1)
                m1s = savgol_filter(m1, 11, 3)
                axes[0, idx].plot(m1, 'b-', alpha=0.5, label='Raw')
                axes[0, idx].plot(m1s, 'r-', linewidth=2, label='Smoothed')
                axes[0, idx].set_title(f'{gesture} - Raw vs Smoothed')
                axes[0, idx].set_xlabel('Frequency Point')
                axes[0, idx].set_ylabel('Magnitude (Ohms)')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
                p1 = np.unwrap(np.angle(c1))
                p1s = savgol_filter(p1, 11, 3)
                axes[1, idx].plot(p1, 'b-', alpha=0.5, label='Raw')
                axes[1, idx].plot(p1s, 'r-', linewidth=2, label='Smoothed')
                axes[1, idx].set_title(f'{gesture} - Phase Response')
                axes[1, idx].set_xlabel('Frequency Point')
                axes[1, idx].set_ylabel('Phase (radians)')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)
                break
        break

plt.suptitle('BI_D4: Raw Bio-Impedance Signal Examples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "BI_D4_Raw_Signal_Example.png", dpi=300)
plt.savefig(FIG_DIR / "BI_D4_Raw_Signal_Example.pdf")
plt.close()
print("  ✅ Saved")

# Save summary table
summary_df = pd.DataFrame({
    'Metric': ['Total Samples', 'Total Features', 'Total Subjects', 'Total Gestures'],
    'Value': [len(X), X.shape[1], len(unique_subjects), len(GESTURES)]
})
summary_df.to_csv(TAB_DIR / "BI_Table_Dataset_Summary.csv", index=False)

print("\n✅ BI_BLOCK_1 COMPLETED")