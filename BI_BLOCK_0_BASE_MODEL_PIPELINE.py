
# ================================================================
# BI_BLOCK_0_BASE_MODEL_PIPELINE.py
# ================================================================
# BLOCK 0 — BASE MODEL PIPELINE (405-FEATURE PRIMARY)
# ================================================================
# PURPOSE: Load data, extract 405 features, train SD and LOSO models
# OUTPUT: Results/BI_SD_Results/, Results/BI_LOSO_Results/, Results/BI_Models/
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import gc
import pickle
from datetime import datetime
import logging
import time as timer
from tqdm import tqdm

# ================================================================
# VERSION LOGGING
# ================================================================
import sklearn
import xgboost
import scipy

# ================================================================
# LOGGING SETUP
# ================================================================
log_dir = Path("./Logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_dir / "execution_log.txt",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_message(msg):
    print(msg)
    logging.info(msg)

# ================================================================
# FIXED RANDOM SEED
# ================================================================
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ================================================================
# CONFIGURATION
# ================================================================
GESTURES = ['HOC', 'TFM', 'IFM', 'MFM', 'RFM', 'LFM', 'PH']
CALIBRATION_SHOTS = [0, 1, 3, 5, 7, 10]
N_SPLITS_SD = 5

XGB_PARAMS = {
    'n_estimators': 600,
    'max_depth': 7,
    'learning_rate': 0.04,
    'subsample': 0.85,
    'colsample_bytree': 0.80,
    'reg_lambda': 3,
    'reg_alpha': 1,
    'min_child_weight': 3,
    'random_state': RANDOM_STATE,
    'tree_method': 'hist',
    'device': 'cpu',
    'n_jobs': 1,
    'eval_metric': 'mlogloss',
    'verbosity': 0
}

RF_PARAMS = {
    'n_estimators': 400,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}

DATASET_ROOT = Path("/mnt/f/Faruk/Bio-impedance_Gesture_Project/Bioimpedance_7Gesture_Final_Dataset")

# Output directories
OUTPUT_DIR = Path("./Results")
CACHE_DIR = Path("./Cached_Data")
SD_DIR = OUTPUT_DIR / "BI_SD_Results"
LOSO_DIR = OUTPUT_DIR / "BI_LOSO_Results"
MODELS_DIR = OUTPUT_DIR / "BI_Models"
PREDS_DIR = OUTPUT_DIR / "BI_Predictions"

for d in [OUTPUT_DIR, CACHE_DIR, SD_DIR, LOSO_DIR, MODELS_DIR, PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_message("="*80)
log_message("BI_BLOCK_0: BASE MODEL PIPELINE (405-FEATURE PRIMARY)")
log_message(f"Python: {sys.version}")
log_message(f"NumPy: {np.__version__}")
log_message(f"Pandas: {pd.__version__}")
log_message(f"Scikit-learn: {sklearn.__version__}")
log_message(f"XGBoost: {xgboost.__version__}")
log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message("="*80)

# ================================================================
# FEATURE EXTRACTION (405 FEATURES)
# ================================================================

def extract_features_405(d1, d2):
    """
    Extract full 405-feature vector (9 groups × 45 points).
    Feature groups: Mag_CH1, Phase_CH1, Mag_CH2, Phase_CH2,
                    Grad_CH1, Grad_CH2, Ratio, Cumsum_CH1, Cumsum_CH2
    """
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return None
    
    c1 = d1[:,1] + 1j * d1[:,2]
    c2 = d2[:,1] + 1j * d2[:,2]
    m1 = np.abs(c1)
    m2 = np.abs(c2)
    p1 = np.unwrap(np.angle(c1))
    p2 = np.unwrap(np.angle(c2))
    
    m1s = savgol_filter(m1, 11, 3)
    m2s = savgol_filter(m2, 11, 3)
    p1s = savgol_filter(p1, 11, 3)
    p2s = savgol_filter(p2, 11, 3)
    
    idx = slice(5, 95, 2)
    m1_roi = m1s[idx]
    p1_roi = p1s[idx]
    m2_roi = m2s[idx]
    p2_roi = p2s[idx]
    
    grad1 = np.gradient(m1_roi)
    grad2 = np.gradient(m2_roi)
    ratio = m1_roi / (m2_roi + 1e-8)
    cumsum1 = np.cumsum(m1_roi)
    cumsum2 = np.cumsum(m2_roi)
    
    features = np.concatenate([m1_roi, p1_roi, m2_roi, p2_roi,
                                grad1, grad2, ratio, cumsum1, cumsum2])
    assert len(features) == 405, f"Feature size mismatch: {len(features)}"
    return features

# ================================================================
# CACHE CHECK
# ================================================================

cache_files_exist = all([
    (CACHE_DIR / "BI_X.npy").exists(),
    (CACHE_DIR / "BI_y.npy").exists(),
    (CACHE_DIR / "BI_subjects.npy").exists(),
    (CACHE_DIR / "BI_feature_names.npy").exists()
])

if cache_files_exist:
    log_message("\n📦 Loading cached data...")
    X = np.load(CACHE_DIR / "BI_X.npy")
    y = np.load(CACHE_DIR / "BI_y.npy")
    subjects = np.load(CACHE_DIR / "BI_subjects.npy")
    set_ids = np.load(CACHE_DIR / "BI_set_ids.npy") if (CACHE_DIR / "BI_set_ids.npy").exists() else np.zeros(len(X))
    feature_names = np.load(CACHE_DIR / "BI_feature_names.npy", allow_pickle=True)
    log_message("✅ Cached data loaded!")
else:
    log_message("\n📊 Extracting 405 features (first time)...")
    
    feature_names = []
    groups = ['Mag_CH1', 'Phase_CH1', 'Mag_CH2', 'Phase_CH2',
              'Grad_CH1', 'Grad_CH2', 'Ratio', 'Cumsum_CH1', 'Cumsum_CH2']
    for g in groups:
        for i in range(45):
            feature_names.append(f'{g}_{i}')
    feature_names = np.array(feature_names)

    memory_vault = {'X': [], 'y': [], 'subjects': [], 'set_ids': []}
    
    for subject_folder in tqdm(sorted(DATASET_ROOT.iterdir()), desc="Processing subjects"):
        if not subject_folder.is_dir() or not subject_folder.name.startswith("Subject_"):
            continue
        sub_id = subject_folder.name
        
        for gesture_folder in sorted(subject_folder.iterdir()):
            gesture = gesture_folder.name.split('_')[-1]
            if gesture not in GESTURES:
                continue
            
            ch1_path = next((gesture_folder / p for p in ["Channel_1", "ch1"] 
                           if (gesture_folder / p).exists()), None)
            ch2_path = next((gesture_folder / p for p in ["Channel_2", "ch2"] 
                           if (gesture_folder / p).exists()), None)
            
            if not ch1_path or not ch2_path:
                continue
            
            f1_list = sorted(ch1_path.glob("*.spec"))
            f2_list = sorted(ch2_path.glob("*.spec"))
            
            for i, (f1, f2) in enumerate(zip(f1_list, f2_list)):
                try:
                    set_id = int(f1.name.split('_')[1].split('.')[0]) if '_' in f1.name else i
                    d1 = pd.read_csv(f1, skiprows=6, header=None).apply(
                        pd.to_numeric, errors='coerce').dropna().values
                    d2 = pd.read_csv(f2, skiprows=6, header=None).apply(
                        pd.to_numeric, errors='coerce').dropna().values
                    features = extract_features_405(d1, d2)
                    if features is None:
                        continue
                    memory_vault['X'].append(features)
                    memory_vault['y'].append(gesture)
                    memory_vault['subjects'].append(sub_id)
                    memory_vault['set_ids'].append(set_id)
                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    continue

    X = np.array(memory_vault['X'])
    y = np.array(memory_vault['y'])
    subjects = np.array(memory_vault['subjects'])
    set_ids = np.array(memory_vault['set_ids'])
    
    np.save(CACHE_DIR / "BI_X.npy", X)
    np.save(CACHE_DIR / "BI_y.npy", y)
    np.save(CACHE_DIR / "BI_subjects.npy", subjects)
    np.save(CACHE_DIR / "BI_set_ids.npy", set_ids)
    np.save(CACHE_DIR / "BI_feature_names.npy", feature_names)
    log_message("✅ Data cached!")

le = LabelEncoder()
y_enc = le.fit_transform(y)
np.save(CACHE_DIR / "BI_y_enc.npy", y_enc)

log_message(f"\n✅ Dataset: {X.shape[0]} samples, {X.shape[1]} features")
log_message(f"   Subjects: {len(np.unique(subjects))}, Gestures: {len(np.unique(y))}")

joblib.dump(le, MODELS_DIR / "BI_label_encoder.pkl")
np.save(OUTPUT_DIR / "BI_feature_names.npy", feature_names)

# ================================================================
# METRICS & MODEL FUNCTIONS
# ================================================================

def calculate_all_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred) * 100,
        'f1_weighted': f1_score(y_true, y_pred, average='weighted') * 100,
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

def get_models():
    return XGBClassifier(**XGB_PARAMS), RandomForestClassifier(**RF_PARAMS)

# ================================================================
# SUBJECT-DEPENDENT EVALUATION
# ================================================================

log_message("\n" + "="*80)
log_message("BI_SD: SUBJECT-DEPENDENT EVALUATION")
log_message("="*80)

sd_results = {'XGBoost': [], 'RandomForest': [], 'Ensemble': []}
sd_subject_details = []
sd_all_predictions = {}
sd_all_probabilities = {}

for sub_id in tqdm(np.unique(subjects), desc="SD Evaluation"):
    mask = (subjects == sub_id)
    X_sub = X[mask]
    y_sub = y_enc[mask]
    
    if len(np.unique(y_sub)) < 2:
        continue
    
    skf = StratifiedKFold(n_splits=N_SPLITS_SD, shuffle=True, random_state=RANDOM_STATE)
    xgb_metrics = []
    rf_metrics = []
    ens_metrics = []
    fold_predictions = []
    fold_probabilities = []
    
    for train_idx, test_idx in skf.split(X_sub, y_sub):
        X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = y_sub[train_idx], y_sub[test_idx]
        
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        xgb, rf = get_models()
        xgb.fit(X_tr, y_tr)
        rf.fit(X_tr, y_tr)
        
        pred_xgb = xgb.predict(X_te)
        pred_rf = rf.predict(X_te)
        proba = (xgb.predict_proba(X_te) + rf.predict_proba(X_te)) / 2
        pred_ens = np.argmax(proba, axis=1)
        
        xgb_metrics.append(calculate_all_metrics(y_te, pred_xgb))
        rf_metrics.append(calculate_all_metrics(y_te, pred_rf))
        ens_metrics.append(calculate_all_metrics(y_te, pred_ens))
        fold_predictions.append({'true': y_te, 'pred': pred_ens})
        fold_probabilities.append({'proba': proba})
        
        del xgb, rf
        gc.collect()
    
    sd_results['XGBoost'].append(np.mean([m['accuracy'] for m in xgb_metrics]))
    sd_results['RandomForest'].append(np.mean([m['accuracy'] for m in rf_metrics]))
    sd_results['Ensemble'].append(np.mean([m['accuracy'] for m in ens_metrics]))
    
    sd_subject_details.append({
        'subject': sub_id,
        'XGBoost_Acc': f"{np.mean([m['accuracy'] for m in xgb_metrics]):.1f}",
        'RF_Acc': f"{np.mean([m['accuracy'] for m in rf_metrics]):.1f}",
        'Ensemble_Acc': f"{np.mean([m['accuracy'] for m in ens_metrics]):.1f}",
        'Ensemble_F1': f"{np.mean([m['f1_weighted'] for m in ens_metrics]):.1f}",
        'Ensemble_MCC': f"{np.mean([m['mcc'] for m in ens_metrics]):.3f}"
    })
    
    sd_all_predictions[sub_id] = fold_predictions
    sd_all_probabilities[sub_id] = fold_probabilities
    log_message(f"  {sub_id}: {np.mean([m['accuracy'] for m in ens_metrics]):.1f}%")

pd.DataFrame(sd_subject_details).to_csv(SD_DIR / "BI_SD_Results.csv", index=False)
with open(PREDS_DIR / "BI_SD_predictions.pkl", 'wb') as f:
    pickle.dump(sd_all_predictions, f)
with open(PREDS_DIR / "BI_SD_probabilities.pkl", 'wb') as f:
    pickle.dump(sd_all_probabilities, f)

# ================================================================
# SUBJECT-INDEPENDENT LOSO EVALUATION
# ================================================================

log_message("\n" + "="*80)
log_message("BI_LOSO: SUBJECT-INDEPENDENT EVALUATION")
log_message("="*80)

rng = np.random.default_rng(RANDOM_STATE)
logo = LeaveOneGroupOut()

loso_results = {s: {'acc': [], 'f1': [], 'mcc': [], 'balanced_acc': [], 'time': [],
                    'acc_xgb': [], 'acc_rf': []} 
                for s in CALIBRATION_SHOTS}
confusion_matrices = {}
loso_subject_wise = {s: {} for s in CALIBRATION_SHOTS}
loso_all_predictions = {s: [] for s in CALIBRATION_SHOTS}
loso_all_probabilities = {s: [] for s in CALIBRATION_SHOTS}

for shots in CALIBRATION_SHOTS:
    log_message(f"\n→ Running {shots}-Shot...")
    acc_list = []
    f1_list = []
    mcc_list = []
    balanced_acc_list = []
    time_list = []
    acc_xgb_list = []
    acc_rf_list = []
    all_preds = []
    all_true = []
    
    for train_idx, test_idx in logo.split(X, y_enc, groups=subjects):
        X_source = X[train_idx]
        y_source = y_enc[train_idx]
        X_target = X[test_idx]
        y_target = y_enc[test_idx]
        test_subject = np.unique(subjects[test_idx])[0]
        start_time = timer.time()
        
        # Scaler fitted ONLY on source data — NO LEAKAGE
        scaler = RobustScaler()
        scaler.fit(X_source)
        X_source_scaled = scaler.transform(X_source)
        X_target_scaled = scaler.transform(X_target)
        
        if shots == 0:
            X_train = X_source_scaled
            y_train = y_source
            X_test = X_target_scaled
            y_test = y_target
        else:
            calib_idx = []
            test_remaining = []
            
            for g in np.unique(y_target):
                idx = np.where(y_target == g)[0]
                if len(idx) == 0:
                    continue
                shuffled_idx = idx.copy()
                rng.shuffle(shuffled_idx)
                take_n = min(shots, len(shuffled_idx))
                calib_idx.extend(shuffled_idx[:take_n])
                test_remaining.extend(shuffled_idx[take_n:])
            
            if len(test_remaining) == 0 or len(calib_idx) == 0:
                continue
            
            X_calib = X_target[calib_idx]
            y_calib = y_target[calib_idx]
            X_test = X_target[test_remaining]
            y_test = y_target[test_remaining]
            
            X_calib_scaled = scaler.transform(X_calib)
            X_test_scaled = scaler.transform(X_test)
            
            X_train = np.vstack([X_source_scaled, X_calib_scaled])
            y_train = np.concatenate([y_source, y_calib])
            X_test = X_test_scaled
        
        xgb, rf = get_models()
        xgb.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        
        prob_xgb = xgb.predict_proba(X_test)
        prob_rf = rf.predict_proba(X_test)
        proba = (prob_xgb + prob_rf) / 2
        
        pred_ens = np.argmax(proba, axis=1)
        pred_xgb = np.argmax(prob_xgb, axis=1)
        pred_rf = np.argmax(prob_rf, axis=1)
        
        elapsed_time = timer.time() - start_time
        metrics = calculate_all_metrics(y_test, pred_ens)
        
        acc_list.append(metrics['accuracy'])
        f1_list.append(metrics['f1_weighted'])
        mcc_list.append(metrics['mcc'])
        balanced_acc_list.append(metrics['balanced_accuracy'])
        time_list.append(elapsed_time)
        acc_xgb_list.append(accuracy_score(y_test, pred_xgb) * 100)
        acc_rf_list.append(accuracy_score(y_test, pred_rf) * 100)
        
        all_preds.extend(pred_ens)
        all_true.extend(y_test)
        loso_subject_wise[shots][test_subject] = metrics['accuracy']
        loso_all_predictions[shots].append({'true': y_test, 'pred': pred_ens, 'subject': test_subject})
        loso_all_probabilities[shots].append({'proba': proba, 'subject': test_subject})
        
        del xgb, rf, scaler
        gc.collect()
    
    if len(acc_list) > 0:
        loso_results[shots]['acc'] = acc_list
        loso_results[shots]['f1'] = f1_list
        loso_results[shots]['mcc'] = mcc_list
        loso_results[shots]['balanced_acc'] = balanced_acc_list
        loso_results[shots]['time'] = time_list
        loso_results[shots]['acc_xgb'] = acc_xgb_list
        loso_results[shots]['acc_rf'] = acc_rf_list
        confusion_matrices[shots] = confusion_matrix(all_true, all_preds, 
                                                     labels=np.arange(len(GESTURES)))
        log_message(f"   {shots}-shot → Ens: {np.mean(acc_list):.2f}% | "
                    f"XGB: {np.mean(acc_xgb_list):.2f}% | "
                    f"RF: {np.mean(acc_rf_list):.2f}% | "
                    f"Time: {np.mean(time_list):.3f}s")

# ================================================================
# SAVE LOSO RESULTS
# ================================================================

loso_summary = []
for shots in CALIBRATION_SHOTS:
    if loso_results[shots]['acc']:
        loso_summary.append({
            'Shots': shots,
            'Accuracy': f"{np.mean(loso_results[shots]['acc']):.2f}",
            'Std': f"{np.std(loso_results[shots]['acc']):.2f}",
            'F1': f"{np.mean(loso_results[shots]['f1']):.2f}",
            'MCC': f"{np.mean(loso_results[shots]['mcc']):.3f}",
            'Balanced_Acc': f"{np.mean(loso_results[shots]['balanced_acc']):.2f}",
            'Avg_Time_s': f"{np.mean(loso_results[shots]['time']):.3f}",
            'XGB_Accuracy': f"{np.mean(loso_results[shots]['acc_xgb']):.2f}",
            'RF_Accuracy': f"{np.mean(loso_results[shots]['acc_rf']):.2f}"
        })

pd.DataFrame(loso_summary).to_csv(LOSO_DIR / "BI_LOSO_Results.csv", index=False)

for shots, cm in confusion_matrices.items():
    pd.DataFrame(cm, index=GESTURES, columns=GESTURES).to_csv(LOSO_DIR / f"BI_CM_{shots}shot.csv")

for shots in CALIBRATION_SHOTS:
    if loso_subject_wise[shots]:
        pd.DataFrame([{'Subject': k, 'Accuracy': v} 
                     for k, v in loso_subject_wise[shots].items()]).to_csv(
            LOSO_DIR / f"BI_LOSO_SubjectWise_{shots}shot.csv", index=False)

with open(PREDS_DIR / "BI_LOSO_predictions.pkl", 'wb') as f:
    pickle.dump(loso_all_predictions, f)
with open(PREDS_DIR / "BI_LOSO_probabilities.pkl", 'wb') as f:
    pickle.dump(loso_all_probabilities, f)

# ================================================================
# SAVE FINAL MODELS
# ================================================================
log_message("\n📦 Training final deployment models...")
xgb_final, rf_final = get_models()
scaler_final = RobustScaler()
X_scaled = scaler_final.fit_transform(X)
xgb_final.fit(X_scaled, y_enc)
rf_final.fit(X_scaled, y_enc)
joblib.dump(xgb_final, MODELS_DIR / "BI_xgb_final.pkl")
joblib.dump(rf_final, MODELS_DIR / "BI_rf_final.pkl")
joblib.dump(scaler_final, MODELS_DIR / "BI_scaler.pkl")

log_message(f"\n✅ Models saved to {MODELS_DIR}")
log_message("\n" + "="*80)
log_message("✅ BI_BLOCK_0 COMPLETED!")
log_message("="*80)