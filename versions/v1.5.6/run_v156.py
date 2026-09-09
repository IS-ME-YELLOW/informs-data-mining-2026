"""run_v156.py — 用 v1.5.6 缓存(163列)跑 LightGBM 基线"""
import sys, os, json
# 必须在所有import之前移除Windows pyarrow
sys.path = [p for p in sys.path if '.python_packages' not in p]
sys.path.insert(0, 'code_phase1')

import numpy as np
import pandas as pd
from config import (FEATURE_VERSION, CV_VERSION, SEED, HORIZONS, HORIZON_HOURS,
                     LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING_ROUNDS,
                     N_FOLDS, LOG_FILE, OUTPUT_FILE, SUBMISSION_FILE,
                     OSI_MAX_OBSERVED, MODEL_DIR)
from cv import get_cv_folds
from model import train_all_horizons, predict, save_model
from evaluate import compute_metrics, compute_all_baselines, feature_importance, pred_stats, post_process
from logger import ExperimentLogger
import lightgbm as lgb

CACHE = 'cache'
VER = 'v1.5.6'

# 直接加载parquet
X_train = pd.read_parquet(f'{CACHE}/features_train_{VER}.parquet')
y_train = pd.read_parquet(f'{CACHE}/targets_train_{VER}.parquet')
meta_train = pd.read_parquet(f'{CACHE}/meta_train_{VER}.parquet')
X_test = pd.read_parquet(f'{CACHE}/features_test_{VER}.parquet')
meta_test = pd.read_parquet(f'{CACHE}/meta_test_{VER}.parquet')
submission = pd.read_csv(SUBMISSION_FILE)

print(f"=== LightGBM ({VER}, {X_train.shape[1]} cols) ===")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# CV
folds = get_cv_folds(meta_train)
last_osis = X_train['last_osi'].values
print(f"Folds: {len(folds)}, fold0: train={len(folds[0][0])}, val={len(folds[0][1])}")

# Train
results = train_all_horizons(X_train, y_train, folds, LGBM_PARAMS)

# Baselines
baselines = compute_all_baselines(y_train, last_osis, folds)

# Predict
preds = {}
for h in HORIZONS:
    preds[h] = predict(results[h]['model'], X_test)
    preds[h] = post_process(preds[h])

# Fill submission
def fill_submission(sub, preds, meta_test, horizon_hours):
    s = sub.copy()
    key_to_idx = {}
    for i, row in s.iterrows():
        key_to_idx[(str(row['fipsCode']), row['timestamp_et'])] = i
    for h in HORIZONS:
        h_val = horizon_hours[h]
        col = s.columns.get_loc(h)
        for j, row in meta_test.iterrows():
            key = (str(row['fipsCode']), row['timestamp_et'])
            if key in key_to_idx:
                idx = key_to_idx[key]
                if row['hour_idx'] + h_val > 215:
                    s.iat[idx, col] = np.nan
                else:
                    s.iat[idx, col] = preds[h][j]
    return s

sub = fill_submission(submission, preds, meta_test, HORIZON_HOURS)
sub.to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")

# Save models
os.makedirs(MODEL_DIR, exist_ok=True)
for h in HORIZONS:
    save_model(results[h]['model'], h, f'{VER}_{CV_VERSION}')

# Log
log = ExperimentLogger(LOG_FILE)
config_dict = {
    'feature_version': VER, 'cv_version': CV_VERSION, 'seed': SEED,
    'feature_count': X_train.shape[1], 'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0], 'n_folds': N_FOLDS,
    'lgbm_params': {k: v for k, v in LGBM_PARAMS.items() if k != 'verbose'},
    'num_boost_round': LGBM_NUM_BOOST_ROUND, 'early_stopping_rounds': LGBM_EARLY_STOPPING_ROUNDS,
}
log.run_header(config_dict)
log.log(f"Train samples per horizon: " + ', '.join(f"{h}: {y_train[h].notna().sum()}" for h in HORIZONS))
log.log(f"Feature count: {X_train.shape[1]}")
log.log('')
for h in HORIZONS:
    log.cv_results(h, results[h]['fold_metrics'], results[h]['summary'])
    log.feature_importance(h, feature_importance(results[h]['model'], top_n=15))
    log.prediction_summary(h, pred_stats(preds[h]))
log.baselines(baselines)
log.notes(f"Feature version {VER}. Features: {X_train.shape[1]}. Models saved. Submission: {OUTPUT_FILE}")
log.run_footer()

print("\n=== Done ===")
for h in HORIZONS:
    s = results[h]['summary']
    print(f"  {h}: RMSE={s['rmse']:.6f} ± {s['rmse_std']:.6f}, MAE={s['mae']:.6f} ± {s['mae_std']:.6f}")
