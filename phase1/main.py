# ============================================================
# main.py — Phase 1 主流程编排
# ============================================================
# 执行9步流程:
#   1. 加载数据(训练/测试/提交模板)
#   2. 预处理(OSI重建、风向编码、温度转换)
#   3. 构建特征(有缓存则加载, 无则计算并存盘)
#   4. 设置CV折数(分层分组5折)
#   5. 训练4个LightGBM模型(逐horizon, 5折CV+early stopping)
#   6. 计算基线(Zero/Mean/Persistence)
#   7. 预测测试集
#   8. 填充提交文件(超3/19的行设为NaN)
#   9. 写实验日志(配置+CV结果+特征重要性+基线对比)
#
# 用法:
#   python main.py                    # 用缓存特征训练
#   python main.py --rebuild-features  # 强制重建特征
# ============================================================

import argparse
import numpy as np
import pandas as pd

from config import (
    SEED, FEATURE_VERSION, HORIZONS, HORIZON_HOURS, LGBM_PARAMS,
    LOG_FILE, OUTPUT_FILE, TRAIN_FILE, TEST_FILE, SUBMISSION_FILE,
    LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING_ROUNDS, N_FOLDS,
)
from data_loader import load_train, load_test, load_submission, preprocess
from cache import get_or_build
from cv import get_cv_folds
from model import train_all_horizons, predict, save_model
from evaluate import compute_metrics, compute_all_baselines, feature_importance, pred_stats
from logger import ExperimentLogger


def parse_args():
    """解析命令行参数: --rebuild-features 强制重建特征缓存"""
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild-features', action='store_true', default=False)
    return p.parse_args()


def fill_submission(submission, preds, meta_test, horizon_hours):
    """
    填充提交文件:
    - 以sample_submission.csv为模板(不改标识列和行顺序)
    - 按行匹配 (fipsCode, timestamp_et)
    - 超出3/19 23:00的目标(hour_idx + h > 215)设为NaN
    - 仅填4个评分目标列
    """
    sub = submission.copy()
    # 建立 (fips, timestamp) → 行索引 的查找表
    key_to_idx = {}
    for i, row in sub.iterrows():
        key = (str(row['fipsCode']), row['timestamp_et'])
        key_to_idx[key] = i

    for h_idx, h in enumerate(HORIZONS):
        h_val = horizon_hours[h]
        col = sub.columns.get_loc(h)
        for j, row in meta_test.iterrows():
            key = (str(row['fipsCode']), row['timestamp_et'])
            if key in key_to_idx:
                idx = key_to_idx[key]
                hour_idx = row['hour_idx']
                if hour_idx + h_val > 215:
                    sub.iat[idx, col] = np.nan      # 超出数据末尾, 保持NaN
                else:
                    sub.iat[idx, col] = preds[h][j]
    return sub


def main():
    args = parse_args()

    log = ExperimentLogger(LOG_FILE)

    print("=== Phase 1: LightGBM Baseline ===")

    # 步骤1: 加载数据
    print("[1/9] Loading data...")
    train_df = load_train()
    test_df = load_test()
    submission = load_submission()

    # 步骤2: 预处理(OSI重建、风向sin/cos编码、温度Kelvin→Celsius)
    print("[2/9] Preprocessing (OSI reconstruction, wind encoding)...")
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # 步骤3: 构建特征(有缓存则秒级加载, 无则计算并存盘)
    print("[3/9] Building features...")
    X_train, y_train, meta_train = get_or_build(train_df, is_train=True,
                                                 force_rebuild=args.rebuild_features)
    X_test, _, meta_test = get_or_build(test_df, is_train=False,
                                      force_rebuild=args.rebuild_features)

    # 步骤4: 设置CV折数(分层分组5折: group=fipsCode, stratify=州×severity_tier)
    print("[4/9] Setting up CV folds...")
    folds = get_cv_folds(meta_train)
    last_osis = X_train['last_osi'].values   # 持续性基线用

    # 步骤5: 训练4个独立LightGBM模型(逐horizon, 5折CV+early stopping+全量重训)
    print("[5/9] Training models...")
    results = train_all_horizons(X_train, y_train, folds, LGBM_PARAMS)

    # 步骤6: 计算基线(Zero/Mean/Persistence)用于对比
    print("[6/9] Computing baselines...")
    baselines = compute_all_baselines(y_train, last_osis, folds)

    # 步骤7: 预测测试集(clip到[0, 0.65])
    print("[7/9] Predicting on test set...")
    preds = {}
    for h in HORIZONS:
        preds[h] = predict(results[h]['model'], X_test)

    # 步骤8: 填充提交文件(超3/19的行设NaN, 不改标识列和行顺序)
    print("[8/9] Filling submission...")
    sub = fill_submission(submission, preds, meta_test, HORIZON_HOURS)
    sub.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved: {OUTPUT_FILE}")

    # 步骤9: 保存模型
    for h in HORIZONS:
        save_model(results[h]['model'], h, FEATURE_VERSION)

    # === 写实验日志 ===
    print("\n[Log] Writing experiment log...")

    config_dict = {
        'feature_version': FEATURE_VERSION,
        'seed': SEED,
        'feature_count': X_train.shape[1],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'n_folds': N_FOLDS,
        'lgbm_params': {k: v for k, v in LGBM_PARAMS.items() if k != 'verbose'},
        'num_boost_round': LGBM_NUM_BOOST_ROUND,
        'early_stopping_rounds': LGBM_EARLY_STOPPING_ROUNDS,
    }
    log.run_header(config_dict)
    log.log(f"Train samples per horizon: " +
            ', '.join(f"{h}: {y_train[h].notna().sum()}" for h in HORIZONS))
    log.log(f"Feature names: {list(X_train.columns)}")
    log.log('')

    # 逐horizon记录: CV结果 + 特征重要性 + 预测分布
    for h in HORIZONS:
        log.cv_results(h, results[h]['fold_metrics'], results[h]['summary'])
        log.feature_importance(h, feature_importance(results[h]['model'], top_n=15))
        log.prediction_summary(h, pred_stats(preds[h]))

    log.baselines(baselines)

    notes = (
        f"Phase 1 baseline. Feature version {FEATURE_VERSION}. "
        f"Models saved to models/. "
        f"Submission: {OUTPUT_FILE}"
    )
    log.notes(notes)
    log.run_footer()

    # 打印汇总
    print("\n=== Done ===")
    print("Summary:")
    for h in HORIZONS:
        s = results[h]['summary']
        print(f"  {h}: RMSE={s['rmse']:.6f} ± {s['rmse_std']:.6f}, "
              f"MAE={s['mae']:.6f} ± {s['mae_std']:.6f}")


if __name__ == '__main__':
    main()
