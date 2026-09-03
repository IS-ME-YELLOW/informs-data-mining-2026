# ============================================================
# model.py — LightGBM 训练与预测
# ============================================================
# 职责:
#   1. train_one_horizon: 对单个horizon做5折CV训练, 返回最终模型+OOF预测+逐折指标
#   2. train_all_horizons: 循环训练4个horizon的独立模型
#   3. predict: 用模型预测测试集, clip到[0, 0.65]
#   4. save_model / load_model: 模型存盘/加载
#
# 关键设计:
#   - 每个horizon独立建模(t+48h与t+1h相关仅0.056, 动态差异大)
#   - Huber损失: 对43%零膨胀和极端值鲁棒
#   - 5折CV: 每折用4折训练+1折验证(early stopping), 再用全量+平均best_iter重训最终模型
#   - OOF(out-of-fold)预测: 用于后续分析和评估
#   - 索引重映射: folds基于全量34416行, 但每个horizon的NaN行数不同, 需映射
# ============================================================

import numpy as np
import lightgbm as lgb

from config import (
    LGBM_PARAMS, LGBM_NUM_BOOST_ROUND, LGBM_EARLY_STOPPING_ROUNDS,
    HORIZONS, SEED, OSI_MAX_OBSERVED, MODEL_DIR,
)
from evaluate import compute_metrics
import os


def train_one_horizon(X, y_col, folds, params=None):
    """
    训练单个horizon的LightGBM模型

    流程:
      1. 过滤NaN目标行(每个horizon末尾有1/6/24/48行NaN)
      2. 建立全量→过滤后的索引映射(folds基于全量, 需重映射)
      3. 5折CV: 每折训练+early stopping, 记录OOF预测和逐折RMSE/MAE
      4. 用全量数据+平均best_iter重训最终模型

    返回: (final_model, oof预测, 逐折指标, 汇总指标)
    """
    if params is None:
        params = LGBM_PARAMS.copy()

    n_full = len(X)
    oof = np.full(n_full, np.nan)          # OOF预测(全量长度, NaN位置保持NaN)
    fold_metrics = []
    models = []

    # 过滤NaN目标行 + 建立索引映射
    valid_mask = y_col.notna().values
    valid_indices = np.where(valid_mask)[0]
    X_valid = X[valid_mask].reset_index(drop=True)
    y_valid = y_col[valid_mask].reset_index(drop=True)
    oof_valid = np.full(len(X_valid), np.nan)

    # 全量索引→过滤后索引的映射(无效行映射为-1)
    full_to_filt = np.full(n_full, -1, dtype=int)
    full_to_filt[valid_indices] = np.arange(len(valid_indices))

    for fi, (train_idx, val_idx) in enumerate(folds):
        # 将全量fold索引映射到过滤后空间, 丢弃-1(无效行)
        tr_f = full_to_filt[train_idx]
        tr_f = tr_f[tr_f >= 0]
        va_f = full_to_filt[val_idx]
        va_f = va_f[va_f >= 0]

        X_tr = X_valid.iloc[tr_f]
        y_tr = y_valid.iloc[tr_f]
        X_va = X_valid.iloc[va_f]
        y_va = y_valid.iloc[va_f]

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr)

        callbacks = [
            lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            params,
            dtr,
            num_boost_round=LGBM_NUM_BOOST_ROUND,
            valid_sets=[dva],
            callbacks=callbacks,
        )

        pred_val = model.predict(X_va)
        oof_valid[va_f] = pred_val
        metrics = compute_metrics(y_va.values, pred_val)
        metrics['best_iter'] = model.best_iteration
        fold_metrics.append(metrics)
        models.append(model)

    # 将OOF从过滤后空间映射回全量空间
    oof[valid_indices] = oof_valid

    # 汇总指标
    summary = {
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'rmse_std': np.std([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'mae_std': np.std([m['mae'] for m in fold_metrics]),
    }

    # 用全量数据重训最终模型(用5折平均best_iter)
    full_dtr = lgb.Dataset(X_valid, label=y_valid)
    best_iter = int(np.mean([m['best_iter'] for m in fold_metrics]))
    final_model = lgb.train(
        params,
        full_dtr,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(0)],
    )

    return final_model, oof, fold_metrics, summary


def train_all_horizons(X, y_df, folds, params=None):
    """循环训练4个horizon的独立模型, 返回dict{horizon: {model, oof, metrics, summary}}"""
    results = {}
    for h in HORIZONS:
        print(f"\n--- Training {h} ---")
        model, oof, fold_metrics, summary = train_one_horizon(
            X, y_df[h], folds, params
        )
        results[h] = {
            'model': model,
            'oof': oof,
            'fold_metrics': fold_metrics,
            'summary': summary,
        }
        print(f"  Mean RMSE={summary['rmse']:.6f} ± {summary['rmse_std']:.6f}, "
              f"MAE={summary['mae']:.6f} ± {summary['mae_std']:.6f}")
    return results


def predict(model, X_test):
    """预测测试集, clip到[0, 0.65](OSI观测最大值)"""
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, OSI_MAX_OBSERVED)
    return pred


def save_model(model, horizon, version='v1'):
    """保存模型到 models/lgbm_{horizon}_{version}.txt"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f'lgbm_{horizon}_{version}.txt')
    model.save_model(path)


def load_model(horizon, version='v1'):
    """从文件加载模型"""
    path = os.path.join(MODEL_DIR, f'lgbm_{horizon}_{version}.txt')
    booster = lgb.Booster(model_file=path)
    return booster
