# ============================================================
# evaluate.py — 评估指标与基线
# ============================================================
# 职责:
#   1. compute_metrics: 计算RMSE/MAE(自动跳过NaN)
#   2. 三个基线: Zero(全零)、Persistence(用last_osi)、Mean(训练均值)
#   3. feature_importance: 提取LightGBM的top-N特征(gain)
#   4. pred_stats: 预测值分布统计(min/max/mean/zero%)
#   5. compute_all_baselines: 对4个horizon统一计算3个基线的CV指标
# ============================================================

import numpy as np
from config import HORIZONS, OBSERVED_END


def compute_metrics(y_true, y_pred):
    """计算RMSE和MAE, 自动跳过NaN行"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {'rmse': np.nan, 'mae': np.nan}
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae = np.mean(np.abs(yt - yp))
    return {'rmse': rmse, 'mae': mae}


def zero_baseline(y_true):
    """零基线: 全部预测为0(OSI约44%为零, 这是最低基准)"""
    return np.zeros_like(np.asarray(y_true, dtype=float))


def persistence_baseline(last_osis):
    """持续性基线: 预测=最后观测到的OSI(短期有效, 长期很差)"""
    return np.asarray(last_osis, dtype=float).copy()


def mean_baseline(y_true):
    """均值基线: 全部预测为训练集OSI均值"""
    y = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y)
    return np.full(len(y), np.nanmean(y[mask]))


def feature_importance(model, top_n=15):
    """提取LightGBM特征重要性(按gain排序, 返回top-N)"""
    importance = model.feature_importance(importance_type='gain')
    names = model.feature_name()
    pairs = sorted(zip(names, importance), key=lambda x: -x[1])
    return pairs[:top_n]


def pred_stats(pred):
    """预测值分布统计: min/max/mean/zero%"""
    pred = np.asarray(pred, dtype=float)
    mask = ~np.isnan(pred)
    p = pred[mask]
    if len(p) == 0:
        return {'min': 0, 'max': 0, 'mean': 0, 'zero_pct': 0}
    return {
        'min': p.min(),
        'max': p.max(),
        'mean': p.mean(),
        'zero_pct': 100 * np.mean(p == 0),
    }


<<<<<<< HEAD
def post_process(preds, threshold=0.001):
    """
    预测后处理: 低于阈值的预测设为0
    解决Huber loss不倾向预测精确零值的问题
    OSI < 0.001 意味着停电比例极低, 可视为无停电
    """
    preds = np.asarray(preds, dtype=float)
    preds = np.where(preds < threshold, 0.0, preds)
    return preds


=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
def compute_all_baselines(y_df, last_osis, folds):
    """
    对4个horizon统一计算3个基线(Zero/Mean/Persistence)的CV指标
    返回: {horizon: {baseline_name: {rmse, mae}}}
    """
    results = {}
    for h in HORIZONS:
        y = y_df[h].values
        zero_pred = zero_baseline(y)
        mean_pred = mean_baseline(y)
        pers_pred = persistence_baseline(last_osis)

        all_metrics = {}
        for name, pred in [('Zero', zero_pred), ('Mean', mean_pred),
                           ('Persistence', pers_pred)]:
            fold_rmse = []
            fold_mae = []
            for _, va_idx in folds:
                m = compute_metrics(y[va_idx], pred[va_idx])
                fold_rmse.append(m['rmse'])
                fold_mae.append(m['mae'])
            all_metrics[name] = {
                'rmse': np.mean(fold_rmse),
                'mae': np.mean(fold_mae),
            }
        results[h] = all_metrics
    return results
