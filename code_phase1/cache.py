# ============================================================
# cache.py — 特征缓存管理
# ============================================================
# 职责:
#   将计算好的特征矩阵存盘, 避免调参时重复计算
#   文件名含 FEATURE_VERSION, 改版本号即废弃旧缓存
#
#   历史通用缓存支持 parquet/pickle；v1.5.5 固定使用 parquet(pyarrow)
#   调参时直接从缓存加载, 秒级启动
#   v1.5.5 完整重建: python versions/v1.5.5/build_features.py --from-raw --overwrite
#   旧版数据集保持只读，不能用仅含基础特征的构建器覆盖
# ============================================================

import os
import pickle
import json
import pandas as pd

from config import CACHE_DIR, FEATURE_VERSION
from features import build_feature_matrix

# 检测是否有 pyarrow, 决定使用 parquet 还是 pickle
USE_PARQUET = True
try:
    import pyarrow
except ImportError:
    USE_PARQUET = False


def _cache_path(is_train):
    """特征矩阵缓存路径: cache/features_{train|test}_{version}.{parquet|pkl}"""
    tag = 'train' if is_train else 'test'
    ext = 'parquet' if USE_PARQUET else 'pkl'
    return os.path.join(CACHE_DIR, f'features_{tag}_{FEATURE_VERSION}.{ext}')


def _meta_path(is_train):
    """元信息缓存路径: cache/meta_{train|test}_{version}.{parquet|pkl}"""
    tag = 'train' if is_train else 'test'
    ext = 'parquet' if USE_PARQUET else 'pkl'
    return os.path.join(CACHE_DIR, f'meta_{tag}_{FEATURE_VERSION}.{ext}')


def _targets_path():
    """训练目标缓存路径(仅训练集)"""
    ext = 'parquet' if USE_PARQUET else 'pkl'
    return os.path.join(CACHE_DIR, f'targets_train_{FEATURE_VERSION}.{ext}')


def _names_path():
    """特征列名JSON路径"""
    return os.path.join(CACHE_DIR, f'feature_names_{FEATURE_VERSION}.json')


def _save_df(df, path):
    """保存DataFrame: parquet优先, 否则pickle"""
    if USE_PARQUET:
        df.to_parquet(path, index=False)
    else:
        with open(path, 'wb') as f:
            pickle.dump(df, f)


def _load_df(path):
    """加载DataFrame"""
    if USE_PARQUET:
        try:
            return pd.read_parquet(path)
        except OSError:
            # Existing feature caches were written by a Parquet implementation
            # that current PyArrow cannot read on this machine.
            return pd.read_parquet(path, engine='fastparquet')
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_features(X, y, meta, is_train):
    """保存特征矩阵 + 元信息 + 目标(仅训练) + 列名JSON"""
    if FEATURE_VERSION in ('v1.5.2', 'v1.5.3', 'v1.5.5', 'v1.5.6', 'v1.5.7'):
        raise ValueError('Versioned datasets cannot be written by the partial feature builder. '
                         'Use the version-specific build_features.py.')
    os.makedirs(CACHE_DIR, exist_ok=True)
    _save_df(X, _cache_path(is_train))
    _save_df(meta, _meta_path(is_train))
    if is_train and y is not None:
        _save_df(y, _targets_path())
    with open(_names_path(), 'w') as f:
        json.dump(list(X.columns), f)
    fmt = 'parquet' if USE_PARQUET else 'pickle'
    print(f"  [cache] Saved ({fmt}): X={X.shape}, meta={meta.shape}")


def load_features(is_train):
    """加载缓存: 不存在返回(None, None, None)"""
    if FEATURE_VERSION == 'v1.5.5':
        from feature_dataset import load_feature_dataset
        data = load_feature_dataset()
        return ((data.X_train, data.y_train, data.meta_train) if is_train
                else (data.X_test, None, data.meta_test))
    if FEATURE_VERSION == 'v1.5.6':
        from feature_dataset_v156 import load_feature_dataset
        data = load_feature_dataset()
        return ((data.X_train, data.y_train, data.meta_train) if is_train
                else (data.X_test, None, data.meta_test))
    if FEATURE_VERSION == 'v1.5.7':
        # v1.5.7 = v1.5.6 + F1 features, loaded directly from parquet
        tag = 'train' if is_train else 'test'
        xp = _cache_path(is_train)
        mp = _meta_path(is_train)
        if not os.path.exists(xp) or not os.path.exists(mp):
            return None, None, None
        X = _load_df(xp)
        meta = _load_df(mp)
        if is_train:
            tp = _targets_path()
            y = _load_df(tp) if os.path.exists(tp) else None
        else:
            y = None
        return X, y, meta
    xp = _cache_path(is_train)
    mp = _meta_path(is_train)
    if not os.path.exists(xp) or not os.path.exists(mp):
        return None, None, None
    X = _load_df(xp)
    meta = _load_df(mp)
    if is_train:
        tp = _targets_path()
        y = _load_df(tp) if os.path.exists(tp) else None
    else:
        y = None
    return X, y, meta


def get_or_build(df, is_train, force_rebuild=False):
    """
    主入口: 有缓存则加载, 无则计算并存盘
    - force_rebuild=True 时强制重新计算(--rebuild-features)
    - 调参时重复调用此函数, 首次计算后续直接加载
    """
    if force_rebuild and FEATURE_VERSION in ('v1.5.2', 'v1.5.3', 'v1.5.5', 'v1.5.6', 'v1.5.7'):
        raise ValueError('Use the version-specific build_features.py '
                         'to rebuild the complete dataset. Historical versions are read-only.')
    if not force_rebuild:
        X, y, meta = load_features(is_train)
        if X is not None:
            print(f"[cache] Loaded {'train' if is_train else 'test'} features: "
                  f"{X.shape[0]} rows x {X.shape[1]} cols")
            return X, y, meta
    if FEATURE_VERSION in ('v1.5.2', 'v1.5.3', 'v1.5.5', 'v1.5.6', 'v1.5.7'):
        raise FileNotFoundError('Versioned cache missing; use the version-specific build_features.py.')
    print(f"[cache] Building {'train' if is_train else 'test'} features...")
    X, y, meta = build_feature_matrix(df, is_train=is_train)
    save_features(X, y, meta, is_train)
    print(f"[cache] Saved: {X.shape[0]} rows x {X.shape[1]} cols")
    return X, y, meta
