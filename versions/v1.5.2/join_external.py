"""
join_external.py — 将外部县级特征拼接到已有特征矩阵上
============================================================
用途: 加载 v1.5.1 缓存特征, join 外部县级特征, 存为 v1.5.2 缓存
不需要重算已有的 129 维特征, 几秒完成。

新增特征(11维静态 + 1维交互 = 12维):
  - rural_urban_code, is_metro, pop_density, n_utilities (城市/基础设施)
  - pct_forest, pct_developed, pct_agriculture, pct_water, pct_wetland (土地覆盖)
  - tree_canopy_pct, tree_canopy_std (树冠)
  - forest_x_customers (交互: pct_forest × log_customers)

用法: python3 join_external.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'phase1'))

from config import (
    CACHE_DIR, DATA_DIR, FEATURE_VERSION,
)
import cache as cache_mod
from features import ALL_FEATURE_NAMES as BASE_FEATURE_NAMES

# 外部特征列名(与county_features.csv一致, 不含population/area中间变量)
EXTERNAL_STATIC = [
    'rural_urban_code', 'is_metro', 'pop_density', 'n_utilities',
    'pct_forest', 'pct_developed', 'pct_agriculture', 'pct_water', 'pct_wetland',
    'tree_canopy_pct', 'tree_canopy_std',
]
# 交互特征(需要log_customers, 在join时计算)
INTERACTION_FEATURES = ['forest_x_customers']

ALL_EXTERNAL = EXTERNAL_STATIC + INTERACTION_FEATURES

# 源版本(从哪个缓存加载)
SRC_VERSION = 'v1.5.1'
DST_VERSION = FEATURE_VERSION  # 'v1.5.2'

COUNTY_FEATURES_PATH = os.path.join(DATA_DIR, 'data', 'county_features.csv')


def load_v151_cache(is_train):
    """手动加载v1.5.1缓存(绕过cache.py的版本检查)"""
    tag = 'train' if is_train else 'test'
    xp = os.path.join(CACHE_DIR, f'features_{tag}_{SRC_VERSION}.parquet')
    mp = os.path.join(CACHE_DIR, f'meta_{tag}_{SRC_VERSION}.parquet')
    X = pd.read_parquet(xp)
    meta = pd.read_parquet(mp)
    y = None
    if is_train:
        tp = os.path.join(CACHE_DIR, f'targets_train_{SRC_VERSION}.parquet')
        y = pd.read_parquet(tp) if os.path.exists(tp) else None
    return X, y, meta


def save_v152_cache(X, y, meta, is_train):
    """保存为v1.5.2缓存"""
    tag = 'train' if is_train else 'test'
    ext = 'parquet'
    X.to_parquet(os.path.join(CACHE_DIR, f'features_{tag}_{DST_VERSION}.{ext}'), index=False)
    meta.to_parquet(os.path.join(CACHE_DIR, f'meta_{tag}_{DST_VERSION}.{ext}'), index=False)
    if is_train and y is not None:
        y.to_parquet(os.path.join(CACHE_DIR, f'targets_train_{DST_VERSION}.{ext}'), index=False)
    # 保存特征列名JSON
    import json
    all_names = list(X.columns)
    with open(os.path.join(CACHE_DIR, f'feature_names_{DST_VERSION}.json'), 'w') as f:
        json.dump(all_names, f)
    print(f'  Saved: X={X.shape}, meta={meta.shape} ({tag})')


def main():
    print(f'=== Join External Features ({SRC_VERSION} → {DST_VERSION}) ===')
    print(f'  基础特征: {len(BASE_FEATURE_NAMES)} 维')
    print(f'  外部静态: {len(EXTERNAL_STATIC)} 维')
    print(f'  交互: {len(INTERACTION_FEATURES)} 维')
    print(f'  合计: {len(BASE_FEATURE_NAMES) + len(ALL_EXTERNAL)} 维')

    # 1. 加载县级外部特征
    print(f'\n[1/3] 加载 county_features.csv')
    county = pd.read_csv(COUNTY_FEATURES_PATH, dtype={'fipsCode': str})
    county['fipsCode'] = county['fipsCode'].str.zfill(5)
    county = county.set_index('fipsCode')
    print(f'  {len(county)} 县 × {len(county.columns)} 特征')

    # 2. 训练集
    print(f'\n[2/3] 训练集 join')
    X_train, y_train, meta_train = load_v151_cache(is_train=True)
    print(f'  原始: {X_train.shape}')

    # 按meta中的fipsCode join外部特征
    fips_train = meta_train['fipsCode'].astype(str).str.zfill(5).values
    ext_train = county.loc[fips_train].reset_index(drop=True)

    # 计算交互特征
    log_c_train = X_train['log_customers'].values
    ext_train['forest_x_customers'] = ext_train['pct_forest'].values * log_c_train

    # 横向拼接
    X_train_new = pd.concat([X_train.reset_index(drop=True), ext_train.reset_index(drop=True)], axis=1)
    assert len(X_train_new) == len(X_train), f'行数不匹配: {len(X_train_new)} vs {len(X_train)}'
    print(f'  拼接后: {X_train_new.shape}')
    save_v152_cache(X_train_new, y_train, meta_train, is_train=True)

    # 3. 测试集
    print(f'\n[3/3] 测试集 join')
    X_test, _, meta_test = load_v151_cache(is_train=False)
    print(f'  原始: {X_test.shape}')

    fips_test = meta_test['fipsCode'].astype(str).str.zfill(5).values
    ext_test = county.loc[fips_test].reset_index(drop=True)

    log_c_test = X_test['log_customers'].values
    ext_test['forest_x_customers'] = ext_test['pct_forest'].values * log_c_test

    X_test_new = pd.concat([X_test.reset_index(drop=True), ext_test.reset_index(drop=True)], axis=1)
    assert len(X_test_new) == len(X_test), f'行数不匹配: {len(X_test_new)} vs {len(X_test)}'
    assert list(X_test_new.columns) == list(X_train_new.columns), '列名不一致!'
    print(f'  拼接后: {X_test_new.shape}')
    save_v152_cache(X_test_new, None, meta_test, is_train=False)

    # 验证
    print(f'\n=== 验证 ===')
    print(f'  训练集: {X_train_new.shape}')
    print(f'  测试集: {X_test_new.shape}')
    print(f'  列名一致: {list(X_train_new.columns) == list(X_test_new.columns)}')
    print(f'  NaN总数: {X_train_new.isna().sum().sum()} (train)')
    print(f'  新增特征: {ALL_EXTERNAL}')
    print(f'\n完成! 下一步: cd phase1 && python3 main.py')


if __name__ == '__main__':
    main()
