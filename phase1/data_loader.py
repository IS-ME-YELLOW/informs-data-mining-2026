# ============================================================
# data_loader.py — 数据加载与预处理
# ============================================================
# 职责:
#   1. 加载 DM_Train.csv / DM_Test.csv / sample_submission.csv
#   2. 解析时间戳(必须用datetime排序, 字符串排序"10:00"<"9:00"是错的)
#   3. 为每行分配全局小时索引 _hour_idx (每县0-215)
#   4. 预处理:
#      - reconstruct_osi: 测试集无osi列, 从P_t/N_t/D_t/R_t按公式重建
#        (直接用存储的组件值, 不从outageCount重新推导N_t/R_t)
#      - encode_wind: 风向0-360°是圆形变量, 编码为sin/cos
#      - add_derived: 温度Kelvin→Celsius, 供派生特征用
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    TRAIN_FILE, TEST_FILE, SUBMISSION_FILE,
    OSI_WEIGHTS, OBSERVED_END, SEED,
)


def _parse_ts(ts):
    """解析时间戳 '3/14/2026 0:00' → datetime对象"""
    return datetime.strptime(ts, '%m/%d/%Y %H:%M')


def load_train():
    """加载训练集: 51,624行×63列, 按县+时间排序, 添加_hour_idx(0-215)"""
    df = pd.read_csv(TRAIN_FILE)
    df['_dt'] = df['timestamp_et'].apply(_parse_ts)
    df = df.sort_values(['fipsCode', '_dt']).reset_index(drop=True)
    df['_hour_idx'] = df.groupby('fipsCode').cumcount()
    return df


def load_test():
    """加载测试集: 13,608行×43列, 同上排序+索引"""
    df = pd.read_csv(TEST_FILE)
    df['_dt'] = df['timestamp_et'].apply(_parse_ts)
    df = df.sort_values(['fipsCode', '_dt']).reset_index(drop=True)
    df['_hour_idx'] = df.groupby('fipsCode').cumcount()
    return df


def load_submission():
    """加载提交模板: 9,072行(63县×144小时), 8列"""
    return pd.read_csv(SUBMISSION_FILE)


def reconstruct_osi(df):
    """
    重建OSI列: OSI = 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t (clip>=0)
    训练集已有osi列(直接保留); 测试集无此列, 需从组件重建(仅3月11-13有值)
    """
    if 'osi' in df.columns:
        return df
    p = df['P_t'].values
    n = df['N_t'].values
    d = df['D_t'].values
    r = df['R_t'].values
    mask = ~(np.isnan(p) | np.isnan(n) | np.isnan(d) | np.isnan(r))
    osi = np.full(len(df), np.nan)
    osi[mask] = (
        OSI_WEIGHTS['P'] * p[mask]
        + OSI_WEIGHTS['N'] * n[mask]
        + OSI_WEIGHTS['D'] * d[mask]
        - OSI_WEIGHTS['R'] * r[mask]
    )
    osi = np.where(osi < 0, 0, osi)
    df['osi'] = osi
    return df


def encode_wind(df):
    """
    风向编码: wind_dir_10m是圆形变量(0-360°), 359°和1°物理相近但数值差358
    必须编码为 sin/cos, 不可直接用原始角度作为特征
    """
    rad = np.radians(df['wind_dir_10m'].values)
    df['wind_sin'] = np.sin(rad)
    df['wind_cos'] = np.cos(rad)
    return df


def add_derived(df):
    """派生温度列: Kelvin → Celsius, 供结冰风险等特征计算"""
    df['temp_c'] = df['t2m'] - 273.15
    df['dewpoint_c'] = df['d2m'] - 273.15
    return df


def preprocess(df):
    """预处理主入口: 依次执行 OSI重建 → 风向编码 → 温度转换"""
    df = reconstruct_osi(df)
    df = encode_wind(df)
    df = add_derived(df)
    return df
