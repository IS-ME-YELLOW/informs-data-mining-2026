# ============================================================
<<<<<<< HEAD
# features.py — 特征工程 (Phase 1.5, v2)
# ============================================================
# 职责: 从原始数据构建114维特征矩阵, 供LightGBM训练/预测使用
#
# 特征分10类:
#   A. 观测窗口停电摘要 (25维): 从3月11-13(72h)停电数据提取, 每县固定
=======
# features.py — 特征工程
# ============================================================
# 职责: 从原始数据构建82维特征矩阵, 供LightGBM训练/预测使用
#
# 特征分6类:
#   A. 观测窗口摘要 (25维): 从3月11-13(72h)停电数据提取, 每县固定
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
#   B. 当前气象 (24维): 预测时刻t的22个气象变量 + 风向sin/cos
#   C. 目标时段气象统计 (20维): [t, t+h]窗口内gust_max/mean等, 4个horizon×5统计
#   D. 派生气象 (6维): 结冰风险/阵风超限/土湿×风速交互等
#   E. 时间特征 (6维): 小时/天/距观测结束小时数/风暴阶段
#   F. 县级特征 (1维): log(customersTracked), 全程有值
#
<<<<<<< HEAD
#   --- Phase 1.5 新增 ---
#   G. 阈值超越时长 (10维): gust>30/40mph的累计小时数 [P1, 文献: Cerrai/Yang系列]
#   H. 峰值窗口条件均值 (4维): 最强风4h窗口均值 [P2, 文献: Cerrai/Yang系列]
#   I. 气象变化率 (6维): gust/pressure/temp的1h/3h/6h差分 [P3, 文献: Alpay 2020]
#   J. 气象×阶段交互 (4维): gust×hso, gust×phase等 [P4, 拆解时间-气象共线性]
#   K. 累计暴露量 (3维): 风暴开始以来的累计gust/tp [P5, 文献: Arora 2023]
#   L. 观测窗口气象边界 (3维): 末尾gust/6h趋势 [P6, 文献: Alpay 2020, STO-CAST]
#
=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
# 合规要点:
#   - 不使用预测窗口(3月14-19)的停电数据作输入(模拟测试条件)
#   - 气象可用任意时刻(含未来, 规则允许)
#   - 仅用观测窗口停电数据(3月11-13)提取县级摘要
# ============================================================

import numpy as np
import pandas as pd

from config import (
    OBSERVED_END, PRED_START, PRED_END,
    HORIZONS, HORIZON_HOURS, WEATHER_COLS,
)

# ============================================================
# 特征列名定义(保证训练/测试列名一致)
# ============================================================

<<<<<<< HEAD
# A. 观测窗口停电摘要 (25→32维, 新增P1观测2+P5观测1+P6边界3+原有25)
OBSERVED_FEATURES = [
    # 原有25维
=======
# A. 观测窗口摘要特征 (25维, 从3月11-13停电数据提取, 每县固定不变)
OBSERVED_FEATURES = [
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    'last_osi', 'last_P_t', 'last_N_t', 'last_D_t', 'last_R_t',
    'last_outage_pct',
    'osi_mean_72h', 'osi_max_72h', 'osi_std_72h',
    'outage_pct_mean_72h', 'outage_pct_max_72h',
    'N_t_mean_72h', 'N_t_max_72h', 'R_t_mean_72h', 'R_t_max_72h',
    'pre_event_mean_osi', 'pre_event_max_outage_pct',
    'storm_onset_max_osi', 'storm_onset_mean_osi',
    'osi_trend_last6h', 'is_worsening', 'frac_zero_72h',
    'hours_since_peak', 'gust_mean_obs', 'gust_max_obs',
<<<<<<< HEAD
    # P1: 观测窗口阈值超越时长 (文献: Cerrai/Yang 4篇)
    'gust_gt30_obs', 'gust_gt40_obs',
    # P5: 观测窗口累计风力暴露 (文献: Arora 2023)
    'cumul_gust_obs',
    # P6: 观测窗口末尾气象边界 (文献: Alpay 2020, STO-CAST 2026)
    'gust_last_obs', 'gust_max_last6h_obs', 'gust_trend_last6h_obs',
=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
]

# B. 预测时刻t的气象特征 (24维, 随t变化)
WEATHER_AT_T_FEATURES = [
    'gust_t', 'wind_speed_t', 'wind_sin_t', 'wind_cos_t',
    't2m_t', 'd2m_t', 'tp_t', 'rain_t', 'csnow_t', 'sdwe_t',
    'r2_t', 'vpd_t', 'soil_moist_t', 'blh_t', 'tcc_t', 'lcc_t',
    'mcc_t', 'hcc_t', 'sp_t', 'mslma_t',
    'sdswrf_t', 'direct_rad_t', 'diffuse_rad_t', 'et0_t',
]

<<<<<<< HEAD
# D. 派生气象特征 (6→10维, 新增P4交互4维)
DERIVED_WEATHER_FEATURES = [
    # 原有6维
    'temp_c_t', 'dewpoint_c_t', 'icing_risk_t',
    'gust_exceed_30', 'soil_moist_x_gust', 'is_snowing',
    # P4: 气象×阶段交互 (拆解时间-气象共线性)
    'gust_x_hso', 'gust_x_phase',
    'soil_gust_x_phase', 'gust_exceed30_x_hso',
]

# C. 目标时段气象统计 (20→32维, 新增P1阈值8+P2峰值4)
HORIZON_STAT_FEATURES = []
for _h in ['1', '6', '24', '48']:
    HORIZON_STAT_FEATURES.extend([
        # 原有5统计
        f'gust_max_next_{_h}h', f'gust_mean_next_{_h}h',
        f'wind_speed_max_next_{_h}h', f'total_tp_next_{_h}h',
        f'min_t2m_next_{_h}h',
        # P1: 阈值超越时长 (文献: Cerrai/Yang系列, 最验证有效的风特征)
        f'gust_gt30_next_{_h}h', f'gust_gt40_next_{_h}h',
        # P2: 峰值窗口条件均值 (文献: 最强风4h窗口均值 > 简单均值)
        f'gust_peak4h_mean_next_{_h}h',
    ])

# I. 气象变化率 (6维, P3新增, 文献: Alpay 2020 互相关分析)
WEATHER_CHANGE_FEATURES = [
    'gust_change_1h', 'gust_change_3h',
    'pressure_change_3h', 'pressure_change_6h',
    'temp_change_6h', 'wind_change_1h',
]

# K. 累计暴露量 (3维, P5新增, 文献: Arora 2023 前期条件)
CUMULATIVE_FEATURES = [
    'cumul_gust_since_onset', 'cumul_tp_since_onset', 'gust_gt30_since_onset',
]

# L. 目标时刻精确气象 (16维, 方向1新增, 文献: STO-CAST 2026 SHAP: 未来精确值>窗口统计)
# 对每个horizon h, 取t+h时刻的精确gust/wind_speed/t2m/tp值
# 数据验证: gust_at_t+48h与target corr=0.1651, vs gust_max_next_48h corr=0.1178 (+40%)
TARGET_TIME_WEATHER_FEATURES = []
for _h in ['1', '6', '24', '48']:
    TARGET_TIME_WEATHER_FEATURES.extend([
        f'gust_at_t{_h}h', f'wind_speed_at_t{_h}h',
        f't2m_at_t{_h}h', f'tp_at_t{_h}h',
=======
# D. 派生气象特征 (6维)
DERIVED_WEATHER_FEATURES = [
    'temp_c_t', 'dewpoint_c_t', 'icing_risk_t',
    'gust_exceed_30', 'soil_moist_x_gust', 'is_snowing',
]

# C. 目标时段气象统计 (20维 = 5统计 × 4个horizon)
HORIZON_STAT_FEATURES = []
for _h in ['1', '6', '24', '48']:
    HORIZON_STAT_FEATURES.extend([
        f'gust_max_next_{_h}h', f'gust_mean_next_{_h}h',
        f'wind_speed_max_next_{_h}h', f'total_tp_next_{_h}h',
        f'min_t2m_next_{_h}h',
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    ])

# E. 时间特征 (6维)
TEMPORAL_FEATURES = [
    'hour_of_day', 'hour_sin', 'hour_cos',
    'days_since_onset', 'hours_since_obs', 'storm_phase',
]

# F. 县级特征 (1维)
COUNTY_FEATURES = ['log_customers']

# 全部特征列名(顺序固定, 训练/测试必须一致)
ALL_FEATURE_NAMES = (
<<<<<<< HEAD
    OBSERVED_FEATURES              # 32
    + WEATHER_AT_T_FEATURES        # 24
    + DERIVED_WEATHER_FEATURES     # 10
    + HORIZON_STAT_FEATURES        # 32
    + WEATHER_CHANGE_FEATURES      # 6
    + CUMULATIVE_FEATURES          # 3
    + TARGET_TIME_WEATHER_FEATURES # 16 (方向1新增)
    + TEMPORAL_FEATURES            # 6
    + COUNTY_FEATURES              # 1
)                                  # 合计 130
=======
    OBSERVED_FEATURES
    + WEATHER_AT_T_FEATURES
    + DERIVED_WEATHER_FEATURES
    + HORIZON_STAT_FEATURES
    + TEMPORAL_FEATURES
    + COUNTY_FEATURES
)
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb


# ============================================================
# 各类特征的计算函数
# ============================================================

def compute_observed_features(county_df):
    """
<<<<<<< HEAD
    从3月11-13观测窗口(hour_idx 0-71)提取32维县级摘要特征
    这些特征对同一县的所有144个预测小时保持不变(静态)
    包括: 最后观测值、72h统计量、预事件基线、第一波信号、趋势、
          阈值超越(P1)、累计暴露(P5)、气象边界(P6)
=======
    从3月11-13观测窗口(hour_idx 0-71)提取25维县级摘要特征
    这些特征对同一县的所有144个预测小时保持不变(静态)
    包括: 最后观测值、72h统计量、预事件基线、第一波信号、趋势
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    """
    obs = county_df[county_df['_hour_idx'] < OBSERVED_END]
    if len(obs) == 0:
        return {name: np.nan for name in OBSERVED_FEATURES}

    last = obs.iloc[-1]           # 观测窗口最后一行(3月13日23:00)
    osi_vals = obs['osi'].dropna().values
    pct_vals = obs['outage_pct'].dropna().values
    n_vals = obs['N_t'].dropna().values
    r_vals = obs['R_t'].dropna().values
    gust_vals = obs['gust'].dropna().values

    pre = obs[obs['_hour_idx'] < 48]     # 3月11-12(预事件窗口)
    storm = obs[obs['_hour_idx'] >= 48]  # 3月13(风暴第一波)

    pre_osi = pre['osi'].dropna().values
    pre_pct = pre['outage_pct'].dropna().values
    storm_osi = storm['osi'].dropna().values

    # OSI趋势: 观测窗口最后6小时的线性回归斜率
    last6 = obs.tail(6)['osi'].dropna().values
    if len(last6) >= 2:
        x = np.arange(len(last6))
        trend = np.polyfit(x, last6, 1)[0] if np.std(last6) > 0 else 0.0
    else:
        trend = 0.0

    # 距观测窗口内OSI峰值已过去多少小时
    if len(osi_vals) > 0 and osi_vals.max() > 0:
        peak_idx = np.argmax(osi_vals)
        hours_since_peak = len(osi_vals) - 1 - peak_idx
    else:
        hours_since_peak = np.nan

<<<<<<< HEAD
    # P6: 观测窗口末尾气象边界 (文献: Alpay 2020 lag=1, STO-CAST O(-6))
    last6_gust = obs.tail(6)['gust'].dropna().values
    if len(last6_gust) >= 2:
        gx = np.arange(len(last6_gust))
        gust_trend = np.polyfit(gx, last6_gust, 1)[0] if np.std(last6_gust) > 0 else 0.0
    else:
        gust_trend = 0.0

    feat = {
        # 原有25维
=======
    feat = {
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
        'last_osi': last['osi'] if not np.isnan(last['osi']) else 0.0,
        'last_P_t': last['P_t'],
        'last_N_t': last['N_t'],
        'last_D_t': last['D_t'],
        'last_R_t': last['R_t'],
        'last_outage_pct': last['outage_pct'],
        'osi_mean_72h': np.mean(osi_vals) if len(osi_vals) > 0 else 0.0,
        'osi_max_72h': np.max(osi_vals) if len(osi_vals) > 0 else 0.0,
        'osi_std_72h': np.std(osi_vals) if len(osi_vals) > 0 else 0.0,
        'outage_pct_mean_72h': np.mean(pct_vals) if len(pct_vals) > 0 else 0.0,
        'outage_pct_max_72h': np.max(pct_vals) if len(pct_vals) > 0 else 0.0,
        'N_t_mean_72h': np.mean(n_vals) if len(n_vals) > 0 else 0.0,
        'N_t_max_72h': np.max(n_vals) if len(n_vals) > 0 else 0.0,
        'R_t_mean_72h': np.mean(r_vals) if len(r_vals) > 0 else 0.0,
        'R_t_max_72h': np.max(r_vals) if len(r_vals) > 0 else 0.0,
        'pre_event_mean_osi': np.mean(pre_osi) if len(pre_osi) > 0 else 0.0,
        'pre_event_max_outage_pct': np.max(pre_pct) if len(pre_pct) > 0 else 0.0,
        'storm_onset_max_osi': np.max(storm_osi) if len(storm_osi) > 0 else 0.0,
        'storm_onset_mean_osi': np.mean(storm_osi) if len(storm_osi) > 0 else 0.0,
        'osi_trend_last6h': trend,
        'is_worsening': 1.0 if (not np.isnan(last['N_t']) and not np.isnan(last['R_t'])
                                and last['N_t'] > last['R_t']) else 0.0,
        'frac_zero_72h': np.mean(osi_vals == 0) if len(osi_vals) > 0 else 1.0,
        'hours_since_peak': hours_since_peak,
        'gust_mean_obs': np.mean(gust_vals) if len(gust_vals) > 0 else 0.0,
        'gust_max_obs': np.max(gust_vals) if len(gust_vals) > 0 else 0.0,
<<<<<<< HEAD
        # P1: 观测窗口阈值超越时长 (30/40mph ≈ 13/18 m/s)
        'gust_gt30_obs': np.sum(gust_vals > 30) if len(gust_vals) > 0 else 0,
        'gust_gt40_obs': np.sum(gust_vals > 40) if len(gust_vals) > 0 else 0,
        # P5: 观测窗口累计风力暴露
        'cumul_gust_obs': np.sum(gust_vals) if len(gust_vals) > 0 else 0.0,
        # P6: 观测窗口末尾气象边界
        'gust_last_obs': last['gust'] if not np.isnan(last['gust']) else 0.0,
        'gust_max_last6h_obs': np.max(last6_gust) if len(last6_gust) > 0 else 0.0,
        'gust_trend_last6h_obs': gust_trend,
=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    }
    return feat


def compute_weather_at_t(row):
    """提取预测时刻t的22个气象变量(加风向sin/cos共24维)"""
    return {
        'gust_t': row['gust'],
        'wind_speed_t': row['wind_speed_10m'],
        'wind_sin_t': row['wind_sin'],
        'wind_cos_t': row['wind_cos'],
        't2m_t': row['t2m'],
        'd2m_t': row['d2m'],
        'tp_t': row['tp'],
        'rain_t': row['rain'],
        'csnow_t': row['csnow'],
        'sdwe_t': row['sdwe'],
        'r2_t': row['r2'],
        'vpd_t': row['vpd'],
        'soil_moist_t': row['soil_moist'],
        'blh_t': row['blh'],
        'tcc_t': row['tcc'],
        'lcc_t': row['lcc'],
        'mcc_t': row['mcc'],
        'hcc_t': row['hcc'],
        'sp_t': row['sp'],
        'mslma_t': row['mslma'],
        'sdswrf_t': row['sdswrf'],
        'direct_rad_t': row['direct_rad'],
        'diffuse_rad_t': row['diffuse_rad'],
        'et0_t': row['et0'],
    }


def compute_derived_weather(row, horizon_stats):
    """
    派生气象特征(6维):
    - temp_c/dewpoint_c: Kelvin→Celsius
    - icing_risk: 温度<0°C且湿度>80% → 结冰风险
    - gust_exceed_30: 未来1h最大阵风超过30mph(损害阈值)的幅度
    - soil_moist_x_gust: 土壤湿度×风速(树木倒伏风险交互)
    - is_snowing: 是否降雪
    """
    temp_c = row['temp_c']
    r2 = row['r2']
    gust = row['gust']
    soil = row['soil_moist']
    return {
        'temp_c_t': temp_c,
        'dewpoint_c_t': row['dewpoint_c'],
        'icing_risk_t': 1.0 if (temp_c < 0 and r2 > 80) else 0.0,
        'gust_exceed_30': max(0.0, horizon_stats.get('gust_max_next_1h', 0) - 30),
        'soil_moist_x_gust': soil * gust,
        'is_snowing': 1.0 if row['csnow'] > 0 else 0.0,
    }


<<<<<<< HEAD
def compute_interactions(row, feat_so_far):
    """
    P4: 气象×阶段交互 (4维)
    拆解时间-气象共线性: 同gust在不同风暴阶段(树未倒vs已倒)效果不同
    数据验证: t+48h中 target=0 时 gust 反而更高(24.8 vs 21.7)
    """
    gust = row['gust']
    hso = feat_so_far.get('hours_since_obs', 0)
    phase = feat_so_far.get('storm_phase', 0)
    soil = row['soil_moist']
    ge30 = feat_so_far.get('gust_exceed_30', 0)
    return {
        'gust_x_hso': gust * hso,
        'gust_x_phase': gust * phase,
        'soil_gust_x_phase': soil * gust * phase,
        'gust_exceed30_x_hso': ge30 * hso,
    }


def compute_horizon_stats(county_df, t_idx, h):
    """
    计算目标时段[t, t+h]的气象统计(原有5维):
=======
def compute_horizon_stats(county_df, t_idx, h):
    """
    计算目标时段[t, t+h]的气象统计:
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    - gust_max/mean: 最大/平均阵风(核心驱动)
    - wind_speed_max: 最大持续风速
    - total_tp: 累计降水
    - min_t2m: 最低温度(结冰风险)
    气象数据全216h可用, 允许查看未来(规则允许)
    """
    end = min(t_idx + h + 1, len(county_df))
    window = county_df.iloc[t_idx:end]
    if len(window) == 0:
        return {name: np.nan for name in
                ['gust_max', 'gust_mean', 'wind_speed_max', 'total_tp', 'min_t2m']}
    return {
        'gust_max': window['gust'].max(),
        'gust_mean': window['gust'].mean(),
        'wind_speed_max': window['wind_speed_10m'].max(),
        'total_tp': window['tp'].sum(),
        'min_t2m': window['t2m'].min(),
    }


<<<<<<< HEAD
def compute_threshold_exceedance(county_df, t_idx, h):
    """
    P1: 阈值超越时长 (文献: Cerrai 2019/2020, Yang 2020a/b — 4篇UConn系列)
    [t, t+h]中 gust>30mph 和 gust>40mph 的小时数
    30mph≈13m/s(损害开始), 40mph≈18m/s(显著损害)
    """
    end = min(t_idx + h + 1, len(county_df))
    window = county_df.iloc[t_idx:end]
    if len(window) == 0:
        return {'gust_gt30': np.nan, 'gust_gt40': np.nan}
    gusts = window['gust'].dropna()
    return {
        'gust_gt30': int((gusts > 30).sum()),
        'gust_gt40': int((gusts > 40).sum()),
    }


def compute_peak_window_mean(county_df, t_idx, h):
    """
    P2: 峰值窗口条件均值 (文献: Cerrai/Yang系列 "最强风4小时窗口均值")
    [t, t+h]中gust最高的min(4, h)小时的均值
    比简单全窗口均值保留更多破坏性信息
    """
    end = min(t_idx + h + 1, len(county_df))
    window = county_df.iloc[t_idx:end]
    if len(window) == 0:
        return np.nan
    gusts = window['gust'].dropna()
    n_top = min(4, len(gusts))
    if n_top == 0:
        return np.nan
    return gusts.nlargest(n_top).mean()


def compute_weather_changes(county_df, t_idx):
    """
    P3: 气象变化率 (文献: Alpay 2020 互相关分析, wind-outage最优滞后l=1)
    gust/pressure/temp/wind的1h/3h/6h差分
    捕捉风暴锋面过境动态, 与时间位置相关性低(不共线)
    """
    def safe_diff(col, lag):
        if t_idx - lag < 0:
            return np.nan
        cur = county_df.iloc[t_idx][col]
        prev = county_df.iloc[t_idx - lag][col]
        if pd.isna(cur) or pd.isna(prev):
            return np.nan
        return cur - prev

    return {
        'gust_change_1h': safe_diff('gust', 1),
        'gust_change_3h': safe_diff('gust', 3),
        'pressure_change_3h': safe_diff('sp', 3),
        'pressure_change_6h': safe_diff('sp', 6),
        'temp_change_6h': safe_diff('t2m', 6),
        'wind_change_1h': safe_diff('wind_speed_10m', 1),
    }


def compute_cumulative_exposure(county_df, t_idx):
    """
    P5: 累计暴露量 (文献: Arora 2023 前期条件SPI6, Cerrai 2020 事件总累积)
    从风暴开始(March 13 0:00, hour_idx=48)到t的累计gust/tp/损害性风力时长
    编码基础设施疲劳累积
    """
    start = 48  # March 13 0:00
    if t_idx < start:
        return {'cumul_gust_since_onset': 0.0, 'cumul_tp_since_onset': 0.0,
                'gust_gt30_since_onset': 0}
    window = county_df.iloc[start:t_idx + 1]
    gusts = window['gust'].dropna()
    return {
        'cumul_gust_since_onset': float(gusts.sum()) if len(gusts) > 0 else 0.0,
        'cumul_tp_since_onset': float(window['tp'].dropna().sum()),
        'gust_gt30_since_onset': int((gusts > 30).sum()) if len(gusts) > 0 else 0,
    }


def compute_target_time_weather(county_df, t_idx, horizons):
    """
    方向1: 目标时刻精确气象 (文献: STO-CAST 2026, SHAP证明未来精确值>窗口统计)
    对每个horizon h, 取 t+h 时刻的精确 gust/wind_speed/t2m/tp 值
    数据验证: gust_at_t+48h corr=0.1651 vs gust_max_next_48h corr=0.1178 (+40%)
    """
    result = {}
    for h_key, h_val in horizons.items():
        target_idx = t_idx + h_val
        h_short = str(h_val)
        if target_idx < len(county_df):
            row = county_df.iloc[target_idx]
            result[f'gust_at_t{h_short}h'] = row['gust']
            result[f'wind_speed_at_t{h_short}h'] = row['wind_speed_10m']
            result[f't2m_at_t{h_short}h'] = row['t2m']
            result[f'tp_at_t{h_short}h'] = row['tp']
        else:
            result[f'gust_at_t{h_short}h'] = np.nan
            result[f'wind_speed_at_t{h_short}h'] = np.nan
            result[f't2m_at_t{h_short}h'] = np.nan
            result[f'tp_at_t{h_short}h'] = np.nan
    return result


=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
def compute_temporal_features(dt, hour_idx):
    """
    时间特征(6维):
    - hour_of_day/sin/cos: 日内周期
    - days_since_onset: 风暴开始(3/13)后天数
    - hours_since_obs: 距最后观测(3/13 23:00)的小时数(1-144), 编码信息衰减
    - storm_phase: 0=第一波(3/14), 1=间歇(3/15), 2=第二波(3/16-17), 3=消退(3/18-19)
    """
    hour = dt.hour
    day = dt.day
    storm_start_day = 13
    return {
        'hour_of_day': hour,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'days_since_onset': max(0, day - storm_start_day),
        'hours_since_obs': hour_idx - (OBSERVED_END - 1),
        'storm_phase': {14: 0, 15: 1, 16: 2, 17: 2, 18: 3, 19: 3}.get(day, 0),
    }


def compute_county_features(row):
    """县级特征: log(customersTracked), 全程有值, 编码县规模"""
    ct = row.get('customersTracked', np.nan)
    return {'log_customers': np.log1p(ct) if ct and ct > 0 else 0.0}


# ============================================================
# 构建完整特征矩阵
# ============================================================

def build_feature_matrix(df, is_train=True):
    """
<<<<<<< HEAD
    主入口: 从原始数据构建114维特征矩阵

    流程:
      对每个县:
        1. 提取观测窗口(3月11-13) → 计算32维县级摘要(对所有预测小时固定)
        2. 对预测窗口(3月14-19)每个小时t:
           a. 提取t时刻气象(24维)
           b. 计算4个horizon的气象统计(20维原有)
           c. 计算阈值超越时长(8维, P1) + 峰值窗口均值(4维, P2)
           d. 计算派生气象(6维原有) + 交互(4维, P4)
           e. 计算气象变化率(6维, P3)
           f. 计算累计暴露(3维, P5)
           g. 计算时间特征(6维) + 县级特征(1维)
           h. 拼装为单行114维特征
      合并所有行 → X(特征), y(目标, 仅训练), meta(元信息)

    返回: X(DataFrame, 114列), y(DataFrame, 4列或None), meta(DataFrame)
=======
    主入口: 从原始数据构建特征矩阵

    流程:
      对每个县:
        1. 提取观测窗口(3月11-13) → 计算25维县级摘要(对所有预测小时固定)
        2. 对预测窗口(3月14-19)每个小时t:
           a. 提取t时刻气象(24维)
           b. 计算4个horizon的气象统计(20维)
           c. 计算派生气象(6维)
           d. 计算时间特征(6维)
           e. 计算县级特征(1维)
           f. 拼装为单行82维特征
      合并所有行 → X(特征), y(目标, 仅训练), meta(元信息)

    返回: X(DataFrame, 82列), y(DataFrame, 4列或None), meta(DataFrame)
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
    """
    rows_out = []
    meta_rows = []

    for fips, county_df in df.groupby('fipsCode'):
        county_df = county_df.sort_values('_hour_idx').reset_index(drop=True)
<<<<<<< HEAD
        obs_feat = compute_observed_features(county_df)  # 每县计算一次(32维静态)
=======
        obs_feat = compute_observed_features(county_df)  # 每县计算一次
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb

        pred_df = county_df[(county_df['_hour_idx'] >= PRED_START)
                            & (county_df['_hour_idx'] < PRED_END)]

        for _, row in pred_df.iterrows():
            t_idx = int(row['_hour_idx'])

<<<<<<< HEAD
            # --- 基础特征(Phase 1) ---
            feat = dict(obs_feat)
            feat.update(compute_weather_at_t(row))

            # 4个horizon的气象统计(原有5维×4=20, 允许查看未来)
=======
            # 拼装特征: 观测摘要(静态) + 气象(随t) + 派生 + 时间 + 县级
            feat = dict(obs_feat)
            feat.update(compute_weather_at_t(row))

            # 4个horizon的气象统计(允许查看未来, 规则允许)
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
            hs = {}
            for h_key, h_val in HORIZON_HOURS.items():
                stats = compute_horizon_stats(county_df, t_idx, h_val)
                h_short = str(h_val)
                for stat_name, stat_val in stats.items():
                    hs[f'{stat_name}_next_{h_short}h'] = stat_val
            feat.update(hs)

            feat.update(compute_derived_weather(row, hs))
            feat.update(compute_temporal_features(row['_dt'], t_idx))
            feat.update(compute_county_features(row))

<<<<<<< HEAD
            # --- Phase 1.5 新增特征 ---
            # P1: 阈值超越时长 (文献最验证有效的风特征)
            for h_key, h_val in HORIZON_HOURS.items():
                te = compute_threshold_exceedance(county_df, t_idx, h_val)
                h_short = str(h_val)
                for name, val in te.items():
                    feat[f'{name}_next_{h_short}h'] = val

            # P2: 峰值窗口条件均值 (最强风4h窗口均值 > 简单均值)
            for h_key, h_val in HORIZON_HOURS.items():
                h_short = str(h_val)
                feat[f'gust_peak4h_mean_next_{h_short}h'] = \
                    compute_peak_window_mean(county_df, t_idx, h_val)

            # P3: 气象变化率 (捕捉锋面过境, 与时间不共线)
            feat.update(compute_weather_changes(county_df, t_idx))

            # P4: 气象×阶段交互 (拆解时间-气象共线性, 需在时间特征后)
            feat.update(compute_interactions(row, feat))

            # P5: 累计暴露量 (基础设施疲劳累积)
            feat.update(compute_cumulative_exposure(county_df, t_idx))

            # 方向1: 目标时刻精确气象 (STO-CAST: 未来精确值>窗口统计, corr+40%)
            feat.update(compute_target_time_weather(county_df, t_idx, HORIZON_HOURS))

=======
>>>>>>> 0541cc420ad1f0f3fd384b1aeb9c3d2f63ee72fb
            rows_out.append(feat)

            # 元信息(用于CV分组、提交匹配)
            meta_row = {
                'fipsCode': fips,
                'timestamp_et': row['timestamp_et'],
                'hour_idx': t_idx,
                'stateAbbr': row['stateAbbr'],
                'hours_since_obs': t_idx - (OBSERVED_END - 1),
                'storm_phase': feat['storm_phase'],
            }
            if is_train and 'severity_tier' in row:
                meta_row['severity_tier'] = row['severity_tier']
            meta_rows.append(meta_row)

    X = pd.DataFrame(rows_out, columns=ALL_FEATURE_NAMES)
    meta = pd.DataFrame(meta_rows)

    # 训练集: 提取目标列(4个horizon)
    if is_train:
        pred_df_all = df[(df['_hour_idx'] >= PRED_START)
                         & (df['_hour_idx'] < PRED_END)].copy()
        pred_df_all = pred_df_all.sort_values(['fipsCode', '_hour_idx']).reset_index(drop=True)
        y = pred_df_all[HORIZONS].copy().reset_index(drop=True)
        assert len(X) == len(y), f"X rows {len(X)} != y rows {len(y)}"
        assert len(X) == len(meta), f"X rows {len(X)} != meta rows {len(meta)}"
    else:
        y = None

    return X, y, meta


def get_feature_names():
    """返回特征列名列表(用于缓存对齐)"""
    return list(ALL_FEATURE_NAMES)
