# ============================================================
# features.py — 特征工程
# ============================================================
# 职责: 从原始数据构建82维特征矩阵, 供LightGBM训练/预测使用
#
# 特征分6类:
#   A. 观测窗口摘要 (25维): 从3月11-13(72h)停电数据提取, 每县固定
#   B. 当前气象 (24维): 预测时刻t的22个气象变量 + 风向sin/cos
#   C. 目标时段气象统计 (20维): [t, t+h]窗口内gust_max/mean等, 4个horizon×5统计
#   D. 派生气象 (6维): 结冰风险/阵风超限/土湿×风速交互等
#   E. 时间特征 (6维): 小时/天/距观测结束小时数/风暴阶段
#   F. 县级特征 (1维): log(customersTracked), 全程有值
#
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

# A. 观测窗口摘要特征 (25维, 从3月11-13停电数据提取, 每县固定不变)
OBSERVED_FEATURES = [
    'last_osi', 'last_P_t', 'last_N_t', 'last_D_t', 'last_R_t',
    'last_outage_pct',
    'osi_mean_72h', 'osi_max_72h', 'osi_std_72h',
    'outage_pct_mean_72h', 'outage_pct_max_72h',
    'N_t_mean_72h', 'N_t_max_72h', 'R_t_mean_72h', 'R_t_max_72h',
    'pre_event_mean_osi', 'pre_event_max_outage_pct',
    'storm_onset_max_osi', 'storm_onset_mean_osi',
    'osi_trend_last6h', 'is_worsening', 'frac_zero_72h',
    'hours_since_peak', 'gust_mean_obs', 'gust_max_obs',
]

# B. 预测时刻t的气象特征 (24维, 随t变化)
WEATHER_AT_T_FEATURES = [
    'gust_t', 'wind_speed_t', 'wind_sin_t', 'wind_cos_t',
    't2m_t', 'd2m_t', 'tp_t', 'rain_t', 'csnow_t', 'sdwe_t',
    'r2_t', 'vpd_t', 'soil_moist_t', 'blh_t', 'tcc_t', 'lcc_t',
    'mcc_t', 'hcc_t', 'sp_t', 'mslma_t',
    'sdswrf_t', 'direct_rad_t', 'diffuse_rad_t', 'et0_t',
]

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
    OBSERVED_FEATURES
    + WEATHER_AT_T_FEATURES
    + DERIVED_WEATHER_FEATURES
    + HORIZON_STAT_FEATURES
    + TEMPORAL_FEATURES
    + COUNTY_FEATURES
)


# ============================================================
# 各类特征的计算函数
# ============================================================

def compute_observed_features(county_df):
    """
    从3月11-13观测窗口(hour_idx 0-71)提取25维县级摘要特征
    这些特征对同一县的所有144个预测小时保持不变(静态)
    包括: 最后观测值、72h统计量、预事件基线、第一波信号、趋势
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

    feat = {
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


def compute_horizon_stats(county_df, t_idx, h):
    """
    计算目标时段[t, t+h]的气象统计:
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
    """
    rows_out = []
    meta_rows = []

    for fips, county_df in df.groupby('fipsCode'):
        county_df = county_df.sort_values('_hour_idx').reset_index(drop=True)
        obs_feat = compute_observed_features(county_df)  # 每县计算一次

        pred_df = county_df[(county_df['_hour_idx'] >= PRED_START)
                            & (county_df['_hour_idx'] < PRED_END)]

        for _, row in pred_df.iterrows():
            t_idx = int(row['_hour_idx'])

            # 拼装特征: 观测摘要(静态) + 气象(随t) + 派生 + 时间 + 县级
            feat = dict(obs_feat)
            feat.update(compute_weather_at_t(row))

            # 4个horizon的气象统计(允许查看未来, 规则允许)
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
