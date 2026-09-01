# ============================================================
# config.py — 全局配置中心
# ============================================================
# 本文件集中管理 Phase 1 的所有配置项,包括:
#   - 文件路径(训练/测试/提交/缓存/日志/模型)
#   - 随机种子(与比赛划分一致, seed=42)
#   - 时间窗口边界(hour_idx: 0-47预事件, 48-71风暴第一波, 72-215预测窗口)
#   - 4个预测目标horizon及其对应小时数
#   - OSI公式权重和观测最大值
#   - 22个气象列名
#   - 禁用特征列表(合规清单: 标识符/停电变量/lag/目标/delta均不可直接用作输入)
#   - LightGBM超参数和CV折数
# 修改特征工程代码时,递增 FEATURE_VERSION 以触发缓存重建。
# ============================================================

import os

# --- 路径配置(全部相对项目根目录, 保证可复现) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')          # 特征缓存目录
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')             # 实验日志目录
LOG_FILE = os.path.join(LOG_DIR, 'experiments.log')     # 日志文件(追加写)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')        # 模型保存目录

TRAIN_FILE = os.path.join(DATA_DIR, 'DM_Train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'DM_Test.csv')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'submission_phase1.csv')

# --- 随机种子与版本 ---
SEED = 42                          # 与比赛 train/test 划分一致
FEATURE_VERSION = 'v1'             # 改特征工程时递增 → 废弃旧缓存

# --- 时间窗口边界(全局小时索引, 每县0-215) ---
PRE_EVENT_END = 48                 # 3月12日23:00 (预事件窗口结束)
OBSERVED_END = 72                  # 3月13日23:00 (观测窗口结束, 预测窗口开始)
PRED_START = 72                    # 3月14日0:00  (预测窗口起点)
PRED_END = 216                     # 3月19日23:00+1 (预测窗口终点, exclusive)

# --- 4个预测目标 ---
HORIZONS = ['osi_target_t01h', 'osi_target_t06h', 'osi_target_t24h', 'osi_target_t48h']
HORIZON_HOURS = {'osi_target_t01h': 1, 'osi_target_t06h': 6,
                  'osi_target_t24h': 24, 'osi_target_t48h': 48}

# --- OSI公式权重 ---
# OSI_t = 0.40*P_t + 0.35*N_t + 0.25*D_t - 0.10*R_t  (clip >= 0)
OSI_WEIGHTS = {'P': 0.40, 'N': 0.35, 'D': 0.25, 'R': 0.10}
OSI_MAX_OBSERVED = 0.65            # 训练数据中观测到的最大OSI值, 用于预测clip

# --- 气象列名(22列, 训练/测试集全216小时可用) ---
WEATHER_COLS = [
    'gust', 'wind_speed_10m', 'wind_dir_10m', 't2m', 'd2m',
    'sp', 'mslma', 'blh', 'tp', 'rain', 'csnow', 'sdwe',
    'tcc', 'lcc', 'mcc', 'hcc', 'sdswrf', 'direct_rad', 'diffuse_rad',
    'r2', 'vpd', 'et0', 'soil_moist'
]

# --- 禁用特征列表(合规: 以下列不可作为模型输入特征) ---
# 包括: 标识符、停电变量(预测窗口NaN)、lag特征(预计算列在预测窗口NaN)、
#        目标列、delta列、事件级结果变量(事后统计)、wind_dir(用sin/cos替代)
FORBIDDEN_AS_FEATURES = [
    'timestamp_et', 'fipsCode', 'countyName', 'stateName', 'stateAbbr',
    'in_event_window', 'split', 'severity_tier', 'event_duration_h',
    'peak_pct', 'peak_customers', 'time_to_restore_h',
    'osi', 'outageCount', 'outage_pct', 'P_t', 'N_t', 'D_t', 'R_t',
    'osi_lag1h', 'osi_lag3h', 'osi_lag6h', 'osi_lag24h', 'osi_lag48h',
    'outage_pct_lag1h', 'outage_pct_lag3h', 'outage_pct_lag6h',
    'outage_pct_lag24h', 'outage_pct_lag48h',
    'osi_target_t01h', 'osi_target_t03h', 'osi_target_t06h',
    'osi_target_t24h', 'osi_target_t48h',
    'osi_delta_t01h', 'osi_delta_t03h', 'osi_delta_t06h',
    'osi_delta_t24h', 'osi_delta_t48h',
    'wind_dir_10m',
]

# --- LightGBM超参数 ---
# Huber损失: 对零膨胀(43%)和极端值鲁棒
# min_child_samples=50: 239县×1事件, 需强正则化防过拟合
# feature_fraction=0.8, bagging_fraction=0.8: 列/行采样防过拟合
LGBM_PARAMS = {
    'objective': 'huber',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
}
LGBM_NUM_BOOST_ROUND = 2000
LGBM_EARLY_STOPPING_ROUNDS = 100
N_FOLDS = 5
