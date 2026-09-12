"""Central configuration for the spatial residual model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
OUTPUT_DIR = PROJECT_ROOT / "code_phase2_dem" / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_FEATURES = CACHE_DIR / "features_train_v1.5.7.parquet"
TEST_FEATURES = CACHE_DIR / "features_test_v1.5.7.parquet"
TRAIN_META = CACHE_DIR / "meta_train_v1.5.7.parquet"
TEST_META = CACHE_DIR / "meta_test_v1.5.7.parquet"
TRAIN_TARGETS = CACHE_DIR / "targets_train_v1.5.7.parquet"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
GEO_DBF = DATA_DIR / "geo" / "c_16ap26.dbf"
TERRAIN_FILE = DATA_DIR / "geo" / "county_terrain.csv"

SEED = 42
HORIZONS = ["osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h"]
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
OBSERVED_END = 72
PRED_START = 72
PRED_END = 216
OSI_MAX = 0.65

# Phase 1's latest model is deliberately retrained here, so phase2 remains
# self-contained and its OOF residuals are generated with the exact same CV.
LGBM_PARAMS = {
    "objective": "huber",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "drop_seed": SEED,
    "data_random_seed": SEED,
}
LGBM_ROUNDS = 2000
LGBM_EARLY_STOPPING = 100
N_FOLDS = 5

# Spatial graph and GAT settings. k=8 keeps the graph local while the
# symmetrisation makes cross-county message passing stable near boundaries.
GRAPH_K = 8
GAT_HIDDEN = 24
GAT_HEADS = 4
GAT_DROPOUT = 0.12
GAT_LR = 0.003
GAT_WEIGHT_DECAY = 2e-4
GAT_EPOCHS = 220
GAT_PATIENCE = 35
GAT_ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)

# All 211 Phase-1 cache columns have already passed the causal feature audit:
# no prediction-window outage/OSI/lag/target column is present. Feeding the
# complete clean cache lets the spatial model use the same weather trajectory,
# land-cover and county context as the LightGBM base. Coordinates/state are
# appended by data.py.
GAT_USE_ALL_FEATURES = True
GAT_FEATURES = [
    "last_osi", "last_P_t", "last_N_t", "last_D_t", "last_R_t",
    "osi_mean_72h", "osi_max_72h", "osi_std_72h", "osi_trend_last6h",
    "gust_t", "wind_speed_t", "gust_max_next_{h}h", "gust_mean_next_{h}h",
    "wind_speed_max_next_{h}h", "total_tp_next_{h}h", "min_t2m_next_{h}h",
    "gust_at_t{h}h", "wind_speed_at_t{h}h", "t2m_at_t{h}h", "tp_at_t{h}h",
    "hours_since_obs", "storm_phase", "log_customers", "pop_density",
    "pct_forest", "tree_canopy_pct", "n_utilities", "rural_urban_code",
]
