"""v1.9 固定配置：分量专用 LightGBM 与前 72 小时县域响应画像。"""

from pathlib import Path

V19_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = V19_DIR.parents[1]
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data"
CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "experiments.log"
RESULTS_FILE = PROJECT_ROOT / "Results.md"
SUBMISSION_TEMPLATE = DATA_DIR / "sample_submission.csv"
RAW_TRAIN_FILE = DATA_DIR / "DM_Train.csv"
RAW_TEST_FILE = DATA_DIR / "DM_Test.csv"
V18_DIR = PROJECT_ROOT / "versions" / "v1.8"

BASE_FEATURE_VERSION = "v1.5.6"
FEATURE_VERSION = "v1.9"
EXPERIMENT_VERSION = "v1.9"
CV_VERSION = "balanced_v1"
SEED = 42
N_FOLDS = 5

HORIZONS = (
    "osi_target_t01h",
    "osi_target_t06h",
    "osi_target_t24h",
    "osi_target_t48h",
)
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}

OSI_MIN = 0.0
OSI_MAX = 0.65
ZERO_THRESHOLD = 0.001
COMPONENT_MIN = 0.0
COMPONENT_MAX = 1.0
MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260911
MATERIAL_THRESHOLD = 0.001
GUST_THRESHOLD = 30.0

BASE_LGBM_PARAMS = {
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
    "num_threads": -1,
}

RESPONSE_FEATURES = (
    "N_nonzero_frac_obs",
    "R_nonzero_frac_obs",
    "P_mean_last24_obs",
    "D_mean_last24_obs",
    "N_mean_last24_obs",
    "R_mean_last24_obs",
    "flow_balance_last24_obs",
    "restoration_ratio_obs",
    "P_D_gap_last_obs",
    "hours_since_P_peak_obs",
    "hours_since_N_peak_obs",
    "hours_since_R_peak_obs",
    "first_material_N_hour_obs",
    "first_material_R_hour_obs",
    "P_trend_last24_obs",
    "D_trend_last24_obs",
    "P_post_peak_slope_obs",
    "P_active_frac_obs",
    "has_material_P_obs",
    "has_material_N_obs",
    "has_material_R_obs",
    "gust_excess30_sum_obs",
    "N_per_gust_excess_obs",
    "P_change_per_gust_excess_obs",
    "gust_to_N_best_corr_obs",
    "gust_to_N_best_lag_obs",
    "gust_at_first_material_N_obs",
)

FEATURES_TRAIN_FILE = V19_DIR / "features_train_v1.9.parquet"
FEATURES_TEST_FILE = V19_DIR / "features_test_v1.9.parquet"
FEATURE_NAMES_FILE = V19_DIR / "feature_names_v1.9.json"
PROFILE_TRAIN_FILE = V19_DIR / "response_profiles_train_v1.9.parquet"
PROFILE_TEST_FILE = V19_DIR / "response_profiles_test_v1.9.parquet"
FEATURE_DEFINITIONS_FILE = V19_DIR / "feature_definitions_v1.9.csv"
FEATURE_VALIDATION_FILE = V19_DIR / "feature_validation.json"

MODELS_DIR = V19_DIR / "models"
CV_MODELS_DIR = MODELS_DIR / "cv"
FINAL_MODELS_DIR = MODELS_DIR / "final"
TARGETS_FILE = V18_DIR / "component_targets_v1.8.parquet"
TARGET_VALIDATION_FILE = V18_DIR / "component_target_validation.json"


def ensure_dirs() -> None:
    V19_DIR.mkdir(parents=True, exist_ok=True)
    CV_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

