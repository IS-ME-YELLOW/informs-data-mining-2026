"""v3.1 时序残差模型的固定配置。"""

from pathlib import Path


VERSION = "v3.1"
BASELINE_VERSION = "v1.5.6"
CV_VERSION = "balanced_v1"
SEED = 42

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CV_DIR = PROJECT_ROOT / "cv"
ARTIFACT_DIR = HERE / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"

TRAIN_RAW = DATA_DIR / "DM_Train.csv"
TEST_RAW = DATA_DIR / "DM_Test.csv"
SUBMISSION_TEMPLATE = DATA_DIR / "sample_submission.csv"

FEATURE_TRAIN = CACHE_DIR / f"features_train_{BASELINE_VERSION}.parquet"
FEATURE_TEST = CACHE_DIR / f"features_test_{BASELINE_VERSION}.parquet"
TARGET_TRAIN = CACHE_DIR / f"targets_train_{BASELINE_VERSION}.parquet"
META_TRAIN = CACHE_DIR / f"meta_train_{BASELINE_VERSION}.parquet"
META_TEST = CACHE_DIR / f"meta_test_{BASELINE_VERSION}.parquet"
FEATURE_NAMES = CACHE_DIR / f"feature_names_{BASELINE_VERSION}.json"
CV_ASSIGNMENTS = CV_DIR / f"cv_assignments_{CV_VERSION}_seed{SEED}.csv"

TRAIN_SEQUENCE_CACHE = ARTIFACT_DIR / "sequences_train_v3.1.npz"
TEST_SEQUENCE_CACHE = ARTIFACT_DIR / "sequences_test_v3.1.npz"
SEQUENCE_MANIFEST = ARTIFACT_DIR / "sequence_manifest.json"
BASELINE_PREDICTIONS = ARTIFACT_DIR / "baseline_predictions_v3.1.npz"
BASELINE_METADATA = ARTIFACT_DIR / "baseline_metadata.json"
DIAGNOSTIC_PREDICTIONS = ARTIFACT_DIR / "diagnostic_predictions.npz"
OOF_PREDICTIONS = ARTIFACT_DIR / "oof_predictions.parquet"
TEST_PREDICTIONS = ARTIFACT_DIR / "test_predictions.parquet"
FOLD_METRICS = ARTIFACT_DIR / "fold_metrics.csv"
SUMMARY_METRICS = ARTIFACT_DIR / "summary_metrics.csv"
RUN_METADATA = ARTIFACT_DIR / "run_metadata.json"
VERIFICATION_FILE = ARTIFACT_DIR / "verification.json"
SUBMISSION_OUTPUT = HERE / "submission_v3.1_balanced_v1.csv"

OBSERVED_HOURS = 72
TOTAL_HOURS = 216
FUTURE_HOURS = TOTAL_HOURS - OBSERVED_HOURS

HORIZONS = [
    "osi_target_t01h",
    "osi_target_t06h",
    "osi_target_t24h",
    "osi_target_t48h",
]
HORIZON_HOURS = [1, 6, 24, 48]

OUTAGE_CHANNELS = ["P_t", "N_t", "D_t", "R_t"]
WEATHER_SOURCE_CHANNELS = [
    "gust",
    "wind_speed_10m",
    "wind_dir_10m",
    "t2m",
    "tp",
    "sp",
    "r2",
    "csnow",
    "soil_moist",
]
WEATHER_CHANNELS = [
    "gust",
    "wind_speed_10m",
    "wind_dir_sin",
    "wind_dir_cos",
    "t2m",
    "tp",
    "sp",
    "r2",
    "csnow",
    "soil_moist",
]
OBSERVED_CHANNELS = OUTAGE_CHANNELS + WEATHER_CHANNELS
LOG1P_CHANNELS = {"tp", "csnow"}

STATIC_NUMERIC_COLUMNS = [
    "pop_density",
    "n_utilities",
    "pct_forest",
    "pct_wetland",
    "tree_canopy_pct",
    "tree_canopy_std",
]
STATIC_CATEGORICAL_COLUMN = "rural_urban_code"
LAST_STATE_COLUMNS = ["last_P_t", "last_N_t", "last_D_t", "last_R_t", "last_osi"]

OSI_MAX = 0.65
ZERO_THRESHOLD = 0.001
N_FOLDS = 5

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
    "num_threads": -1,
}
LGBM_NUM_BOOST_ROUND = 2000
LGBM_EARLY_STOPPING_ROUNDS = 100

GRU_HIDDEN_SIZE = 32
STATIC_HIDDEN_SIZE = 16
HEAD_HIDDEN_SIZE = 32
GRU_DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS_DIAGNOSTIC = 30
EARLY_STOPPING_PATIENCE = 6
GRADIENT_CLIP = 1.0
ALPHA_BY_HORIZON = [1.0, 1.0, 1.0, 1.0]


def ensure_artifact_dirs():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
