"""v3.2 统一目标小时动态融合模型的固定配置。"""

from pathlib import Path


VERSION = "v3.2"
BASELINE_VERSION = "v1.5.6"
CV_VERSION = "balanced_v1"
SEED = 42

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
CV_DIR = PROJECT_ROOT / "cv"
V31_DIR = HERE.parent / "v3.1"
V31_ARTIFACT_DIR = V31_DIR / "artifacts"
V31_M4_DIR = V31_ARTIFACT_DIR / "m4"

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

# 只读复用的 v3.1 资产。
V31_TRAIN_SEQUENCE = V31_ARTIFACT_DIR / "sequences_train_v3.1.npz"
V31_TEST_SEQUENCE = V31_ARTIFACT_DIR / "sequences_test_v3.1.npz"
V31_SEQUENCE_MANIFEST = V31_ARTIFACT_DIR / "sequence_manifest.json"
V31_BASELINE_PREDICTIONS = V31_ARTIFACT_DIR / "baseline_predictions_v3.1.npz"
V31_M4_METADATA = V31_M4_DIR / "run_metadata.json"
V31_M4_FOLD_DIR = V31_M4_DIR / "folds"

ARTIFACT_DIR = HERE / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
TIMELINE_TRAIN = ARTIFACT_DIR / "target_timeline_train_v3.2.npz"
TIMELINE_TEST = ARTIFACT_DIR / "target_timeline_test_v3.2.npz"
TIMELINE_MANIFEST = ARTIFACT_DIR / "target_timeline_manifest.json"
DIAGNOSTIC_PREDICTIONS = ARTIFACT_DIR / "diagnostic_predictions.npz"
DIAGNOSTIC_OOF_PARQUET = ARTIFACT_DIR / "diagnostic_oof_predictions.parquet"
DIAGNOSTIC_TEST_PARQUET = ARTIFACT_DIR / "diagnostic_test_predictions.parquet"
DIAGNOSTIC_SUMMARY = ARTIFACT_DIR / "diagnostic_summary_metrics.csv"
DIAGNOSTIC_FOLDS = ARTIFACT_DIR / "diagnostic_fold_metrics.csv"
DIAGNOSTIC_METADATA = ARTIFACT_DIR / "diagnostic_run_metadata.json"
DIAGNOSTIC_VERIFICATION = ARTIFACT_DIR / "diagnostic_verification.json"
DIAGNOSTIC_SUBMISSION = HERE / "submission_v3.2_diagnostic_only.csv"

M4_DIR = ARTIFACT_DIR / "m4"
M4_FOLD_DIR = M4_DIR / "folds"
M4_CHECKPOINT_DIR = M4_DIR / "checkpoints"
M4_PREDICTIONS = M4_DIR / "nested_oof_predictions.npz"
M4_OOF_PARQUET = M4_DIR / "nested_oof_predictions.parquet"
M4_WEIGHTS_PARQUET = M4_DIR / "fusion_weights.parquet"
M4_SUMMARY = M4_DIR / "summary_metrics.csv"
M4_FOLDS = M4_DIR / "fold_metrics.csv"
M4_COUNTIES = M4_DIR / "county_metrics.csv"
M4_STAGES = M4_DIR / "stage_metrics.csv"
M4_BOOTSTRAP = M4_DIR / "paired_bootstrap.csv"
M4_METADATA = M4_DIR / "run_metadata.json"
M4_VERIFICATION = M4_DIR / "verification.json"

OBSERVED_HOURS = 72
TOTAL_HOURS = 216
FUTURE_HOURS = TOTAL_HOURS - OBSERVED_HOURS
TARGET_HOURS = list(range(73, 216))
N_TARGET_HOURS = len(TARGET_HOURS)

HORIZONS = [
    "osi_target_t01h",
    "osi_target_t06h",
    "osi_target_t24h",
    "osi_target_t48h",
]
HORIZON_HOURS = [1, 6, 24, 48]
N_HORIZONS = len(HORIZONS)
N_FOLDS = 5

OUTAGE_CHANNELS = ["P_t", "N_t", "D_t", "R_t"]
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
TARGET_EQUALITY_ATOL = 1e-7
RELOAD_ATOL = 1e-6

GRU_HIDDEN_SIZE = 32
STATIC_HIDDEN_SIZE = 16
HEAD_HIDDEN_SIZE = 32
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
MAX_EPOCHS_DIAGNOSTIC = 30
MAX_EPOCHS_M4 = 50
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP = 1.0
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
BOOTSTRAP_REPLICATES = 2000


def ensure_artifact_dirs() -> None:
    for path in [
        ARTIFACT_DIR,
        CHECKPOINT_DIR,
        M4_DIR,
        M4_FOLD_DIR,
        M4_CHECKPOINT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
