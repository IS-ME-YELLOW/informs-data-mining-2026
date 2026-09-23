"""Frozen tree-only v1.8 experiment with isolated nested county validation."""

from pathlib import Path

V18_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = V18_DIR.parents[2]
CACHE_DIR = PROJECT_ROOT / "versions" / "xyy" / "v1.5.6"
DATA_DIR = PROJECT_ROOT / "data"
CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
LOG_FILE = PROJECT_ROOT / "logs" / "experiments.log"
SUBMISSION_TEMPLATE = DATA_DIR / "sample_submission.csv"
RAW_TRAIN_FILE = DATA_DIR / "DM_Train.csv"
RAW_TEST_FILE = DATA_DIR / "DM_Test.csv"
PROTOCOL = "v18_tree_nested_v2"
RUNS_DIR = V18_DIR / "runs"

FEATURE_VERSION = "v1.5.6"
EXPERIMENT_VERSION = "v1.8"
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
BOOTSTRAP_SEED = 20260910

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
    "num_threads": 1,
    "force_col_wise": True,
    "deterministic": True,
}

MODELS_DIR = V18_DIR / "models"
CV_MODELS_DIR = MODELS_DIR / "cv"
FINAL_MODELS_DIR = MODELS_DIR / "final"
TARGETS_FILE = V18_DIR / "component_targets_v1.8.parquet"
TARGET_VALIDATION_FILE = V18_DIR / "component_target_validation.json"


def make_lgbm_params(model_seed: int = SEED) -> dict:
    params = dict(LGBM_PARAMS)
    for key in ("seed", "feature_fraction_seed", "bagging_seed", "drop_seed", "data_random_seed"):
        params[key] = int(model_seed)
    return params
