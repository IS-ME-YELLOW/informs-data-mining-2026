"""Frozen configuration for the dem_v158_nested_v2 protocol.

This module contains no data loading side effects. Paths are resolved relative
to this file and the default feature source is one complete v1.5.6 package,
never a per-file cache fallback.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"

PROTOCOL = "dem_v158_nested_v2"
# v1.5.8 is the experiment/component-target release.  Its own config pins the
# model feature package to v1.5.6; keeping both identifiers explicit prevents
# accidentally treating the release number as a feature schema version.
EXPERIMENT_VERSION = "v1.5.8"
FEATURE_VERSION = "v1.5.6"
GRAPH_INPUT_DIM = 173
GRAPH_SCHEMA_VERSION = "direct_gat_no_neighbor_summary_v1"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "versions" / "xyy" / FEATURE_VERSION
FEATURE_NAMES_FILE = f"feature_names_{FEATURE_VERSION}.json"
FEATURE_MANIFEST_CANDIDATES = ("manifest.json", f"manifest_{FEATURE_VERSION}.json")
PARQUET_ENGINE = "pyarrow"

CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
CV_VERSION = "balanced_v1_seed42"
CV_SEED = 42
N_FOLDS = 5

DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
GEO_DBF = DATA_DIR / "geo" / "c_16ap26.dbf"
TERRAIN_FILE = DATA_DIR / "geo" / "county_terrain.csv"
COMPONENT_TARGETS_FILE = (
    PROJECT_ROOT / "versions" / "xyy" / "v1.5.8" / "component_targets_v1.8.parquet"
)

HORIZONS = ("osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h")
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}

OBSERVED_END = 72
PRED_START = 72
PRED_END = 216
OSI_MAX = 0.65
# OOF covers all 239 training counties. Submission counts below cover 63 test
# counties and must never be used to validate the training OOF table.
TRAIN_SCOREABLE_ROWS = {
    horizon: 239 * (PRED_END - PRED_START - hours)
    for horizon, hours in HORIZON_HOURS.items()
}
OFFICIAL_SCOREABLE_ROWS = {
    "osi_target_t01h": 9009,
    "osi_target_t06h": 8694,
    "osi_target_t24h": 7560,
    "osi_target_t48h": 6048,
}

# v1.5.6 tree configuration. Seeds and thread count are filled per scope by
# make_lgbm_params(), so --seed cannot silently remain coupled to module 42.
LGBM_BASE_PARAMS = {
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
}
LGBM_ROUNDS = 2000
LGBM_EARLY_STOPPING = 100
FIXED_THREADS = 1


def make_lgbm_params(seed: int) -> dict:
    """Return a fully seeded, fixed-thread LightGBM parameter dictionary."""

    seed = int(seed)
    params = dict(LGBM_BASE_PARAMS)
    params.update({
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "drop_seed": seed,
        "data_random_seed": seed,
        "num_threads": FIXED_THREADS,
        "force_col_wise": True,
    })
    return params


# These names may be used for alignment/grouping but never become model input.
# The actual input contract is the ordered frozen feature_names file, not this
# blacklist. The blacklist remains useful for diagnostic error messages.
FORBIDDEN_INPUT_COLUMNS = frozenset({
    "timestamp_et", "fipsCode", "countyName", "stateName", "stateAbbr",
    "state", "state_id", "state_code", "state_IN", "state_OH", "state_PA",
    "state_WV", "in_event_window", "split", "severity_tier",
    "event_duration_h", "peak_pct", "peak_customers", "time_to_restore_h",
    "outageCount", "outage_pct", "P_t", "N_t", "D_t", "R_t", "osi",
})

GRAPH_K = 8
GAT_HIDDEN = 24
GAT_HEADS = 4
GAT_DROPOUT = 0.12
GAT_LR = 0.003
GAT_WEIGHT_DECAY = 2e-4
GAT_EPOCHS = 220
GAT_PATIENCE = 35
GAT_TIME_STRIDE = 1
GAT_LOSS = "mse"
GAT_ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)

# County-clustered paired bootstrap is a reporting diagnostic, never a model
# selection input.  The seed and replicate count are part of the run identity.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260910

TRAIN_TARGET_COLUMNS = set(HORIZONS)
