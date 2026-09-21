"""Frozen configuration for direct component GAT training.

The first graph feature is retained as the current GAT ``base`` channel for
schema comparability, but is a constant zero placeholder because this protocol
has no LightGBM learner.  The remaining 204 columns are the current GAT
features: v1.5.6 features, coordinates, DEM and 32 approved neighbour
aggregates.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
PROTOCOL = "direct_component_gat_v1"
EXPERIMENT_VERSION = "v1.5.8"
FEATURE_VERSION = "v1.5.6"
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "versions" / "xyy" / FEATURE_VERSION
FEATURE_NAMES_FILE = f"feature_names_{FEATURE_VERSION}.json"
FEATURE_MANIFEST_CANDIDATES = ("manifest.json", f"manifest_{FEATURE_VERSION}.json")
PARQUET_ENGINE = "pyarrow"

CV_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
CV_VERSION = "balanced_v1_seed42"
N_FOLDS = 5
CV_FOLD_COUNTS = {0: 48, 1: 47, 2: 48, 3: 48, 4: 48}

DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
GEO_DBF = DATA_DIR / "geo" / "c_16ap26.dbf"
TERRAIN_FILE = DATA_DIR / "geo" / "county_terrain.csv"
COMPONENT_TARGETS_FILE = PROJECT_ROOT / "versions" / "xyy" / "v1.5.8" / "component_targets_v1.8.parquet"

HORIZONS = ("osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h")
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}
PRED_START, PRED_END = 72, 216
OSI_MAX = 0.65
OFFICIAL_SCOREABLE_ROWS = {
    "osi_target_t01h": 9009, "osi_target_t06h": 8694,
    "osi_target_t24h": 7560, "osi_target_t48h": 6048,
}
TRAIN_SCOREABLE_ROWS = {
    h: 239 * (PRED_END - PRED_START - hours)
    for h, hours in HORIZON_HOURS.items()
}

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
FIXED_THREADS = 1
TARGET_SCALE_FLOOR = 0.003

FORBIDDEN_INPUT_COLUMNS = frozenset({
    "timestamp_et", "fipsCode", "countyName", "stateName", "stateAbbr",
    "state", "state_id", "state_code", "state_IN", "state_OH", "state_PA",
    "state_WV", "in_event_window", "split", "severity_tier", "event_duration_h",
    "peak_pct", "peak_customers", "time_to_restore_h", "outageCount", "outage_pct",
    "P_t", "N_t", "D_t", "R_t", "osi",
})


def component_weighted_osi(parts):
    values = 0.0
    for component, weight in COMPONENT_WEIGHTS.items():
        values = values + float(weight) * parts[component]
    return values
