from __future__ import annotations

from itertools import product

PROTOCOL = "stella_v112_nested_v1"
SPLIT_SEED = 42
MODEL_SEED = 42
FOLDS = (0, 1, 2, 3, 4)
HORIZONS = ("osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h")
LONG_HORIZONS = ("osi_target_t24h", "osi_target_t48h")
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}
FAMILIES = ("lightgbm", "xgboost", "catboost")
SOURCE_ORDER = (
    "lgbm_direct",
    "lgbm_component",
    "xgboost_direct",
    "catboost_direct",
    "xgboost_component",
    "catboost_component",
)
QUANTILE = 0.95
QUANTILE_METHOD = "linear"
OSI_MIN = 0.0
OSI_MAX = 0.65
ZERO_THRESHOLD = 0.001
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260910
ATOL = 1e-12

TARGETS = tuple(
    [horizon]
    + [f"{component}_target_{horizon.rsplit('_', 1)[-1]}" for component in COMPONENTS]
    for horizon in LONG_HORIZONS
)
TARGETS = tuple(target for group in TARGETS for target in group)

FAMILY_CONFIG = {
    "lightgbm": {
        "max_rounds": 2000,
        "patience": 100,
        "params": {
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
            "seed": MODEL_SEED,
            "feature_fraction_seed": MODEL_SEED,
            "bagging_seed": MODEL_SEED,
            "drop_seed": MODEL_SEED,
            "data_random_seed": MODEL_SEED,
            "num_threads": 1,
            "force_col_wise": True,
            "deterministic": True,
        },
    },
    "xgboost_direct": {
        "max_rounds": 4000,
        "patience": 100,
    },
    "xgboost_component": {
        "max_rounds": 8000,
        "patience": 100,
    },
    "catboost": {
        "max_rounds": 4000,
        "patience": 100,
    },
}

EXPECTED_SCOREABLE_ROWS = {
    "osi_target_t01h": 34177,
    "osi_target_t06h": 32982,
    "osi_target_t24h": 28680,
    "osi_target_t48h": 22944,
}


def component_target(component: str, horizon: str) -> str:
    return f"{component}_target_{horizon.rsplit('_', 1)[-1]}"


def target_horizon(target: str) -> str:
    suffix = target.rsplit("_", 1)[-1]
    return f"osi_target_{suffix}"


def resolved_config() -> dict:
    return {
        "protocol": PROTOCOL,
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "folds": list(FOLDS),
        "features": {"count": 163, "source": "versions/xyy/v1.5.6"},
        "horizons": list(HORIZONS),
        "long_horizons": list(LONG_HORIZONS),
        "components": list(COMPONENTS),
        "component_weights": COMPONENT_WEIGHTS,
        "source_order": list(SOURCE_ORDER),
        "gate": {
            "quantile_probability": QUANTILE,
            "method": QUANTILE_METHOD,
            "comparison": "L0 > theta uses L0; equality uses L1",
        },
        "post_process": {
            "clip": [OSI_MIN, OSI_MAX],
            "strictly_below_zero_threshold": ZERO_THRESHOLD,
        },
        "families": FAMILY_CONFIG,
        "round_rule": "max(1, floor(mean(scope-internal probe best round counts)))",
        "short_horizons": "exact pinned F2 main prediction, identical for L0/L1/L2",
        "bootstrap": {
            "unit": "county_full_trajectory",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "numeric_comparison": {"atol": ATOL, "rtol": 0},
        "expected_scoreable_rows": EXPECTED_SCOREABLE_ROWS,
    }

