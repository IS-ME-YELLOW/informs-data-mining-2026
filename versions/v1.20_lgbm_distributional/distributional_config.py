"""Fixed configuration for the v1.20 LightGBM distributional experiment."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
V18_DIR = PROJECT_ROOT / "versions" / "v1.8"
V112_DIR = PROJECT_ROOT / "versions" / "v1.12_tail_protected_ensemble"

MODELS_DIR = VERSION_DIR / "models"
PREDICTIONS_DIR = VERSION_DIR / "predictions"
FOLDS_DIR = VERSION_DIR / "folds"
RESULTS_DIR = VERSION_DIR / "results"
MANIFESTS_DIR = VERSION_DIR / "manifests"
ARTIFACTS_DIR = VERSION_DIR / "artifacts"
OUTPUT_DIRS = (MODELS_DIR, PREDICTIONS_DIR, FOLDS_DIR, RESULTS_DIR, MANIFESTS_DIR, ARTIFACTS_DIR)

PRIMARY_SPLIT = "balanced_v1"
PRIMARY_FOLD_FILE = PROJECT_ROOT / "cv" / "cv_assignments_balanced_v1_seed42.csv"
REPEAT_SEEDS = (20260921, 20260922)
N_FOLDS = 5
HORIZONS = ("osi_target_t24h", "osi_target_t48h")
COMPONENTS = ("P_t", "D_t")
ALL_COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0)
MODES = ("plain", "protected")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260920

V18_OOF_COMPONENTS = V18_DIR / "oof_component_predictions.parquet"
V18_TEST_COMPONENTS = V18_DIR / "test_component_predictions.parquet"
V112_OOF = V112_DIR / "artifacts" / "oof_predictions.parquet"
V112_TEST = V112_DIR / "artifacts" / "test_predictions.csv"

BASE_LGBM_PARAMS: dict[str, Any] = {
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
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "drop_seed": 42,
    "data_random_seed": 42,
}

CANDIDATE_SPECS: dict[str, dict[str, Any]] = {
    "dart_huber": {
        "params": {
            "boosting": "dart", "objective": "huber", "learning_rate": 0.03,
            "num_leaves": 31, "min_child_samples": 50, "feature_fraction": 0.8,
            "lambda_l1": 0.1, "lambda_l2": 1.0, "drop_rate": 0.1,
            "skip_drop": 0.5, "max_drop": 50,
        },
        "rounds": 600, "early_stopping": None,
    },
    "tweedie_p13": {"params": {"objective": "tweedie", "tweedie_variance_power": 1.3}, "rounds": 2000, "early_stopping": 100},
    "tweedie_p15": {"params": {"objective": "tweedie", "tweedie_variance_power": 1.5}, "rounds": 2000, "early_stopping": 100},
    "tweedie_p17": {"params": {"objective": "tweedie", "tweedie_variance_power": 1.7}, "rounds": 2000, "early_stopping": 100},
    "quantile_q50": {"params": {"objective": "quantile", "alpha": 0.5}, "rounds": 2000, "early_stopping": 100},
    "quantile_q90": {"params": {"objective": "quantile", "alpha": 0.9}, "rounds": 2000, "early_stopping": 100},
}


def ensure_output_dirs() -> None:
    for path in OUTPUT_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def candidate_names() -> tuple[str, ...]:
    return tuple(CANDIDATE_SPECS)


def resolved_spec(name: str, num_threads: int, quick: bool = False) -> dict[str, Any]:
    if name not in CANDIDATE_SPECS:
        raise KeyError(f"Unknown candidate: {name}")
    spec = deepcopy(CANDIDATE_SPECS[name])
    spec["params"] = {**BASE_LGBM_PARAMS, **spec["params"], "num_threads": int(num_threads)}
    if quick:
        if name == "dart_huber":
            spec["rounds"], spec["early_stopping"] = 100, None
        else:
            spec["rounds"], spec["early_stopping"] = 300, 30
    return spec


def stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def horizon_suffix(horizon: str) -> str:
    return horizon.rsplit("_", 1)[-1]
