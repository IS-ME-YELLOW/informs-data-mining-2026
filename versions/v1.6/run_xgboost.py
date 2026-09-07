"""Run the v1.6 XGBoost trial on the frozen v1.5.2 feature cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


VERSIONS_DIR = Path(__file__).resolve().parents[1]
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

from gbdt_experiment import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    PROJECT_ROOT,
    SEED,
    run_experiment,
)

import xgboost as xgb  # noqa: E402


class XGBoostAdapter:
    family = "XGBoost"
    slug = "xgboost"
    version = "v1.6"
    model_extension = ".json"

    @staticmethod
    def _params(n_estimators: int) -> dict:
        return {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.01,
            "eval_metric": "rmse",
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def config(self, max_rounds: int, early_stopping_rounds: int) -> dict:
        return {
            **self._params(max_rounds),
            "early_stopping_rounds": early_stopping_rounds,
            "final_round_rule": "rounded mean of the five CV best rounds",
        }

    @staticmethod
    def library_versions() -> dict[str, str]:
        return {"xgboost": xgb.__version__}

    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int):
        return xgb.XGBRegressor(
            **self._params(max_rounds),
            early_stopping_rounds=early_stopping_rounds,
        )

    @staticmethod
    def fit_cv(model, X_train, y_train, X_valid, y_valid) -> None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    @staticmethod
    def best_iteration(model) -> int:
        return int(model.best_iteration) + 1

    def build_final_model(self, n_rounds: int):
        return xgb.XGBRegressor(**self._params(n_rounds))

    @staticmethod
    def fit_final(model, X, y) -> None:
        model.fit(X, y, verbose=False)

    @staticmethod
    def predict(model, X):
        return model.predict(X)

    @staticmethod
    def feature_importance(model, feature_names):
        return model.feature_importances_

    @staticmethod
    def save_model(model, path: Path) -> None:
        model.save_model(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        XGBoostAdapter(),
        PROJECT_ROOT / "versions" / "v1.6",
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        validate_only=args.validate_only,
    )
