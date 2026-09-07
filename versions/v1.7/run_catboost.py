"""Run the v1.7 CatBoost trial on the frozen v1.5.2 feature cache."""

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

import catboost  # noqa: E402
from catboost import CatBoostRegressor  # noqa: E402


class CatBoostAdapter:
    family = "CatBoost"
    slug = "catboost"
    version = "v1.7"
    model_extension = ".cbm"

    @staticmethod
    def _params(iterations: int) -> dict:
        return {
            "loss_function": "Huber:delta=0.01",
            "eval_metric": "RMSE",
            "iterations": iterations,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "rsm": 0.8,
            "random_seed": SEED,
            "thread_count": -1,
            "allow_writing_files": False,
            "verbose": False,
        }

    def config(self, max_rounds: int, early_stopping_rounds: int) -> dict:
        return {
            **self._params(max_rounds),
            "early_stopping_rounds": early_stopping_rounds,
            "final_round_rule": "rounded mean of the five CV best rounds",
        }

    @staticmethod
    def library_versions() -> dict[str, str]:
        return {"catboost": catboost.__version__}

    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int):
        return CatBoostRegressor(
            **self._params(max_rounds),
            early_stopping_rounds=early_stopping_rounds,
        )

    @staticmethod
    def fit_cv(model, X_train, y_train, X_valid, y_valid) -> None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    @staticmethod
    def best_iteration(model) -> int:
        best = int(model.get_best_iteration())
        if best < 0:
            return int(model.tree_count_)
        return best + 1

    def build_final_model(self, n_rounds: int):
        return CatBoostRegressor(**self._params(n_rounds))

    @staticmethod
    def fit_final(model, X, y) -> None:
        model.fit(X, y)

    @staticmethod
    def predict(model, X):
        return model.predict(X)

    @staticmethod
    def feature_importance(model, feature_names):
        return model.get_feature_importance()

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
        CatBoostAdapter(),
        PROJECT_ROOT / "versions" / "v1.7",
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        validate_only=args.validate_only,
    )
