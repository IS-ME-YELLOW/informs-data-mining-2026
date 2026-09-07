"""Model-family adapters used by the v2 component-first experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SEED = 42


class LightGBMAdapter:
    family = "LightGBM"
    slug = "lightgbm"
    model_extension = ".txt"

    def __init__(self) -> None:
        import lightgbm as lgb

        self.lgb = lgb

    @staticmethod
    def _params(n_estimators: int) -> dict:
        return {
            "objective": "huber",
            "metric": "rmse",
            "n_estimators": n_estimators,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": SEED,
            "feature_fraction_seed": SEED,
            "bagging_seed": SEED,
            "data_random_seed": SEED,
            "importance_type": "gain",
            "verbosity": -1,
            "n_jobs": -1,
        }

    def config(self, max_rounds: int, early_stopping_rounds: int) -> dict:
        return {
            **self._params(max_rounds),
            "early_stopping_rounds": early_stopping_rounds,
            "final_round_rule": "rounded mean of the five CV best rounds",
        }

    def library_versions(self) -> dict[str, str]:
        return {"lightgbm": self.lgb.__version__}

    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int):
        return self.lgb.LGBMRegressor(**self._params(max_rounds))

    def fit_cv(self, model, X_train, y_train, X_valid, y_valid,
               early_stopping_rounds: int) -> None:
        model.fit(
            X_train,
            y_train,
            eval_X=X_valid,
            eval_y=y_valid,
            callbacks=[
                self.lgb.early_stopping(early_stopping_rounds, verbose=False),
                self.lgb.log_evaluation(0),
            ],
        )

    @staticmethod
    def best_iteration(model) -> int:
        return int(model.best_iteration_ or model.n_estimators_)

    def build_final_model(self, n_rounds: int):
        return self.lgb.LGBMRegressor(**self._params(n_rounds))

    def fit_final(self, model, X, y) -> None:
        model.fit(X, y, callbacks=[self.lgb.log_evaluation(0)])

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return model.predict(X)

    @staticmethod
    def feature_importance(model, feature_names: list[str]) -> np.ndarray:
        return model.booster_.feature_importance(importance_type="gain")

    @staticmethod
    def save_model(model, path: Path) -> None:
        model.booster_.save_model(str(path))


class XGBoostAdapter:
    family = "XGBoost"
    slug = "xgboost"
    model_extension = ".json"

    def __init__(self) -> None:
        import xgboost as xgb

        self.xgb = xgb

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

    def library_versions(self) -> dict[str, str]:
        return {"xgboost": self.xgb.__version__}

    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int):
        return self.xgb.XGBRegressor(
            **self._params(max_rounds),
            early_stopping_rounds=early_stopping_rounds,
        )

    @staticmethod
    def fit_cv(model, X_train, y_train, X_valid, y_valid,
               early_stopping_rounds: int) -> None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    @staticmethod
    def best_iteration(model) -> int:
        return int(model.best_iteration) + 1

    def build_final_model(self, n_rounds: int):
        return self.xgb.XGBRegressor(**self._params(n_rounds))

    @staticmethod
    def fit_final(model, X, y) -> None:
        model.fit(X, y, verbose=False)

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return model.predict(X)

    @staticmethod
    def feature_importance(model, feature_names: list[str]) -> np.ndarray:
        return model.feature_importances_

    @staticmethod
    def save_model(model, path: Path) -> None:
        model.save_model(path)


class CatBoostAdapter:
    family = "CatBoost"
    slug = "catboost"
    model_extension = ".cbm"

    def __init__(self) -> None:
        import catboost
        from catboost import CatBoostRegressor

        self.catboost = catboost
        self.regressor = CatBoostRegressor

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

    def library_versions(self) -> dict[str, str]:
        return {"catboost": self.catboost.__version__}

    def build_cv_model(self, max_rounds: int, early_stopping_rounds: int):
        return self.regressor(
            **self._params(max_rounds),
            early_stopping_rounds=early_stopping_rounds,
        )

    @staticmethod
    def fit_cv(model, X_train, y_train, X_valid, y_valid,
               early_stopping_rounds: int) -> None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    @staticmethod
    def best_iteration(model) -> int:
        best = int(model.get_best_iteration())
        return int(model.tree_count_) if best < 0 else best + 1

    def build_final_model(self, n_rounds: int):
        return self.regressor(**self._params(n_rounds))

    @staticmethod
    def fit_final(model, X, y) -> None:
        model.fit(X, y)

    @staticmethod
    def predict(model, X) -> np.ndarray:
        return model.predict(X)

    @staticmethod
    def feature_importance(model, feature_names: list[str]) -> np.ndarray:
        return model.get_feature_importance()

    @staticmethod
    def save_model(model, path: Path) -> None:
        model.save_model(path)
