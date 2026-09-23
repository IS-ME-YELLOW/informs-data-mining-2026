from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import FAMILY_CONFIG, MODEL_SEED, target_horizon
from .data_access import digest_object, expected_mask, scope_record, sha256_file, values_for_target, write_json
from .planning import ModelTask


def family_role(task: ModelTask) -> str:
    return "direct" if task.target == target_horizon(task.target) else "component"


def model_extension(family: str) -> str:
    return {"lightgbm": ".txt", "xgboost": ".json", "catboost": ".cbm"}[family]


def family_settings(task: ModelTask) -> dict:
    if task.family == "lightgbm":
        return FAMILY_CONFIG["lightgbm"]
    if task.family == "xgboost":
        return FAMILY_CONFIG[f"xgboost_{family_role(task)}"]
    return FAMILY_CONFIG["catboost"]


def selected_rows(data, protocol, target: str, folds: tuple[int, ...]) -> np.ndarray:
    valid = expected_mask(protocol, data.meta_train, target)
    rows = np.flatnonzero(valid & np.isin(data.row_folds, np.asarray(folds, dtype=int)))
    values = values_for_target(data, target)
    if not np.isfinite(values[rows]).all():
        raise ValueError(f"Non-finite target inside selected scope: {target}, folds={folds}")
    actual_folds = set(np.unique(data.row_folds[rows]).tolist())
    if actual_folds != set(folds):
        raise ValueError(f"Scope fold coverage mismatch: expected={folds}, actual={sorted(actual_folds)}")
    return rows


def _matrix(data, rows: np.ndarray, family: str):
    if family == "lightgbm":
        return data.X_train.iloc[rows]
    if not hasattr(data, "_x_train_float32"):
        data._x_train_float32 = data.X_train.astype(np.float32)
    return data._x_train_float32.iloc[rows]


def _xgb_params(n_estimators: int, *, early_stopping_rounds: int | None = None) -> dict:
    params = {
        "objective": "reg:pseudohubererror",
        "huber_slope": 0.01,
        "eval_metric": "rmse",
        "n_estimators": int(n_estimators),
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "random_state": MODEL_SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    return params


def _cat_params(iterations: int, *, early_stopping_rounds: int | None = None) -> dict:
    params = {
        "loss_function": "Huber:delta=0.01",
        "eval_metric": "RMSE",
        "iterations": int(iterations),
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "rsm": 0.8,
        "random_seed": MODEL_SEED,
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
    }
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    return params


def effective_params(task: ModelTask, rounds: int, *, probe: bool) -> dict:
    settings = family_settings(task)
    patience = int(settings["patience"]) if probe else None
    if task.family == "lightgbm":
        return {**settings["params"], "num_boost_round": int(rounds), "early_stopping_rounds": patience}
    if task.family == "xgboost":
        return _xgb_params(rounds, early_stopping_rounds=patience)
    return _cat_params(rounds, early_stopping_rounds=patience)


def _fit_probe(task: ModelTask, data, y: np.ndarray, train_rows: np.ndarray, valid_rows: np.ndarray):
    settings = family_settings(task)
    max_rounds = int(settings["max_rounds"])
    patience = int(settings["patience"])
    if task.family == "lightgbm":
        import lightgbm as lgb

        train_set = lgb.Dataset(_matrix(data, train_rows, task.family), label=y[train_rows], free_raw_data=True)
        valid_set = lgb.Dataset(_matrix(data, valid_rows, task.family), label=y[valid_rows], reference=train_set, free_raw_data=True)
        model = lgb.train(
            dict(settings["params"]),
            train_set,
            num_boost_round=max_rounds,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(patience, verbose=False), lgb.log_evaluation(0)],
        )
        best = int(model.best_iteration or model.num_trees())
    elif task.family == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(**_xgb_params(max_rounds, early_stopping_rounds=patience))
        model.fit(
            _matrix(data, train_rows, task.family),
            y[train_rows],
            eval_set=[(_matrix(data, valid_rows, task.family), y[valid_rows])],
            verbose=False,
        )
        best = int(model.best_iteration) + 1
    else:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**_cat_params(max_rounds, early_stopping_rounds=patience))
        model.fit(
            _matrix(data, train_rows, task.family),
            y[train_rows],
            eval_set=(_matrix(data, valid_rows, task.family), y[valid_rows]),
            use_best_model=True,
        )
        index = int(model.get_best_iteration())
        best = int(model.tree_count_) if index < 0 else index + 1
    if best < 1 or best > max_rounds:
        raise ValueError(f"Invalid best round count {best} for {task.task_id}")
    return best


def _fit_refit(task: ModelTask, data, y: np.ndarray, rows: np.ndarray, rounds: int):
    if task.family == "lightgbm":
        import lightgbm as lgb

        settings = family_settings(task)
        train_set = lgb.Dataset(_matrix(data, rows, task.family), label=y[rows], free_raw_data=True)
        model = lgb.train(
            dict(settings["params"]),
            train_set,
            num_boost_round=int(rounds),
            callbacks=[lgb.log_evaluation(0)],
        )
    elif task.family == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor(**_xgb_params(rounds))
        model.fit(_matrix(data, rows, task.family), y[rows], verbose=False)
    else:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**_cat_params(rounds))
        model.fit(_matrix(data, rows, task.family), y[rows])
    return model


def _save_model(model, family: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if family == "lightgbm":
        model.save_model(str(temporary))
    else:
        model.save_model(temporary)
    temporary.replace(path)


def load_model(family: str, path: Path):
    if family == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if family == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(path)
    return model


def predict_model(model, family: str, data, rows: np.ndarray, rounds: int) -> np.ndarray:
    matrix = _matrix(data, rows, family)
    if family == "lightgbm":
        values = model.predict(matrix, num_iteration=int(rounds), num_threads=1)
    elif family == "xgboost":
        values = model.predict(matrix, iteration_range=(0, int(rounds)))
    else:
        values = model.predict(matrix, ntree_end=int(rounds), thread_count=1)
    values = np.asarray(values, dtype=float)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError(f"Invalid prediction from {family}: shape={values.shape}")
    return values


def task_paths(run_dir: Path, task: ModelTask) -> dict[str, Path]:
    stem = run_dir / "models" / task.family / task.scope_id / task.target
    return {
        "model": stem.with_suffix(model_extension(task.family)),
        "receipt": stem.with_suffix(".receipt.json"),
        "prediction": run_dir / "predictions" / task.family / task.scope_id / f"{task.target}.npz",
    }


def _relative(path: Path, run_dir: Path) -> str:
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def validate_saved_task(run_dir: Path, task: ModelTask, identity_hash: str) -> dict | None:
    paths = task_paths(run_dir, task)
    if not paths["receipt"].is_file():
        return None
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    if receipt.get("task") != task.as_dict() or receipt.get("run_identity_hash") != identity_hash:
        raise ValueError(f"Saved receipt identity mismatch: {task.task_id}")
    for key in ("model", "prediction"):
        if not paths[key].is_file() or sha256_file(paths[key]) != receipt[f"{key}_sha256"]:
            raise ValueError(f"Saved {key} differs from receipt: {task.task_id}")
    return receipt


def fit_task(run_dir: Path, task: ModelTask, data, protocol, identity_hash: str) -> dict:
    existing = validate_saved_task(run_dir, task, identity_hash)
    if existing is not None:
        return existing

    y = values_for_target(data, task.target)
    probe_records = []
    best_rounds = []
    for validation_fold in task.scope:
        train_folds = tuple(fold for fold in task.scope if fold != validation_fold)
        train_rows = selected_rows(data, protocol, task.target, train_folds)
        valid_rows = selected_rows(data, protocol, task.target, (validation_fold,))
        if set(data.row_folds[train_rows]) & set(data.row_folds[valid_rows]):
            raise ValueError("Probe train and early-stop folds overlap")
        best = _fit_probe(task, data, y, train_rows, valid_rows)
        best_rounds.append(best)
        probe_records.append({
            "train_folds": list(train_folds),
            "early_stop_folds": [validation_fold],
            "train": scope_record(data, train_rows),
            "early_stop": scope_record(data, valid_rows),
            "best_round_count": best,
        })

    rounds = max(1, math.floor(float(np.mean(best_rounds))))
    refit_rows = selected_rows(data, protocol, task.target, task.scope)
    model = _fit_refit(task, data, y, refit_rows, rounds)
    paths = task_paths(run_dir, task)
    _save_model(model, task.family, paths["model"])

    predict_rows = selected_rows(data, protocol, task.target, task.predict_folds)
    predictions = predict_model(model, task.family, data, predict_rows, rounds)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["prediction"].with_suffix(".tmp.npz")
    np.savez_compressed(temporary, rows=predict_rows.astype(np.int64), predictions=predictions)
    temporary.replace(paths["prediction"])

    spec = {
        "task": task.as_dict(),
        "run_identity_hash": identity_hash,
        "feature_names": list(data.X_train.columns),
        "requested_rounds": rounds,
    }
    model_id = digest_object(spec)
    receipt = {
        **spec,
        "model_id": model_id,
        "fit": {
            "probes": probe_records,
            "best_round_counts": best_rounds,
            "round_rule": "max(1, floor(mean(best_round_counts)))",
            "requested_rounds": rounds,
            "refit_folds": list(task.scope),
            "refit": scope_record(data, refit_rows),
            "params": effective_params(task, rounds, probe=False),
        },
        "prediction": {
            "folds": list(task.predict_folds),
            **scope_record(data, predict_rows),
        },
        "model_path": _relative(paths["model"], run_dir),
        "prediction_path": _relative(paths["prediction"], run_dir),
        "receipt_path": _relative(paths["receipt"], run_dir),
        "model_sha256": sha256_file(paths["model"]),
        "prediction_sha256": sha256_file(paths["prediction"]),
    }
    write_json(paths["receipt"], receipt)
    return receipt


def load_prediction(run_dir: Path, task: ModelTask, identity_hash: str) -> tuple[np.ndarray, np.ndarray, dict]:
    receipt = validate_saved_task(run_dir, task, identity_hash)
    if receipt is None:
        raise FileNotFoundError(f"Task is incomplete: {task.task_id}")
    values = np.load(task_paths(run_dir, task)["prediction"])
    return values["rows"].astype(int), values["predictions"].astype(float), receipt
