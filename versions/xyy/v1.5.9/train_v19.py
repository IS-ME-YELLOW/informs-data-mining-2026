"""训练 v1.9 E1/E2/E3 分量候选，并保存完整 OOF、测试预测和模型。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    BASE_FEATURE_VERSION,
    BASE_LGBM_PARAMS,
    CACHE_DIR,
    CV_MODELS_DIR,
    CV_VERSION,
    EARLY_STOPPING_ROUNDS,
    EXPERIMENT_VERSION,
    FEATURE_NAMES_FILE,
    FINAL_MODELS_DIR,
    HORIZONS,
    MAX_BOOST_ROUNDS,
    PROJECT_ROOT,
    RESPONSE_FEATURES,
    SEED,
    V18_DIR,
    V19_DIR,
    ensure_dirs,
)
from protocol import (
    build_component_targets,
    clip_component,
    component_target_name,
    json_ready,
    load_data,
    metrics,
    sha256_file,
)

COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
REGRESSION_CANDIDATES = {
    "P_t": ("huber", "l2", "weighted_l2"),
    "D_t": ("huber", "l2", "weighted_l2"),
    "N_t": ("huber", "tweedie", "hurdle"),
    "R_t": ("huber", "tweedie", "hurdle"),
}


def candidate_key(feature_set: str, component: str, candidate: str, horizon: str) -> str:
    return f"{feature_set}__{component}__{candidate}__{horizon}"


def safe_name(text: str) -> str:
    return text.replace("_target_", "_").replace("osi_target_", "")


def model_path(
    role: str,
    feature_set: str,
    component: str,
    candidate: str,
    horizon: str,
    stage: str,
    fold: int | None = None,
) -> Path:
    stem = f"lgbm_{feature_set}_{component}_{candidate}_{safe_name(horizon)}_{stage}"
    if role == "cv":
        return CV_MODELS_DIR / f"{stem}_{EXPERIMENT_VERSION}_{CV_VERSION}_fold{fold}.txt"
    return FINAL_MODELS_DIR / f"{stem}_{EXPERIMENT_VERSION}_{CV_VERSION}_final.txt"


def params_for(candidate: str, stage: str = "regression") -> dict:
    params = dict(BASE_LGBM_PARAMS)
    if stage == "classification":
        params.update(objective="binary", metric="binary_logloss")
    elif stage == "magnitude":
        params.update(objective="gamma", metric="rmse")
    elif candidate == "huber":
        params.update(objective="huber", metric="rmse")
    elif candidate in {"l2", "weighted_l2"}:
        params.update(objective="regression", metric="rmse")
    elif candidate == "tweedie":
        params.update(objective="tweedie", metric="rmse", tweedie_variance_power=1.5)
    else:
        raise ValueError(f"未知候选: {candidate}/{stage}")
    return params


def high_value_weights(y: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(y, dtype=float)
    q95 = float(np.quantile(values[np.isfinite(values)], 0.95))
    if q95 <= 0:
        return np.ones_like(values), q95
    return 1.0 + 2.0 * np.clip(values / q95, 0.0, 1.0), q95


def fit_or_load(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    params: dict,
    path: Path,
    resume: bool,
    train_weights: np.ndarray | None = None,
    fixed_rounds: int | None = None,
) -> lgb.Booster:
    if resume and path.exists():
        existing = lgb.Booster(model_file=str(path))
        if fixed_rounds is None or existing.num_trees() == int(fixed_rounds):
            return existing
    train_set = lgb.Dataset(
        X.iloc[train_idx],
        label=y[train_idx],
        weight=None if train_weights is None else train_weights,
        free_raw_data=True,
    )
    if fixed_rounds is None:
        valid_set = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx], reference=train_set)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=MAX_BOOST_ROUNDS,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    else:
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=int(fixed_rounds),
            callbacks=[lgb.log_evaluation(0)],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return booster


def manifest_record(
    booster: lgb.Booster,
    path: Path,
    role: str,
    feature_set: str,
    component: str,
    candidate: str,
    horizon: str,
    stage: str,
    fold: int | None,
) -> dict:
    return {
        "role": role,
        "feature_set": feature_set,
        "component": component,
        "candidate": candidate,
        "horizon": horizon,
        "stage": stage,
        "fold": fold,
        "rounds": int(booster.num_trees()),
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(path),
    }


def importance_records(
    booster: lgb.Booster,
    feature_names: list[str],
    feature_set: str,
    component: str,
    candidate: str,
    horizon: str,
    stage: str,
) -> list[dict]:
    gain = booster.feature_importance(importance_type="gain")
    total = float(gain.sum())
    return [
        {
            "feature_set": feature_set,
            "component": component,
            "candidate": candidate,
            "horizon": horizon,
            "stage": stage,
            "feature": feature_names[idx],
            "gain": float(value),
            "gain_fraction": float(value / total) if total > 0 else 0.0,
        }
        for idx, value in enumerate(gain)
    ]


def train_regular_candidate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_series: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_set: str,
    component: str,
    candidate: str,
    horizon: str,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict], list[dict], dict]:
    y = y_series.to_numpy(dtype=float)
    valid = np.isfinite(y)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict] = []
    manifests: list[dict] = []
    best_rounds: list[int] = []
    q95_values: list[float] = []
    params = params_for(candidate)

    for fold, (train_all, valid_all) in enumerate(folds):
        train_idx = train_all[valid[train_all]]
        valid_idx = valid_all[valid[valid_all]]
        weights = None
        q95 = np.nan
        if candidate == "weighted_l2":
            weights, q95 = high_value_weights(y[train_idx])
        path = model_path("cv", feature_set, component, candidate, horizon, "regression", fold)
        booster = fit_or_load(
            X_train, y, train_idx, valid_idx, params, path, resume,
            train_weights=weights,
        )
        pred = clip_component(booster.predict(X_train.iloc[valid_idx]))
        oof[valid_idx] = pred
        score = metrics(y[valid_idx], pred)
        fold_rows.append({
            "feature_set": feature_set,
            "component": component,
            "candidate": candidate,
            "horizon": horizon,
            "fold": fold,
            **score,
            "best_rounds": int(booster.num_trees()),
            "train_q95": float(q95),
        })
        manifests.append(manifest_record(
            booster, path, "cv", feature_set, component, candidate, horizon, "regression", fold
        ))
        best_rounds.append(int(booster.num_trees()))
        q95_values.append(float(q95))

    final_rounds = max(1, int(np.mean(best_rounds)))
    all_idx = np.flatnonzero(valid)
    final_weights = None
    final_q95 = np.nan
    if candidate == "weighted_l2":
        final_weights, final_q95 = high_value_weights(y[all_idx])
    final_path = model_path("final", feature_set, component, candidate, horizon, "regression")
    final = fit_or_load(
        X_train, y, all_idx, all_idx[:1], params, final_path, resume,
        train_weights=final_weights, fixed_rounds=final_rounds,
    )
    test_pred = clip_component(final.predict(X_test))
    manifests.append(manifest_record(
        final, final_path, "final", feature_set, component, candidate, horizon, "regression", None
    ))
    importance = importance_records(
        final, list(X_train.columns), feature_set, component, candidate, horizon, "regression"
    )
    summary = {
        "feature_set": feature_set,
        "component": component,
        "candidate": candidate,
        "horizon": horizon,
        **metrics(y, oof),
        "best_rounds": json.dumps(best_rounds),
        "final_rounds": int(final.num_trees()),
        "fold_train_q95": json.dumps(q95_values),
        "final_train_q95": float(final_q95),
    }
    return oof, test_pred, fold_rows, manifests, importance, summary


def train_hurdle_candidate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_series: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    feature_set: str,
    component: str,
    horizon: str,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict], list[dict], dict]:
    y = y_series.to_numpy(dtype=float)
    valid = np.isfinite(y)
    positive = valid & (y > 0)
    binary_y = (y > 0).astype(float)
    oof = np.full(len(y), np.nan, dtype=float)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    oof_magnitude = np.full(len(y), np.nan, dtype=float)
    fold_rows: list[dict] = []
    manifests: list[dict] = []
    cls_rounds: list[int] = []
    mag_rounds: list[int] = []

    for fold, (train_all, valid_all) in enumerate(folds):
        train_idx = train_all[valid[train_all]]
        valid_idx = valid_all[valid[valid_all]]
        pos_train_idx = train_all[positive[train_all]]
        pos_valid_idx = valid_all[positive[valid_all]]
        if len(pos_train_idx) == 0 or len(pos_valid_idx) == 0:
            raise ValueError(f"{component}/{horizon}/fold{fold} 没有正值样本")

        cls_path = model_path("cv", feature_set, component, "hurdle", horizon, "classification", fold)
        cls = fit_or_load(
            X_train, binary_y, train_idx, valid_idx,
            params_for("hurdle", "classification"), cls_path, resume,
        )
        mag_path = model_path("cv", feature_set, component, "hurdle", horizon, "magnitude", fold)
        mag = fit_or_load(
            X_train, y, pos_train_idx, pos_valid_idx,
            params_for("hurdle", "magnitude"), mag_path, resume,
        )
        prob = np.clip(cls.predict(X_train.iloc[valid_idx]), 0.0, 1.0)
        magnitude = np.maximum(mag.predict(X_train.iloc[valid_idx]), 0.0)
        pred = clip_component(prob * magnitude)
        oof[valid_idx] = pred
        oof_prob[valid_idx] = prob
        oof_magnitude[valid_idx] = magnitude
        fold_rows.append({
            "feature_set": feature_set,
            "component": component,
            "candidate": "hurdle",
            "horizon": horizon,
            "fold": fold,
            **metrics(y[valid_idx], pred),
            "classification_rounds": int(cls.num_trees()),
            "magnitude_rounds": int(mag.num_trees()),
            "positive_train_rows": int(len(pos_train_idx)),
            "positive_valid_rows": int(len(pos_valid_idx)),
        })
        manifests.extend([
            manifest_record(cls, cls_path, "cv", feature_set, component, "hurdle", horizon, "classification", fold),
            manifest_record(mag, mag_path, "cv", feature_set, component, "hurdle", horizon, "magnitude", fold),
        ])
        cls_rounds.append(int(cls.num_trees()))
        mag_rounds.append(int(mag.num_trees()))

    all_idx = np.flatnonzero(valid)
    all_pos_idx = np.flatnonzero(positive)
    final_cls_rounds = max(1, int(np.mean(cls_rounds)))
    final_mag_rounds = max(1, int(np.mean(mag_rounds)))
    cls_path = model_path("final", feature_set, component, "hurdle", horizon, "classification")
    cls = fit_or_load(
        X_train, binary_y, all_idx, all_idx[:1], params_for("hurdle", "classification"),
        cls_path, resume, fixed_rounds=final_cls_rounds,
    )
    mag_path = model_path("final", feature_set, component, "hurdle", horizon, "magnitude")
    mag = fit_or_load(
        X_train, y, all_pos_idx, all_pos_idx[:1], params_for("hurdle", "magnitude"),
        mag_path, resume, fixed_rounds=final_mag_rounds,
    )
    test_prob = np.clip(cls.predict(X_test), 0.0, 1.0)
    test_magnitude = np.maximum(mag.predict(X_test), 0.0)
    test_pred = clip_component(test_prob * test_magnitude)
    manifests.extend([
        manifest_record(cls, cls_path, "final", feature_set, component, "hurdle", horizon, "classification", None),
        manifest_record(mag, mag_path, "final", feature_set, component, "hurdle", horizon, "magnitude", None),
    ])
    importance = importance_records(
        cls, list(X_train.columns), feature_set, component, "hurdle", horizon, "classification"
    ) + importance_records(
        mag, list(X_train.columns), feature_set, component, "hurdle", horizon, "magnitude"
    )
    summary = {
        "feature_set": feature_set,
        "component": component,
        "candidate": "hurdle",
        "horizon": horizon,
        **metrics(y, oof),
        "classification_rounds": json.dumps(cls_rounds),
        "magnitude_rounds": json.dumps(mag_rounds),
        "final_classification_rounds": int(cls.num_trees()),
        "final_magnitude_rounds": int(mag.num_trees()),
        "positive_rows": int(positive.sum()),
    }
    return (
        oof, test_pred, oof_prob, oof_magnitude,
        fold_rows, manifests, importance, summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    started = time.perf_counter()

    print("[1/5] 加载 v1.9 特征、v1.8 分量目标和固定县级折...", flush=True)
    data = load_data()
    targets, target_validation = build_component_targets(data)
    all_names = json.loads(FEATURE_NAMES_FILE.read_text(encoding="utf-8"))
    base_names = json.loads(
        (CACHE_DIR / f"feature_names_{BASE_FEATURE_VERSION}.json").read_text(encoding="utf-8")
    )
    if all_names != [*base_names, *RESPONSE_FEATURES]:
        raise ValueError("v1.9 特征清单与配置不一致")
    feature_sets = {
        "base": (data.X_train[base_names], data.X_test[base_names]),
        "profile": (data.X_train[all_names], data.X_test[all_names]),
    }

    baseline_oof = pd.read_parquet(V18_DIR / "oof_component_predictions.parquet")
    baseline_test = pd.read_parquet(V18_DIR / "test_component_predictions.parquet")
    if len(baseline_oof) != len(data.X_train) or len(baseline_test) != len(data.X_test):
        raise ValueError("v1.8 基线预测行数不一致")

    oof_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target = component_target_name(component, horizon)
            key = candidate_key("base", component, "huber", horizon)
            oof_predictions[key] = baseline_oof[f"pred_{target}"].to_numpy(dtype=float)
            test_predictions[key] = baseline_test[f"pred_{target}"].to_numpy(dtype=float)

    tasks: list[tuple[str, str, str, str]] = []
    for feature_set in ("base", "profile"):
        for component in COMPONENTS:
            for candidate in REGRESSION_CANDIDATES[component]:
                if feature_set == "base" and candidate == "huber":
                    continue
                for horizon in HORIZONS:
                    tasks.append((feature_set, component, candidate, horizon))

    fold_rows: list[dict] = []
    manifests: list[dict] = []
    importance: list[dict] = []
    summaries: list[dict] = []
    hurdle_oof: dict[str, np.ndarray] = {}
    hurdle_test: dict[str, np.ndarray] = {}

    print(f"[2/5] 训练 {len(tasks)} 个特征集/分量/候选/horizon 任务...", flush=True)
    for task_no, (feature_set, component, candidate, horizon) in enumerate(tasks, start=1):
        print(
            f"  [{task_no:02d}/{len(tasks)}] {feature_set} {component} {candidate} {horizon}",
            flush=True,
        )
        X_train, X_test = feature_sets[feature_set]
        target = component_target_name(component, horizon)
        key = candidate_key(feature_set, component, candidate, horizon)
        if candidate == "hurdle":
            (
                oof, test_pred, oof_prob, oof_magnitude,
                task_folds, task_manifest, task_importance, task_summary,
            ) = train_hurdle_candidate(
                X_train, X_test, targets[target], data.folds,
                feature_set, component, horizon, args.resume,
            )
            hurdle_oof[f"prob__{key}"] = oof_prob
            hurdle_oof[f"magnitude__{key}"] = oof_magnitude
            hurdle_test[f"prob__{key}"] = np.full(len(X_test), np.nan)
            hurdle_test[f"magnitude__{key}"] = np.full(len(X_test), np.nan)
            # 测试概率和条件均值可由最终模型重载复算；主文件保存乘积预测。
        else:
            (
                oof, test_pred, task_folds, task_manifest, task_importance, task_summary,
            ) = train_regular_candidate(
                X_train, X_test, targets[target], data.folds,
                feature_set, component, candidate, horizon, args.resume,
            )
        oof_predictions[key] = oof
        test_predictions[key] = test_pred
        fold_rows.extend(task_folds)
        manifests.extend(task_manifest)
        importance.extend(task_importance)
        summaries.append(task_summary)

    print("[3/5] 保存 OOF、测试预测、指标和模型清单...", flush=True)
    oof_frame = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    test_frame = data.meta_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    for key, values in oof_predictions.items():
        oof_frame[f"pred__{key}"] = values
    for key, values in test_predictions.items():
        test_frame[f"pred__{key}"] = values
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target = component_target_name(component, horizon)
            oof_frame[f"actual__{component}__{horizon}"] = targets[target].to_numpy(dtype=float)
    oof_frame.to_parquet(V19_DIR / "oof_component_candidates.parquet", index=False)
    test_frame.to_parquet(V19_DIR / "test_component_candidates.parquet", index=False)

    hurdle_oof_frame = data.meta_train[["fipsCode", "timestamp_et", "hour_idx"]].copy()
    for key, values in hurdle_oof.items():
        hurdle_oof_frame[key] = values
    hurdle_oof_frame.to_parquet(V19_DIR / "hurdle_oof_diagnostics.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(V19_DIR / "training_fold_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(V19_DIR / "training_target_summary.csv", index=False)
    pd.DataFrame(importance).to_csv(V19_DIR / "feature_importance.csv", index=False)
    manifest_frame = pd.DataFrame(manifests)
    manifest_frame.to_csv(V19_DIR / "model_manifest.csv", index=False)

    print("[4/5] 校验覆盖、NaN 掩码和模型哈希...", flush=True)
    for horizon in HORIZONS:
        for component in COMPONENTS:
            truth = targets[component_target_name(component, horizon)].to_numpy(dtype=float)
            valid = np.isfinite(truth)
            for feature_set in ("base", "profile"):
                for candidate in REGRESSION_CANDIDATES[component]:
                    if feature_set == "base" and candidate == "huber":
                        key = candidate_key(feature_set, component, candidate, horizon)
                    else:
                        key = candidate_key(feature_set, component, candidate, horizon)
                    values = oof_predictions[key]
                    if not np.array_equal(np.isfinite(values), valid):
                        raise AssertionError(f"OOF 掩码错误: {key}")
    for _, row in manifest_frame.iterrows():
        path = PROJECT_ROOT / row["path"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise AssertionError(f"模型哈希错误: {path}")

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "experiment_version": EXPERIMENT_VERSION,
        "base_feature_version": BASE_FEATURE_VERSION,
        "feature_count_base": len(base_names),
        "feature_count_profile": len(all_names),
        "response_feature_count": len(RESPONSE_FEATURES),
        "cv_version": CV_VERSION,
        "seed": SEED,
        "task_count": len(tasks),
        "saved_models": int(len(manifest_frame)),
        "candidate_prediction_count_including_v18_baseline": len(oof_predictions),
        "runtime_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "lightgbm": getattr(lgb, "__version__", "unknown"),
        "target_validation": target_validation,
        "inputs_sha256": {
            "features_train_v1.9.parquet": sha256_file(V19_DIR / "features_train_v1.9.parquet"),
            "features_test_v1.9.parquet": sha256_file(V19_DIR / "features_test_v1.9.parquet"),
            "component_targets_v1.8.parquet": sha256_file(V18_DIR / "component_targets_v1.8.parquet"),
            "oof_component_predictions_v1.8.parquet": sha256_file(V18_DIR / "oof_component_predictions.parquet"),
        },
    }
    (V19_DIR / "training_metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[5/5] 训练阶段完成。", flush=True)
    print(json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
