"""训练并评估 v1.8：v1.5.6 特征上的直接 OSI 与 P/N/D/R 分量 LightGBM。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CACHE_DIR,
    COMPONENTS,
    COMPONENT_WEIGHTS,
    CV_FILE,
    CV_MODELS_DIR,
    CV_VERSION,
    EARLY_STOPPING_ROUNDS,
    EXPERIMENT_VERSION,
    FEATURE_VERSION,
    FINAL_MODELS_DIR,
    HORIZON_HOURS,
    HORIZONS,
    LGBM_PARAMS,
    LOG_FILE,
    MAX_BOOST_ROUNDS,
    PROJECT_ROOT,
    RAW_TRAIN_FILE,
    SEED,
    SUBMISSION_TEMPLATE,
    V18_DIR,
    ensure_dirs,
)
from protocol import (
    align_same_target,
    build_component_targets,
    clip_component,
    component_target_name,
    compose_osi,
    fill_submission,
    json_ready,
    load_data,
    metrics,
    post_process,
    sha256_file,
)


MODELS = ("C0_direct_osi", "C1_component_osi", "C2_equal_blend", "C3_aligned_component")


def _safe_target(target: str) -> str:
    return target.replace("_target_", "_")


def _model_path(role: str, target: str, fold: int | None = None) -> Path:
    safe = _safe_target(target)
    if role == "cv":
        return CV_MODELS_DIR / f"lgbm_{safe}_{EXPERIMENT_VERSION}_{CV_VERSION}_fold{fold}.txt"
    return FINAL_MODELS_DIR / f"lgbm_{safe}_{EXPERIMENT_VERSION}_{CV_VERSION}_final.txt"


def _train_target(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target: str,
    target_kind: str,
    max_rounds: int,
    early_stopping_rounds: int,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict], dict, list[dict], list[dict]]:
    valid_target = y.notna().to_numpy()
    oof_raw = np.full(len(y), np.nan, dtype=float)
    fold_records: list[dict] = []
    model_records: list[dict] = []
    best_rounds: list[int] = []
    fold_scores = []

    for fold_number, (train_full, valid_full) in enumerate(folds):
        train_idx = train_full[valid_target[train_full]]
        valid_idx = valid_full[valid_target[valid_full]]
        path = _model_path("cv", target, fold_number)
        if resume:
            if not path.exists():
                raise FileNotFoundError(f"恢复模式缺少折模型: {path}")
            booster = lgb.Booster(model_file=str(path))
            best_round = int(booster.num_trees())
        else:
            dtrain = lgb.Dataset(X_train.iloc[train_idx], label=y.iloc[train_idx])
            dvalid = lgb.Dataset(X_train.iloc[valid_idx], label=y.iloc[valid_idx], reference=dtrain)
            booster = lgb.train(
                {**LGBM_PARAMS, "metric": "rmse"},
                dtrain,
                num_boost_round=max_rounds,
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_round = int(booster.best_iteration)
            booster.save_model(str(path), num_iteration=best_round)
        raw = booster.predict(X_train.iloc[valid_idx], num_iteration=best_round)
        oof_raw[valid_idx] = raw
        evaluated = clip_component(raw) if target_kind == "component" else post_process(raw)
        score = metrics(y.iloc[valid_idx].to_numpy(), evaluated)
        best_rounds.append(best_round)
        fold_scores.append(score)
        fold_records.append({
            "target_kind": target_kind,
            "target": target,
            "fold": fold_number,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "best_round": best_round,
            **score,
        })
        model_records.append({
            "role": "cv",
            "target_kind": target_kind,
            "target": target,
            "fold": fold_number,
            "rounds": best_round,
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
        })
        print(
            f"      fold={fold_number} RMSE={score['rmse']:.6f} "
            f"MAE={score['mae']:.6f} best_round={best_round}", flush=True
        )

    if np.isnan(oof_raw[valid_target]).any() or np.isfinite(oof_raw[~valid_target]).any():
        raise ValueError(f"{target} OOF 覆盖异常")
    evaluated_oof = clip_component(oof_raw) if target_kind == "component" else post_process(oof_raw)
    pooled = metrics(y.to_numpy(), evaluated_oof)
    final_rounds = max(1, int(np.mean(best_rounds)))
    summary = {
        "target_kind": target_kind,
        "target": target,
        **pooled,
        "rmse_std": float(np.std([item["rmse"] for item in fold_scores])),
        "mae_std": float(np.std([item["mae"] for item in fold_scores])),
        "best_rounds": json.dumps(best_rounds),
        "final_rounds": final_rounds,
    }

    final_path = _model_path("final", target)
    if resume:
        if not final_path.exists():
            raise FileNotFoundError(f"恢复模式缺少全量模型: {final_path}")
        final = lgb.Booster(model_file=str(final_path))
        if int(final.num_trees()) != final_rounds:
            raise ValueError(
                f"恢复模型轮数不一致: {target} {final.num_trees()} != {final_rounds}"
            )
    else:
        valid_idx = np.flatnonzero(valid_target)
        final = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_train.iloc[valid_idx], label=y.iloc[valid_idx]),
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        final.save_model(str(final_path), num_iteration=final_rounds)
    model_records.append({
        "role": "final",
        "target_kind": target_kind,
        "target": target,
        "fold": -1,
        "rounds": final_rounds,
        "path": str(final_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(final_path),
    })
    test_raw = final.predict(X_test, num_iteration=final_rounds)
    importance = final.feature_importance(importance_type="gain")
    total = float(importance.sum())
    importance_records = [
        {
            "target_kind": target_kind,
            "target": target,
            "rank": rank,
            "feature": X_train.columns[idx],
            "importance_gain": float(importance[idx]),
            "importance_fraction": float(importance[idx] / total) if total > 0 else 0.0,
        }
        for rank, idx in enumerate(np.argsort(-importance), start=1)
    ]
    print(
        f"      pooled RMSE={pooled['rmse']:.6f} MAE={pooled['mae']:.6f}; "
        f"final_rounds={final_rounds}", flush=True
    )
    return oof_raw, test_raw, fold_records, summary, importance_records, model_records


def _build_controls(meta: pd.DataFrame, direct_raw: dict, component_raw: dict) -> tuple[dict, dict]:
    raw_controls: dict[str, dict[str, np.ndarray]] = {name: {} for name in MODELS}
    aligned_components: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZONS}
    unique_component_frames = []
    for component in COMPONENTS:
        by_horizon = {
            horizon: clip_component(component_raw[horizon][component])
            for horizon in HORIZONS
        }
        mapped, unique = align_same_target(meta, by_horizon)
        for horizon in HORIZONS:
            aligned_components[horizon][component] = mapped[horizon]
        unique.insert(0, "component", component)
        unique_component_frames.append(unique)

    for horizon in HORIZONS:
        c0_raw = np.asarray(direct_raw[horizon], dtype=float)
        c1_raw = compose_osi({
            component: clip_component(component_raw[horizon][component])
            for component in COMPONENTS
        })
        c3_raw = compose_osi(aligned_components[horizon])
        raw_controls["C0_direct_osi"][horizon] = c0_raw
        raw_controls["C1_component_osi"][horizon] = c1_raw
        raw_controls["C2_equal_blend"][horizon] = 0.5 * (c0_raw + c1_raw)
        raw_controls["C3_aligned_component"][horizon] = c3_raw
    controls = {
        model: {horizon: post_process(raw_controls[model][horizon]) for horizon in HORIZONS}
        for model in MODELS
    }
    return controls, {
        "raw_controls": raw_controls,
        "aligned_components": aligned_components,
        "unique_components": pd.concat(unique_component_frames, ignore_index=True),
    }


def _evaluation_tables(data, targets, controls, component_oof) -> dict[str, pd.DataFrame]:
    summary_rows = []
    fold_rows = []
    county_rows = []
    severity_rows = []
    day_rows = []
    correlation_rows = []
    contribution_rows = []
    component_rows = []

    c0_metrics = {
        horizon: metrics(data.y_train[horizon], controls["C0_direct_osi"][horizon])
        for horizon in HORIZONS
    }
    for model in MODELS:
        for horizon in HORIZONS:
            truth = data.y_train[horizon].to_numpy(dtype=float)
            pred = controls[model][horizon]
            score = metrics(truth, pred)
            base = c0_metrics[horizon]
            summary_rows.append({
                "model": model,
                "horizon": horizon,
                **score,
                "rmse_change_pct_vs_C0": 100 * (score["rmse"] / base["rmse"] - 1),
                "mae_change_pct_vs_C0": 100 * (score["mae"] / base["mae"] - 1),
                "prediction_mean": float(np.nanmean(pred)),
                "prediction_max": float(np.nanmax(pred)),
                "zero_pct": float(100 * np.mean(pred[np.isfinite(pred)] == 0)),
            })
            for fold in range(5):
                hit = data.row_folds == fold
                fold_rows.append({"model": model, "horizon": horizon, "fold": fold,
                                  **metrics(truth[hit], pred[hit])})
            for fips in data.meta_train["fipsCode"].drop_duplicates():
                hit = data.meta_train["fipsCode"].to_numpy() == fips
                county_rows.append({"model": model, "horizon": horizon, "fipsCode": fips,
                                    **metrics(truth[hit], pred[hit])})

            target_time = data.meta_train["timestamp_et"] + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h")
            valid = np.isfinite(truth) & np.isfinite(pred)
            bins = np.select(
                [truth == 0, (truth > 0) & (truth <= 0.01), (truth > 0.01) & (truth <= 0.05), truth > 0.05],
                ["zero", "0_to_0.01", "0.01_to_0.05", "above_0.05"], default="invalid"
            )
            for severity in ("zero", "0_to_0.01", "0.01_to_0.05", "above_0.05"):
                hit = valid & (bins == severity)
                if hit.any():
                    severity_rows.append({"model": model, "horizon": horizon, "severity": severity,
                                          **metrics(truth[hit], pred[hit])})
            for day in sorted(target_time[valid].dt.strftime("%Y-%m-%d").unique()):
                hit = valid & (target_time.dt.strftime("%Y-%m-%d").to_numpy() == day)
                day_rows.append({"model": model, "horizon": horizon, "target_day": day,
                                 **metrics(truth[hit], pred[hit])})

    for horizon in HORIZONS:
        truth = data.y_train[horizon].to_numpy(dtype=float)
        valid = np.isfinite(truth)
        e0 = controls["C0_direct_osi"][horizon][valid] - truth[valid]
        e1 = controls["C1_component_osi"][horizon][valid] - truth[valid]
        correlation_rows.append({
            "horizon": horizon,
            "pearson_error_correlation_C0_C1": float(np.corrcoef(e0, e1)[0, 1]),
            "sign_disagreement_pct": float(100 * np.mean(np.sign(e0) != np.sign(e1))),
        })
        osi_error = e1
        for component in COMPONENTS:
            target_name = component_target_name(component, horizon)
            actual_component = targets[target_name].to_numpy(dtype=float)
            pred_component = clip_component(component_oof[horizon][component])
            component_rows.append({
                "horizon": horizon,
                "component": component,
                **metrics(actual_component, pred_component),
            })
            contribution = COMPONENT_WEIGHTS[component] * (pred_component[valid] - actual_component[valid])
            contribution_rows.append({
                "horizon": horizon,
                "component": component,
                "weight": COMPONENT_WEIGHTS[component],
                "contribution_rmse": float(np.sqrt(np.mean(contribution ** 2))),
                "contribution_mae": float(np.mean(np.abs(contribution))),
                "correlation_with_C1_osi_error": float(np.corrcoef(contribution, osi_error)[0, 1]),
            })

    return {
        "summary_metrics.csv": pd.DataFrame(summary_rows),
        "control_fold_metrics.csv": pd.DataFrame(fold_rows),
        "county_metrics.csv": pd.DataFrame(county_rows),
        "severity_metrics.csv": pd.DataFrame(severity_rows),
        "target_day_metrics.csv": pd.DataFrame(day_rows),
        "error_correlation.csv": pd.DataFrame(correlation_rows),
        "component_contribution.csv": pd.DataFrame(contribution_rows),
        "component_summary_metrics.csv": pd.DataFrame(component_rows),
    }


def _paired_bootstrap(data, controls, replicates: int, seed: int) -> pd.DataFrame:
    comparisons = [
        ("C1_component_osi", "C0_direct_osi"),
        ("C2_equal_blend", "C0_direct_osi"),
        ("C2_equal_blend", "C1_component_osi"),
        ("C3_aligned_component", "C0_direct_osi"),
        ("C3_aligned_component", "C1_component_osi"),
    ]
    counties = data.meta_train["fipsCode"].drop_duplicates().to_numpy()
    county_index = {fips: idx for idx, fips in enumerate(counties)}
    row_county = data.meta_train["fipsCode"].map(county_index).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(counties), size=(replicates, len(counties)))
    rows = []
    for horizon in HORIZONS:
        truth = data.y_train[horizon].to_numpy(dtype=float)
        stats = {}
        for model in MODELS:
            pred = controls[model][horizon]
            valid = np.isfinite(truth) & np.isfinite(pred)
            err = pred - truth
            stats[model] = {
                "sse": np.bincount(row_county[valid], weights=err[valid] ** 2, minlength=len(counties)),
                "sae": np.bincount(row_county[valid], weights=np.abs(err[valid]), minlength=len(counties)),
                "n": np.bincount(row_county[valid], minlength=len(counties)),
            }
        for model, reference in comparisons:
            left, right = stats[model], stats[reference]
            left_n = left["n"][sampled].sum(axis=1)
            right_n = right["n"][sampled].sum(axis=1)
            metric_values = {
                "rmse": (
                    np.sqrt(left["sse"][sampled].sum(axis=1) / left_n),
                    np.sqrt(right["sse"][sampled].sum(axis=1) / right_n),
                    np.sqrt(left["sse"].sum() / left["n"].sum()),
                    np.sqrt(right["sse"].sum() / right["n"].sum()),
                ),
                "mae": (
                    left["sae"][sampled].sum(axis=1) / left_n,
                    right["sae"][sampled].sum(axis=1) / right_n,
                    left["sae"].sum() / left["n"].sum(),
                    right["sae"].sum() / right["n"].sum(),
                ),
            }
            for metric_name, (left_rep, right_rep, left_point, right_point) in metric_values.items():
                diff = left_rep - right_rep
                low, high = np.quantile(diff, [0.025, 0.975])
                rows.append({
                    "model": model,
                    "reference": reference,
                    "horizon": horizon,
                    "metric": metric_name,
                    "difference": float(left_point - right_point),
                    "change_pct": float(100 * (left_point / right_point - 1)),
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "improvement_probability": float(np.mean(diff < 0)),
                    "replicates": replicates,
                    "unit": "county_full_trajectory",
                })
    return pd.DataFrame(rows)


def _prediction_frames(data, targets, controls, component_oof, component_test) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    for horizon in HORIZONS:
        oof[f"actual_{horizon}"] = data.y_train[horizon]
        for model in MODELS:
            oof[f"pred_{model}_{horizon}"] = controls[model][horizon]
    component_train = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    component_test_frame = data.meta_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    for horizon in HORIZONS:
        for component in COMPONENTS:
            name = component_target_name(component, horizon)
            component_train[f"actual_{name}"] = targets[name]
            component_train[f"pred_{name}"] = clip_component(component_oof[horizon][component])
            component_test_frame[f"pred_{name}"] = clip_component(component_test[horizon][component])
    return oof, component_train, component_test_frame


def _append_log(metadata: dict) -> None:
    lines = [
        "",
        "=" * 72,
        f"Experiment: {EXPERIMENT_VERSION} | {metadata['created_at']}",
        "Strategy: v1.5.6 direct OSI vs P/N/D/R LightGBM and controls",
        f"Feature version: {FEATURE_VERSION} (163 columns); CV: {CV_VERSION}; seed: {SEED}",
    ]
    summary = pd.read_csv(V18_DIR / "summary_metrics.csv")
    for _, row in summary.iterrows():
        lines.append(
            f"{row['model']} {row['horizon']}: RMSE={row['rmse']:.9f}, "
            f"MAE={row['mae']:.9f}, deltaRMSEvsC0={row['rmse_change_pct_vs_C0']:+.4f}%"
        )
    lines.extend([
        f"Models saved: {metadata['model_count']} ({metadata['cv_model_count']} CV + {metadata['final_model_count']} final)",
        f"Runtime seconds: {metadata['runtime_seconds']:.1f}",
        f"Report: versions/v1.8/Report_v1.8.md",
        "=" * 72,
    ])
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force-targets", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    parser.add_argument(
        "--resume", action="store_true",
        help="加载已保存的全部折模型和全量模型，重算预测并继续生成评估产物",
    )
    args = parser.parse_args()
    if args.max_rounds < 1 or args.early_stopping_rounds < 1:
        parser.error("训练轮数和早停耐心必须为正数")
    ensure_dirs()
    started = time.perf_counter()
    print(f"=== {EXPERIMENT_VERSION}: v1.5.6 P/N/D/R component LightGBM ===", flush=True)
    print("[1/8] 校验特征、固定县级折与分量标签...", flush=True)
    data = load_data()
    targets, target_validation = build_component_targets(data, force=args.force_targets)
    print(
        f"      train={data.X_train.shape}, test={data.X_test.shape}, "
        f"county={data.meta_train['fipsCode'].nunique()}, fold_count="
        f"{pd.Series(data.row_folds).value_counts().sort_index().to_dict()}", flush=True
    )
    if args.validate_only:
        print(json.dumps(target_validation, ensure_ascii=False, indent=2))
        return

    direct_oof: dict[str, np.ndarray] = {}
    direct_test: dict[str, np.ndarray] = {}
    component_oof: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZONS}
    component_test: dict[str, dict[str, np.ndarray]] = {h: {} for h in HORIZONS}
    training_fold_records = []
    training_summaries = []
    importance_records = []
    model_records = []

    print("[2/8] 训练 4 个直接 OSI LightGBM...", flush=True)
    for horizon in HORIZONS:
        print(f"  [C0 / {horizon}]", flush=True)
        result = _train_target(
            data.X_train, data.X_test, data.y_train[horizon], data.folds,
            horizon, "direct_osi", args.max_rounds, args.early_stopping_rounds,
            args.resume,
        )
        direct_oof[horizon], direct_test[horizon] = result[0], result[1]
        training_fold_records.extend(result[2]); training_summaries.append(result[3])
        importance_records.extend(result[4]); model_records.extend(result[5])

    print("[3/8] 训练 16 个 P/N/D/R LightGBM...", flush=True)
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target_name = component_target_name(component, horizon)
            print(f"  [C1 / {horizon} / {component}]", flush=True)
            result = _train_target(
                data.X_train, data.X_test, targets[target_name], data.folds,
                target_name, "component", args.max_rounds, args.early_stopping_rounds,
                args.resume,
            )
            component_oof[horizon][component], component_test[horizon][component] = result[0], result[1]
            training_fold_records.extend(result[2]); training_summaries.append(result[3])
            importance_records.extend(result[4]); model_records.extend(result[5])

    print("[4/8] 构造 C0/C1/C2/C3 并计算诊断...", flush=True)
    oof_controls, oof_aux = _build_controls(data.meta_train, direct_oof, component_oof)
    test_controls, test_aux = _build_controls(data.meta_test, direct_test, component_test)
    tables = _evaluation_tables(data, targets, oof_controls, component_oof)
    tables["paired_county_bootstrap.csv"] = _paired_bootstrap(
        data, oof_controls, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
    )
    for filename, frame in tables.items():
        frame.to_csv(V18_DIR / filename, index=False)
    pd.DataFrame(training_fold_records).to_csv(V18_DIR / "training_fold_metrics.csv", index=False)
    pd.DataFrame(training_summaries).to_csv(V18_DIR / "training_target_summary.csv", index=False)
    pd.DataFrame(importance_records).to_csv(V18_DIR / "feature_importance.csv", index=False)
    model_manifest = pd.DataFrame(model_records)
    model_manifest.to_csv(V18_DIR / "model_manifest.csv", index=False)

    print("[5/8] 保存 OOF、分量与测试预测...", flush=True)
    oof_frame, component_oof_frame, component_test_frame = _prediction_frames(
        data, targets, oof_controls, component_oof, component_test
    )
    oof_frame.to_parquet(V18_DIR / "oof_predictions.parquet", index=False)
    component_oof_frame.to_parquet(V18_DIR / "oof_component_predictions.parquet", index=False)
    component_test_frame.to_parquet(V18_DIR / "test_component_predictions.parquet", index=False)
    oof_aux["unique_components"].to_parquet(V18_DIR / "oof_aligned_unique_components.parquet", index=False)
    test_aux["unique_components"].to_parquet(V18_DIR / "test_aligned_unique_components.parquet", index=False)
    test_control_frame = data.meta_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    for model in MODELS:
        for horizon in HORIZONS:
            test_control_frame[f"pred_{model}_{horizon}"] = test_controls[model][horizon]
    test_control_frame.to_parquet(V18_DIR / "test_control_predictions.parquet", index=False)

    print("[6/8] 生成并校验四套提交文件...", flush=True)
    for model in MODELS:
        submission = fill_submission(data.meta_test, test_controls[model])
        submission.to_csv(
            V18_DIR / f"submission_{EXPERIMENT_VERSION}_{model}_{CV_VERSION}.csv",
            index=False, date_format="%m/%d/%Y %H:%M",
        )

    print("[7/8] 写入运行元数据和实验日志...", flush=True)
    input_paths = [
        CACHE_DIR / f"features_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"features_test_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"targets_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"meta_train_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"meta_test_{FEATURE_VERSION}.parquet",
        CACHE_DIR / f"feature_names_{FEATURE_VERSION}.json",
        CV_FILE, RAW_TRAIN_FILE, SUBMISSION_TEMPLATE,
    ]
    runtime = time.perf_counter() - started
    summary_records = tables["summary_metrics.csv"].to_dict(orient="records")
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "new_feature_count": 0,
        "future_weather_policy": "不增加 v1.5.7 F1 或其他新未来天气特征；冻结复用 v1.5.6 的既有列",
        "cv_version": CV_VERSION,
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "train_shape": list(data.X_train.shape),
        "test_shape": list(data.X_test.shape),
        "model_count": len(model_manifest),
        "cv_model_count": int((model_manifest["role"] == "cv").sum()),
        "final_model_count": int((model_manifest["role"] == "final").sum()),
        "lgbm_params": LGBM_PARAMS,
        "max_boost_rounds": args.max_rounds,
        "early_stopping_rounds": args.early_stopping_rounds,
        "final_round_rule": "floor(mean(five CV best iterations)), matching phase1",
        "component_formula": COMPONENT_WEIGHTS,
        "component_clip": [0.0, 1.0],
        "osi_post_process": {"clip": [0.0, 0.65], "zero_threshold": 0.001},
        "controls": list(MODELS),
        "summary_metrics": summary_records,
        "component_target_validation": target_validation,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "unit": "county_full_trajectory"},
        "runtime_seconds": runtime,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in input_paths
        },
    }
    (V18_DIR / "run_metadata.json").write_text(
        json.dumps(json_ready(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _append_log(metadata)

    print("[8/8] v1.8 训练完成。", flush=True)
    for row in summary_records:
        print(
            f"  {row['model']} {row['horizon']}: RMSE={row['rmse']:.6f}, "
            f"MAE={row['mae']:.6f}, vs C0={row['rmse_change_pct_vs_C0']:+.3f}%",
            flush=True,
        )
    print(f"  runtime={runtime:.1f}s, models={len(model_manifest)}", flush=True)


if __name__ == "__main__":
    main()
