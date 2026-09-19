from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .arithmetic import compose_component_osi, gated_prediction, positive_quantile, post_process, six_source_mean
from .config import (
    ATOL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPONENTS,
    FOLDS,
    HORIZONS,
    HORIZON_HOURS,
    LONG_HORIZONS,
    PROTOCOL,
    SOURCE_ORDER,
    component_target,
    resolved_config,
)
from .data_access import digest_object, load_data, sha256_file, write_json
from .models import fit_task, load_model, load_prediction, predict_model, task_paths, validate_saved_task
from .planning import ModelTask, all_new_tasks, dependency_plan, inner_tasks, outer_tasks
from .preflight import environment_report, replay_reference_lgb, run_preflight


def _code_hashes(experiment_root: Path) -> dict:
    files = [experiment_root / "run.py", experiment_root / "Protocol_Alignment.md"]
    files.extend(sorted((experiment_root / "src").glob("*.py")))
    return {path.relative_to(experiment_root).as_posix(): sha256_file(path) for path in files}


def _input_identity(data, project_root: Path, experiment_root: Path) -> dict:
    return {
        "protocol": PROTOCOL,
        "config": resolved_config(),
        "environment": environment_report(),
        "inputs": data.loaded_hashes,
        "input_paths": {name: str(Path(path).resolve().relative_to(project_root)) for name, path in data.input_paths.items()},
        "code": _code_hashes(experiment_root),
        "preflight_report_sha256": sha256_file(experiment_root / "preflight" / "preflight_report.json"),
        "source_verification_sha256": sha256_file(experiment_root / "preflight" / "source_verification.json"),
        "dependency_plan_sha256": sha256_file(experiment_root / "dependency_plan.json"),
    }


def prepare_run(project_root: Path, experiment_root: Path, run_id: str):
    preflight_path = experiment_root / "preflight" / "preflight_report.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("A passing preflight is required before training")
    data, baseline_config, protocol = load_data(project_root)
    identity = _input_identity(data, project_root, experiment_root)
    identity_hash = digest_object(identity)
    run_dir = experiment_root / "runs" / run_id
    manifest = {"run_id": run_id, "identity_hash": identity_hash, "identity": identity}
    if run_dir.exists():
        path = run_dir / "run_manifest.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing run identity differs; refusing to resume")
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "resolved_config.json", resolved_config())
        write_json(run_dir / "environment.json", identity["environment"])
        write_json(run_dir / "input_manifest.json", {
            key: {"path": identity["input_paths"][key], "sha256": value}
            for key, value in identity["inputs"].items()
        })
        data.assignment.to_csv(run_dir / "cv_assignments.csv", index=False, lineterminator="\n")
        write_json(run_dir / "dependency_plan.json", dependency_plan())
    return run_dir, identity_hash, data, protocol


def _progress(run_dir: Path, tasks: list[ModelTask], identity_hash: str, started: float) -> dict:
    complete = []
    for task in tasks:
        if validate_saved_task(run_dir, task, identity_hash) is not None:
            complete.append(task.task_id)
    value = {
        "status": "TRAINING" if len(complete) < len(tasks) else "TRAINING_COMPLETE",
        "completed_models": len(complete),
        "expected_models": len(tasks),
        "completed_task_ids": complete,
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / "training_progress.json", value)
    return value


def train(project_root: Path, experiment_root: Path, run_id: str) -> dict:
    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, run_id)
    tasks = all_new_tasks()
    started = time.time()
    progress = _progress(run_dir, tasks, identity_hash, started)
    for index, task in enumerate(tasks, start=1):
        existed = validate_saved_task(run_dir, task, identity_hash) is not None
        receipt = fit_task(run_dir, task, data, protocol, identity_hash)
        state = "resumed" if existed else "trained"
        print(
            f"[{index:03d}/{len(tasks):03d}] {state} {task.task_id} rounds={receipt['fit']['requested_rounds']}",
            flush=True,
        )
        progress = _progress(run_dir, tasks, identity_hash, started)
    receipts = [validate_saved_task(run_dir, task, identity_hash) for task in tasks]
    if any(receipt is None for receipt in receipts):
        raise RuntimeError("Training ended with incomplete model receipts")
    write_json(run_dir / "model_manifest.json", receipts)
    write_json(run_dir / "fit_records.json", [receipt["fit"] | {"task": receipt["task"]} for receipt in receipts])
    return {"status": "TRAINING_COMPLETE", "run_id": run_id, **progress}


def _empty(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=float)


def _new_outer_predictions(run_dir: Path, identity_hash: str, data) -> tuple[dict, dict]:
    n = len(data.X_train)
    predictions = {
        family: {target: _empty(n) for target in {task.target for task in outer_tasks() if task.family == family}}
        for family in ("xgboost", "catboost")
    }
    sources = {
        family: {target: np.full(n, "", dtype=object) for target in predictions[family]}
        for family in predictions
    }
    for task in outer_tasks():
        rows, values, receipt = load_prediction(run_dir, task, identity_hash)
        target_values = predictions[task.family][task.target]
        if np.isfinite(target_values[rows]).any():
            raise ValueError(f"Repeated outer prediction coverage: {task.task_id}")
        target_values[rows] = values
        sources[task.family][task.target][rows] = receipt["model_id"]
    return predictions, sources


def _reference_outer(project_root: Path, data) -> tuple[dict, dict, pd.DataFrame]:
    base = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "runs" / "v18_tree_nested_v2_split42_model42"
    frame = pd.read_parquet(base / "base_predictions_cv.parquet", engine="pyarrow")
    predictions = {target: frame[f"raw_{target}"].to_numpy(dtype=float) for target in [
        horizon for horizon in LONG_HORIZONS
    ] + [component_target(component, horizon) for horizon in LONG_HORIZONS for component in COMPONENTS]}
    sources = {target: frame[f"source_{target}"].to_numpy() for target in predictions}
    return predictions, sources, frame


def _inner_predictions(run_dir: Path, identity_hash: str, data) -> tuple[pd.DataFrame, dict]:
    records = []
    lookup = {}
    task_map = {(task.scope, task.target): task for task in inner_tasks()}
    for outer_q in FOLDS:
        for horizon in LONG_HORIZONS:
            for inner_r in FOLDS:
                if inner_r == outer_q:
                    continue
                scope = tuple(fold for fold in FOLDS if fold not in (outer_q, inner_r))
                raw = {}
                model_ids = {}
                expected_rows = None
                for component in COMPONENTS:
                    target = component_target(component, horizon)
                    task = task_map[(scope, target)]
                    all_rows, all_values, receipt = load_prediction(run_dir, task, identity_hash)
                    choose = data.row_folds[all_rows] == inner_r
                    rows = all_rows[choose]
                    values = all_values[choose]
                    if expected_rows is None:
                        expected_rows = rows
                    elif not np.array_equal(expected_rows, rows):
                        raise ValueError("Inner component row alignment differs")
                    raw[component] = values
                    model_ids[component] = receipt["model_id"]
                anchor = compose_component_osi(raw)
                key = (outer_q, horizon, inner_r)
                lookup[key] = (expected_rows, anchor, raw, model_ids, scope)
                meta = data.meta_train.iloc[expected_rows]
                frame = meta[["fipsCode", "timestamp_et", "hour_idx"]].copy()
                frame["outer_fold"] = outer_q
                frame["predicted_fold"] = inner_r
                frame["horizon"] = horizon
                frame["scope"] = ",".join(map(str, scope))
                for component in COMPONENTS:
                    frame[f"raw_{component}"] = raw[component]
                    frame[f"model_id_{component}"] = model_ids[component]
                frame["processed_anchor"] = anchor
                records.append(frame)
    result = pd.concat(records, ignore_index=True)
    if result.duplicated(["outer_fold", "horizon", "fipsCode", "timestamp_et"]).any():
        raise ValueError("Duplicate inner anchor row")
    return result, lookup


def _thresholds(data, lookup: dict) -> pd.DataFrame:
    rows = []
    for outer_q in FOLDS:
        for horizon in LONG_HORIZONS:
            anchors = []
            sources = []
            for inner_r in FOLDS:
                if inner_r == outer_q:
                    continue
                prediction_rows, anchor, raw, model_ids, scope = lookup[(outer_q, horizon, inner_r)]
                anchors.append(anchor)
                sources.append({
                    "predicted_fold": inner_r,
                    "scope": list(scope),
                    "rows": int(len(prediction_rows)),
                    "model_ids": model_ids,
                })
            combined = np.concatenate(anchors)
            theta, positive_count = positive_quantile(combined)
            rows.append({
                "outer_fold": outer_q,
                "horizon": horizon,
                "quantile_probability": 0.95,
                "method": "linear",
                "valid_count": int(len(combined)),
                "positive_count": positive_count,
                "theta": theta,
                "source_manifest": json.dumps(sources, sort_keys=True),
                "inner_prediction_hash": digest_object({"values": combined.tolist(), "sources": sources}),
            })
    return pd.DataFrame(rows)


def _processed_sources(horizon: str, lgb: dict, new: dict) -> tuple[dict, dict]:
    raw = {}
    processed = {}
    for family, values in (("lgbm", lgb), ("xgboost", new["xgboost"]), ("catboost", new["catboost"])):
        direct_name = f"{family}_direct"
        component_name = f"{family}_component"
        raw[direct_name] = {"direct": values[horizon]}
        raw[component_name] = {component: values[component_target(component, horizon)] for component in COMPONENTS}
        processed[direct_name] = post_process(values[horizon])
        processed[component_name] = compose_component_osi(raw[component_name])
    return raw, processed


def _candidate_frames(project_root: Path, run_dir: Path, identity_hash: str, data, protocol):
    n = len(data.X_train)
    new, new_sources = _new_outer_predictions(run_dir, identity_hash, data)
    lgb, lgb_sources, baseline_frame = _reference_outer(project_root, data)
    inner_frame, inner_lookup = _inner_predictions(run_dir, identity_hash, data)
    threshold_frame = _thresholds(data, inner_lookup)

    keys = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    keys["fold"] = data.row_folds
    keys["split_id"] = "seed42"
    candidate = keys.copy()
    outer_raw = keys.copy()
    processed_by_horizon = {}

    f2_path = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "p_information_increment" / "runs" / "v18_pinfo_v1_split42_model42" / "control_predictions.parquet"
    f2 = pd.read_parquet(f2_path, engine="pyarrow")

    for horizon in HORIZONS:
        mask = np.asarray(protocol.expected_mask(data.meta_train, horizon), dtype=bool)
        truth = data.y_train[horizon].to_numpy(dtype=float)
        candidate[f"scoreable_{horizon}"] = mask
        candidate[f"truth_{horizon}"] = truth
        if horizon not in LONG_HORIZONS:
            fixed = f2[f"pred_F2_v18_rule_{horizon}"].to_numpy(dtype=float)
            for level in ("L0", "L1", "L2"):
                candidate[f"pred_{level}_{horizon}"] = fixed
            candidate[f"theta_{horizon}"] = np.nan
            candidate[f"anchor_gt_theta_{horizon}"] = False
            continue

        raw, processed = _processed_sources(horizon, lgb, new)
        processed_by_horizon[horizon] = processed
        for source in SOURCE_ORDER:
            values = processed[source].copy()
            values[~mask] = np.nan
            candidate[f"processed_{source}_{horizon}"] = values
            if "direct" in source:
                outer_raw[f"raw_{source}_{horizon}"] = raw[source]["direct"]
            else:
                for component in COMPONENTS:
                    outer_raw[f"raw_{source}_{component}_{horizon}"] = raw[source][component]

        l0 = processed["lgbm_component"].copy()
        l1 = six_source_mean([processed[source] for source in SOURCE_ORDER])
        l2 = _empty(n)
        theta_per_row = _empty(n)
        gate = np.zeros(n, dtype=bool)
        for outer_q in FOLDS:
            qmask = mask & (data.row_folds == outer_q)
            theta = float(threshold_frame.query("outer_fold == @outer_q and horizon == @horizon").theta.iloc[0])
            theta_per_row[qmask] = theta
            l2[qmask] = gated_prediction(l0[qmask], l1[qmask], theta)
            gate[qmask] = l0[qmask] > theta
        for values in (l0, l1, l2):
            values[~mask] = np.nan
        candidate[f"pred_L0_{horizon}"] = l0
        candidate[f"pred_L1_{horizon}"] = l1
        candidate[f"pred_L2_{horizon}"] = l2
        candidate[f"theta_{horizon}"] = theta_per_row
        candidate[f"anchor_gt_theta_{horizon}"] = gate
    return {
        "candidate_predictions.parquet": candidate,
        "outer_raw_predictions.parquet": outer_raw,
        "inner_anchor_predictions.parquet": inner_frame,
        "thresholds.parquet": threshold_frame,
    }


def _metrics(data, frames: dict) -> dict[str, pd.DataFrame]:
    candidate = frames["candidate_predictions.parquet"]
    summary = []
    fold_rows = []
    county_rows = []
    day_rows = []
    gate_rows = []
    correlation_rows = []
    for horizon in HORIZONS:
        truth = candidate[f"truth_{horizon}"].to_numpy(dtype=float)
        valid = candidate[f"scoreable_{horizon}"].to_numpy(dtype=bool)
        for level in ("L0", "L1", "L2"):
            pred = candidate[f"pred_{level}_{horizon}"].to_numpy(dtype=float)
            if not np.isfinite(pred[valid]).all() or not np.isnan(pred[~valid]).all():
                raise ValueError(f"Candidate coverage invalid: {level}, {horizon}")
            error = pred[valid] - truth[valid]
            summary.append({
                "candidate": level,
                "horizon": horizon,
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(np.abs(error))),
                "n": int(valid.sum()),
                "prediction_mean": float(np.mean(pred[valid])),
                "prediction_max": float(np.max(pred[valid])),
                "zero_pct": float(100 * np.mean(pred[valid] == 0)),
            })
            for fold in FOLDS:
                hit = valid & (data.row_folds == fold)
                delta = pred[hit] - truth[hit]
                fold_rows.append({"candidate": level, "horizon": horizon, "fold": fold,
                                  "rmse": float(np.sqrt(np.mean(delta ** 2))), "mae": float(np.mean(np.abs(delta))), "n": int(hit.sum())})
            for fips in data.meta_train.fipsCode.drop_duplicates():
                hit = valid & (data.meta_train.fipsCode.to_numpy() == fips)
                delta = pred[hit] - truth[hit]
                county_rows.append({"candidate": level, "horizon": horizon, "fipsCode": fips,
                                    "rmse": float(np.sqrt(np.mean(delta ** 2))), "mae": float(np.mean(np.abs(delta))), "n": int(hit.sum())})
            target_dates = (pd.to_datetime(data.meta_train.timestamp_et) + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h")).dt.strftime("%Y-%m-%d")
            for day in sorted(target_dates[valid].unique()):
                hit = valid & (target_dates.to_numpy() == day)
                delta = pred[hit] - truth[hit]
                day_rows.append({"candidate": level, "horizon": horizon, "target_day": day,
                                 "rmse": float(np.sqrt(np.mean(delta ** 2))), "mae": float(np.mean(np.abs(delta))), "n": int(hit.sum())})
        if horizon in LONG_HORIZONS:
            gate = candidate[f"anchor_gt_theta_{horizon}"].to_numpy(dtype=bool) & valid
            top_cut = float(np.quantile(truth[valid], 0.95, method="linear"))
            actual_top = valid & (truth > top_cut)
            gate_rows.append({
                "horizon": horizon,
                "gate_rows": int(gate.sum()),
                "gate_pct": float(100 * gate.sum() / valid.sum()),
                "actual_top5_threshold": top_cut,
                "actual_top5_rows": int(actual_top.sum()),
                "gate_and_actual_top5": int((gate & actual_top).sum()),
            })
            errors = {}
            for source in SOURCE_ORDER:
                pred = candidate[f"processed_{source}_{horizon}"].to_numpy(dtype=float)
                errors[source] = pred[valid] - truth[valid]
            corr = np.corrcoef(np.stack([errors[source] for source in SOURCE_ORDER]))
            for i, left in enumerate(SOURCE_ORDER):
                for j, right in enumerate(SOURCE_ORDER):
                    if j <= i:
                        continue
                    correlation_rows.append({"horizon": horizon, "left": left, "right": right, "pearson_error_correlation": float(corr[i, j])})
    summary_frame = pd.DataFrame(summary)
    deltas = []
    for horizon in LONG_HORIZONS:
        scores = summary_frame.query("horizon == @horizon").set_index("candidate")
        for candidate_name, reference in (("L1", "L0"), ("L2", "L0"), ("L2", "L1")):
            deltas.append({
                "horizon": horizon,
                "candidate": candidate_name,
                "reference": reference,
                "delta_rmse": float(scores.loc[candidate_name, "rmse"] - scores.loc[reference, "rmse"]),
                "delta_mae": float(scores.loc[candidate_name, "mae"] - scores.loc[reference, "mae"]),
                "rmse_change_pct": float(100 * (scores.loc[candidate_name, "rmse"] / scores.loc[reference, "rmse"] - 1)),
            })
    return {
        "summary_metrics.csv": summary_frame,
        "candidate_deltas.csv": pd.DataFrame(deltas),
        "fold_metrics.csv": pd.DataFrame(fold_rows),
        "county_metrics.csv": pd.DataFrame(county_rows),
        "target_day_metrics.csv": pd.DataFrame(day_rows),
        "gate_diagnostics.csv": pd.DataFrame(gate_rows),
        "source_error_correlations.csv": pd.DataFrame(correlation_rows),
    }


def _paired_bootstrap(data, candidate: pd.DataFrame) -> pd.DataFrame:
    comparisons = (("L1", "L0"), ("L2", "L0"), ("L2", "L1"))
    counties = data.meta_train.fipsCode.drop_duplicates().to_numpy()
    county_map = {county: index for index, county in enumerate(counties)}
    row_county = data.meta_train.fipsCode.map(county_map).to_numpy(dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(counties), size=(BOOTSTRAP_REPLICATES, len(counties)))
    rows = []
    for horizon in LONG_HORIZONS:
        truth = candidate[f"truth_{horizon}"].to_numpy(dtype=float)
        valid = candidate[f"scoreable_{horizon}"].to_numpy(dtype=bool)
        stats = {}
        for level in ("L0", "L1", "L2"):
            error = candidate[f"pred_{level}_{horizon}"].to_numpy(dtype=float) - truth
            stats[level] = {
                "sse": np.bincount(row_county[valid], weights=error[valid] ** 2, minlength=len(counties)),
                "n": np.bincount(row_county[valid], minlength=len(counties)),
            }
        for level, reference in comparisons:
            left, right = stats[level], stats[reference]
            left_rep = np.sqrt(left["sse"][sampled].sum(axis=1) / left["n"][sampled].sum(axis=1))
            right_rep = np.sqrt(right["sse"][sampled].sum(axis=1) / right["n"][sampled].sum(axis=1))
            difference = left_rep - right_rep
            point_left = np.sqrt(left["sse"].sum() / left["n"].sum())
            point_right = np.sqrt(right["sse"].sum() / right["n"].sum())
            low, high = np.quantile(difference, [0.025, 0.975])
            rows.append({
                "horizon": horizon,
                "candidate": level,
                "reference": reference,
                "delta_rmse": float(point_left - point_right),
                "ci_2_5": float(low),
                "ci_97_5": float(high),
                "improvement_probability": float(np.mean(difference < 0)),
                "replicates": BOOTSTRAP_REPLICATES,
                "unit": "county_full_trajectory",
            })
    return pd.DataFrame(rows)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, engine="pyarrow")
    else:
        frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _results_markdown(run_id: str, metrics: dict[str, pd.DataFrame], bootstrap: pd.DataFrame) -> str:
    summary = metrics["summary_metrics.csv"]
    deltas = metrics["candidate_deltas.csv"]
    lines = [f"# Stella v1.12 strict seed42 results", "", f"Run: `{run_id}`", "", "## Pooled metrics", "", summary.to_markdown(index=False), "", "## Long-horizon differences", "", deltas.to_markdown(index=False), "", "## County bootstrap", "", bootstrap.to_markdown(index=False), ""]
    return "\n".join(lines)


def assemble(project_root: Path, experiment_root: Path, run_id: str, *, write: bool = True):
    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, run_id)
    tasks = all_new_tasks()
    if any(validate_saved_task(run_dir, task, identity_hash) is None for task in tasks):
        raise RuntimeError("All 180 model tasks must complete before assembly")
    frames = _candidate_frames(project_root, run_dir, identity_hash, data, protocol)
    metrics = _metrics(data, frames)
    bootstrap = _paired_bootstrap(data, frames["candidate_predictions.parquet"])
    metrics["paired_county_bootstrap.csv"] = bootstrap
    if write:
        for name, frame in {**frames, **metrics}.items():
            _write_frame(run_dir / name, frame)
        results = _results_markdown(run_id, metrics, bootstrap)
        (run_dir / "Results.md").write_text(results, encoding="utf-8", newline="\n")
        artifacts = {name: sha256_file(run_dir / name) for name in [*frames, *metrics, "Results.md"]}
        write_json(run_dir / "assembly_manifest.json", {"identity_hash": identity_hash, "artifacts": artifacts})
    return run_dir, identity_hash, data, protocol, frames, metrics


def _compare_frame(saved_path: Path, expected: pd.DataFrame) -> None:
    if saved_path.suffix == ".parquet":
        actual = pd.read_parquet(saved_path, engine="pyarrow")
    else:
        actual = pd.read_csv(saved_path, dtype={"fipsCode": str}, float_precision="round_trip")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False, check_exact=False, atol=ATOL, rtol=0)


def verify(project_root: Path, experiment_root: Path, run_id: str) -> dict:
    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, run_id)
    reloaded = 0
    max_difference = 0.0
    for task in all_new_tasks():
        receipt = validate_saved_task(run_dir, task, identity_hash)
        if receipt is None:
            raise RuntimeError(f"Missing task during verification: {task.task_id}")
        paths = task_paths(run_dir, task)
        saved = np.load(paths["prediction"])
        rows = saved["rows"].astype(int)
        expected = saved["predictions"].astype(float)
        model = load_model(task.family, paths["model"])
        actual = predict_model(model, task.family, data, rows, int(receipt["fit"]["requested_rounds"]))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
        max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
        reloaded += 1

    replay = replay_reference_lgb(project_root, experiment_root / "preflight", data, protocol)
    run_dir2, identity_hash2, data2, protocol2, frames, metrics = assemble(project_root, experiment_root, run_id, write=False)
    for name, frame in {**frames, **metrics}.items():
        _compare_frame(run_dir / name, frame)

    candidate = frames["candidate_predictions.parquet"]
    short_exact = {}
    for horizon in ("osi_target_t01h", "osi_target_t06h"):
        l0 = candidate[f"pred_L0_{horizon}"].to_numpy(dtype=float)
        for level in ("L1", "L2"):
            values = candidate[f"pred_{level}_{horizon}"].to_numpy(dtype=float)
            if not np.array_equal(l0, values, equal_nan=True):
                raise AssertionError(f"Short prediction changed for {level}, {horizon}")
        short_exact[horizon] = True

    report = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "run_id": run_id,
        "identity_hash": identity_hash,
        "new_models_reloaded": reloaded,
        "referenced_models_reloaded": replay["models_reloaded"],
        "new_prediction_max_abs_difference": max_difference,
        "reference_prediction_max_abs_difference": max(record["max_abs_difference"] for record in replay["records"]),
        "comparison": {"atol": ATOL, "rtol": 0},
        "short_predictions_exactly_unchanged": short_exact,
        "artifacts_recomputed_from_models": sorted([*frames, *metrics]),
        "training_performed_by_verifier": False,
        "test_inference_performed": False,
    }
    write_json(run_dir / "verification.json", report)
    included = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "CV_COMPLETE":
            included.append(path)
    marker = {
        "status": "CV_COMPLETE",
        "identity_hash": identity_hash,
        "verification_sha256": sha256_file(run_dir / "verification.json"),
        "files": {path.relative_to(run_dir).as_posix(): sha256_file(path) for path in included},
    }
    write_json(run_dir / "CV_COMPLETE", marker)
    return report


def run_command(command: str, project_root: Path, experiment_root: Path, run_id: str) -> dict:
    if command == "train":
        return train(project_root, experiment_root, run_id)
    if command == "assemble":
        run_dir, identity_hash, data, protocol, frames, metrics = assemble(project_root, experiment_root, run_id, write=True)
        return {"status": "ASSEMBLY_COMPLETE", "run_id": run_id, "run_dir": str(run_dir), "artifacts": sorted([*frames, *metrics])}
    if command == "verify":
        return verify(project_root, experiment_root, run_id)
    if command == "all":
        run_preflight(project_root, experiment_root)
        training = train(project_root, experiment_root, run_id)
        assemble(project_root, experiment_root, run_id, write=True)
        return {"status": "READY_FOR_INDEPENDENT_VERIFY", "training": training}
    raise ValueError(f"Unknown command: {command}")
