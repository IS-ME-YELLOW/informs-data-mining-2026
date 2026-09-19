from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .arithmetic import compose_component_osi, gated_prediction, positive_quantile, post_process, six_source_mean
from .config import (
    ATOL,
    COMPONENTS,
    EXPECTED_SCOREABLE_ROWS,
    FOLDS,
    HORIZONS,
    LONG_HORIZONS,
    PROTOCOL,
    component_target,
    resolved_config,
)
from .data_access import (
    canonical_lf_bytes,
    load_data,
    sha256_bytes,
    sha256_file,
    verify_handoff_sources,
    write_json,
)
from .models import selected_rows
from .planning import all_new_tasks, dependency_plan


def environment_report() -> dict:
    import catboost
    import lightgbm
    import numpy
    import pandas
    import pyarrow
    import scipy
    import sklearn
    import xgboost

    modules = [numpy, pandas, pyarrow, scipy, sklearn, lightgbm, xgboost, catboost]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": {
            module.__name__: {
                "version": module.__version__,
                "path": str(Path(module.__file__).resolve()),
            }
            for module in modules
        },
    }


def validate_protocol_examples(project_root: Path) -> dict:
    examples = json.loads((project_root / "versions" / "stella_v112_handoff" / "protocol_examples.json").read_text(encoding="utf-8"))
    tolerance = examples["numeric_comparison"]
    atol = float(tolerance["atol"])

    np.testing.assert_allclose(
        post_process(examples["post_process"]["input"]),
        examples["post_process"]["expected"], atol=atol, rtol=0,
    )
    component = examples["component_then_post"]
    parts = dict(zip(COMPONENTS, (np.asarray([value]) for value in component["raw_PNDR"])))
    np.testing.assert_allclose(compose_component_osi(parts), [component["expected_osi"]], atol=atol, rtol=0)
    quantile = examples["positive_quantile"]
    theta, count = positive_quantile(quantile["inner_anchor"])
    np.testing.assert_allclose(theta, quantile["expected_theta"], atol=atol, rtol=0)
    if count != quantile["positive_count"]:
        raise AssertionError("Positive count differs from protocol example")
    gate = examples["gate_equality"]
    gated = gated_prediction(gate["outer_anchor"], gate["simple_processed"], gate["theta"])
    np.testing.assert_allclose(gated, gate["expected"], atol=atol, rtol=0)
    actual_flags = (np.asarray(gate["outer_anchor"]) > gate["theta"]).tolist()
    if actual_flags != gate["expected_anchor_used"]:
        raise AssertionError("Gate equality behavior differs")
    average = examples["post_after_average"]
    mean = six_source_mean([np.asarray([value]) for value in average["processed_sources"]])
    np.testing.assert_allclose(mean, [average["expected"]], atol=atol, rtol=0)
    for key in ("no_positive_inner", "invalid_inner"):
        values = examples[key].get("inner_anchor", examples[key].get("inner_anchor_on_scoreable_rows"))
        try:
            positive_quantile(values)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"Protocol example {key} should fail")
    rounds = examples["floor_rounds"]
    if max(1, int(np.floor(np.mean(rounds["best_round_counts"])))) != rounds["expected_refit_rounds"]:
        raise AssertionError("Floor round example differs")
    return {"status": "PASS", "examples_checked": 8, "atol": atol, "rtol": 0}


def validate_data(data, protocol) -> dict:
    counts = {horizon: int(protocol.expected_mask(data.meta_train, horizon).sum()) for horizon in HORIZONS}
    if counts != EXPECTED_SCOREABLE_ROWS:
        raise AssertionError(f"Scoreable counts differ: {counts}")
    return {
        "status": "PASS",
        "train_rows": len(data.X_train),
        "test_rows": len(data.X_test),
        "train_counties": int(data.meta_train.fipsCode.nunique()),
        "test_counties": int(data.meta_test.fipsCode.nunique()),
        "feature_count": data.X_train.shape[1],
        "feature_dtypes": {name: str(dtype) for name, dtype in data.X_train.dtypes.items()},
        "scoreable_rows": counts,
        "input_paths": {name: str(path) for name, path in data.input_paths.items()},
        "raw_input_hashes": data.loaded_hashes,
    }


def validate_scope_and_sentinels(data, protocol) -> dict:
    tasks = all_new_tasks()
    checks = 0
    poison_checks = 0
    for task in tasks:
        original = (
            data.y_train[task.target].to_numpy(dtype=float)
            if task.target in data.y_train
            else data.component_targets[task.target].to_numpy(dtype=float)
        )
        for validation_fold in task.scope:
            train_folds = tuple(fold for fold in task.scope if fold != validation_fold)
            train_rows = selected_rows(data, protocol, task.target, train_folds)
            valid_rows = selected_rows(data, protocol, task.target, (validation_fold,))
            forbidden = np.isin(data.row_folds, [fold for fold in FOLDS if fold not in task.scope])
            if forbidden[train_rows].any() or forbidden[valid_rows].any():
                raise AssertionError(f"Forbidden fold entered probe scope: {task.task_id}")
            if set(train_rows) & set(valid_rows):
                raise AssertionError(f"Probe scopes overlap: {task.task_id}")
            poisoned = original.copy()
            poisoned[forbidden] = 1e9
            np.testing.assert_array_equal(poisoned[train_rows], original[train_rows])
            np.testing.assert_array_equal(poisoned[valid_rows], original[valid_rows])
            poison_checks += 1
            checks += 1
        refit_rows = selected_rows(data, protocol, task.target, task.scope)
        if np.isin(data.row_folds[refit_rows], [fold for fold in FOLDS if fold not in task.scope]).any():
            raise AssertionError(f"Forbidden fold entered refit scope: {task.task_id}")
        checks += 1
    return {
        "status": "PASS",
        "production_tasks_checked": len(tasks),
        "scope_edges_checked": checks,
        "poisoned_label_checks": poison_checks,
        "real_fit_called": False,
    }


def _assert_keys_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["fipsCode", "timestamp_et", "hour_idx"]
    left_keys = left[columns].copy()
    right_keys = right[columns].copy()
    for frame in (left_keys, right_keys):
        frame["fipsCode"] = frame.fipsCode.astype(str).str.zfill(5)
        frame["timestamp_et"] = pd.to_datetime(frame.timestamp_et)
    pd.testing.assert_frame_equal(left_keys.reset_index(drop=True), right_keys.reset_index(drop=True), check_dtype=False)


def replay_reference_lgb(project_root: Path, preflight_dir: Path, data, protocol) -> dict:
    import lightgbm as lgb

    baseline_run = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "runs" / "v18_tree_nested_v2_split42_model42"
    manifest = json.loads((baseline_run / "model_manifest_cv.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(baseline_run / "base_predictions_cv.parquet", engine="pyarrow")
    _assert_keys_equal(predictions, data.meta_train)
    if not np.array_equal(predictions.fold.to_numpy(dtype=int), data.row_folds):
        raise AssertionError("Saved baseline fold column differs from current seed42 folds")

    wanted = []
    for receipt in manifest:
        target = receipt["spec"]["target"]
        if target.rsplit("_", 1)[-1] not in {"t24h", "t48h"}:
            continue
        wanted.append(receipt)
    if len(wanted) != 50:
        raise AssertionError(f"Expected 50 referenced LGB models, found {len(wanted)}")

    rows_report = []
    for receipt in wanted:
        target = receipt["spec"]["target"]
        scope = tuple(receipt["spec"]["scope"])
        excluded = tuple(fold for fold in FOLDS if fold not in scope)
        if len(excluded) != 1:
            raise AssertionError("Referenced model is not a four-fold outer model")
        outer_fold = excluded[0]
        source_path = baseline_run / receipt["model_path"]
        canonical_bytes = canonical_lf_bytes(source_path.read_bytes())
        canonical_hash = sha256_bytes(canonical_bytes)
        if canonical_hash != receipt["model_sha256"]:
            raise AssertionError(f"Canonical source model hash differs: {receipt['model_path']}")
        canonical_path = preflight_dir / "reference_models" / receipt["model_path"]
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        if not canonical_path.exists() or sha256_file(canonical_path) != canonical_hash:
            temporary = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
            temporary.write_bytes(canonical_bytes)
            temporary.replace(canonical_path)

        model = lgb.Booster(model_file=str(canonical_path))
        valid = protocol.expected_mask(data.meta_train, f"osi_target_{target.rsplit('_', 1)[-1]}")
        predict_rows = np.flatnonzero(valid & (data.row_folds == outer_fold))
        actual = model.predict(
            data.X_train.iloc[predict_rows],
            num_iteration=int(receipt["fit"]["requested_rounds"]),
            num_threads=1,
        )
        expected = predictions[f"raw_{target}"].to_numpy(dtype=float)[predict_rows]
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0, equal_nan=True)
        source_ids = predictions[f"source_{target}"].to_numpy()[predict_rows]
        if set(source_ids) != {receipt["model_id"]}:
            raise AssertionError(f"Prediction lineage differs for {target}, outer{outer_fold}")
        rows_report.append({
            "target": target,
            "outer_fold": outer_fold,
            "model_id": receipt["model_id"],
            "source_path": str(source_path.relative_to(project_root)),
            "canonical_copy": str(canonical_path.relative_to(project_root)),
            "canonical_sha256": canonical_hash,
            "rows": int(len(predict_rows)),
            "requested_rounds": int(receipt["fit"]["requested_rounds"]),
            "trees_loaded": int(model.num_trees()),
            "max_abs_difference": float(np.max(np.abs(actual - expected))),
        })
    return {
        "status": "PASS",
        "models_reloaded": len(rows_report),
        "comparison": {"atol": ATOL, "rtol": 0},
        "records": rows_report,
    }


def validate_f2_control(project_root: Path, data, protocol) -> dict:
    path = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "p_information_increment" / "runs" / "v18_pinfo_v1_split42_model42" / "control_predictions.parquet"
    frame = pd.read_parquet(path, engine="pyarrow")
    _assert_keys_equal(frame, data.meta_train)
    if not np.array_equal(frame.fold.to_numpy(dtype=int), data.row_folds):
        raise AssertionError("F2 fold column differs")
    result = {"status": "PASS", "path": str(path.relative_to(project_root)), "sha256": sha256_file(path), "columns": {}}
    for horizon in ("osi_target_t01h", "osi_target_t06h"):
        column = f"pred_F2_v18_rule_{horizon}"
        values = frame[column].to_numpy(dtype=float)
        mask = protocol.expected_mask(data.meta_train, horizon)
        if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
            raise AssertionError(f"Invalid F2 control coverage: {horizon}")
        result["columns"][horizon] = {"column": column, "scoreable_rows": int(mask.sum())}
    return result


def run_preflight(project_root: Path, experiment_root: Path) -> dict:
    preflight_dir = experiment_root / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    source_report = verify_handoff_sources(project_root)
    write_json(preflight_dir / "source_verification.json", source_report)
    if source_report["status"] != "PASS":
        raise RuntimeError("Handoff source verification failed; see preflight/source_verification.json")

    environment = environment_report()
    write_json(preflight_dir / "environment.json", environment)
    config = resolved_config()
    write_json(experiment_root / "resolved_config.json", config)
    plan = dependency_plan()
    if plan["budget"] != {
        "new_fit_calls": 820,
        "new_saved_models": 180,
        "referenced_models": 50,
        "outer_new_fit_calls": 500,
        "inner_new_fit_calls": 320,
    }:
        raise AssertionError(f"Dependency budget differs: {plan['budget']}")
    write_json(experiment_root / "dependency_plan.json", plan)

    data, baseline_config, protocol = load_data(project_root)
    data_report = validate_data(data, protocol)
    write_json(preflight_dir / "data_validation.json", data_report)
    example_report = validate_protocol_examples(project_root)
    write_json(preflight_dir / "protocol_examples.json", example_report)
    scope_report = validate_scope_and_sentinels(data, protocol)
    write_json(preflight_dir / "scope_sentinel_tests.json", scope_report)
    f2_report = validate_f2_control(project_root, data, protocol)
    write_json(preflight_dir / "f2_control_validation.json", f2_report)
    lgb_report = replay_reference_lgb(project_root, preflight_dir, data, protocol)
    write_json(preflight_dir / "reference_lgb_replay.json", lgb_report)

    result = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "source_verification": source_report["counts"],
        "data_validation": data_report,
        "protocol_examples": example_report,
        "scope_sentinel_tests": scope_report,
        "f2_control_validation": f2_report,
        "reference_lgb_replay": {
            "status": lgb_report["status"],
            "models_reloaded": lgb_report["models_reloaded"],
            "max_abs_difference": max(record["max_abs_difference"] for record in lgb_report["records"]),
        },
        "dependency_budget": plan["budget"],
        "training_performed": False,
        "test_inference_performed": False,
    }
    write_json(preflight_dir / "preflight_report.json", result)
    return result
