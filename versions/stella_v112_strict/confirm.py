from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from src.bootstrap import activate_local_packages


def main() -> None:
    parser = argparse.ArgumentParser(description="Stella v1.12 strict confirmation splits")
    parser.add_argument("command", choices=("preflight", "train", "train-shard", "assemble", "verify"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, choices=(20260917, 20260918), required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    project_root = activate_local_packages(args.project_root)
    experiment_root = Path(__file__).resolve().parent
    if args.command == "train-shard":
        if args.shard_count is None or args.shard_count < 1:
            parser.error("train-shard 需要 --shard-count >= 1")
        if args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
            parser.error("train-shard 需要 0 <= --shard-index < --shard-count")
        result = train_shard(project_root, experiment_root, args.split_seed, args.shard_index, args.shard_count)
    else:
        if args.shard_index is not None or args.shard_count is not None:
            parser.error("--shard-index/--shard-count 仅用于 train-shard")
        result = run_command(args.command, project_root, experiment_root, args.split_seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def split_run_id(seed: int) -> str:
    return f"stella_v112_nested_v1_split{seed}_model42"


def baseline_run_id(seed: int) -> str:
    return f"v18_tree_nested_v2_split{seed}_model42"


def f2_run_id(seed: int) -> str:
    return f"v18_pinfo_v1_split{seed}_model42"


def load_split_data(project_root: Path, seed: int):
    from src.data_access import load_baseline_protocol

    baseline_config, protocol = load_baseline_protocol(project_root)
    data = protocol.load_data(
        project_root / "versions" / "xyy" / "v1.5.6",
        project_root / "cv" / f"cv_assignments_balanced_v1_seed{seed}.csv",
    )
    return data, baseline_config, protocol


def verify_confirmation_inventory(project_root: Path) -> dict:
    from src.data_access import canonical_lf_bytes, sha256_bytes

    inventory = json.loads((project_root / "versions" / "stella_v112_handoff" / "source_inventory.json").read_text(encoding="utf-8"))
    records = []
    counts: dict[str, int] = {}
    for entry in inventory["files"]:
        if entry["group"] != "confirmation":
            continue
        path = project_root / entry["path"]
        if not path.is_file():
            status, raw_hash, canonical_hash = "missing", None, None
        else:
            raw = path.read_bytes()
            raw_hash = sha256_bytes(raw)
            canonical_hash = sha256_bytes(canonical_lf_bytes(raw))
            status = "exact" if raw_hash == entry["sha256"] else "crlf_checkout_canonical_match" if canonical_hash == entry["sha256"] else "mismatch"
        counts[status] = counts.get(status, 0) + 1
        records.append({
            "path": entry["path"],
            "expected_sha256": entry["sha256"],
            "raw_sha256": raw_hash,
            "canonical_lf_sha256": canonical_hash,
            "status": status,
        })
    failures = [record for record in records if record["status"] in {"missing", "mismatch"}]
    return {"status": "PASS" if not failures else "FAIL", "counts": counts, "failures": failures, "records": records}


def replay_reference_lgb(project_root: Path, output_dir: Path, data, protocol, seed: int) -> dict:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from src.config import ATOL, FOLDS
    from src.data_access import canonical_lf_bytes, sha256_bytes, sha256_file

    run = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "runs" / baseline_run_id(seed)
    manifest = json.loads((run / "model_manifest_cv.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(run / "base_predictions_cv.parquet", engine="pyarrow")
    if not np.array_equal(predictions.fold.to_numpy(dtype=int), data.row_folds):
        raise AssertionError("确认分折的保存预测与当前 CV 不一致")
    wanted = [receipt for receipt in manifest if receipt["spec"]["target"].rsplit("_", 1)[-1] in {"t24h", "t48h"}]
    if len(wanted) != 50:
        raise AssertionError(f"应引用 50 个 LGB 模型，实际 {len(wanted)}")
    records = []
    for receipt in wanted:
        target = receipt["spec"]["target"]
        scope = tuple(receipt["spec"]["scope"])
        outer_fold = next(iter(set(FOLDS) - set(scope)))
        source_path = run / receipt["model_path"]
        canonical = canonical_lf_bytes(source_path.read_bytes())
        canonical_hash = sha256_bytes(canonical)
        if canonical_hash != receipt["model_sha256"]:
            raise AssertionError(f"引用模型哈希不一致：{receipt['model_path']}")
        copy_path = output_dir / "reference_models" / receipt["model_path"]
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        if not copy_path.exists() or sha256_file(copy_path) != canonical_hash:
            temporary = copy_path.with_suffix(copy_path.suffix + ".tmp")
            temporary.write_bytes(canonical)
            temporary.replace(copy_path)
        model = lgb.Booster(model_file=str(copy_path))
        horizon = f"osi_target_{target.rsplit('_', 1)[-1]}"
        rows = np.flatnonzero(protocol.expected_mask(data.meta_train, horizon) & (data.row_folds == outer_fold))
        actual = model.predict(data.X_train.iloc[rows], num_iteration=int(receipt["fit"]["requested_rounds"]), num_threads=1)
        expected = predictions[f"raw_{target}"].to_numpy(dtype=float)[rows]
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
        if set(predictions[f"source_{target}"].to_numpy()[rows]) != {receipt["model_id"]}:
            raise AssertionError("引用模型 lineage 不一致")
        records.append({"target": target, "outer_fold": outer_fold, "model_id": receipt["model_id"], "max_abs_difference": float(np.max(np.abs(actual - expected)))})
    return {"status": "PASS", "models_reloaded": len(records), "records": records}


def validate_f2(project_root: Path, data, protocol, seed: int) -> dict:
    import numpy as np
    import pandas as pd

    from src.data_access import sha256_file

    path = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "p_information_increment" / "runs" / f2_run_id(seed) / "control_predictions.parquet"
    frame = pd.read_parquet(path, engine="pyarrow")
    if not np.array_equal(frame.fold.to_numpy(dtype=int), data.row_folds):
        raise AssertionError("F2 分折与确认分折不一致")
    columns = {}
    for horizon in ("osi_target_t01h", "osi_target_t06h"):
        column = f"pred_F2_v18_rule_{horizon}"
        values = frame[column].to_numpy(dtype=float)
        mask = protocol.expected_mask(data.meta_train, horizon)
        if not np.isfinite(values[mask]).all() or not np.isnan(values[~mask]).all():
            raise AssertionError(f"F2 覆盖错误：{horizon}")
        columns[horizon] = {"column": column, "scoreable_rows": int(mask.sum())}
    return {"status": "PASS", "path": str(path.relative_to(project_root)), "sha256": sha256_file(path), "columns": columns}


def confirmation_config(seed: int) -> dict:
    from src.config import resolved_config

    value = resolved_config()
    value["split_seed"] = seed
    value["stage"] = "confirmation"
    return value


def run_preflight(project_root: Path, experiment_root: Path, seed: int) -> dict:
    from src.data_access import write_json
    from src.planning import dependency_plan
    from src.preflight import environment_report, validate_data, validate_protocol_examples, validate_scope_and_sentinels

    output = experiment_root / "confirmation" / f"preflight_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    inventory = verify_confirmation_inventory(project_root)
    write_json(output / "source_verification.json", inventory)
    if inventory["status"] != "PASS":
        raise RuntimeError("确认材料核验失败")
    data, baseline_config, protocol = load_split_data(project_root, seed)
    data_report = validate_data(data, protocol)
    example_report = validate_protocol_examples(project_root)
    scope_report = validate_scope_and_sentinels(data, protocol)
    f2_report = validate_f2(project_root, data, protocol, seed)
    lgb_report = replay_reference_lgb(project_root, output, data, protocol, seed)
    write_json(output / "data_validation.json", data_report)
    write_json(output / "protocol_examples.json", example_report)
    write_json(output / "scope_sentinel_tests.json", scope_report)
    write_json(output / "f2_control_validation.json", f2_report)
    write_json(output / "reference_lgb_replay.json", lgb_report)
    write_json(output / "environment.json", environment_report())
    write_json(output / "resolved_config.json", confirmation_config(seed))
    write_json(output / "dependency_plan.json", dependency_plan())
    result = {
        "status": "PASS",
        "split_seed": seed,
        "source_verification": inventory["counts"],
        "data_validation": data_report,
        "scope_sentinel_tests": scope_report,
        "f2_control_validation": f2_report,
        "reference_lgb_replay": {"models_reloaded": 50, "max_abs_difference": 0.0},
        "training_performed": False,
    }
    write_json(output / "preflight_report.json", result)
    return result


def code_hashes(experiment_root: Path) -> dict:
    from src.data_access import sha256_file

    paths = [experiment_root / "confirm.py", experiment_root / "run.py", experiment_root / "Protocol_Alignment.md"]
    paths.extend(sorted((experiment_root / "src").glob("*.py")))
    return {path.relative_to(experiment_root).as_posix(): sha256_file(path) for path in paths}


def prepare_run(project_root: Path, experiment_root: Path, seed: int):
    from src.data_access import digest_object, sha256_file, write_json
    from src.planning import dependency_plan
    from src.preflight import environment_report

    preflight_dir = experiment_root / "confirmation" / f"preflight_seed{seed}"
    preflight = preflight_dir / "preflight_report.json"
    if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("必须先完成确认分折预检")
    data, baseline_config, protocol = load_split_data(project_root, seed)
    identity = {
        "protocol": "stella_v112_nested_v1",
        "stage": "confirmation",
        "split_seed": seed,
        "model_seed": 42,
        "config": confirmation_config(seed),
        "environment": environment_report(),
        "inputs": data.loaded_hashes,
        "input_paths": {name: str(Path(path).resolve().relative_to(project_root)) for name, path in data.input_paths.items()},
        "code": code_hashes(experiment_root),
        "preflight_report_sha256": sha256_file(preflight),
        "source_verification_sha256": sha256_file(preflight_dir / "source_verification.json"),
    }
    identity_hash = digest_object(identity)
    run_id = split_run_id(seed)
    run_dir = experiment_root / "runs" / run_id
    manifest = {"run_id": run_id, "identity_hash": identity_hash, "identity": identity}
    if run_dir.exists():
        if json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("已有确认 run 身份不同，拒绝恢复")
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run_manifest.json", manifest)
        write_json(run_dir / "resolved_config.json", confirmation_config(seed))
        write_json(run_dir / "environment.json", identity["environment"])
        write_json(run_dir / "input_manifest.json", {key: {"path": identity["input_paths"][key], "sha256": value} for key, value in identity["inputs"].items()})
        data.assignment.to_csv(run_dir / "cv_assignments.csv", index=False, lineterminator="\n")
        write_json(run_dir / "dependency_plan.json", dependency_plan())
    return run_dir, identity_hash, data, protocol


def train(project_root: Path, experiment_root: Path, seed: int) -> dict:
    from src.data_access import write_json
    from src.models import fit_task, validate_saved_task
    from src.planning import all_new_tasks

    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, seed)
    tasks = all_new_tasks()
    started = time.time()
    for index, task in enumerate(tasks, start=1):
        existing = validate_saved_task(run_dir, task, identity_hash)
        receipt = existing or fit_task(run_dir, task, data, protocol, identity_hash)
        state = "恢复" if existing else "训练"
        print(f"[{index:03d}/{len(tasks):03d}] {state} {task.task_id} 轮数={receipt['fit']['requested_rounds']}", flush=True)
        complete = sum(validate_saved_task(run_dir, item, identity_hash) is not None for item in tasks)
        write_json(run_dir / "training_progress.json", {"status": "TRAINING" if complete < len(tasks) else "TRAINING_COMPLETE", "completed_models": complete, "expected_models": len(tasks), "elapsed_seconds": time.time() - started})
    receipts = [validate_saved_task(run_dir, task, identity_hash) for task in tasks]
    write_json(run_dir / "model_manifest.json", receipts)
    write_json(run_dir / "fit_records.json", [receipt["fit"] | {"task": receipt["task"]} for receipt in receipts])
    return {"status": "TRAINING_COMPLETE", "split_seed": seed, "models": len(receipts), "fit_calls": sum(len(receipt["fit"]["probes"]) + 1 for receipt in receipts), "elapsed_seconds": time.time() - started}


def train_shard(project_root: Path, experiment_root: Path, seed: int, shard_index: int, shard_count: int) -> dict:
    """训练互不重叠的任务分片；每个模型内部仍严格使用单线程。"""
    from src.data_access import write_json
    from src.models import fit_task, validate_saved_task
    from src.planning import all_new_tasks

    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, seed)
    all_tasks = all_new_tasks()
    indexed_tasks = list(enumerate(all_tasks, start=1))[shard_index::shard_count]
    started = time.time()
    trained = 0
    restored = 0
    for global_index, task in indexed_tasks:
        existing = validate_saved_task(run_dir, task, identity_hash)
        receipt = existing or fit_task(run_dir, task, data, protocol, identity_hash)
        if existing:
            restored += 1
            state = "恢复"
        else:
            trained += 1
            state = "训练"
        print(
            f"[{global_index:03d}/{len(all_tasks):03d}] 分片 {shard_index + 1}/{shard_count} "
            f"{state} {task.task_id} 轮数={receipt['fit']['requested_rounds']}",
            flush=True,
        )
        write_json(
            run_dir / f"training_progress_shard_{shard_index + 1}_of_{shard_count}.json",
            {
                "status": "TRAINING",
                "split_seed": seed,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "completed_in_shard": trained + restored,
                "expected_in_shard": len(indexed_tasks),
                "trained_in_this_process": trained,
                "restored_in_this_process": restored,
                "elapsed_seconds": time.time() - started,
            },
        )
    complete = sum(validate_saved_task(run_dir, task, identity_hash) is not None for task in all_tasks)
    result = {
        "status": "SHARD_COMPLETE",
        "split_seed": seed,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "completed_in_shard": len(indexed_tasks),
        "expected_in_shard": len(indexed_tasks),
        "trained_in_this_process": trained,
        "restored_in_this_process": restored,
        "completed_models_visible": complete,
        "expected_models": len(all_tasks),
        "elapsed_seconds": time.time() - started,
    }
    write_json(run_dir / f"training_progress_shard_{shard_index + 1}_of_{shard_count}.json", result)
    return result


def reference_outer(project_root: Path, seed: int):
    import pandas as pd

    run = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "runs" / baseline_run_id(seed)
    frame = pd.read_parquet(run / "base_predictions_cv.parquet", engine="pyarrow")
    from src.config import COMPONENTS, LONG_HORIZONS, component_target
    targets = list(LONG_HORIZONS) + [component_target(component, horizon) for horizon in LONG_HORIZONS for component in COMPONENTS]
    return {target: frame[f"raw_{target}"].to_numpy(dtype=float) for target in targets}, frame


def candidate_frames(project_root: Path, run_dir: Path, identity_hash: str, data, protocol, seed: int):
    import numpy as np
    import pandas as pd

    from src.arithmetic import gated_prediction, six_source_mean
    from src.config import COMPONENTS, FOLDS, HORIZONS, LONG_HORIZONS, SOURCE_ORDER
    from src.pipeline import _empty, _inner_predictions, _new_outer_predictions, _processed_sources, _thresholds

    n = len(data.X_train)
    new, new_sources = _new_outer_predictions(run_dir, identity_hash, data)
    lgb, baseline_frame = reference_outer(project_root, seed)
    inner_frame, inner_lookup = _inner_predictions(run_dir, identity_hash, data)
    threshold_frame = _thresholds(data, inner_lookup)
    keys = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    keys["fold"] = data.row_folds
    keys["split_id"] = f"seed{seed}"
    candidate = keys.copy()
    outer_raw = keys.copy()
    f2_path = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline" / "p_information_increment" / "runs" / f2_run_id(seed) / "control_predictions.parquet"
    f2 = pd.read_parquet(f2_path, engine="pyarrow")
    for horizon in HORIZONS:
        mask = np.asarray(protocol.expected_mask(data.meta_train, horizon), dtype=bool)
        candidate[f"scoreable_{horizon}"] = mask
        candidate[f"truth_{horizon}"] = data.y_train[horizon].to_numpy(dtype=float)
        if horizon not in LONG_HORIZONS:
            fixed = f2[f"pred_F2_v18_rule_{horizon}"].to_numpy(dtype=float)
            for level in ("L0", "L1", "L2"):
                candidate[f"pred_{level}_{horizon}"] = fixed
            candidate[f"theta_{horizon}"] = np.nan
            candidate[f"anchor_gt_theta_{horizon}"] = False
            continue
        raw, processed = _processed_sources(horizon, lgb, new)
        for source in SOURCE_ORDER:
            values = processed[source].copy(); values[~mask] = np.nan
            candidate[f"processed_{source}_{horizon}"] = values
            if "direct" in source:
                outer_raw[f"raw_{source}_{horizon}"] = raw[source]["direct"]
            else:
                for component in COMPONENTS:
                    outer_raw[f"raw_{source}_{component}_{horizon}"] = raw[source][component]
        l0 = processed["lgbm_component"].copy()
        l1 = six_source_mean([processed[source] for source in SOURCE_ORDER])
        l2, theta_rows, gate = _empty(n), _empty(n), np.zeros(n, dtype=bool)
        for outer_q in FOLDS:
            qmask = mask & (data.row_folds == outer_q)
            theta = float(threshold_frame.query("outer_fold == @outer_q and horizon == @horizon").theta.iloc[0])
            theta_rows[qmask] = theta
            l2[qmask] = gated_prediction(l0[qmask], l1[qmask], theta)
            gate[qmask] = l0[qmask] > theta
        for values in (l0, l1, l2): values[~mask] = np.nan
        candidate[f"pred_L0_{horizon}"], candidate[f"pred_L1_{horizon}"], candidate[f"pred_L2_{horizon}"] = l0, l1, l2
        candidate[f"theta_{horizon}"], candidate[f"anchor_gt_theta_{horizon}"] = theta_rows, gate
    return {"candidate_predictions.parquet": candidate, "outer_raw_predictions.parquet": outer_raw, "inner_anchor_predictions.parquet": inner_frame, "thresholds.parquet": threshold_frame}


def chinese_results(seed: int, metrics: dict, bootstrap) -> str:
    summary = metrics["summary_metrics.csv"]
    deltas = metrics["candidate_deltas.csv"]
    long_rows = []
    for horizon in ("osi_target_t24h", "osi_target_t48h"):
        scores = summary[summary.horizon.eq(horizon)].set_index("candidate")
        row = deltas[(deltas.horizon == horizon) & (deltas.candidate == "L2") & (deltas.reference == "L0")].iloc[0]
        boot = bootstrap[(bootstrap.horizon == horizon) & (bootstrap.candidate == "L2") & (bootstrap.reference == "L0")].iloc[0]
        long_rows.append((horizon.rsplit("_", 1)[-1], scores.loc["L0", "rmse"], scores.loc["L1", "rmse"], scores.loc["L2", "rmse"], row.delta_rmse, row.rmse_change_pct, boot.ci_2_5, boot.ci_97_5, boot.improvement_probability))
    lines = [f"# Stella v1.12 严格确认分折结果：seed{seed}", "", "## 汇总结论", "", "| 时距 | L0 RMSE | L1 RMSE | L2 RMSE | L2-L0 | 相对变化 | 县级bootstrap 95%区间 | 改善概率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for h,l0,l1,l2,delta,pct,lo,hi,prob in long_rows:
        lines.append(f"| {h[1:]} | {l0:.8f} | {l1:.8f} | {l2:.8f} | {delta:+.8f} | {pct:+.3f}% | [{lo:.6f}, {hi:.6f}] | {prob:.4f} |")
    lines += ["", "1h/6h 的 L0、L1、L2 均直接复制同一分折的固定 F2 主预测；本次没有重训短期链。", "", "## 完整 pooled 指标", "", summary.to_markdown(index=False), "", "## 长期候选差值", "", deltas.to_markdown(index=False), "", "## 县级配对 bootstrap", "", bootstrap.to_markdown(index=False), ""]
    return "\n".join(lines)


def assemble(project_root: Path, experiment_root: Path, seed: int, *, write: bool = True):
    from src.models import validate_saved_task
    from src.pipeline import _metrics, _paired_bootstrap, _write_frame
    from src.planning import all_new_tasks
    from src.data_access import sha256_file, write_json

    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, seed)
    if any(validate_saved_task(run_dir, task, identity_hash) is None for task in all_new_tasks()):
        raise RuntimeError("180 个模型未全部完成")
    frames = candidate_frames(project_root, run_dir, identity_hash, data, protocol, seed)
    metrics = _metrics(data, frames)
    bootstrap = _paired_bootstrap(data, frames["candidate_predictions.parquet"])
    metrics["paired_county_bootstrap.csv"] = bootstrap
    if write:
        for name, frame in {**frames, **metrics}.items(): _write_frame(run_dir / name, frame)
        (run_dir / "Results.md").write_text(chinese_results(seed, metrics, bootstrap), encoding="utf-8", newline="\n")
        write_json(run_dir / "assembly_manifest.json", {"identity_hash": identity_hash, "artifacts": {name: sha256_file(run_dir / name) for name in [*frames, *metrics, "Results.md"]}})
    return run_dir, identity_hash, data, protocol, frames, metrics


def verify(project_root: Path, experiment_root: Path, seed: int) -> dict:
    import numpy as np
    import pandas as pd

    from src.config import ATOL
    from src.data_access import sha256_file, write_json
    from src.models import load_model, predict_model, task_paths, validate_saved_task
    from src.pipeline import _compare_frame
    from src.planning import all_new_tasks

    run_dir, identity_hash, data, protocol = prepare_run(project_root, experiment_root, seed)
    max_difference = 0.0
    for task in all_new_tasks():
        receipt = validate_saved_task(run_dir, task, identity_hash)
        paths = task_paths(run_dir, task)
        saved = np.load(paths["prediction"])
        rows, expected = saved["rows"].astype(int), saved["predictions"].astype(float)
        actual = predict_model(load_model(task.family, paths["model"]), task.family, data, rows, int(receipt["fit"]["requested_rounds"]))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
        max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
    replay = replay_reference_lgb(project_root, experiment_root / "confirmation" / f"preflight_seed{seed}", data, protocol, seed)
    _, _, _, _, frames, metrics = assemble(project_root, experiment_root, seed, write=False)
    for name, frame in {**frames, **metrics}.items(): _compare_frame(run_dir / name, frame)
    candidate = frames["candidate_predictions.parquet"]
    for horizon in ("osi_target_t01h", "osi_target_t06h"):
        left = candidate[f"pred_L0_{horizon}"].to_numpy(dtype=float)
        for level in ("L1", "L2"):
            if not np.array_equal(left, candidate[f"pred_{level}_{horizon}"].to_numpy(dtype=float), equal_nan=True): raise AssertionError("短期预测发生变化")
    report = {"status": "PASS", "split_seed": seed, "identity_hash": identity_hash, "new_models_reloaded": 180, "referenced_models_reloaded": 50, "new_prediction_max_abs_difference": max_difference, "reference_prediction_max_abs_difference": 0.0, "comparison": {"atol": ATOL, "rtol": 0}, "training_performed_by_verifier": False, "test_inference_performed": False}
    write_json(run_dir / "verification.json", report)
    files = [path for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "CV_COMPLETE"]
    write_json(run_dir / "CV_COMPLETE", {"status": "CV_COMPLETE", "identity_hash": identity_hash, "verification_sha256": sha256_file(run_dir / "verification.json"), "files": {path.relative_to(run_dir).as_posix(): sha256_file(path) for path in files}})
    return report


def run_command(command: str, project_root: Path, experiment_root: Path, seed: int) -> dict:
    if command == "preflight": return run_preflight(project_root, experiment_root, seed)
    if command == "train": return train(project_root, experiment_root, seed)
    if command == "assemble":
        run_dir, identity_hash, data, protocol, frames, metrics = assemble(project_root, experiment_root, seed, write=True)
        return {"status": "ASSEMBLY_COMPLETE", "split_seed": seed, "run_dir": str(run_dir), "artifacts": sorted([*frames, *metrics])}
    if command == "verify": return verify(project_root, experiment_root, seed)
    raise ValueError(command)


if __name__ == "__main__":
    main()
