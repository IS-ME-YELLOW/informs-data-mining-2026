"""独立重载并复核 v3.2 产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
import torch

from config import (
    CHECKPOINT_DIR,
    DIAGNOSTIC_METADATA,
    DIAGNOSTIC_PREDICTIONS,
    DIAGNOSTIC_SUBMISSION,
    DIAGNOSTIC_SUMMARY,
    DIAGNOSTIC_VERIFICATION,
    META_TEST,
    M4_BOOTSTRAP,
    M4_CHECKPOINT_DIR,
    M4_COUNTIES,
    M4_FOLD_DIR,
    M4_FOLDS,
    M4_METADATA,
    M4_OOF_PARQUET,
    M4_PREDICTIONS,
    M4_STAGES,
    M4_SUMMARY,
    M4_VERIFICATION,
    M4_WEIGHTS_PARQUET,
    BOOTSTRAP_REPLICATES,
    RELOAD_ATOL,
    SEED,
    SUBMISSION_TEMPLATE,
    TIMELINE_TEST,
    TIMELINE_TRAIN,
    V31_BASELINE_PREDICTIONS,
    V31_TEST_SEQUENCE,
    V31_TRAIN_SEQUENCE,
)
from dataset import (
    FusionPreprocessor,
    align_candidates,
    broadcast_unique,
    equal_mean,
    load_sequence,
    load_timeline,
    rows_to_county_matrix,
)
from evaluate import (
    apply_alpha,
    assert_unique_mapping,
    build_prediction_frame,
    build_submission,
    postprocess,
)
from m4_evaluate import (
    build_evaluation_tables,
    paired_county_bootstrap,
    weights_frame,
)
from model import restore_model
from train_diagnostic import county_folds
from training import MODEL_VARIANTS, predict_dynamic, validate_weights


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    left = np.asarray(actual)
    right = np.asarray(expected)
    np.testing.assert_allclose(left, right, rtol=0, atol=RELOAD_ATOL, equal_nan=True)
    mask = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else 0.0


def compare_frames(name: str, expected: pd.DataFrame, path: Path) -> float:
    if path.suffix == ".parquet":
        actual = pd.read_parquet(path)
    else:
        string_columns = {
            column: str
            for column in expected.columns
            if not pd.api.types.is_numeric_dtype(expected[column])
        }
        actual = pd.read_csv(path, dtype=string_columns)
    if actual.shape != expected.shape or actual.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"{name} 表结构不一致")
    maximum = 0.0
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            maximum = max(maximum, assert_close(name + "." + column, expected[column], actual[column]))
        else:
            left = expected[column].replace("", np.nan).fillna("<NA>").astype(str).reset_index(drop=True)
            right = actual[column].replace("", np.nan).fillna("<NA>").astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise AssertionError(f"{name}.{column} 内容不一致")
    return maximum


def verify_m3() -> dict:
    metadata = json.loads(DIAGNOSTIC_METADATA.read_text(encoding="utf-8"))
    if metadata.get("evaluation_status") != "diagnostic_only":
        raise AssertionError("M3 状态标记错误")
    train_sequence = load_sequence(V31_TRAIN_SEQUENCE)
    test_sequence = load_sequence(V31_TEST_SEQUENCE)
    train_timeline = load_timeline(TIMELINE_TRAIN)
    test_timeline = load_timeline(TIMELINE_TEST)
    with np.load(V31_BASELINE_PREDICTIONS, allow_pickle=False) as baseline_data:
        baseline_train = rows_to_county_matrix(
            baseline_data["oof_raw"], train_sequence.row_index
        )
        baseline_test = rows_to_county_matrix(
            baseline_data["test_raw"], test_sequence.row_index
        )
    train_candidates = align_candidates(baseline_train, train_timeline)
    test_candidates = align_candidates(baseline_test, test_timeline)
    train_equal = equal_mean(train_candidates)
    test_equal = equal_mean(test_candidates)
    folds = county_folds(train_sequence.county_fips)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    with np.load(DIAGNOSTIC_PREDICTIONS, allow_pickle=False) as saved:
        names = saved["model_names"].astype(str).tolist()
        saved_oof = {name: saved["unique_oof"][idx].copy() for idx, name in enumerate(names)}
        saved_test = {name: saved["unique_test"][idx].copy() for idx, name in enumerate(names)}
        weight_names = saved["weight_model_names"].astype(str).tolist()
        saved_weights = {
            name: saved["oof_weights"][idx].copy() for idx, name in enumerate(weight_names)
        }
        if not np.array_equal(saved["outer_folds"], folds):
            raise AssertionError("M3 折分数组不一致")
    max_differences = {
        "B1_oof": assert_close("B1_oof", train_equal, saved_oof["B1_equal_mean"]),
        "B1_test": assert_close("B1_test", test_equal, saved_test["B1_equal_mean"]),
    }
    test_reconstructed = {name: [] for name in MODEL_VARIANTS}
    checkpoint_count = 0
    for outer_fold in range(5):
        validation = np.flatnonzero(folds == outer_fold)
        train = np.flatnonzero(folds != outer_fold)
        for variant_name, variant in MODEL_VARIANTS.items():
            checkpoint = CHECKPOINT_DIR / f"{variant_name}_fold{outer_fold}.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            if (
                payload["outer_fold"] != outer_fold
                or payload["variant_name"] != variant_name
                or set(payload["train_counties"]) != set(train_sequence.county_fips[train].astype(str))
                or set(payload["validation_counties"]) != set(train_sequence.county_fips[validation].astype(str))
            ):
                raise AssertionError(f"{variant_name} fold {outer_fold} 训练边界错误")
            model = restore_model(payload["architecture"], payload["model_config"]).to(device)
            model.load_state_dict(payload["state_dict"])
            preprocessor = FusionPreprocessor.from_dict(payload["preprocessor"])
            prepared = preprocessor.transform(train_sequence, train_timeline, train_candidates)
            dynamic, weights = predict_dynamic(model, prepared, validation, variant, device)
            alpha = float(payload["alpha"])
            combined = apply_alpha(train_equal[validation], dynamic, alpha)
            max_differences[f"{variant_name}.fold{outer_fold}.prediction"] = assert_close(
                f"{variant_name}.fold{outer_fold}.prediction",
                combined,
                saved_oof[variant_name][validation],
            )
            equal_weights = (
                train_timeline.candidate_mask[None, :, :]
                / train_timeline.candidate_mask.sum(axis=1)[None, :, None]
            )
            final_weights = (1.0 - alpha) * equal_weights + alpha * weights
            validate_weights(final_weights, prepared["candidate_mask"][validation])
            max_differences[f"{variant_name}.fold{outer_fold}.weights"] = assert_close(
                f"{variant_name}.fold{outer_fold}.weights",
                final_weights,
                saved_weights[variant_name][validation],
            )
            test_prepared = preprocessor.transform(test_sequence, test_timeline, test_candidates)
            test_dynamic, _ = predict_dynamic(
                model, test_prepared, np.arange(len(test_sequence.county_fips)), variant, device
            )
            test_reconstructed[variant_name].append(apply_alpha(test_equal, test_dynamic, alpha))
            checkpoint_count += 1
    for name, values in test_reconstructed.items():
        mean_prediction = np.mean(np.stack(values), axis=0)
        max_differences[f"{name}.test"] = assert_close(
            f"{name}.test", mean_prediction, saved_test[name]
        )
    for name, values in saved_oof.items():
        assert_unique_mapping(values, broadcast_unique(values, train_timeline), train_timeline)
    for name, values in saved_test.items():
        assert_unique_mapping(values, broadcast_unique(values, test_timeline), test_timeline)

    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    meta_test = pd.read_parquet(META_TEST)
    rebuilt_submission = build_submission(
        template,
        meta_test,
        test_timeline.row_index,
        broadcast_unique(postprocess(saved_test["B5_gru_future"]), test_timeline),
    )
    actual_submission = pd.read_csv(DIAGNOSTIC_SUBMISSION, dtype={"fipsCode": str})
    if rebuilt_submission.columns.tolist() != actual_submission.columns.tolist():
        raise AssertionError("提交列结构不一致")
    for column in rebuilt_submission.columns:
        if not pd.api.types.is_numeric_dtype(rebuilt_submission[column]):
            left = rebuilt_submission[column].fillna("<NA>").astype(str).reset_index(drop=True)
            right = actual_submission[column].fillna("<NA>").astype(str).reset_index(drop=True)
            if not left.equals(right):
                raise AssertionError(f"提交标识列 {column} 不一致")
        else:
            max_differences[f"submission.{column}"] = assert_close(
                f"submission.{column}", rebuilt_submission[column], actual_submission[column]
            )

    hash_mismatches = []
    for path_text, expected in {**metadata["input_sha256"], **metadata["output_sha256"]}.items():
        if sha256_file(Path(path_text)) != expected:
            hash_mismatches.append(path_text)
    if hash_mismatches:
        raise AssertionError(f"文件哈希不一致: {hash_mismatches}")
    result = {
        "version": "v3.2",
        "milestone": "M3",
        "evaluation_status": "diagnostic_only",
        "status": "passed",
        "checkpoints_reloaded": checkpoint_count,
        "unique_mapping": "raw/postprocessed 预测由唯一轨迹广播，逐值一致",
        "verified_hashes": len(metadata["input_sha256"]) + len(metadata["output_sha256"]),
        "tolerance": RELOAD_ATOL,
        "max_absolute_difference": max(max_differences.values()),
        "max_absolute_differences": max_differences,
    }
    DIAGNOSTIC_VERIFICATION.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def verify_m4() -> dict:
    from train_nested_cv import M4_UNIQUE_MODELS, load_v31_fold

    metadata = json.loads(M4_METADATA.read_text(encoding="utf-8"))
    if metadata.get("evaluation_status") != "strict_nested_cv":
        raise AssertionError("M4 状态标记错误")
    sequence = load_sequence(V31_TRAIN_SEQUENCE)
    timeline = load_timeline(TIMELINE_TRAIN)
    folds = county_folds(sequence.county_fips)
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with np.load(M4_PREDICTIONS, allow_pickle=False) as saved:
        if str(saved["contract_hash"]) != metadata["contract_hash"]:
            raise AssertionError("M4 合同哈希不一致")
        if not np.array_equal(saved["outer_folds"], folds):
            raise AssertionError("M4 外层折数组不一致")
        saved_baseline = saved["baseline_oof"].copy()
        names = saved["unique_model_names"].astype(str).tolist()
        if names != M4_UNIQUE_MODELS:
            raise AssertionError("M4 唯一模型顺序不一致")
        saved_unique = {name: saved["unique_predictions"][idx].copy() for idx, name in enumerate(names)}
        weight_names = saved["weight_model_names"].astype(str).tolist()
        saved_weights = {name: saved["weights"][idx].copy() for idx, name in enumerate(weight_names)}

    reconstructed_baseline = np.full_like(saved_baseline, np.nan)
    reconstructed_unique = {name: np.full_like(timeline.unique_targets, np.nan) for name in names}
    reconstructed_weights = {name: np.zeros_like(saved_weights[name]) for name in weight_names}
    coverage = np.zeros(len(folds), dtype=int)
    maximums = {}
    checkpoint_count = 0
    for outer_fold in range(5):
        validation = np.flatnonzero(folds == outer_fold)
        train = np.flatnonzero(folds != outer_fold)
        baseline, _, source_hash = load_v31_fold(outer_fold, sequence, folds)
        candidates = align_candidates(baseline, timeline)
        equal = equal_mean(candidates)
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        fold_record = json.loads(fold_json.read_text(encoding="utf-8"))
        if sha256_file(fold_npz) != fold_record["fold_npz_sha256"]:
            raise AssertionError(f"v3.2 fold {outer_fold} NPZ 哈希错误")
        if fold_record["source_v31_fold_sha256"] != source_hash:
            raise AssertionError(f"v3.2 fold {outer_fold} 基线来源哈希错误")
        with np.load(fold_npz, allow_pickle=False) as fold_data:
            maximums[f"fold{outer_fold}.baseline_file"] = assert_close(
                f"fold{outer_fold}.baseline_file", baseline[validation], fold_data["baseline_validation"]
            )
            fold_names = fold_data["unique_model_names"].astype(str).tolist()
            for idx, name in enumerate(fold_names):
                maximums[f"fold{outer_fold}.{name}.file"] = assert_close(
                    f"fold{outer_fold}.{name}.file",
                    saved_unique[name][validation],
                    fold_data["unique_predictions"][idx],
                )
        reconstructed_baseline[validation] = baseline[validation]
        reconstructed_unique["B1_equal_mean"][validation] = equal[validation]
        equal_weights = (
            timeline.candidate_mask[None, :, :]
            / timeline.candidate_mask.sum(axis=1)[None, :, None]
        )
        for variant_name, variant in MODEL_VARIANTS.items():
            checkpoint = M4_CHECKPOINT_DIR / f"{variant_name}_outer_fold{outer_fold}.pt"
            expected_hash = fold_record["models"][variant_name]["checkpoint_sha256"]
            if sha256_file(checkpoint) != expected_hash:
                raise AssertionError(f"{variant_name} fold {outer_fold} checkpoint 哈希错误")
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            if (
                payload["evaluation_status"] != "strict_nested_cv"
                or payload["outer_fold"] != outer_fold
                or set(payload["train_counties"]) != set(sequence.county_fips[train].astype(str))
                or set(payload["validation_counties"]) != set(sequence.county_fips[validation].astype(str))
            ):
                raise AssertionError(f"{variant_name} fold {outer_fold} checkpoint 边界错误")
            model = restore_model(payload["architecture"], payload["model_config"]).to(device)
            model.load_state_dict(payload["state_dict"])
            preprocessor = FusionPreprocessor.from_dict(payload["preprocessor"])
            prepared = preprocessor.transform(sequence, timeline, candidates)
            dynamic, weights = predict_dynamic(model, prepared, validation, variant, device)
            alpha = float(payload["alpha"])
            combined = apply_alpha(equal[validation], dynamic, alpha)
            final_weights = np.broadcast_to(
                (1.0 - alpha) * equal_weights + alpha * weights,
                weights.shape,
            ).copy()
            validate_weights(final_weights, np.isfinite(candidates[validation]))
            reconstructed_unique[variant_name][validation] = combined
            reconstructed_weights[variant_name][validation] = final_weights
            maximums[f"fold{outer_fold}.{variant_name}.prediction"] = assert_close(
                f"fold{outer_fold}.{variant_name}.prediction", combined, saved_unique[variant_name][validation]
            )
            maximums[f"fold{outer_fold}.{variant_name}.weights"] = assert_close(
                f"fold{outer_fold}.{variant_name}.weights", final_weights, saved_weights[variant_name][validation]
            )
            checkpoint_count += 1
        minimum = np.nanmin(candidates[validation], axis=-1)
        maximum = np.nanmax(candidates[validation], axis=-1)
        reconstructed_unique["D0_oracle_convex_hull"][validation] = np.clip(
            timeline.unique_targets[validation], minimum, maximum
        )
        coverage[validation] += 1
    if not np.all(coverage == 1):
        raise AssertionError("239 个县没有各且仅各进入一次外层验证")
    maximums["aggregate.B0"] = assert_close("aggregate.B0", reconstructed_baseline, saved_baseline)
    for name in names:
        maximums[f"aggregate.{name}"] = assert_close(
            f"aggregate.{name}", reconstructed_unique[name], saved_unique[name]
        )
        assert_unique_mapping(
            reconstructed_unique[name], broadcast_unique(reconstructed_unique[name], timeline), timeline
        )
    for name in weight_names:
        maximums[f"aggregate.{name}.weights"] = assert_close(
            f"aggregate.{name}.weights", reconstructed_weights[name], saved_weights[name]
        )

    summary, fold_metrics, county_metrics, stage_metrics = build_evaluation_tables(
        sequence.targets, timeline.unique_targets, reconstructed_baseline,
        reconstructed_unique, timeline, sequence.county_fips, folds,
    )
    maximums["summary"] = compare_frames("summary", summary, M4_SUMMARY)
    maximums["fold_metrics"] = compare_frames("fold_metrics", fold_metrics, M4_FOLDS)
    maximums["county_metrics"] = compare_frames("county_metrics", county_metrics, M4_COUNTIES)
    maximums["stage_metrics"] = compare_frames("stage_metrics", stage_metrics, M4_STAGES)
    bootstrap = paired_county_bootstrap(
        sequence.targets, reconstructed_baseline, reconstructed_unique,
        timeline, BOOTSTRAP_REPLICATES, SEED + 3200,
    )
    maximums["bootstrap"] = compare_frames("bootstrap", bootstrap, M4_BOOTSTRAP)
    weight_table = weights_frame(sequence.county_fips, timeline.target_hours, reconstructed_weights)
    maximums["weight_table"] = compare_frames("weight_table", weight_table, M4_WEIGHTS_PARQUET)
    prediction_table = build_prediction_frame(
        sequence.county_fips, timeline.target_hours, timeline.unique_targets, reconstructed_unique
    )
    maximums["prediction_table"] = compare_frames("prediction_table", prediction_table, M4_OOF_PARQUET)

    mismatches = []
    for relative, expected in {**metadata["input_sha256"], **metadata["output_sha256"]}.items():
        if sha256_file(ROOT / relative) != expected:
            mismatches.append(relative)
    if mismatches:
        raise AssertionError(f"M4 输入输出哈希不一致: {mismatches}")
    result = {
        "version": "v3.2",
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "status": "passed",
        "outer_county_coverage": "239 个县各且仅各进入一次外层验证",
        "source_reuse": "v3.1 M4 baseline_inner_and_outer 已按来源哈希复核",
        "checkpoints_reloaded": checkpoint_count,
        "verified_hashes": len(metadata["input_sha256"]) + len(metadata["output_sha256"]),
        "unique_mapping": "B1-B5 与 oracle 在 raw/postprocessed 映射下均为同一目标小时一个值",
        "tolerance": RELOAD_ATOL,
        "max_absolute_difference": max(maximums.values()),
        "max_absolute_differences": maximums,
    }
    M4_VERIFICATION.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=["m3", "m4"], default="m3")
    args = parser.parse_args()
    if args.milestone == "m3":
        print(json.dumps(verify_m3(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify_m4(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
