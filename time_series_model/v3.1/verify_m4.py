"""独立重载 M4 外层模型，并复算严格 OOF 与全部诊断表。"""

from __future__ import annotations

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

from baseline_crossfit import lgb, load_row_folds
from config import (
    FEATURE_TRAIN,
    HORIZONS,
    M4_BOOTSTRAP,
    M4_BOOTSTRAP_REPLICATES,
    M4_CONSISTENCY,
    M4_COUNTY_METRICS,
    M4_FOLD_DIR,
    M4_FOLD_METRICS,
    M4_METADATA,
    M4_OOF_PARQUET,
    M4_PREDICTIONS,
    M4_SUMMARY_METRICS,
    M4_TIME_METRICS,
    M4_VERIFICATION,
    META_TRAIN,
    N_FOLDS,
    SEED,
    TARGET_TRAIN,
    TRAIN_SEQUENCE_CACHE,
)
from dataset import FoldPreprocessor, county_matrix_to_rows, load_bundle
from evaluate import postprocess
from m4_evaluate import (
    build_evaluation_tables,
    build_target_phase,
    paired_county_bootstrap,
    target_aligned_average,
    target_time_consistency,
    time_and_phase_metrics,
)
from train_nested_cv import (
    ALL_MODELS,
    MODEL_VARIANTS,
    county_folds,
    predict_standardized,
    restore_model,
)


ATOL = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} 形状不一致: {actual.shape} != {expected.shape}")
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL, equal_nan=True)
    mask = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[mask] - expected[mask]))) if mask.any() else 0.0


def compare_frames(name: str, actual_path: Path, expected: pd.DataFrame) -> float:
    if actual_path.suffix == ".parquet":
        actual = pd.read_parquet(actual_path)
    else:
        actual = pd.read_csv(actual_path, dtype={"fipsCode": str})
    if actual.columns.tolist() != expected.columns.tolist() or actual.shape != expected.shape:
        raise AssertionError(f"{name} 表结构不一致")
    numeric = expected.select_dtypes(include=[np.number]).columns.tolist()
    max_difference = 0.0
    for column in numeric:
        max_difference = max(
            max_difference,
            assert_close(name + "." + column, actual[column], expected[column]),
        )
    for column in [item for item in expected.columns if item not in numeric]:
        left = actual[column].fillna("<NA>").astype(str).reset_index(drop=True)
        right = expected[column].fillna("<NA>").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise AssertionError(f"{name}.{column} 内容不一致")
    return max_difference


def main() -> None:
    metadata = json.loads(M4_METADATA.read_text(encoding="utf-8"))
    if metadata.get("evaluation_status") != "strict_nested_cv":
        raise AssertionError("M4 元数据未标记 strict_nested_cv")
    contract_hash = metadata["contract_hash"]
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    X = pd.read_parquet(FEATURE_TRAIN).reset_index(drop=True)
    target_frame = pd.read_parquet(TARGET_TRAIN).reset_index(drop=True)
    meta = pd.read_parquet(META_TRAIN).reset_index(drop=True)
    bundle = load_bundle(TRAIN_SEQUENCE_CACHE)
    folds = county_folds(bundle.county_fips)
    row_folds = load_row_folds(meta)

    with np.load(M4_PREDICTIONS, allow_pickle=False) as saved:
        if str(saved["contract_hash"]) != contract_hash:
            raise AssertionError("M4 聚合预测合同哈希不一致")
        names = saved["model_names"].astype(str).tolist()
        if names != ALL_MODELS:
            raise AssertionError("M4 聚合模型顺序不一致")
        saved_predictions = {
            name: saved["predictions"][idx].copy()
            for idx, name in enumerate(names)
        }
        if not np.array_equal(saved["outer_folds"], folds):
            raise AssertionError("M4 聚合折编号不一致")

    reconstructed = {
        name: np.full_like(bundle.targets, np.nan, dtype=np.float32)
        for name in ALL_MODELS
    }
    coverage = np.zeros(len(bundle.county_fips), dtype=int)
    county_lookup = {str(fips): idx for idx, fips in enumerate(bundle.county_fips)}
    max_differences = {}
    neural_checkpoint_count = 0
    lightgbm_checkpoint_count = 0

    for outer_fold in range(N_FOLDS):
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        fold_metadata = json.loads(fold_json.read_text(encoding="utf-8"))
        if fold_metadata["contract_hash"] != contract_hash:
            raise AssertionError(f"outer fold {outer_fold} JSON 合同哈希不一致")
        if sha256_file(fold_npz) != fold_metadata["fold_npz_sha256"]:
            raise AssertionError(f"outer fold {outer_fold} NPZ 哈希不一致")
        with np.load(fold_npz, allow_pickle=False) as fold_data:
            validation_indices = np.asarray([
                county_lookup[str(fips)] for fips in fold_data["county_fips"]
            ], dtype=int)
            train_counties = set(fold_data["outer_train_counties"].astype(str).tolist())
            fold_names = fold_data["model_names"].astype(str).tolist()
            fold_predictions = fold_data["predictions"].copy()
            baseline_full = fold_data["baseline_inner_and_outer"].copy()
            constant_value = fold_data["constant_value"].copy()
            constant_alpha = fold_data["constant_alpha"].copy()
        validation_counties = set(bundle.county_fips[validation_indices].astype(str).tolist())
        if fold_names != ALL_MODELS or not np.all(folds[validation_indices] == outer_fold):
            raise AssertionError(f"outer fold {outer_fold} 县或模型顺序不一致")
        if train_counties & validation_counties:
            raise AssertionError(f"outer fold {outer_fold} 训练县和验证县重叠")
        if train_counties | validation_counties != set(bundle.county_fips.astype(str).tolist()):
            raise AssertionError(f"outer fold {outer_fold} 县集合不完整")
        coverage[validation_indices] += 1

        baseline_reloaded = np.full(
            (len(validation_indices), 144, len(HORIZONS)), np.nan, dtype=np.float64
        )
        for horizon_idx, horizon in enumerate(HORIZONS):
            record = fold_metadata["lgbm"][horizon]
            checkpoint = ROOT / record["checkpoint"]
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise AssertionError(f"{horizon} outer fold {outer_fold} checkpoint 哈希不一致")
            model = lgb.Booster(model_file=str(checkpoint))
            valid_rows = target_frame[horizon].notna().to_numpy() & (row_folds == outer_fold)
            row_prediction = np.full(len(X), np.nan, dtype=float)
            row_prediction[valid_rows] = model.predict(X.loc[valid_rows])
            baseline_reloaded[:, :, horizon_idx] = row_prediction[
                bundle.row_index[validation_indices]
            ]
            lightgbm_checkpoint_count += 1
        max_differences[f"fold{outer_fold}.B0_checkpoint"] = assert_close(
            f"fold{outer_fold}.B0_checkpoint",
            baseline_reloaded,
            fold_predictions[ALL_MODELS.index("B0_lightgbm")],
        )
        reconstructed["B0_lightgbm"][validation_indices] = baseline_reloaded

        aligned = target_aligned_average(baseline_reloaded)
        max_differences[f"fold{outer_fold}.B1_rebuild"] = assert_close(
            f"fold{outer_fold}.B1_rebuild",
            aligned,
            fold_predictions[ALL_MODELS.index("B1_target_aligned")],
        )
        reconstructed["B1_target_aligned"][validation_indices] = aligned

        constant_prediction = (
            baseline_reloaded
            + constant_alpha.reshape(1, 1, -1)
            * constant_value.reshape(1, 1, -1)
        )
        max_differences[f"fold{outer_fold}.B4_rebuild"] = assert_close(
            f"fold{outer_fold}.B4_rebuild",
            constant_prediction,
            fold_predictions[ALL_MODELS.index("B4_constant_mean")],
        )
        reconstructed["B4_constant_mean"][validation_indices] = constant_prediction

        for variant_name, variant in MODEL_VARIANTS.items():
            checkpoint_record = fold_metadata["neural_checkpoints"][variant_name]
            checkpoint = ROOT / checkpoint_record["checkpoint"]
            if sha256_file(checkpoint) != checkpoint_record["checkpoint_sha256"]:
                raise AssertionError(
                    f"{variant_name} outer fold {outer_fold} checkpoint 哈希不一致"
                )
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            if (
                payload["outer_fold"] != outer_fold
                or payload["variant_name"] != variant_name
                or set(payload["train_counties"]) != train_counties
                or set(payload["validation_counties"]) != validation_counties
                or set(payload["train_counties"]) & set(payload["validation_counties"])
            ):
                raise AssertionError(f"{variant_name} outer fold {outer_fold} 训练边界不一致")
            model = restore_model(payload, device)
            preprocessor = FoldPreprocessor.from_dict(payload["preprocessor"])
            prepared = preprocessor.transform(bundle, baseline_full)
            delta_std = predict_standardized(
                model,
                prepared,
                validation_indices,
                variant["use_future_weather"],
                device,
            )
            delta = delta_std * preprocessor.residual_scale.reshape(1, 1, -1)
            alpha = np.asarray(payload["alpha"], dtype=float).reshape(1, 1, -1)
            prediction = baseline_reloaded + alpha * delta
            max_differences[f"fold{outer_fold}.{variant_name}_checkpoint"] = assert_close(
                f"fold{outer_fold}.{variant_name}_checkpoint",
                prediction,
                fold_predictions[ALL_MODELS.index(variant_name)],
            )
            reconstructed[variant_name][validation_indices] = prediction
            neural_checkpoint_count += 1

    if not np.all(coverage == 1):
        raise AssertionError("每个县没有恰好作为一次外层验证集")
    target_mask = np.isfinite(bundle.targets)
    for name in ALL_MODELS:
        reconstructed[name][~target_mask] = np.nan
        max_differences[f"aggregate.{name}"] = assert_close(
            f"aggregate.{name}", reconstructed[name], saved_predictions[name]
        )

    summary, fold_metrics, county_metrics = build_evaluation_tables(
        bundle.targets, reconstructed, bundle.county_fips, folds
    )
    summary["evaluation_status"] = "strict_nested_cv"
    fold_metrics["evaluation_status"] = "strict_nested_cv"
    county_metrics["evaluation_status"] = "strict_nested_cv"
    max_differences["summary_metrics"] = compare_frames(
        "summary_metrics", M4_SUMMARY_METRICS, summary
    )
    max_differences["fold_metrics"] = compare_frames(
        "fold_metrics", M4_FOLD_METRICS, fold_metrics
    )
    max_differences["county_metrics"] = compare_frames(
        "county_metrics", M4_COUNTY_METRICS, county_metrics
    )

    bootstrap = paired_county_bootstrap(
        bundle.targets, reconstructed, M4_BOOTSTRAP_REPLICATES, SEED + 4000
    )
    bootstrap["evaluation_status"] = "strict_nested_cv"
    max_differences["paired_bootstrap"] = compare_frames(
        "paired_bootstrap", M4_BOOTSTRAP, bootstrap
    )
    target_phase = build_target_phase(X, bundle.row_index)
    time_metrics = time_and_phase_metrics(bundle.targets, reconstructed, target_phase)
    time_metrics["evaluation_status"] = "strict_nested_cv"
    max_differences["time_phase_metrics"] = compare_frames(
        "time_phase_metrics", M4_TIME_METRICS, time_metrics
    )
    consistency = target_time_consistency(reconstructed)
    consistency["evaluation_status"] = "strict_nested_cv"
    max_differences["target_time_consistency"] = compare_frames(
        "target_time_consistency", M4_CONSISTENCY, consistency
    )

    oof_frame = meta.copy()
    target_rows = county_matrix_to_rows(bundle.targets, bundle.row_index)
    for horizon_idx, horizon in enumerate(HORIZONS):
        oof_frame[f"target_{horizon}"] = target_rows[:, horizon_idx]
    for model_name, values in reconstructed.items():
        rows = county_matrix_to_rows(postprocess(values), bundle.row_index)
        for horizon_idx, horizon in enumerate(HORIZONS):
            oof_frame[f"{model_name}_{horizon}"] = rows[:, horizon_idx]
    max_differences["oof_parquet"] = compare_frames(
        "oof_parquet", M4_OOF_PARQUET, oof_frame
    )

    hash_mismatches = []
    for relative, expected_hash in {
        **metadata["input_sha256"], **metadata["output_sha256"]
    }.items():
        if sha256_file(ROOT / relative) != expected_hash:
            hash_mismatches.append(relative)
    for relative, record in metadata.get("supplemental_outputs", {}).items():
        if sha256_file(ROOT / relative) != record["sha256"]:
            hash_mismatches.append(relative)
    if hash_mismatches:
        raise AssertionError(f"运行元数据哈希不一致: {hash_mismatches}")

    result = {
        "version": "v3.1",
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "status": "passed",
        "tolerance": ATOL,
        "outer_county_coverage": "239 个县各且仅各作为一次外层验证集",
        "lightgbm_checkpoints_reloaded": lightgbm_checkpoint_count,
        "neural_checkpoints_reloaded": neural_checkpoint_count,
        "verified_run_hashes": (
            len(metadata["input_sha256"])
            + len(metadata["output_sha256"])
            + len(metadata.get("supplemental_outputs", {}))
        ),
        "max_absolute_difference": max(max_differences.values()),
        "max_absolute_differences": max_differences,
    }
    M4_VERIFICATION.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
