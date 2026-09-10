"""独立重载 v3.1 M3 检查点，并复算预测、指标与提交文件。"""

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

from config import (
    ALPHA_BY_HORIZON,
    BASELINE_PREDICTIONS,
    CHECKPOINT_DIR,
    DIAGNOSTIC_PREDICTIONS,
    FOLD_METRICS,
    HORIZONS,
    HORIZON_HOURS,
    META_TEST,
    META_TRAIN,
    N_FOLDS,
    OOF_PREDICTIONS,
    RUN_METADATA,
    SUBMISSION_OUTPUT,
    SUBMISSION_TEMPLATE,
    SUMMARY_METRICS,
    TEST_PREDICTIONS,
    TEST_SEQUENCE_CACHE,
    TRAIN_SEQUENCE_CACHE,
    VERIFICATION_FILE,
)
from dataset import load_bundle, rows_to_county_matrix
from evaluate import build_submission, postprocess, prediction_frame, summarize_predictions
from train_diagnostic import county_folds, load_checkpoint, predict_standardized


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


def verify_prediction_frame(
    name: str,
    saved_path: Path,
    expected: pd.DataFrame,
) -> float:
    saved = pd.read_parquet(saved_path)
    if saved.shape != expected.shape or saved.columns.tolist() != expected.columns.tolist():
        raise AssertionError(f"{name} 表结构不一致")
    numeric_columns = expected.select_dtypes(include=[np.number]).columns
    max_diff = 0.0
    for column in numeric_columns:
        max_diff = max(
            max_diff,
            assert_close(
                f"{name}.{column}",
                saved[column].to_numpy(),
                expected[column].to_numpy(),
            ),
        )
    for column in expected.columns.difference(numeric_columns):
        if not saved[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"{name}.{column} 内容不一致")
    return max_diff


def main() -> None:
    required = [
        TRAIN_SEQUENCE_CACHE,
        TEST_SEQUENCE_CACHE,
        BASELINE_PREDICTIONS,
        DIAGNOSTIC_PREDICTIONS,
        OOF_PREDICTIONS,
        TEST_PREDICTIONS,
        FOLD_METRICS,
        SUMMARY_METRICS,
        RUN_METADATA,
        SUBMISSION_OUTPUT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 M3 产物: {missing}")

    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    train_bundle = load_bundle(TRAIN_SEQUENCE_CACHE)
    test_bundle = load_bundle(TEST_SEQUENCE_CACHE)
    folds = county_folds(train_bundle.county_fips)
    if set(np.unique(folds).tolist()) != set(range(N_FOLDS)):
        raise AssertionError("县级折分不完整")

    with np.load(BASELINE_PREDICTIONS, allow_pickle=False) as baseline_data:
        baseline_train = rows_to_county_matrix(
            baseline_data["oof_raw"], train_bundle.row_index
        )
        baseline_test = rows_to_county_matrix(
            baseline_data["test_raw"], test_bundle.row_index
        )
    with np.load(DIAGNOSTIC_PREDICTIONS, allow_pickle=False) as diagnostic:
        saved_oof_delta = diagnostic["gru_oof_delta"].copy()
        saved_test_by_fold = diagnostic["gru_test_delta_by_fold"].copy()
        saved_combined_oof = diagnostic["combined_oof_raw"].copy()
        saved_combined_test = diagnostic["combined_test_raw"].copy()

    valid_structure = (
        np.arange(144).reshape(-1, 1) + np.asarray(HORIZON_HOURS).reshape(1, -1) < 144
    )
    reconstructed_oof = np.full_like(saved_oof_delta, np.nan)
    reconstructed_test_folds = []
    covered = np.zeros(len(train_bundle.county_fips), dtype=int)
    checkpoint_details = []

    for fold in range(N_FOLDS):
        checkpoint = CHECKPOINT_DIR / f"gru_residual_fold{fold}.pt"
        model, preprocessor, payload = load_checkpoint(checkpoint, device)
        if payload["evaluation_status"] != "diagnostic_only" or payload["fold"] != fold:
            raise AssertionError(f"fold {fold} 检查点元数据不一致")
        validation_indices = np.flatnonzero(folds == fold)
        training_indices = np.flatnonzero(folds != fold)
        expected_validation = set(
            train_bundle.county_fips[validation_indices].astype(str).tolist()
        )
        expected_training = set(
            train_bundle.county_fips[training_indices].astype(str).tolist()
        )
        saved_validation = set(payload["validation_counties"])
        saved_training = set(payload["train_counties"])
        if saved_validation != expected_validation or saved_training != expected_training:
            raise AssertionError(f"fold {fold} 检查点的训练/验证县清单不一致")
        if saved_validation & saved_training:
            raise AssertionError(f"fold {fold} 训练县与验证县重叠")
        covered[validation_indices] += 1
        prepared_train = preprocessor.transform(train_bundle, baseline_train)
        prepared_test = preprocessor.transform(test_bundle, baseline_test)
        val_standardized = predict_standardized(
            model, prepared_train, validation_indices, device
        )
        test_standardized = predict_standardized(
            model, prepared_test, np.arange(len(test_bundle.county_fips)), device
        )
        scale = preprocessor.residual_scale.reshape(1, 1, -1)
        val_delta = val_standardized * scale
        test_delta = test_standardized * scale
        val_delta[:, ~valid_structure] = np.nan
        test_delta[:, ~valid_structure] = np.nan
        reconstructed_oof[validation_indices] = val_delta
        reconstructed_test_folds.append(test_delta)
        checkpoint_details.append({
            "fold": fold,
            "best_epoch": int(payload["best_epoch"]),
            "validation_counties": int(len(validation_indices)),
            "sha256": sha256_file(checkpoint),
        })

    if not np.all(covered == 1):
        raise AssertionError("OOF 县覆盖不是恰好一次")
    reconstructed_test_folds = np.stack(reconstructed_test_folds, axis=0)
    max_differences = {
        "oof_delta_from_checkpoints": assert_close(
            "oof_delta_from_checkpoints", reconstructed_oof, saved_oof_delta
        ),
        "test_delta_from_checkpoints": assert_close(
            "test_delta_from_checkpoints", reconstructed_test_folds, saved_test_by_fold
        ),
    }

    reconstructed_test = np.full_like(reconstructed_test_folds[0], np.nan)
    reconstructed_test[:, valid_structure] = reconstructed_test_folds[
        :, :, valid_structure
    ].mean(axis=0)
    alpha = np.asarray(ALPHA_BY_HORIZON, dtype=float).reshape(1, 1, -1)
    reconstructed_combined_oof = baseline_train + alpha * reconstructed_oof
    reconstructed_combined_test = baseline_test + alpha * reconstructed_test
    reconstructed_combined_oof[~np.isfinite(train_bundle.targets)] = np.nan
    reconstructed_combined_test[:, ~valid_structure] = np.nan
    max_differences["combined_oof"] = assert_close(
        "combined_oof", reconstructed_combined_oof, saved_combined_oof
    )
    max_differences["combined_test"] = assert_close(
        "combined_test", reconstructed_combined_test, saved_combined_test
    )

    expected_summary = summarize_predictions(
        train_bundle.targets, baseline_train, reconstructed_combined_oof
    )
    expected_summary["evaluation_status"] = "diagnostic_only"
    saved_summary = pd.read_csv(SUMMARY_METRICS)
    if saved_summary[["horizon", "stage", "evaluation_status"]].astype(str).to_dict("records") \
            != expected_summary[["horizon", "stage", "evaluation_status"]].astype(str).to_dict("records"):
        raise AssertionError("summary_metrics 的标签或顺序不一致")
    for column in ["horizon_hours", "rmse", "mae", "n", "prediction_mean", "prediction_max", "zero_pct"]:
        max_differences[f"summary.{column}"] = assert_close(
            f"summary.{column}",
            saved_summary[column].to_numpy(),
            expected_summary[column].to_numpy(),
        )

    meta_train = pd.read_parquet(META_TRAIN)
    meta_test = pd.read_parquet(META_TEST)
    expected_oof_frame = prediction_frame(
        meta_train,
        train_bundle.row_index,
        train_bundle.targets,
        baseline_train,
        reconstructed_oof,
        reconstructed_combined_oof,
    )
    expected_test_frame = prediction_frame(
        meta_test,
        test_bundle.row_index,
        test_bundle.targets,
        baseline_test,
        reconstructed_test,
        reconstructed_combined_test,
    )
    max_differences["oof_prediction_file"] = verify_prediction_frame(
        "oof_predictions", OOF_PREDICTIONS, expected_oof_frame
    )
    max_differences["test_prediction_file"] = verify_prediction_frame(
        "test_predictions", TEST_PREDICTIONS, expected_test_frame
    )

    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    expected_submission = build_submission(
        template,
        meta_test,
        test_bundle.row_index,
        postprocess(reconstructed_combined_test),
    )
    saved_submission = pd.read_csv(SUBMISSION_OUTPUT, dtype={"fipsCode": str})
    if not saved_submission[["fipsCode", "timestamp_et"]].astype(str).equals(
        expected_submission[["fipsCode", "timestamp_et"]].astype(str)
    ):
        raise AssertionError("提交文件的键或顺序不一致")
    for horizon in HORIZONS:
        max_differences[f"submission.{horizon}"] = assert_close(
            f"submission.{horizon}",
            saved_submission[horizon].to_numpy(),
            expected_submission[horizon].to_numpy(),
        )

    metadata = json.loads(RUN_METADATA.read_text(encoding="utf-8"))
    checksum_mismatches = []
    for relative_path, expected_hash in {
        **metadata["input_sha256"], **metadata["output_sha256"]
    }.items():
        path = ROOT / relative_path
        if sha256_file(path) != expected_hash:
            checksum_mismatches.append(relative_path)
    if checksum_mismatches:
        raise AssertionError(f"运行元数据中的哈希不一致: {checksum_mismatches}")

    result = {
        "version": "v3.1",
        "milestone": "M3",
        "evaluation_status": "diagnostic_only",
        "status": "passed",
        "tolerance": ATOL,
        "county_fold_coverage": "每个训练县恰好作为一次外层验证集",
        "checkpoint_details": checkpoint_details,
        "max_absolute_differences": max_differences,
        "submission_rows": int(len(saved_submission)),
        "fold_metric_rows": int(len(pd.read_csv(FOLD_METRICS))),
        "verified_checksums": len(metadata["input_sha256"]) + len(metadata["output_sha256"]),
    }
    VERIFICATION_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
