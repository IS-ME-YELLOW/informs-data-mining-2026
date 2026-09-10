"""运行 v3.2 M3 诊断闭环；结果不得作为正式收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
import torch

from build_target_timeline import sha256_file
from config import (
    CHECKPOINT_DIR,
    CV_ASSIGNMENTS,
    DIAGNOSTIC_FOLDS,
    DIAGNOSTIC_METADATA,
    DIAGNOSTIC_OOF_PARQUET,
    DIAGNOSTIC_PREDICTIONS,
    DIAGNOSTIC_SUBMISSION,
    DIAGNOSTIC_SUMMARY,
    DIAGNOSTIC_TEST_PARQUET,
    EARLY_STOPPING_PATIENCE,
    HORIZONS,
    MAX_EPOCHS_DIAGNOSTIC,
    META_TEST,
    N_FOLDS,
    SEED,
    SUBMISSION_TEMPLATE,
    TIMELINE_MANIFEST,
    TIMELINE_TEST,
    TIMELINE_TRAIN,
    V31_BASELINE_PREDICTIONS,
    V31_TEST_SEQUENCE,
    V31_TRAIN_SEQUENCE,
    VERSION,
    ensure_artifact_dirs,
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
    build_summary,
    metrics,
    postprocess,
)
from model import restore_model
from training import (
    MODEL_VARIANTS,
    UNIQUE_MODELS,
    fit_fixed_epochs,
    predict_dynamic,
    select_epoch_and_alpha,
)


def normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def county_folds(county_fips: np.ndarray) -> np.ndarray:
    assignment = pd.read_csv(CV_ASSIGNMENTS, dtype={"fipsCode": str})
    assignment["fipsCode"] = normalize_fips(assignment["fipsCode"])
    mapping = assignment.set_index("fipsCode")["fold"]
    folds = pd.Series(county_fips.astype(str)).map(mapping)
    if folds.isna().any() or set(folds.astype(int)) != set(range(N_FOLDS)):
        raise ValueError("县级折分配不完整")
    return folds.to_numpy(dtype=int)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    preprocessor: FusionPreprocessor,
    variant_name: str,
    variant: dict,
    outer_fold: int,
    train_fips: np.ndarray,
    validation_fips: np.ndarray,
    epoch: int,
    alpha: float,
    evaluation_status: str = "diagnostic_only",
) -> None:
    torch.save({
        "version": VERSION,
        "evaluation_status": evaluation_status,
        "architecture": variant["architecture"],
        "variant_name": variant_name,
        "model_config": model.model_config,
        "state_dict": model.state_dict(),
        "preprocessor": preprocessor.to_dict(),
        "outer_fold": outer_fold,
        "train_counties": train_fips.astype(str).tolist(),
        "validation_counties": validation_fips.astype(str).tolist(),
        "epoch": epoch,
        "alpha": alpha,
    }, path)


def b0_summary(sequence, baseline: np.ndarray) -> pd.DataFrame:
    rows = []
    for stage, values in {
        "raw": baseline,
        "clipped": np.clip(baseline, 0.0, 0.65),
        "postprocessed": postprocess(baseline),
    }.items():
        for horizon_idx, horizon in enumerate(HORIZONS):
            result = metrics(sequence.targets[:, :, horizon_idx], values[:, :, horizon_idx])
            valid = values[:, :, horizon_idx][np.isfinite(sequence.targets[:, :, horizon_idx])]
            rows.append({
                "model": "B0_lightgbm",
                "stage": stage,
                "horizon": horizon,
                "horizon_hours": [1, 6, 24, 48][horizon_idx],
                **result,
                "prediction_mean": float(valid.mean()),
                "prediction_max": float(valid.max()),
                "zero_pct": float(100 * np.mean(valid == 0)),
                "evaluation_status": "diagnostic_only",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DIAGNOSTIC)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()
    if DIAGNOSTIC_PREDICTIONS.exists() and not args.overwrite:
        print(f"已存在，跳过: {DIAGNOSTIC_PREDICTIONS}")
        return
    required = [
        TIMELINE_TRAIN, TIMELINE_TEST, TIMELINE_MANIFEST,
        V31_TRAIN_SEQUENCE, V31_TEST_SEQUENCE, V31_BASELINE_PREDICTIONS,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    run_start = time.perf_counter()
    device = torch.device("cpu")
    train_sequence = load_sequence(V31_TRAIN_SEQUENCE)
    test_sequence = load_sequence(V31_TEST_SEQUENCE)
    train_timeline = load_timeline(TIMELINE_TRAIN)
    test_timeline = load_timeline(TIMELINE_TEST)
    if not np.array_equal(train_sequence.county_fips, train_timeline.county_fips):
        raise ValueError("训练序列和目标轴县顺序不一致")
    if not np.array_equal(test_sequence.county_fips, test_timeline.county_fips):
        raise ValueError("测试序列和目标轴县顺序不一致")

    with np.load(V31_BASELINE_PREDICTIONS, allow_pickle=False) as baseline_data:
        baseline_train = rows_to_county_matrix(
            baseline_data["oof_raw"], train_sequence.row_index
        ).astype(np.float32)
        baseline_test = rows_to_county_matrix(
            baseline_data["test_raw"], test_sequence.row_index
        ).astype(np.float32)
    train_candidates = align_candidates(baseline_train, train_timeline)
    test_candidates = align_candidates(baseline_test, test_timeline)
    train_equal = equal_mean(train_candidates)
    test_equal = equal_mean(test_candidates)
    folds = county_folds(train_sequence.county_fips)

    unique_oof = {
        name: np.full_like(train_equal, np.nan, dtype=np.float32)
        for name in UNIQUE_MODELS
    }
    unique_oof["B1_equal_mean"] = train_equal.copy()
    test_by_fold = {name: [] for name in MODEL_VARIANTS}
    oof_weights = {
        name: np.zeros_like(train_candidates, dtype=np.float32)
        for name in MODEL_VARIANTS
    }
    fold_records = []
    run_records = []

    for outer_fold in range(N_FOLDS):
        outer_train = np.flatnonzero(folds != outer_fold)
        outer_validation = np.flatnonzero(folds == outer_fold)
        early_fold = (outer_fold + 1) % N_FOLDS
        selection_validation = np.flatnonzero(folds == early_fold)
        selection_fit = np.flatnonzero((folds != outer_fold) & (folds != early_fold))
        fold_record = {"outer_fold": outer_fold, "early_fold": early_fold, "models": {}}

        for variant_idx, (variant_name, variant) in enumerate(MODEL_VARIANTS.items()):
            model_seed = SEED + outer_fold + 100 * variant_idx
            epoch, alpha, selection_history, alpha_history = select_epoch_and_alpha(
                train_sequence,
                train_timeline,
                train_candidates,
                selection_fit,
                selection_validation,
                variant,
                args.max_epochs,
                EARLY_STOPPING_PATIENCE,
                model_seed,
                device,
            )
            model, preprocessor, validation_dynamic, validation_weights, training_history = fit_fixed_epochs(
                train_sequence,
                train_timeline,
                train_candidates,
                outer_train,
                outer_validation,
                variant,
                epoch,
                model_seed,
                device,
            )
            unique_oof[variant_name][outer_validation] = apply_alpha(
                train_equal[outer_validation], validation_dynamic, alpha
            )
            oof_weights[variant_name][outer_validation] = (
                (1.0 - alpha)
                * train_timeline.candidate_mask[None, :, :]
                / train_timeline.candidate_mask.sum(axis=1)[None, :, None]
                + alpha * validation_weights
            )

            checkpoint = CHECKPOINT_DIR / f"{variant_name}_fold{outer_fold}.pt"
            save_checkpoint(
                checkpoint, model, preprocessor, variant_name, variant, outer_fold,
                train_sequence.county_fips[outer_train],
                train_sequence.county_fips[outer_validation], epoch, alpha,
            )
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            reloaded_model = restore_model(payload["architecture"], payload["model_config"]).to(device)
            reloaded_model.load_state_dict(payload["state_dict"])
            reloaded_preprocessor = FusionPreprocessor.from_dict(payload["preprocessor"])
            reloaded_prepared = reloaded_preprocessor.transform(
                train_sequence, train_timeline, train_candidates
            )
            reloaded_dynamic, reloaded_weights = predict_dynamic(
                reloaded_model, reloaded_prepared, outer_validation, variant, device
            )
            np.testing.assert_allclose(validation_dynamic, reloaded_dynamic, rtol=0, atol=1e-7)
            np.testing.assert_allclose(validation_weights, reloaded_weights, rtol=0, atol=1e-7)

            test_prepared = preprocessor.transform(test_sequence, test_timeline, test_candidates)
            test_dynamic, _ = predict_dynamic(
                model, test_prepared, np.arange(len(test_sequence.county_fips)), variant, device
            )
            test_by_fold[variant_name].append(apply_alpha(test_equal, test_dynamic, alpha))
            fold_record["models"][variant_name] = {
                "epoch": epoch,
                "alpha": alpha,
                "selection_history": selection_history,
                "alpha_history": alpha_history,
                "training_history": training_history,
                "checkpoint": str(checkpoint.relative_to(checkpoint.parents[2])),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
            for horizon_idx, horizon in enumerate(HORIZONS):
                base_metric = metrics(
                    train_sequence.targets[outer_validation, :, horizon_idx],
                    postprocess(baseline_train[outer_validation, :, horizon_idx]),
                )
                mapped = broadcast_unique(
                    postprocess(unique_oof[variant_name][outer_validation]), train_timeline
                )
                model_metric = metrics(
                    train_sequence.targets[outer_validation, :, horizon_idx],
                    mapped[:, :, horizon_idx],
                )
                fold_records.append({
                    "outer_fold": outer_fold,
                    "model": variant_name,
                    "horizon": horizon,
                    "rmse": model_metric["rmse"],
                    "mae": model_metric["mae"],
                    "rmse_change_vs_B0_pct": 100 * (model_metric["rmse"] / base_metric["rmse"] - 1),
                    "mae_change_vs_B0_pct": 100 * (model_metric["mae"] / base_metric["mae"] - 1),
                    "evaluation_status": "diagnostic_only",
                })
            print(
                f"outer_fold={outer_fold} {variant_name}: epoch={epoch}, alpha={alpha}",
                flush=True,
            )
        run_records.append(fold_record)

    for name, values in unique_oof.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} OOF 唯一预测不完整")
        assert_unique_mapping(values, broadcast_unique(values, train_timeline), train_timeline)

    unique_test = {"B1_equal_mean": test_equal}
    for name, fold_values in test_by_fold.items():
        unique_test[name] = np.mean(np.stack(fold_values), axis=0).astype(np.float32)
        assert_unique_mapping(
            unique_test[name], broadcast_unique(unique_test[name], test_timeline), test_timeline
        )

    model_names = list(unique_oof)
    np.savez_compressed(
        DIAGNOSTIC_PREDICTIONS,
        evaluation_status=np.asarray("diagnostic_only"),
        model_names=np.asarray(model_names, dtype="U32"),
        unique_oof=np.stack([unique_oof[name] for name in model_names]).astype(np.float32),
        unique_test=np.stack([unique_test[name] for name in model_names]).astype(np.float32),
        baseline_oof_raw=baseline_train,
        baseline_test_raw=baseline_test,
        weight_model_names=np.asarray(list(MODEL_VARIANTS), dtype="U32"),
        oof_weights=np.stack([oof_weights[name] for name in MODEL_VARIANTS]).astype(np.float32),
        outer_folds=folds,
    )

    summary = pd.concat(
        [
            b0_summary(train_sequence, baseline_train),
            build_summary(
                train_sequence.targets,
                train_timeline.unique_targets,
                unique_oof,
                train_timeline,
                "diagnostic_only",
            ),
        ],
        ignore_index=True,
    )
    summary.to_csv(DIAGNOSTIC_SUMMARY, index=False)
    pd.DataFrame(fold_records).to_csv(DIAGNOSTIC_FOLDS, index=False)
    build_prediction_frame(
        train_sequence.county_fips,
        train_timeline.target_hours,
        train_timeline.unique_targets,
        unique_oof,
    ).to_parquet(DIAGNOSTIC_OOF_PARQUET, index=False)
    build_prediction_frame(
        test_sequence.county_fips,
        test_timeline.target_hours,
        test_timeline.unique_targets,
        unique_test,
    ).to_parquet(DIAGNOSTIC_TEST_PARQUET, index=False)

    meta_test = pd.read_parquet(META_TEST)
    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    main_test = postprocess(unique_test["B5_gru_future"])
    submission = build_submission(
        template,
        meta_test,
        test_timeline.row_index,
        broadcast_unique(main_test, test_timeline),
    )
    submission.to_csv(DIAGNOSTIC_SUBMISSION, index=False)

    metadata = {
        "version": VERSION,
        "milestone": "M3",
        "evaluation_status": "diagnostic_only",
        "warning": "使用 v3.1 诊断级 OOF 候选，只验证工程闭环，不作为正式收益。",
        "models": MODEL_VARIANTS,
        "max_epochs": args.max_epochs,
        "folds": run_records,
        "elapsed_seconds": time.perf_counter() - run_start,
        "input_sha256": {
            str(path): sha256_file(path) for path in required
        },
        "output_sha256": {
            str(path): sha256_file(path)
            for path in [
                DIAGNOSTIC_PREDICTIONS, DIAGNOSTIC_SUMMARY, DIAGNOSTIC_FOLDS,
                DIAGNOSTIC_OOF_PARQUET, DIAGNOSTIC_TEST_PARQUET, DIAGNOSTIC_SUBMISSION,
            ]
        },
    }
    DIAGNOSTIC_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        summary.loc[
            (summary["stage"] == "postprocessed")
            & (summary["horizon"] != "unique_trajectory")
        ].to_string(index=False)
    )
    print(f"M3 诊断闭环完成，用时 {metadata['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
