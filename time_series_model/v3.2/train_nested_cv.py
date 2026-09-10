"""运行 v3.2 M4：复用经验证的 v3.1 严格 LightGBM 候选。"""

from __future__ import annotations

import argparse
import copy
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
    BOOTSTRAP_REPLICATES,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
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
    M4_WEIGHTS_PARQUET,
    MAX_EPOCHS_M4,
    N_FOLDS,
    SEED,
    TIMELINE_TRAIN,
    V31_M4_FOLD_DIR,
    V31_M4_METADATA,
    V31_M4_DIR,
    V31_TRAIN_SEQUENCE,
    VERSION,
    WEIGHT_DECAY,
    ensure_artifact_dirs,
)
from dataset import (
    FusionPreprocessor,
    align_candidates,
    broadcast_unique,
    equal_mean,
    load_sequence,
    load_timeline,
)
from evaluate import apply_alpha, assert_unique_mapping, build_prediction_frame
from m4_evaluate import (
    build_evaluation_tables,
    paired_county_bootstrap,
    weights_frame,
)
from optim import LocalAdamW
from train_diagnostic import county_folds, save_checkpoint
from training import (
    MODEL_VARIANTS,
    build_variant_model,
    choose_alpha,
    fit_fixed_epochs,
    predict_dynamic,
    seed_everything,
    train_epoch,
    validation_loss,
    validate_weights,
)


M4_UNIQUE_MODELS = ["B1_equal_mean", *MODEL_VARIANTS.keys(), "D0_oracle_convex_hull"]


def source_contract(max_epochs: int) -> str:
    digest = hashlib.sha256()
    for name in [
        "config.py", "dataset.py", "model.py", "evaluate.py", "training.py",
        "m4_evaluate.py", "train_nested_cv.py",
    ]:
        digest.update((Path(__file__).parent / name).read_bytes())
    digest.update(V31_M4_METADATA.read_bytes())
    digest.update(json.dumps({"max_epochs": max_epochs, "models": MODEL_VARIANTS}, sort_keys=True).encode())
    return digest.hexdigest()


def validate_v31_source() -> dict:
    metadata = json.loads(V31_M4_METADATA.read_text(encoding="utf-8"))
    verification_path = V31_M4_DIR / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if metadata.get("evaluation_status") != "strict_nested_cv" or verification.get("status") != "passed":
        raise ValueError("v3.1 M4 严格基线没有通过独立验证")
    expected = metadata["input_sha256"].get(
        "time_series_model\\v3.1\\artifacts\\sequences_train_v3.1.npz"
    )
    if expected != sha256_file(V31_TRAIN_SEQUENCE):
        raise ValueError("v3.1 序列缓存哈希已变化")
    return {
        "metadata_sha256": sha256_file(V31_M4_METADATA),
        "verification_sha256": sha256_file(verification_path),
        "source_contract_hash": metadata["contract_hash"],
        "source_verification_max_difference": verification["max_absolute_difference"],
    }


def load_v31_fold(
    outer_fold: int,
    sequence,
    folds: np.ndarray,
) -> tuple[np.ndarray, dict, str]:
    npz_path = V31_M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
    json_path = V31_M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if sha256_file(npz_path) != metadata["fold_npz_sha256"]:
        raise ValueError(f"v3.1 outer fold {outer_fold} NPZ 哈希错误")
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_validation = np.flatnonzero(folds == outer_fold)
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data["outer_train_counties"].astype(str)) != set(sequence.county_fips[outer_train].astype(str)):
            raise ValueError(f"v3.1 outer fold {outer_fold} 训练县集合不一致")
        if data["county_fips"].astype(str).tolist() != sequence.county_fips[outer_validation].astype(str).tolist():
            raise ValueError(f"v3.1 outer fold {outer_fold} 验证县顺序不一致")
        baseline = data["baseline_inner_and_outer"].copy().astype(np.float32)
    return baseline, metadata, sha256_file(npz_path)


def select_one_inner_fold(
    sequence,
    timeline,
    candidates: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant: dict,
    max_epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[int, np.ndarray, list[dict]]:
    seed_everything(seed)
    preprocessor = FusionPreprocessor.fit(sequence, candidates, fit_indices)
    prepared = preprocessor.transform(sequence, timeline, candidates)
    model = build_variant_model(variant, prepared).to(device)
    optimizer = LocalAdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    best_epoch = 1
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss, gradient_norm = train_epoch(
            model, prepared, fit_indices, optimizer, variant, device, rng
        )
        valid_loss = validation_loss(model, prepared, validation_indices, variant, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "gradient_norm_before_clip": gradient_norm,
        })
        if valid_loss < best_loss - 1e-10:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("内层选择没有保存模型")
    model.load_state_dict(best_state)
    dynamic, _ = predict_dynamic(model, prepared, validation_indices, variant, device)
    return best_epoch, dynamic, history


def pooled_inner_selection(
    sequence,
    timeline,
    candidates: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    variant_name: str,
    variant: dict,
    max_epochs: int,
    device: torch.device,
) -> tuple[int, float, dict]:
    outer_train = np.flatnonzero(folds != outer_fold)
    pooled_dynamic = np.full_like(timeline.unique_targets, np.nan, dtype=np.float32)
    inner_records = []
    selected_epochs = []
    variant_index = list(MODEL_VARIANTS).index(variant_name)
    for inner_fold in [fold for fold in range(N_FOLDS) if fold != outer_fold]:
        fit_indices = np.flatnonzero((folds != outer_fold) & (folds != inner_fold))
        validation_indices = np.flatnonzero(folds == inner_fold)
        fold_seed = SEED + outer_fold * 1000 + variant_index * 100 + inner_fold
        epoch, dynamic, history = select_one_inner_fold(
            sequence, timeline, candidates, fit_indices, validation_indices,
            variant, max_epochs, fold_seed, device,
        )
        pooled_dynamic[validation_indices] = dynamic
        selected_epochs.append(epoch)
        inner_records.append({
            "inner_validation_fold": inner_fold,
            "fit_counties": int(len(fit_indices)),
            "validation_counties": int(len(validation_indices)),
            "selected_epoch": epoch,
            "history": history,
        })
    if not np.isfinite(pooled_dynamic[outer_train]).all():
        raise ValueError("四折融合器 OOF 预测不完整")
    equal = equal_mean(candidates)
    alpha, alpha_history = choose_alpha(
        sequence.targets[outer_train],
        equal[outer_train],
        pooled_dynamic[outer_train],
        timeline,
    )
    final_epoch = max(1, int(np.rint(np.median(selected_epochs))))
    return final_epoch, alpha, {
        "inner_folds": inner_records,
        "inner_selected_epochs": selected_epochs,
        "final_epoch_median": final_epoch,
        "pooled_alpha": alpha,
        "alpha_history": alpha_history,
    }


def run_outer_fold(
    outer_fold: int,
    contract_hash: str,
    max_epochs: int,
    sequence,
    timeline,
    folds: np.ndarray,
    device: torch.device,
) -> tuple[Path, Path]:
    start = time.perf_counter()
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_validation = np.flatnonzero(folds == outer_fold)
    baseline, source_metadata, source_npz_hash = load_v31_fold(outer_fold, sequence, folds)
    candidates = align_candidates(baseline, timeline)
    equal = equal_mean(candidates)
    fold_unique = {"B1_equal_mean": equal[outer_validation].copy()}
    fold_weights = {}
    model_records = {}
    equal_weight = (
        timeline.candidate_mask[None, :, :]
        / timeline.candidate_mask.sum(axis=1)[None, :, None]
    )

    for variant_idx, (variant_name, variant) in enumerate(MODEL_VARIANTS.items()):
        epoch, alpha, selection_record = pooled_inner_selection(
            sequence, timeline, candidates, folds, outer_fold,
            variant_name, variant, max_epochs, device,
        )
        final_seed = SEED + outer_fold + 100 * variant_idx
        model, preprocessor, dynamic, weights, training_history = fit_fixed_epochs(
            sequence, timeline, candidates, outer_train, outer_validation,
            variant, epoch, final_seed, device,
        )
        combined = apply_alpha(equal[outer_validation], dynamic, alpha)
        final_weights = (1.0 - alpha) * equal_weight + alpha * weights
        final_weights = np.broadcast_to(final_weights, weights.shape).copy().astype(np.float32)
        validate_weights(final_weights, np.isfinite(candidates[outer_validation]))
        fold_unique[variant_name] = combined
        fold_weights[variant_name] = final_weights

        checkpoint = M4_CHECKPOINT_DIR / f"{variant_name}_outer_fold{outer_fold}.pt"
        save_checkpoint(
            checkpoint, model, preprocessor, variant_name, variant, outer_fold,
            sequence.county_fips[outer_train], sequence.county_fips[outer_validation],
            epoch, alpha, "strict_nested_cv",
        )
        model_records[variant_name] = {
            "epoch": epoch,
            "alpha": alpha,
            "selection": selection_record,
            "final_training_history": training_history,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
        print(
            f"outer_fold={outer_fold} {variant_name}: epoch={epoch}, alpha={alpha}",
            flush=True,
        )

    validation_candidates = candidates[outer_validation]
    minimum = np.nanmin(validation_candidates, axis=-1)
    maximum = np.nanmax(validation_candidates, axis=-1)
    fold_unique["D0_oracle_convex_hull"] = np.clip(
        timeline.unique_targets[outer_validation], minimum, maximum
    ).astype(np.float32)
    for name, values in fold_unique.items():
        assert_unique_mapping(values, broadcast_unique(values, timeline), timeline)

    fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
    np.savez_compressed(
        fold_npz,
        contract_hash=np.asarray(contract_hash),
        county_fips=sequence.county_fips[outer_validation],
        unique_model_names=np.asarray(M4_UNIQUE_MODELS, dtype="U32"),
        unique_predictions=np.stack([fold_unique[name] for name in M4_UNIQUE_MODELS]).astype(np.float32),
        weight_model_names=np.asarray(list(MODEL_VARIANTS), dtype="U32"),
        weights=np.stack([fold_weights[name] for name in MODEL_VARIANTS]).astype(np.float32),
        baseline_validation=baseline[outer_validation].astype(np.float32),
        source_v31_fold_sha256=np.asarray(source_npz_hash),
    )
    fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
    fold_record = {
        "version": VERSION,
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "outer_fold": outer_fold,
        "outer_train_counties": int(len(outer_train)),
        "outer_validation_counties": int(len(outer_validation)),
        "contract_hash": contract_hash,
        "source_v31_fold_sha256": source_npz_hash,
        "source_v31_lgbm": source_metadata["lgbm"],
        "models": model_records,
        "fold_npz_sha256": sha256_file(fold_npz),
        "elapsed_seconds": time.perf_counter() - start,
    }
    fold_json.write_text(json.dumps(fold_record, ensure_ascii=False, indent=2), encoding="utf-8")
    return fold_npz, fold_json


def aggregate(
    contract_hash: str,
    max_epochs: int,
    sequence,
    timeline,
    folds: np.ndarray,
    source_record: dict,
    run_start: float,
) -> None:
    baseline = np.full_like(sequence.targets, np.nan, dtype=np.float32)
    unique = {
        name: np.full_like(timeline.unique_targets, np.nan, dtype=np.float32)
        for name in M4_UNIQUE_MODELS
    }
    weight_arrays = {
        name: np.zeros((len(folds), len(timeline.target_hours), 4), dtype=np.float32)
        for name in MODEL_VARIANTS
    }
    coverage = np.zeros(len(folds), dtype=int)
    fold_records = []
    lookup = {str(fips): idx for idx, fips in enumerate(sequence.county_fips)}
    for outer_fold in range(N_FOLDS):
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        record = json.loads(fold_json.read_text(encoding="utf-8"))
        if sha256_file(fold_npz) != record["fold_npz_sha256"]:
            raise ValueError(f"outer fold {outer_fold} 产物哈希错误")
        with np.load(fold_npz, allow_pickle=False) as data:
            indices = np.asarray([lookup[str(fips)] for fips in data["county_fips"]], dtype=int)
            if not np.all(folds[indices] == outer_fold):
                raise ValueError(f"outer fold {outer_fold} 县索引错误")
            names = data["unique_model_names"].astype(str).tolist()
            weight_names = data["weight_model_names"].astype(str).tolist()
            baseline[indices] = data["baseline_validation"]
            for idx, name in enumerate(names):
                unique[name][indices] = data["unique_predictions"][idx]
            for idx, name in enumerate(weight_names):
                weight_arrays[name][indices] = data["weights"][idx]
            coverage[indices] += 1
        fold_records.append(record)
    if not np.all(coverage == 1):
        raise ValueError("每县没有恰好作为一次外层验证")
    if not np.isfinite(baseline[np.isfinite(sequence.targets)]).all():
        raise ValueError("B0 严格 OOF 不完整")
    for name, values in unique.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 严格 OOF 不完整")
        assert_unique_mapping(values, broadcast_unique(values, timeline), timeline)

    np.savez_compressed(
        M4_PREDICTIONS,
        contract_hash=np.asarray(contract_hash),
        baseline_oof=baseline,
        unique_model_names=np.asarray(M4_UNIQUE_MODELS, dtype="U32"),
        unique_predictions=np.stack([unique[name] for name in M4_UNIQUE_MODELS]),
        weight_model_names=np.asarray(list(MODEL_VARIANTS), dtype="U32"),
        weights=np.stack([weight_arrays[name] for name in MODEL_VARIANTS]),
        outer_folds=folds,
    )
    summary, fold_metrics, county_metrics, stage_metrics = build_evaluation_tables(
        sequence.targets, timeline.unique_targets, baseline, unique,
        timeline, sequence.county_fips, folds,
    )
    summary.to_csv(M4_SUMMARY, index=False)
    fold_metrics.to_csv(M4_FOLDS, index=False)
    county_metrics.to_csv(M4_COUNTIES, index=False)
    stage_metrics.to_csv(M4_STAGES, index=False)
    bootstrap = paired_county_bootstrap(
        sequence.targets, baseline, unique, timeline, BOOTSTRAP_REPLICATES, SEED + 3200
    )
    bootstrap.to_csv(M4_BOOTSTRAP, index=False)
    weights_frame(
        sequence.county_fips, timeline.target_hours, weight_arrays
    ).to_parquet(M4_WEIGHTS_PARQUET, index=False)
    build_prediction_frame(
        sequence.county_fips, timeline.target_hours, timeline.unique_targets, unique
    ).to_parquet(M4_OOF_PARQUET, index=False)

    output_paths = [
        M4_PREDICTIONS, M4_SUMMARY, M4_FOLDS, M4_COUNTIES, M4_STAGES,
        M4_BOOTSTRAP, M4_WEIGHTS_PARQUET, M4_OOF_PARQUET,
    ]
    metadata = {
        "version": VERSION,
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "contract_hash": contract_hash,
        "baseline_version": "v1.5.6",
        "source_reuse": source_record,
        "protocol": {
            "outer_folds": 5,
            "unit": "county_full_trajectory",
            "lgbm_candidates": "v3.1 M4 已验证的 baseline_inner_and_outer；outer-train 为固定轮数 inner OOF，outer-validation 为 final 预测",
            "fusion_selection": "outer-train 内四折分别早停，epoch 取中位数，alpha 用四折 pooled OOF 选择",
            "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "outer_validation_usage": "仅最终评分",
        },
        "models": MODEL_VARIANTS,
        "max_epochs": max_epochs,
        "folds": fold_records,
        "elapsed_seconds": time.perf_counter() - run_start,
        "input_sha256": {
            str(V31_TRAIN_SEQUENCE.relative_to(ROOT)): sha256_file(V31_TRAIN_SEQUENCE),
            str(TIMELINE_TRAIN.relative_to(ROOT)): sha256_file(TIMELINE_TRAIN),
            str(V31_M4_METADATA.relative_to(ROOT)): sha256_file(V31_M4_METADATA),
        },
        "output_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths},
    }
    M4_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.loc[summary["scope"].isin(["horizon", "mean_horizon"])].to_string(index=False))
    print(f"M4 严格嵌套验证完成，用时 {metadata['elapsed_seconds']:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_M4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()
    source_record = validate_v31_source()
    sequence = load_sequence(V31_TRAIN_SEQUENCE)
    timeline = load_timeline(TIMELINE_TRAIN)
    folds = county_folds(sequence.county_fips)
    device = torch.device("cpu")
    run_start = time.perf_counter()
    contract_hash = source_contract(args.max_epochs)
    for outer_fold in range(N_FOLDS):
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        if args.overwrite or not (fold_npz.exists() and fold_json.exists()):
            run_outer_fold(
                outer_fold, contract_hash, args.max_epochs,
                sequence, timeline, folds, device,
            )
        else:
            record = json.loads(fold_json.read_text(encoding="utf-8"))
            if record.get("contract_hash") != contract_hash:
                raise ValueError(f"outer fold {outer_fold} 合同哈希变化，请使用 --overwrite")
    aggregate(contract_hash, args.max_epochs, sequence, timeline, folds, source_record, run_start)


if __name__ == "__main__":
    main()
