"""运行 v3.1 M4 严格外层五折，并完成 B0-B7 同折对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
import torch.nn.functional as F

from baseline_crossfit import lgb, load_row_folds
from config import (
    BATCH_SIZE,
    CV_ASSIGNMENTS,
    EARLY_STOPPING_PATIENCE,
    FEATURE_TRAIN,
    GRADIENT_CLIP,
    GRU_DROPOUT,
    GRU_HIDDEN_SIZE,
    HEAD_HIDDEN_SIZE,
    HORIZONS,
    HORIZON_HOURS,
    LEARNING_RATE,
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_NUM_BOOST_ROUND,
    LGBM_PARAMS,
    M4_ALPHA_GRID,
    M4_BOOTSTRAP,
    M4_BOOTSTRAP_REPLICATES,
    M4_CHECKPOINT_DIR,
    M4_CONSISTENCY,
    M4_COUNTY_METRICS,
    M4_DIR,
    M4_EARLY_STOPPING_PATIENCE,
    M4_FOLD_DIR,
    M4_FOLD_METRICS,
    M4_MAX_EPOCHS,
    M4_METADATA,
    M4_OOF_PARQUET,
    M4_PREDICTIONS,
    M4_SUMMARY_METRICS,
    M4_TIME_METRICS,
    META_TRAIN,
    N_FOLDS,
    SEED,
    STATIC_HIDDEN_SIZE,
    TARGET_TRAIN,
    TRAIN_SEQUENCE_CACHE,
    WEIGHT_DECAY,
    ensure_artifact_dirs,
)
from dataset import FoldPreprocessor, county_matrix_to_rows, load_bundle
from evaluate import metrics, postprocess
from m4_evaluate import (
    build_evaluation_tables,
    build_target_phase,
    paired_county_bootstrap,
    target_aligned_average,
    target_time_consistency,
    time_and_phase_metrics,
)
from model import (
    GRUResidualModel,
    MLPResidualModel,
    build_mlp_from_prepared,
    build_model_from_prepared,
)
from train_diagnostic import LocalAdamW


MODEL_VARIANTS = {
    "B2_mlp_huber": {
        "architecture": "mlp",
        "loss": "huber",
        "use_future_weather": False,
        "purpose": "非时序二级校准",
    },
    "B3_gru_huber": {
        "architecture": "gru",
        "loss": "huber",
        "use_future_weather": True,
        "purpose": "v3.1 主模型",
    },
    "B5_gru_no_future": {
        "architecture": "gru",
        "loss": "huber",
        "use_future_weather": False,
        "purpose": "未来气象消融",
    },
    "B6_gru_mse": {
        "architecture": "gru",
        "loss": "mse",
        "use_future_weather": True,
        "purpose": "RMSE 对齐损失",
    },
    "B7_gru_weighted_mse": {
        "architecture": "gru",
        "loss": "weighted_mse",
        "use_future_weather": True,
        "purpose": "高 OSI 加权 RMSE 损失",
    },
}
ALL_MODELS = [
    "B0_lightgbm",
    "B1_target_aligned",
    "B2_mlp_huber",
    "B3_gru_huber",
    "B4_constant_mean",
    "B5_gru_no_future",
    "B6_gru_mse",
    "B7_gru_weighted_mse",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def county_folds(county_fips: np.ndarray) -> np.ndarray:
    assignment = pd.read_csv(CV_ASSIGNMENTS, dtype={"fipsCode": str})
    assignment["fipsCode"] = (
        assignment["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    mapping = assignment.set_index("fipsCode")["fold"]
    values = pd.Series(county_fips.astype(str)).map(mapping)
    if values.isna().any():
        raise ValueError("至少一个训练县缺少 balanced_v1 折分配")
    return values.to_numpy(dtype=int)


def source_contract(max_epochs: int) -> str:
    digest = hashlib.sha256()
    for name in [
        "config.py", "dataset.py", "model.py", "evaluate.py",
        "m4_evaluate.py", "train_nested_cv.py",
    ]:
        digest.update((Path(__file__).parent / name).read_bytes())
    digest.update(json.dumps({
        "max_epochs": max_epochs,
        "patience": M4_EARLY_STOPPING_PATIENCE,
        "alpha_grid": M4_ALPHA_GRID,
        "variants": MODEL_VARIANTS,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def build_model(variant: dict, prepared: dict) -> torch.nn.Module:
    common = {
        "hidden_size": GRU_HIDDEN_SIZE,
        "static_hidden_size": STATIC_HIDDEN_SIZE,
        "head_hidden_size": HEAD_HIDDEN_SIZE,
        "dropout": GRU_DROPOUT,
    }
    if variant["architecture"] == "gru":
        return build_model_from_prepared(prepared, **common)
    if variant["architecture"] == "mlp":
        return build_mlp_from_prepared(prepared, **common)
    raise ValueError(f"未知 architecture: {variant['architecture']}")


def restore_model(payload: dict, device: torch.device) -> torch.nn.Module:
    if payload["architecture"] == "gru":
        model = GRUResidualModel(**payload["model_config"])
    elif payload["architecture"] == "mlp":
        model = MLPResidualModel(**payload["model_config"])
    else:
        raise ValueError(f"未知 checkpoint architecture: {payload['architecture']}")
    model.load_state_dict(payload["state_dict"])
    return model.to(device)


def make_batch(prepared: dict, indices: np.ndarray, device: torch.device) -> dict:
    result = {}
    for key in [
        "observed", "future_weather", "static", "last_state",
        "baseline_scaled", "residual_std", "valid_mask", "targets",
    ]:
        tensor = torch.from_numpy(prepared[key][indices])
        tensor = tensor.bool() if key == "valid_mask" else tensor.float()
        result[key] = tensor.to(device)
    return result


def forward_variant(model: torch.nn.Module, batch: dict, use_future_weather: bool) -> torch.Tensor:
    future = batch["future_weather"]
    if not use_future_weather:
        future = torch.zeros_like(future)
    return model(
        batch["observed"], future, batch["static"],
        batch["last_state"], batch["baseline_scaled"],
    )


def county_balanced_loss(
    prediction: torch.Tensor,
    batch: dict,
    loss_name: str,
) -> torch.Tensor:
    target = batch["residual_std"]
    mask = batch["valid_mask"]
    if loss_name == "huber":
        element = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    elif loss_name in {"mse", "weighted_mse"}:
        element = (prediction - target) ** 2
        if loss_name == "weighted_mse":
            raw_target = torch.nan_to_num(batch["targets"], nan=0.0)
            weights = 1.0 + 4.0 * torch.clamp(raw_target / 0.05, min=0.0, max=1.0)
            element = element * weights
    else:
        raise ValueError(f"未知 loss: {loss_name}")

    mask_float = mask.float()
    count = mask_float.sum(dim=1)
    per_county_horizon = (element * mask_float).sum(dim=1) / count.clamp_min(1.0)
    valid_county_horizon = count > 0
    return per_county_horizon[valid_county_horizon].mean()


def train_epoch(
    model: torch.nn.Module,
    prepared: dict,
    indices: np.ndarray,
    optimizer: LocalAdamW,
    variant: dict,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model.train()
    shuffled = np.asarray(indices, dtype=int).copy()
    rng.shuffle(shuffled)
    loss_sum = 0.0
    norm_sum = 0.0
    item_count = 0
    for start in range(0, len(shuffled), BATCH_SIZE):
        batch_indices = shuffled[start:start + BATCH_SIZE]
        batch = make_batch(prepared, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_variant(model, batch, variant["use_future_weather"])
        loss = county_balanced_loss(prediction, batch, variant["loss"])
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * len(batch_indices)
        norm_sum += float(gradient_norm.detach().cpu()) * len(batch_indices)
        item_count += len(batch_indices)
    return loss_sum / item_count, norm_sum / item_count


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    prepared: dict,
    indices: np.ndarray,
    variant: dict,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = make_batch(prepared, batch_indices, device)
        prediction = forward_variant(model, batch, variant["use_future_weather"])
        loss = county_balanced_loss(prediction, batch, variant["loss"])
        total += float(loss.cpu()) * len(batch_indices)
        count += len(batch_indices)
    return total / count


@torch.no_grad()
def predict_standardized(
    model: torch.nn.Module,
    prepared: dict,
    indices: np.ndarray,
    use_future_weather: bool,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = make_batch(prepared, batch_indices, device)
        outputs.append(
            forward_variant(model, batch, use_future_weather).cpu().numpy()
        )
    return np.concatenate(outputs, axis=0)


def choose_alpha(
    targets: np.ndarray,
    baseline: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(len(HORIZONS), dtype=np.float32)
    for horizon_idx in range(len(HORIZONS)):
        best_score = float("inf")
        best_alpha = 0.0
        for alpha in M4_ALPHA_GRID:
            prediction = postprocess(
                baseline[:, :, horizon_idx] + alpha * delta[:, :, horizon_idx]
            )
            score = metrics(targets[:, :, horizon_idx], prediction)["rmse"]
            if score < best_score - 1e-12:
                best_score = score
                best_alpha = alpha
        selected[horizon_idx] = best_alpha
    return selected


def select_epoch_and_alpha(
    bundle,
    baseline: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant: dict,
    max_epochs: int,
    fold_seed: int,
    device: torch.device,
) -> tuple[int, np.ndarray, list[dict]]:
    seed_everything(fold_seed)
    preprocessor = FoldPreprocessor.fit(bundle, baseline, fit_indices)
    prepared = preprocessor.transform(bundle, baseline)
    model = build_model(variant, prepared).to(device)
    optimizer = LocalAdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(fold_seed)
    best_epoch = 1
    best_loss = float("inf")
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss, gradient_norm = train_epoch(
            model, prepared, fit_indices, optimizer, variant, device, rng
        )
        valid_loss = validation_loss(
            model, prepared, validation_indices, variant, device
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "gradient_norm_before_clip": gradient_norm,
        })
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= M4_EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("未选出神经网络 epoch")
    model.load_state_dict(best_state)
    validation_std = predict_standardized(
        model, prepared, validation_indices,
        variant["use_future_weather"], device,
    )
    validation_delta = validation_std * preprocessor.residual_scale.reshape(1, 1, -1)
    alpha = choose_alpha(
        bundle.targets[validation_indices],
        baseline[validation_indices],
        validation_delta,
    )
    return best_epoch, alpha, history


def fit_final_neural(
    bundle,
    baseline: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant_name: str,
    variant: dict,
    epochs: int,
    alpha: np.ndarray,
    fold_seed: int,
    outer_fold: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    seed_everything(fold_seed)
    preprocessor = FoldPreprocessor.fit(bundle, baseline, train_indices)
    prepared = preprocessor.transform(bundle, baseline)
    model = build_model(variant, prepared).to(device)
    optimizer = LocalAdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(fold_seed)
    training_history = []
    for epoch in range(1, epochs + 1):
        train_loss_value, gradient_norm = train_epoch(
            model, prepared, train_indices, optimizer, variant, device, rng
        )
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss_value,
            "gradient_norm_before_clip": gradient_norm,
        })
    delta_std = predict_standardized(
        model, prepared, validation_indices,
        variant["use_future_weather"], device,
    )
    delta = delta_std * preprocessor.residual_scale.reshape(1, 1, -1)

    checkpoint_path = M4_CHECKPOINT_DIR / f"{variant_name}_outer_fold{outer_fold}.pt"
    payload = {
        "version": "v3.1",
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "variant_name": variant_name,
        "architecture": variant["architecture"],
        "loss": variant["loss"],
        "use_future_weather": variant["use_future_weather"],
        "outer_fold": outer_fold,
        "epochs": epochs,
        "alpha": alpha.tolist(),
        "model_config": model.model_config,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "preprocessor": preprocessor.to_dict(),
        "train_counties": bundle.county_fips[train_indices].astype(str).tolist(),
        "validation_counties": bundle.county_fips[validation_indices].astype(str).tolist(),
        "training_history": training_history,
    }
    torch.save(payload, checkpoint_path)
    reloaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_model = restore_model(reloaded_payload, device)
    reloaded_preprocessor = FoldPreprocessor.from_dict(reloaded_payload["preprocessor"])
    reloaded_prepared = reloaded_preprocessor.transform(bundle, baseline)
    reloaded = predict_standardized(
        reloaded_model, reloaded_prepared, validation_indices,
        variant["use_future_weather"], device,
    )
    if not np.allclose(delta_std, reloaded, rtol=0, atol=1e-7):
        raise ValueError(f"{variant_name} outer fold {outer_fold} 重载预测不一致")
    return delta.astype(np.float32), {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_history": training_history,
    }


def train_lgbm_outer_fold(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    row_folds: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, dict]:
    """返回全县矩阵：外层训练县为固定轮数 inner OOF，外层验证县为 final 预测。"""
    row_predictions = np.full((len(X), len(HORIZONS)), np.nan, dtype=np.float64)
    inner_fold_ids = [fold for fold in range(N_FOLDS) if fold != outer_fold]
    records = {}
    for horizon_idx, horizon in enumerate(HORIZONS):
        y = targets[horizon].to_numpy(dtype=float)
        valid = np.isfinite(y)
        best_iterations = []
        tuning_records = []
        for inner_fold in inner_fold_ids:
            train_mask = valid & (row_folds != outer_fold) & (row_folds != inner_fold)
            validation_mask = valid & (row_folds == inner_fold)
            train_data = lgb.Dataset(X.loc[train_mask], label=y[train_mask])
            validation_data = lgb.Dataset(
                X.loc[validation_mask], label=y[validation_mask], reference=train_data
            )
            booster = lgb.train(
                dict(LGBM_PARAMS),
                train_data,
                num_boost_round=LGBM_NUM_BOOST_ROUND,
                valid_sets=[validation_data],
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_iterations.append(int(booster.best_iteration))
            tuning_records.append({
                "inner_validation_fold": inner_fold,
                "best_iteration": int(booster.best_iteration),
            })
        fixed_rounds = max(1, int(np.rint(np.median(best_iterations))))

        for inner_fold in inner_fold_ids:
            train_mask = valid & (row_folds != outer_fold) & (row_folds != inner_fold)
            validation_mask = valid & (row_folds == inner_fold)
            booster = lgb.train(
                dict(LGBM_PARAMS),
                lgb.Dataset(X.loc[train_mask], label=y[train_mask]),
                num_boost_round=fixed_rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            row_predictions[validation_mask, horizon_idx] = booster.predict(
                X.loc[validation_mask]
            )

        outer_train_mask = valid & (row_folds != outer_fold)
        outer_validation_mask = valid & (row_folds == outer_fold)
        outer_model = lgb.train(
            dict(LGBM_PARAMS),
            lgb.Dataset(X.loc[outer_train_mask], label=y[outer_train_mask]),
            num_boost_round=fixed_rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        validation_prediction = outer_model.predict(X.loc[outer_validation_mask])
        row_predictions[outer_validation_mask, horizon_idx] = validation_prediction
        model_path = M4_CHECKPOINT_DIR / f"B0_{horizon}_outer_fold{outer_fold}.txt"
        outer_model.save_model(str(model_path))
        reloaded = lgb.Booster(model_file=str(model_path))
        reloaded_prediction = reloaded.predict(X.loc[outer_validation_mask])
        if not np.allclose(validation_prediction, reloaded_prediction, rtol=0, atol=1e-12):
            raise ValueError(f"LightGBM {horizon} outer fold {outer_fold} 重载不一致")
        records[horizon] = {
            "fixed_rounds": fixed_rounds,
            "inner_tuning": tuning_records,
            "checkpoint": str(model_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(model_path),
        }
        print(
            f"outer_fold={outer_fold} {horizon}: fixed_rounds={fixed_rounds}, "
            f"inner_best={best_iterations}", flush=True
        )

    if not np.isfinite(row_predictions[np.isfinite(targets.to_numpy(dtype=float))]).all():
        raise ValueError(f"outer fold {outer_fold} LightGBM 预测不完整")
    return row_predictions, records


def constant_control(
    bundle,
    baseline: np.ndarray,
    fit_indices: np.ndarray,
    alpha_validation_indices: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_validation_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit_residual = bundle.targets[fit_indices] - baseline[fit_indices]
    fit_constant = np.nanmean(fit_residual, axis=(0, 1))
    validation_delta = np.broadcast_to(
        fit_constant.reshape(1, 1, -1),
        (len(alpha_validation_indices), 144, len(HORIZONS)),
    ).copy()
    alpha = choose_alpha(
        bundle.targets[alpha_validation_indices],
        baseline[alpha_validation_indices],
        validation_delta,
    )
    full_residual = bundle.targets[outer_train_indices] - baseline[outer_train_indices]
    full_constant = np.nanmean(full_residual, axis=(0, 1))
    outer_delta = np.broadcast_to(
        full_constant.reshape(1, 1, -1),
        (len(outer_validation_indices), 144, len(HORIZONS)),
    ).copy()
    return outer_delta.astype(np.float32), alpha, full_constant.astype(np.float32)


def run_outer_fold(
    outer_fold: int,
    contract_hash: str,
    max_epochs: int,
    X: pd.DataFrame,
    target_frame: pd.DataFrame,
    meta: pd.DataFrame,
    row_folds: np.ndarray,
    bundle,
    folds: np.ndarray,
    device: torch.device,
) -> tuple[Path, Path]:
    fold_start = time.perf_counter()
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_validation = np.flatnonzero(folds == outer_fold)
    early_fold = (outer_fold + 1) % N_FOLDS
    alpha_validation = np.flatnonzero(folds == early_fold)
    neural_fit = np.flatnonzero((folds != outer_fold) & (folds != early_fold))
    if set(bundle.county_fips[outer_train]) & set(bundle.county_fips[outer_validation]):
        raise ValueError("外层训练县和验证县重叠")

    row_prediction, lgbm_record = train_lgbm_outer_fold(
        X, target_frame, meta, row_folds, outer_fold
    )
    baseline = row_prediction[bundle.row_index]
    valid_structure = np.isfinite(bundle.targets)
    if not np.isfinite(baseline[valid_structure]).all():
        raise ValueError(f"outer fold {outer_fold} 基线矩阵不完整")

    validation_predictions = {
        "B0_lightgbm": baseline[outer_validation].copy(),
    }
    validation_predictions["B1_target_aligned"] = target_aligned_average(
        baseline[outer_validation]
    )
    selection_records = {}
    alpha_records = {}
    epoch_records = {}
    checkpoint_records = {}

    constant_delta, constant_alpha, constant_value = constant_control(
        bundle, baseline, neural_fit, alpha_validation, outer_train, outer_validation
    )
    validation_predictions["B4_constant_mean"] = (
        baseline[outer_validation]
        + constant_alpha.reshape(1, 1, -1) * constant_delta
    )
    alpha_records["B4_constant_mean"] = constant_alpha.tolist()

    for variant_name, variant in MODEL_VARIANTS.items():
        variant_seed = SEED + outer_fold
        best_epoch, alpha, selection_history = select_epoch_and_alpha(
            bundle,
            baseline,
            neural_fit,
            alpha_validation,
            variant,
            max_epochs,
            variant_seed,
            device,
        )
        delta, checkpoint_record = fit_final_neural(
            bundle,
            baseline,
            outer_train,
            outer_validation,
            variant_name,
            variant,
            best_epoch,
            alpha,
            variant_seed,
            outer_fold,
            device,
        )
        validation_predictions[variant_name] = (
            baseline[outer_validation] + alpha.reshape(1, 1, -1) * delta
        )
        selection_records[variant_name] = selection_history
        alpha_records[variant_name] = alpha.tolist()
        epoch_records[variant_name] = best_epoch
        checkpoint_records[variant_name] = checkpoint_record
        print(
            f"outer_fold={outer_fold} {variant_name}: epoch={best_epoch}, "
            f"alpha={alpha.tolist()}", flush=True
        )

    for model_name, values in validation_predictions.items():
        if not np.isfinite(values[np.isfinite(bundle.targets[outer_validation])]).all():
            raise ValueError(f"outer fold {outer_fold} {model_name} 验证预测不完整")

    fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
    np.savez_compressed(
        fold_npz,
        contract_hash=np.asarray(contract_hash),
        county_fips=bundle.county_fips[outer_validation],
        model_names=np.asarray(ALL_MODELS, dtype="U32"),
        predictions=np.stack(
            [validation_predictions[name] for name in ALL_MODELS], axis=0
        ).astype(np.float32),
        baseline_inner_and_outer=baseline.astype(np.float32),
        outer_train_counties=bundle.county_fips[outer_train],
        constant_value=constant_value,
        constant_alpha=constant_alpha,
    )
    fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
    fold_metadata = {
        "version": "v3.1",
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "outer_fold": outer_fold,
        "early_stopping_and_alpha_fold": early_fold,
        "contract_hash": contract_hash,
        "max_epochs": max_epochs,
        "outer_train_counties": int(len(outer_train)),
        "outer_validation_counties": int(len(outer_validation)),
        "lgbm": lgbm_record,
        "selected_epochs": epoch_records,
        "selected_alpha": alpha_records,
        "constant_full_train_mean": dict(zip(HORIZONS, constant_value.tolist())),
        "selection_histories": selection_records,
        "neural_checkpoints": checkpoint_records,
        "fold_npz_sha256": sha256_file(fold_npz),
        "elapsed_seconds": time.perf_counter() - fold_start,
    }
    fold_json.write_text(
        json.dumps(fold_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return fold_npz, fold_json


def aggregate_results(
    contract_hash: str,
    bundle,
    folds: np.ndarray,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    max_epochs: int,
    run_start: float,
) -> None:
    predictions = {
        name: np.full_like(bundle.targets, np.nan, dtype=np.float32)
        for name in ALL_MODELS
    }
    fold_metadata = []
    county_lookup = {str(fips): idx for idx, fips in enumerate(bundle.county_fips)}
    coverage = np.zeros(len(bundle.county_fips), dtype=int)
    for outer_fold in range(N_FOLDS):
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        if not fold_npz.exists() or not fold_json.exists():
            raise FileNotFoundError(f"缺少 outer fold {outer_fold} 产物")
        metadata = json.loads(fold_json.read_text(encoding="utf-8"))
        if metadata["contract_hash"] != contract_hash:
            raise ValueError(f"outer fold {outer_fold} 合同哈希不一致，请使用 --overwrite")
        with np.load(fold_npz, allow_pickle=False) as data:
            if str(data["contract_hash"]) != contract_hash:
                raise ValueError(f"outer fold {outer_fold} NPZ 合同哈希不一致")
            names = data["model_names"].astype(str).tolist()
            if names != ALL_MODELS:
                raise ValueError(f"outer fold {outer_fold} 模型顺序不一致")
            indices = np.asarray(
                [county_lookup[str(fips)] for fips in data["county_fips"]], dtype=int
            )
            if not np.all(folds[indices] == outer_fold):
                raise ValueError(f"outer fold {outer_fold} 县清单不一致")
            coverage[indices] += 1
            for model_idx, model_name in enumerate(ALL_MODELS):
                predictions[model_name][indices] = data["predictions"][model_idx]
        fold_metadata.append(metadata)
    if not np.all(coverage == 1):
        raise ValueError("严格 OOF 没有恰好覆盖每个县一次")
    target_mask = np.isfinite(bundle.targets)
    for model_name, values in predictions.items():
        if not np.isfinite(values[target_mask]).all():
            raise ValueError(f"{model_name} 严格 OOF 不完整")
        values[~target_mask] = np.nan

    np.savez_compressed(
        M4_PREDICTIONS,
        model_names=np.asarray(ALL_MODELS, dtype="U32"),
        predictions=np.stack([predictions[name] for name in ALL_MODELS], axis=0),
        county_fips=bundle.county_fips,
        outer_folds=folds,
        contract_hash=np.asarray(contract_hash),
    )

    summary, fold_metrics, county_metrics = build_evaluation_tables(
        bundle.targets, predictions, bundle.county_fips, folds
    )
    summary["evaluation_status"] = "strict_nested_cv"
    fold_metrics["evaluation_status"] = "strict_nested_cv"
    county_metrics["evaluation_status"] = "strict_nested_cv"
    summary.to_csv(M4_SUMMARY_METRICS, index=False)
    fold_metrics.to_csv(M4_FOLD_METRICS, index=False)
    county_metrics.to_csv(M4_COUNTY_METRICS, index=False)

    bootstrap = paired_county_bootstrap(
        bundle.targets, predictions, M4_BOOTSTRAP_REPLICATES, SEED + 4000
    )
    bootstrap["evaluation_status"] = "strict_nested_cv"
    bootstrap.to_csv(M4_BOOTSTRAP, index=False)

    target_phase = build_target_phase(X, bundle.row_index)
    time_metrics = time_and_phase_metrics(bundle.targets, predictions, target_phase)
    time_metrics["evaluation_status"] = "strict_nested_cv"
    time_metrics.to_csv(M4_TIME_METRICS, index=False)
    consistency = target_time_consistency(predictions)
    consistency["evaluation_status"] = "strict_nested_cv"
    consistency.to_csv(M4_CONSISTENCY, index=False)

    oof_frame = meta.reset_index(drop=True).copy()
    target_rows = county_matrix_to_rows(bundle.targets, bundle.row_index)
    for horizon_idx, horizon in enumerate(HORIZONS):
        oof_frame[f"target_{horizon}"] = target_rows[:, horizon_idx]
    for model_name, values in predictions.items():
        rows = county_matrix_to_rows(postprocess(values), bundle.row_index)
        for horizon_idx, horizon in enumerate(HORIZONS):
            oof_frame[f"{model_name}_{horizon}"] = rows[:, horizon_idx]
    oof_frame.to_parquet(M4_OOF_PARQUET, index=False)

    input_paths = [
        FEATURE_TRAIN, TARGET_TRAIN, META_TRAIN, CV_ASSIGNMENTS,
        TRAIN_SEQUENCE_CACHE,
    ]
    output_paths = [
        M4_PREDICTIONS, M4_OOF_PARQUET, M4_FOLD_METRICS,
        M4_SUMMARY_METRICS, M4_COUNTY_METRICS, M4_BOOTSTRAP,
        M4_TIME_METRICS, M4_CONSISTENCY,
    ]
    metadata = {
        "version": "v3.1",
        "milestone": "M4",
        "evaluation_status": "strict_nested_cv",
        "baseline_version": "v1.5.6",
        "cv_version": "balanced_v1",
        "seed": SEED,
        "device": "cpu",
        "torch_version": torch.__version__,
        "lightgbm_version": lgb.__version__,
        "contract_hash": contract_hash,
        "protocol": {
            "outer_folds": N_FOLDS,
            "lgbm_round_selection": "outer-train 内四折早停轮数的中位数",
            "residual_labels": "固定轮数、outer-train 内四折 LightGBM OOF 原始预测",
            "neural_epoch_selection": "outer-train 内固定一折，按各自训练损失早停",
            "alpha_selection_metric": "inner validation postprocessed RMSE",
            "alpha_grid": M4_ALPHA_GRID,
            "outer_validation_usage": "仅最终评分",
        },
        "training": {
            "max_epochs": max_epochs,
            "early_stopping_patience": M4_EARLY_STOPPING_PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size_counties": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "loss_aggregation": "先按 county/horizon 对有效时间求均值，再整体平均",
        },
        "models": {
            "B0_lightgbm": "严格外层 v1.5.6 LightGBM",
            "B1_target_aligned": "同一 county/target_hour 的 horizon 原始预测无权平均",
            **MODEL_VARIANTS,
            "B4_constant_mean": "outer-train OOF 残差均值，alpha 在内层验证折选择",
        },
        "folds": fold_metadata,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths
        },
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in output_paths
        },
        "elapsed_seconds": time.perf_counter() - run_start,
    }
    M4_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nM4 pooled strict OOF:")
    print(summary[[
        "model", "horizon", "rmse", "mae",
        "rmse_change_vs_B0_pct", "mae_change_vs_B0_pct",
    ]].to_string(index=False))
    print(f"M4 完成: {M4_SUMMARY_METRICS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=M4_MAX_EPOCHS)
    parser.add_argument("--outer-fold", type=int, choices=range(N_FOLDS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_epochs < 1:
        raise ValueError("max_epochs 必须为正整数")
    ensure_artifact_dirs()
    run_start = time.perf_counter()
    contract_hash = source_contract(args.max_epochs)
    seed_everything(SEED)
    device = torch.device("cpu")

    X = pd.read_parquet(FEATURE_TRAIN).reset_index(drop=True)
    target_frame = pd.read_parquet(TARGET_TRAIN).reset_index(drop=True)
    meta = pd.read_parquet(META_TRAIN).reset_index(drop=True)
    bundle = load_bundle(TRAIN_SEQUENCE_CACHE)
    row_folds = load_row_folds(meta)
    folds = county_folds(bundle.county_fips)
    if set(np.unique(folds).tolist()) != set(range(N_FOLDS)):
        raise ValueError("外层五折不完整")
    if not np.array_equal(row_folds[bundle.row_index[:, 0]], folds):
        raise ValueError("行级与县级折编号不一致")

    requested_folds = [args.outer_fold] if args.outer_fold is not None else list(range(N_FOLDS))
    for outer_fold in requested_folds:
        fold_npz = M4_FOLD_DIR / f"outer_fold{outer_fold}.npz"
        fold_json = M4_FOLD_DIR / f"outer_fold{outer_fold}.json"
        reusable = False
        if fold_npz.exists() and fold_json.exists() and not args.overwrite:
            metadata = json.loads(fold_json.read_text(encoding="utf-8"))
            reusable = (
                metadata.get("contract_hash") == contract_hash
                and metadata.get("fold_npz_sha256") == sha256_file(fold_npz)
            )
            if not reusable:
                raise ValueError(
                    f"outer fold {outer_fold} 已有产物与当前合同不一致，请使用 --overwrite"
                )
        if reusable:
            print(f"outer_fold={outer_fold}: 复用已验证的折产物", flush=True)
        else:
            print(f"outer_fold={outer_fold}: 开始严格嵌套训练", flush=True)
            run_outer_fold(
                outer_fold, contract_hash, args.max_epochs,
                X, target_frame, meta, row_folds, bundle, folds, device,
            )

    if args.outer_fold is None:
        aggregate_results(contract_hash, bundle, folds, X, meta, args.max_epochs, run_start)
    else:
        print(f"outer_fold={args.outer_fold} 已完成；全部五折完成后不带 --outer-fold 聚合。")


if __name__ == "__main__":
    main()
