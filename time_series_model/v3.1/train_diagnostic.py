"""运行 v3.1 M3 诊断闭环；结果不得作为正式模型收益。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import (
    ALPHA_BY_HORIZON,
    BASELINE_PREDICTIONS,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CV_ASSIGNMENTS,
    DIAGNOSTIC_PREDICTIONS,
    EARLY_STOPPING_PATIENCE,
    FOLD_METRICS,
    GRADIENT_CLIP,
    GRU_DROPOUT,
    GRU_HIDDEN_SIZE,
    HEAD_HIDDEN_SIZE,
    HORIZONS,
    HORIZON_HOURS,
    LEARNING_RATE,
    MAX_EPOCHS_DIAGNOSTIC,
    META_TEST,
    META_TRAIN,
    N_FOLDS,
    OOF_PREDICTIONS,
    RUN_METADATA,
    SEED,
    SEQUENCE_MANIFEST,
    STATIC_HIDDEN_SIZE,
    SUBMISSION_OUTPUT,
    SUBMISSION_TEMPLATE,
    SUMMARY_METRICS,
    TEST_PREDICTIONS,
    TEST_SEQUENCE_CACHE,
    TRAIN_SEQUENCE_CACHE,
    WEIGHT_DECAY,
    ensure_artifact_dirs,
)
from dataset import FoldPreprocessor, load_bundle, rows_to_county_matrix
from evaluate import build_submission, metrics, postprocess, prediction_frame, summarize_predictions
from model import GRUResidualModel, build_model_from_prepared


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
    # v3.1 固定在 CPU 上运行；CPU GRU 配合固定线程数和随机种子可重复。
    # 不调用 torch.use_deterministic_algorithms，避免其在当前精简运行环境中
    # 仅为配置检查而强制导入训练本身不需要的 sympy/inductor 组件。


class LocalAdamW:
    """精简的标准 AdamW，避免当前 PyTorch 优化器导入可选编译依赖。"""

    def __init__(
        self,
        parameters,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_number = 0
        self.first_moment = [torch.zeros_like(parameter) for parameter in self.parameters]
        self.second_moment = [torch.zeros_like(parameter) for parameter in self.parameters]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_number += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_number
        bias_correction2 = 1.0 - self.beta2 ** self.step_number
        step_size = self.lr / bias_correction1
        correction = bias_correction2 ** 0.5
        for parameter, first, second in zip(
            self.parameters, self.first_moment, self.second_moment
        ):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            if self.weight_decay:
                parameter.mul_(1.0 - self.lr * self.weight_decay)
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            denominator = second.sqrt().div_(correction).add_(self.eps)
            parameter.addcdiv_(first, denominator, value=-step_size)


def normalize_fips(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def county_folds(county_fips: np.ndarray) -> np.ndarray:
    assignment = pd.read_csv(CV_ASSIGNMENTS, dtype={"fipsCode": str})
    assignment["fipsCode"] = normalize_fips(assignment["fipsCode"])
    fold_map = assignment.set_index("fipsCode")["fold"]
    folds = pd.Series(county_fips.astype(str)).map(fold_map)
    if folds.isna().any():
        raise ValueError("训练序列中至少一个县缺少折分配")
    return folds.to_numpy(dtype=int)


def to_device_batch(prepared: dict, indices: np.ndarray, device: torch.device) -> dict:
    result = {}
    for key in [
        "observed", "future_weather", "static", "last_state",
        "baseline_scaled", "residual_std", "valid_mask",
    ]:
        values = prepared[key][indices]
        tensor = torch.from_numpy(values)
        if key == "valid_mask":
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        result[key] = tensor.to(device)
    return result


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for horizon_idx in range(len(HORIZONS)):
        horizon_mask = mask[:, :, horizon_idx]
        if horizon_mask.any():
            losses.append(
                F.smooth_l1_loss(
                    prediction[:, :, horizon_idx][horizon_mask],
                    target[:, :, horizon_idx][horizon_mask],
                    reduction="mean",
                    beta=1.0,
                )
            )
    if not losses:
        raise ValueError("当前 batch 没有有效残差标签")
    return torch.stack(losses).mean()


def train_epoch(
    model: GRUResidualModel,
    prepared: dict,
    indices: np.ndarray,
    optimizer: LocalAdamW,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.train()
    shuffled = np.asarray(indices, dtype=int).copy()
    rng.shuffle(shuffled)
    losses = []
    for start in range(0, len(shuffled), BATCH_SIZE):
        batch_indices = shuffled[start:start + BATCH_SIZE]
        batch = to_device_batch(prepared, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch["observed"], batch["future_weather"], batch["static"],
            batch["last_state"], batch["baseline_scaled"],
        )
        loss = masked_huber(prediction, batch["residual_std"], batch["valid_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_loss(
    model: GRUResidualModel,
    prepared: dict,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = to_device_batch(prepared, batch_indices, device)
        prediction = model(
            batch["observed"], batch["future_weather"], batch["static"],
            batch["last_state"], batch["baseline_scaled"],
        )
        loss = masked_huber(prediction, batch["residual_std"], batch["valid_mask"])
        losses.append(float(loss.cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict_standardized(
    model: GRUResidualModel,
    prepared: dict,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = to_device_batch(prepared, batch_indices, device)
        prediction = model(
            batch["observed"], batch["future_weather"], batch["static"],
            batch["last_state"], batch["baseline_scaled"],
        )
        outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def select_epoch(
    bundle,
    baseline_raw: np.ndarray,
    fit_indices: np.ndarray,
    early_indices: np.ndarray,
    max_epochs: int,
    fold_seed: int,
    device: torch.device,
) -> tuple[int, list[dict]]:
    preprocessor = FoldPreprocessor.fit(bundle, baseline_raw, fit_indices)
    prepared = preprocessor.transform(bundle, baseline_raw)
    model = build_model_from_prepared(
        prepared,
        hidden_size=GRU_HIDDEN_SIZE,
        static_hidden_size=STATIC_HIDDEN_SIZE,
        head_hidden_size=HEAD_HIDDEN_SIZE,
        dropout=GRU_DROPOUT,
    ).to(device)
    optimizer = LocalAdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(fold_seed)
    best_epoch = 1
    best_loss = float("inf")
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, prepared, fit_indices, optimizer, device, rng)
        valid_loss = evaluate_loss(model, prepared, early_indices, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            break
    return best_epoch, history


def train_fixed_epochs(
    bundle,
    baseline_raw: np.ndarray,
    train_indices: np.ndarray,
    epochs: int,
    fold_seed: int,
    device: torch.device,
) -> tuple[GRUResidualModel, FoldPreprocessor, list[float]]:
    seed_everything(fold_seed)
    preprocessor = FoldPreprocessor.fit(bundle, baseline_raw, train_indices)
    prepared = preprocessor.transform(bundle, baseline_raw)
    model = build_model_from_prepared(
        prepared,
        hidden_size=GRU_HIDDEN_SIZE,
        static_hidden_size=STATIC_HIDDEN_SIZE,
        head_hidden_size=HEAD_HIDDEN_SIZE,
        dropout=GRU_DROPOUT,
    ).to(device)
    optimizer = LocalAdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = np.random.default_rng(fold_seed)
    losses = []
    for _epoch in range(1, epochs + 1):
        losses.append(train_epoch(model, prepared, train_indices, optimizer, device, rng))
    return model, preprocessor, losses


def save_checkpoint(
    path: Path,
    model: GRUResidualModel,
    preprocessor: FoldPreprocessor,
    fold: int,
    best_epoch: int,
    train_counties: list[str],
    validation_counties: list[str],
) -> None:
    payload = {
        "version": "v3.1",
        "evaluation_status": "diagnostic_only",
        "fold": fold,
        "best_epoch": best_epoch,
        "model_config": model.model_config,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "preprocessor": preprocessor.to_dict(),
        "train_counties": train_counties,
        "validation_counties": validation_counties,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GRUResidualModel, FoldPreprocessor, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = GRUResidualModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    preprocessor = FoldPreprocessor.from_dict(payload["preprocessor"])
    return model, preprocessor, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS_DIAGNOSTIC)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ensure_artifact_dirs()
    if DIAGNOSTIC_PREDICTIONS.exists() and not args.overwrite:
        print(f"已存在，跳过训练: {DIAGNOSTIC_PREDICTIONS}")
        return
    if args.max_epochs < 1:
        raise ValueError("max_epochs 必须为正整数")

    seed_everything(SEED)
    device = torch.device("cpu")
    train_bundle = load_bundle(TRAIN_SEQUENCE_CACHE)
    test_bundle = load_bundle(TEST_SEQUENCE_CACHE)
    with np.load(BASELINE_PREDICTIONS, allow_pickle=False) as baseline_data:
        baseline_oof_rows = baseline_data["oof_raw"].copy()
        baseline_test_rows = baseline_data["test_raw"].copy()
    baseline_train = rows_to_county_matrix(baseline_oof_rows, train_bundle.row_index)
    baseline_test = rows_to_county_matrix(baseline_test_rows, test_bundle.row_index)
    folds = county_folds(train_bundle.county_fips)

    valid_structure = (
        np.arange(144).reshape(-1, 1) + np.asarray(HORIZON_HOURS).reshape(1, -1) < 144
    )
    oof_delta = np.full_like(train_bundle.targets, np.nan, dtype=np.float32)
    test_delta_by_fold = []
    fold_records = []
    histories = {}

    for outer_fold in range(N_FOLDS):
        outer_train = np.flatnonzero(folds != outer_fold)
        outer_valid = np.flatnonzero(folds == outer_fold)
        early_fold = (outer_fold + 1) % N_FOLDS
        early_indices = np.flatnonzero(folds == early_fold)
        fit_indices = np.flatnonzero((folds != outer_fold) & (folds != early_fold))
        fold_seed = SEED + outer_fold
        seed_everything(fold_seed)
        best_epoch, history = select_epoch(
            train_bundle,
            baseline_train,
            fit_indices,
            early_indices,
            args.max_epochs,
            fold_seed,
            device,
        )
        histories[str(outer_fold)] = history
        model, preprocessor, train_losses = train_fixed_epochs(
            train_bundle,
            baseline_train,
            outer_train,
            best_epoch,
            fold_seed,
            device,
        )
        prepared_train = preprocessor.transform(train_bundle, baseline_train)
        prepared_test = preprocessor.transform(test_bundle, baseline_test)
        val_delta_std = predict_standardized(model, prepared_train, outer_valid, device)
        test_delta_std = predict_standardized(
            model, prepared_test, np.arange(len(test_bundle.county_fips)), device
        )
        scale = preprocessor.residual_scale.reshape(1, 1, -1)
        val_delta = val_delta_std * scale
        test_delta = test_delta_std * scale
        val_delta[:, ~valid_structure] = np.nan
        test_delta[:, ~valid_structure] = np.nan
        oof_delta[outer_valid] = val_delta.astype(np.float32)
        test_delta_by_fold.append(test_delta.astype(np.float32))

        checkpoint_path = CHECKPOINT_DIR / f"gru_residual_fold{outer_fold}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            preprocessor,
            outer_fold,
            best_epoch,
            train_bundle.county_fips[outer_train].astype(str).tolist(),
            train_bundle.county_fips[outer_valid].astype(str).tolist(),
        )
        reloaded_model, reloaded_preprocessor, _ = load_checkpoint(checkpoint_path, device)
        reloaded_prepared = reloaded_preprocessor.transform(train_bundle, baseline_train)
        reloaded_val = predict_standardized(
            reloaded_model, reloaded_prepared, outer_valid, device
        )
        if not np.allclose(val_delta_std, reloaded_val, rtol=0, atol=1e-7):
            raise ValueError(f"fold {outer_fold} 模型重载预测不一致")

        baseline_val = baseline_train[outer_valid]
        combined_val = baseline_val + np.asarray(ALPHA_BY_HORIZON).reshape(1, 1, -1) * val_delta
        for horizon_idx, horizon in enumerate(HORIZONS):
            y = train_bundle.targets[outer_valid, :, horizon_idx]
            baseline_metric = metrics(y, postprocess(baseline_val[:, :, horizon_idx]))
            combined_metric = metrics(y, postprocess(combined_val[:, :, horizon_idx]))
            fold_records.append({
                "fold": outer_fold,
                "early_stopping_fold": early_fold,
                "best_epoch": best_epoch,
                "horizon": horizon,
                "baseline_rmse": baseline_metric["rmse"],
                "baseline_mae": baseline_metric["mae"],
                "combined_rmse": combined_metric["rmse"],
                "combined_mae": combined_metric["mae"],
                "rmse_change_pct": 100 * (combined_metric["rmse"] / baseline_metric["rmse"] - 1),
                "mae_change_pct": 100 * (combined_metric["mae"] / baseline_metric["mae"] - 1),
                "last_train_loss": train_losses[-1],
            })
        print(f"outer_fold={outer_fold}: best_epoch={best_epoch}, checkpoint={checkpoint_path.name}")

    if not np.isfinite(oof_delta[np.isfinite(train_bundle.targets)]).all():
        raise ValueError("GRU OOF 残差预测不完整")

    stacked_test_delta = np.stack(test_delta_by_fold, axis=0)
    test_delta = np.full_like(stacked_test_delta[0], np.nan, dtype=np.float32)
    test_delta[:, valid_structure] = stacked_test_delta[:, :, valid_structure].mean(axis=0)
    alpha = np.asarray(ALPHA_BY_HORIZON, dtype=float).reshape(1, 1, -1)
    combined_oof = baseline_train + alpha * oof_delta
    combined_test = baseline_test + alpha * test_delta
    combined_oof[~np.isfinite(train_bundle.targets)] = np.nan
    combined_test[:, ~valid_structure] = np.nan

    np.savez_compressed(
        DIAGNOSTIC_PREDICTIONS,
        baseline_oof_raw=baseline_train,
        gru_oof_delta=oof_delta,
        combined_oof_raw=combined_oof,
        baseline_test_raw=baseline_test,
        gru_test_delta_by_fold=np.stack(test_delta_by_fold, axis=0),
        gru_test_delta=test_delta,
        combined_test_raw=combined_test,
        alpha=np.asarray(ALPHA_BY_HORIZON, dtype=np.float32),
        horizons=np.asarray(HORIZONS, dtype="U32"),
    )

    meta_train = pd.read_parquet(META_TRAIN)
    meta_test = pd.read_parquet(META_TEST)
    oof_frame = prediction_frame(
        meta_train, train_bundle.row_index, train_bundle.targets,
        baseline_train, oof_delta, combined_oof,
    )
    test_frame = prediction_frame(
        meta_test, test_bundle.row_index, test_bundle.targets,
        baseline_test, test_delta, combined_test,
    )
    oof_frame.to_parquet(OOF_PREDICTIONS, index=False)
    test_frame.to_parquet(TEST_PREDICTIONS, index=False)

    summary = summarize_predictions(train_bundle.targets, baseline_train, combined_oof)
    summary["evaluation_status"] = "diagnostic_only"
    summary.to_csv(SUMMARY_METRICS, index=False)
    pd.DataFrame(fold_records).to_csv(FOLD_METRICS, index=False)

    template = pd.read_csv(SUBMISSION_TEMPLATE, dtype={"fipsCode": str})
    submission = build_submission(
        template, meta_test, test_bundle.row_index, postprocess(combined_test)
    )
    submission.to_csv(SUBMISSION_OUTPUT, index=False)

    metadata = {
        "version": "v3.1",
        "milestone": "M3",
        "evaluation_status": "diagnostic_only",
        "warning": "本结果使用开发级交叉拟合，不得作为正式模型收益；M4需要嵌套CV。",
        "baseline_version": "v1.5.6",
        "cv_version": "balanced_v1",
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": {
            "hidden_size": GRU_HIDDEN_SIZE,
            "static_hidden_size": STATIC_HIDDEN_SIZE,
            "head_hidden_size": HEAD_HIDDEN_SIZE,
            "dropout": GRU_DROPOUT,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size_counties": BATCH_SIZE,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "gradient_clip": GRADIENT_CLIP,
            "alpha_by_horizon": dict(zip(HORIZONS, ALPHA_BY_HORIZON)),
        },
        "fold_histories": histories,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [TRAIN_SEQUENCE_CACHE, TEST_SEQUENCE_CACHE, SEQUENCE_MANIFEST, BASELINE_PREDICTIONS]
        },
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                DIAGNOSTIC_PREDICTIONS, OOF_PREDICTIONS, TEST_PREDICTIONS,
                FOLD_METRICS, SUMMARY_METRICS, SUBMISSION_OUTPUT,
            ]
        },
    }
    RUN_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.loc[summary["stage"].isin(["baseline_postprocessed", "combined_postprocessed"])].to_string(index=False))
    print(f"M3 诊断闭环完成: {SUBMISSION_OUTPUT}")


if __name__ == "__main__":
    main()
