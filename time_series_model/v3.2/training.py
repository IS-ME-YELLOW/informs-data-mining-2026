"""v3.2 融合模型共用训练、选择和预测函数。"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from config import (
    ALPHA_GRID,
    BATCH_SIZE,
    DROPOUT,
    GRADIENT_CLIP,
    GRU_HIDDEN_SIZE,
    HEAD_HIDDEN_SIZE,
    LEARNING_RATE,
    STATIC_HIDDEN_SIZE,
    WEIGHT_DECAY,
)
from dataset import FusionPreprocessor, SequenceBundle, TimelineBundle
from evaluate import apply_alpha, primary_score
from model import build_model
from optim import LocalAdamW


MODEL_VARIANTS = {
    "B2_static_convex": {
        "architecture": "static",
        "use_future_weather": False,
        "purpose": "按候选数量学习静态凸权重",
    },
    "B3_mlp_convex": {
        "architecture": "mlp",
        "use_future_weather": True,
        "purpose": "逐目标小时 MLP 动态凸融合",
    },
    "B4_gru_no_future": {
        "architecture": "gru",
        "use_future_weather": False,
        "purpose": "不读取未来天气的 GRU 动态凸融合",
    },
    "B5_gru_future": {
        "architecture": "gru",
        "use_future_weather": True,
        "purpose": "完整 GRU 动态凸融合",
    },
}
UNIQUE_MODELS = ["B1_equal_mean", *MODEL_VARIANTS.keys()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def build_variant_model(variant: dict, prepared: dict) -> torch.nn.Module:
    return build_model(
        variant["architecture"],
        prepared,
        hidden_size=GRU_HIDDEN_SIZE,
        static_hidden_size=STATIC_HIDDEN_SIZE,
        head_hidden_size=HEAD_HIDDEN_SIZE,
        dropout=DROPOUT,
    )


def make_batch(prepared: dict, indices: np.ndarray, device: torch.device) -> dict:
    result = {}
    for key in [
        "observed", "future_weather", "sequence_extra", "static", "last_state",
        "candidate_raw", "candidate_mask", "equal_raw", "targets",
    ]:
        tensor = torch.from_numpy(prepared[key][indices])
        tensor = tensor.bool() if key == "candidate_mask" else tensor.float()
        result[key] = tensor.to(device)
    result["loss_weights"] = torch.from_numpy(prepared["loss_weights"]).float().to(device)
    return result


def forward_variant(
    model: torch.nn.Module,
    batch: dict,
    use_future_weather: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    future = batch["future_weather"]
    if not use_future_weather:
        future = torch.zeros_like(future)
    return model(
        batch["observed"],
        future,
        batch["sequence_extra"],
        batch["static"],
        batch["last_state"],
        batch["candidate_raw"],
        batch["candidate_mask"],
    )


def horizon_balanced_mse(prediction: torch.Tensor, batch: dict) -> torch.Tensor:
    error_squared = (prediction - batch["targets"]) ** 2
    weights = batch["loss_weights"].view(1, -1)
    per_county = (error_squared * weights).sum(dim=1) / weights.sum()
    return per_county.mean()


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
    total_loss = 0.0
    total_norm = 0.0
    total_items = 0
    for start in range(0, len(shuffled), BATCH_SIZE):
        batch_indices = shuffled[start:start + BATCH_SIZE]
        batch = make_batch(prepared, batch_indices, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = forward_variant(model, batch, variant["use_future_weather"])
        loss = horizon_balanced_mse(prediction, batch)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(batch_indices)
        total_norm += float(gradient_norm.detach().cpu()) * len(batch_indices)
        total_items += len(batch_indices)
    return total_loss / total_items, total_norm / total_items


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
        prediction, _ = forward_variant(model, batch, variant["use_future_weather"])
        loss = horizon_balanced_mse(prediction, batch)
        total += float(loss.cpu()) * len(batch_indices)
        count += len(batch_indices)
    return total / count


@torch.no_grad()
def predict_dynamic(
    model: torch.nn.Module,
    prepared: dict,
    indices: np.ndarray,
    variant: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    weights = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch = make_batch(prepared, batch_indices, device)
        prediction, weight = forward_variant(model, batch, variant["use_future_weather"])
        predictions.append(prediction.cpu().numpy())
        weights.append(weight.cpu().numpy())
    prediction_array = np.concatenate(predictions, axis=0).astype(np.float32)
    weight_array = np.concatenate(weights, axis=0).astype(np.float32)
    validate_weights(weight_array, prepared["candidate_mask"][indices])
    return prediction_array, weight_array


def validate_weights(weights: np.ndarray, mask: np.ndarray, atol: float = 1e-6) -> None:
    values = np.asarray(weights, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("融合权重与候选 mask 形状不一致")
    if np.max(np.abs(values[~valid]), initial=0.0) > atol:
        raise ValueError("不可用候选获得了非零权重")
    np.testing.assert_allclose(values.sum(axis=-1), 1.0, rtol=0, atol=atol)
    if np.min(values) < -atol:
        raise ValueError("融合权重出现负数")


def choose_alpha(
    source_targets: np.ndarray,
    equal_raw: np.ndarray,
    dynamic_raw: np.ndarray,
    timeline: TimelineBundle,
) -> tuple[float, list[dict]]:
    records = []
    best_alpha = 0.0
    best_score = float("inf")
    for alpha in ALPHA_GRID:
        combined = apply_alpha(equal_raw, dynamic_raw, alpha)
        score = primary_score(source_targets, combined, timeline)
        records.append({"alpha": alpha, "primary_score": score})
        if score < best_score - 1e-12:
            best_score = score
            best_alpha = alpha
    return best_alpha, records


def select_epoch_and_alpha(
    sequence: SequenceBundle,
    timeline: TimelineBundle,
    candidates: np.ndarray,
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    variant: dict,
    max_epochs: int,
    patience_limit: int,
    seed: int,
    device: torch.device,
) -> tuple[int, float, list[dict], list[dict]]:
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
        if patience >= patience_limit:
            break
    if best_state is None:
        raise RuntimeError("没有保存最佳模型状态")
    model.load_state_dict(best_state)
    dynamic, _ = predict_dynamic(model, prepared, validation_indices, variant, device)
    alpha, alpha_records = choose_alpha(
        sequence.targets[validation_indices],
        prepared["equal_raw"][validation_indices],
        dynamic,
        timeline,
    )
    return best_epoch, alpha, history, alpha_records


def fit_fixed_epochs(
    sequence: SequenceBundle,
    timeline: TimelineBundle,
    candidates: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    variant: dict,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, FusionPreprocessor, np.ndarray, np.ndarray, list[dict]]:
    seed_everything(seed)
    preprocessor = FusionPreprocessor.fit(sequence, candidates, train_indices)
    prepared = preprocessor.transform(sequence, timeline, candidates)
    model = build_variant_model(variant, prepared).to(device)
    optimizer = LocalAdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss, gradient_norm = train_epoch(
            model, prepared, train_indices, optimizer, variant, device, rng
        )
        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "gradient_norm_before_clip": gradient_norm,
        })
    prediction, weights = predict_dynamic(
        model, prepared, prediction_indices, variant, device
    )
    return model, preprocessor, prediction, weights, history
