"""v3.2 非负、和为一的统一目标小时融合模型。"""

from __future__ import annotations

import torch
from torch import nn

from config import FUTURE_HOURS, N_HORIZONS, N_TARGET_HOURS, OBSERVED_HOURS


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape:
        raise ValueError("logits 与 candidate mask 形状不一致")
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("至少一个目标小时没有可用候选")
    masked = logits.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked, dim=-1)
    return weights * mask.float()


def convex_prediction(
    logits: torch.Tensor,
    candidate_raw: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = masked_softmax(logits, candidate_mask)
    prediction = (weights * candidate_raw).sum(dim=-1)
    return prediction, weights


class StaticConvexModel(nn.Module):
    """按候选数量学习一组静态凸权重。"""

    architecture = "static"

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_HORIZONS, N_HORIZONS))
        self.model_config = {}

    def forward(
        self,
        observed: torch.Tensor,
        future_weather: torch.Tensor,
        sequence_extra: torch.Tensor,
        static: torch.Tensor,
        last_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del observed, future_weather, sequence_extra, static, last_state
        stage = candidate_mask.sum(dim=-1).long() - 1
        logits = self.logits[stage]
        return convex_prediction(logits, candidate_raw, candidate_mask)


class MLPConvexModel(nn.Module):
    """只读当前目标时刻输入的动态凸融合对照。"""

    architecture = "mlp"

    def __init__(
        self,
        weather_dim: int,
        extra_dim: int,
        static_dim: int,
        last_state_dim: int,
        static_hidden_size: int = 16,
        head_hidden_size: int = 32,
        dropout: float = 0.1,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.model_config = {
            "weather_dim": weather_dim,
            "extra_dim": extra_dim,
            "static_dim": static_dim,
            "last_state_dim": last_state_dim,
            "static_hidden_size": static_hidden_size,
            "head_hidden_size": head_hidden_size,
            "dropout": dropout,
        }
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, static_hidden_size), nn.SiLU(), nn.Dropout(dropout)
        )
        input_dim = weather_dim + extra_dim + static_hidden_size + last_state_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, N_HORIZONS),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        observed: torch.Tensor,
        future_weather: torch.Tensor,
        sequence_extra: torch.Tensor,
        static: torch.Tensor,
        last_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del observed
        static_state = self.static_net(static)
        repeated_static = static_state[:, None, :].expand(-1, N_TARGET_HOURS, -1)
        repeated_last = last_state[:, None, :].expand(-1, N_TARGET_HOURS, -1)
        inputs = torch.cat(
            [future_weather[:, 1:, :], sequence_extra[:, 1:, :], repeated_static, repeated_last],
            dim=-1,
        )
        logits = self.head(inputs)
        return convex_prediction(logits, candidate_raw, candidate_mask)


class GRUConvexModel(nn.Module):
    """观察期编码 + 未来逐小时编码的动态凸融合器。"""

    architecture = "gru"

    def __init__(
        self,
        observed_dim: int,
        weather_dim: int,
        extra_dim: int,
        static_dim: int,
        last_state_dim: int,
        hidden_size: int = 32,
        static_hidden_size: int = 16,
        head_hidden_size: int = 32,
        dropout: float = 0.1,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.model_config = {
            "observed_dim": observed_dim,
            "weather_dim": weather_dim,
            "extra_dim": extra_dim,
            "static_dim": static_dim,
            "last_state_dim": last_state_dim,
            "hidden_size": hidden_size,
            "static_hidden_size": static_hidden_size,
            "head_hidden_size": head_hidden_size,
            "dropout": dropout,
        }
        self.observed_dropout = nn.Dropout(dropout)
        self.future_dropout = nn.Dropout(dropout)
        self.observed_gru = nn.GRU(observed_dim, hidden_size, batch_first=True)
        self.future_init = nn.Linear(hidden_size, hidden_size)
        self.future_gru = nn.GRU(weather_dim + extra_dim, hidden_size, batch_first=True)
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, static_hidden_size), nn.SiLU(), nn.Dropout(dropout)
        )
        head_input = hidden_size * 2 + static_hidden_size + last_state_dim + extra_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, head_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, N_HORIZONS),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        observed: torch.Tensor,
        future_weather: torch.Tensor,
        sequence_extra: torch.Tensor,
        static: torch.Tensor,
        last_state: torch.Tensor,
        candidate_raw: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observed.shape[1] != OBSERVED_HOURS or future_weather.shape[1] != FUTURE_HOURS:
            raise ValueError("输入序列长度与 v3.2 契约不一致")
        _, observed_hidden = self.observed_gru(self.observed_dropout(observed))
        county_state = observed_hidden[-1]
        future_initial = torch.tanh(self.future_init(county_state)).unsqueeze(0)
        future_input = torch.cat([future_weather, sequence_extra], dim=-1)
        future_states, _ = self.future_gru(self.future_dropout(future_input), future_initial)
        target_states = future_states[:, 1:, :]
        static_state = self.static_net(static)
        repeated_county = county_state[:, None, :].expand(-1, N_TARGET_HOURS, -1)
        repeated_static = static_state[:, None, :].expand(-1, N_TARGET_HOURS, -1)
        repeated_last = last_state[:, None, :].expand(-1, N_TARGET_HOURS, -1)
        head_input = torch.cat(
            [target_states, repeated_county, repeated_static, repeated_last, sequence_extra[:, 1:, :]],
            dim=-1,
        )
        logits = self.head(head_input)
        return convex_prediction(logits, candidate_raw, candidate_mask)


def build_model(architecture: str, prepared: dict, **overrides) -> nn.Module:
    common = {
        "observed_dim": prepared["observed"].shape[-1],
        "weather_dim": prepared["future_weather"].shape[-1],
        "extra_dim": prepared["sequence_extra"].shape[-1],
        "static_dim": prepared["static"].shape[-1],
        "last_state_dim": prepared["last_state"].shape[-1],
    }
    common.update(overrides)
    if architecture == "static":
        return StaticConvexModel(**common)
    if architecture == "mlp":
        return MLPConvexModel(**common)
    if architecture == "gru":
        return GRUConvexModel(**common)
    raise ValueError(f"未知模型结构: {architecture}")


def restore_model(architecture: str, model_config: dict) -> nn.Module:
    if architecture == "static":
        return StaticConvexModel(**model_config)
    if architecture == "mlp":
        return MLPConvexModel(**model_config)
    if architecture == "gru":
        return GRUConvexModel(**model_config)
    raise ValueError(f"未知模型结构: {architecture}")
