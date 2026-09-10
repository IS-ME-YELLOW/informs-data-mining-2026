"""v3.1 双段 GRU 残差模型。"""

from __future__ import annotations

import torch
from torch import nn

from config import FUTURE_HOURS, HORIZON_HOURS, OBSERVED_HOURS


class GRUResidualModel(nn.Module):
    def __init__(
        self,
        observed_dim: int,
        weather_dim: int,
        static_dim: int,
        last_state_dim: int,
        hidden_size: int = 32,
        static_hidden_size: int = 16,
        head_hidden_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_config = {
            "observed_dim": observed_dim,
            "weather_dim": weather_dim,
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
        self.future_gru = nn.GRU(weather_dim, hidden_size, batch_first=True)
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, static_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        origin = torch.arange(FUTURE_HOURS, dtype=torch.long).view(-1, 1)
        horizon = torch.tensor(HORIZON_HOURS, dtype=torch.long).view(1, -1)
        target_index = origin + horizon
        valid = target_index < FUTURE_HOURS
        self.register_buffer("target_index", target_index.clamp(max=FUTURE_HOURS - 1))
        self.register_buffer("target_valid", valid)

        origin_norm = origin.float().expand(-1, len(HORIZON_HOURS)) / (FUTURE_HOURS - 1)
        target_norm = target_index.float().clamp(max=FUTURE_HOURS - 1) / (FUTURE_HOURS - 1)
        horizon_norm = horizon.float().expand(FUTURE_HOURS, -1) / max(HORIZON_HOURS)
        one_hot = torch.eye(len(HORIZON_HOURS)).view(1, len(HORIZON_HOURS), -1)
        one_hot = one_hot.expand(FUTURE_HOURS, -1, -1)
        time_features = torch.cat(
            [
                origin_norm.unsqueeze(-1),
                target_norm.unsqueeze(-1),
                horizon_norm.unsqueeze(-1),
                one_hot,
            ],
            dim=-1,
        )
        self.register_buffer("time_features", time_features)

        head_input_dim = (
            hidden_size * 2
            + static_hidden_size
            + last_state_dim
            + 1
            + time_features.shape[-1]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_input_dim, head_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_size, 1),
                )
                for _ in HORIZON_HOURS
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        observed: torch.Tensor,
        future_weather: torch.Tensor,
        static: torch.Tensor,
        last_state: torch.Tensor,
        baseline_scaled: torch.Tensor,
    ) -> torch.Tensor:
        if observed.shape[1] != OBSERVED_HOURS or future_weather.shape[1] != FUTURE_HOURS:
            raise ValueError("时序长度与 v3.1 契约不一致")
        _, observed_hidden = self.observed_gru(self.observed_dropout(observed))
        county_state = observed_hidden[-1]
        future_initial = torch.tanh(self.future_init(county_state)).unsqueeze(0)
        future_states, _ = self.future_gru(
            self.future_dropout(future_weather), future_initial
        )
        static_state = self.static_net(static)

        batch_size = observed.shape[0]
        flat_index = self.target_index.reshape(-1)
        selected_future = future_states[:, flat_index, :].reshape(
            batch_size, FUTURE_HOURS, len(HORIZON_HOURS), -1
        )
        county_repeated = county_state[:, None, None, :].expand(
            -1, FUTURE_HOURS, len(HORIZON_HOURS), -1
        )
        static_repeated = static_state[:, None, None, :].expand(
            -1, FUTURE_HOURS, len(HORIZON_HOURS), -1
        )
        last_repeated = last_state[:, None, None, :].expand(
            -1, FUTURE_HOURS, len(HORIZON_HOURS), -1
        )
        time_repeated = self.time_features[None, :, :, :].expand(batch_size, -1, -1, -1)

        outputs = []
        for horizon_idx, head in enumerate(self.heads):
            head_input = torch.cat(
                [
                    county_repeated[:, :, horizon_idx, :],
                    selected_future[:, :, horizon_idx, :],
                    static_repeated[:, :, horizon_idx, :],
                    last_repeated[:, :, horizon_idx, :],
                    baseline_scaled[:, :, horizon_idx].unsqueeze(-1),
                    time_repeated[:, :, horizon_idx, :],
                ],
                dim=-1,
            )
            outputs.append(head(head_input).squeeze(-1))
        return torch.stack(outputs, dim=-1)


class MLPResidualModel(nn.Module):
    """不读取逐小时时序的残差对照，只使用静态、时间和基线输入。"""

    def __init__(
        self,
        observed_dim: int,
        weather_dim: int,
        static_dim: int,
        last_state_dim: int,
        static_hidden_size: int = 16,
        head_hidden_size: int = 32,
        dropout: float = 0.1,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.model_config = {
            "observed_dim": observed_dim,
            "weather_dim": weather_dim,
            "static_dim": static_dim,
            "last_state_dim": last_state_dim,
            "static_hidden_size": static_hidden_size,
            "head_hidden_size": head_hidden_size,
            "dropout": dropout,
            "hidden_size": hidden_size,
        }
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, static_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        origin = torch.arange(FUTURE_HOURS, dtype=torch.long).view(-1, 1)
        horizon = torch.tensor(HORIZON_HOURS, dtype=torch.long).view(1, -1)
        target_index = origin + horizon
        valid = target_index < FUTURE_HOURS
        self.register_buffer("target_valid", valid)
        origin_norm = origin.float().expand(-1, len(HORIZON_HOURS)) / (FUTURE_HOURS - 1)
        target_norm = target_index.float().clamp(max=FUTURE_HOURS - 1) / (FUTURE_HOURS - 1)
        horizon_norm = horizon.float().expand(FUTURE_HOURS, -1) / max(HORIZON_HOURS)
        one_hot = torch.eye(len(HORIZON_HOURS)).view(1, len(HORIZON_HOURS), -1)
        one_hot = one_hot.expand(FUTURE_HOURS, -1, -1)
        time_features = torch.cat(
            [origin_norm.unsqueeze(-1), target_norm.unsqueeze(-1),
             horizon_norm.unsqueeze(-1), one_hot], dim=-1
        )
        self.register_buffer("time_features", time_features)
        head_input_dim = static_hidden_size + last_state_dim + 1 + time_features.shape[-1]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_input_dim, head_hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_size, 1),
            )
            for _ in HORIZON_HOURS
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        observed: torch.Tensor,
        future_weather: torch.Tensor,
        static: torch.Tensor,
        last_state: torch.Tensor,
        baseline_scaled: torch.Tensor,
    ) -> torch.Tensor:
        del future_weather
        batch_size = observed.shape[0]
        static_state = self.static_net(static)
        static_repeated = static_state[:, None, :].expand(-1, FUTURE_HOURS, -1)
        last_repeated = last_state[:, None, :].expand(-1, FUTURE_HOURS, -1)
        time_repeated = self.time_features[None, :, :, :].expand(batch_size, -1, -1, -1)
        outputs = []
        for horizon_idx, head in enumerate(self.heads):
            head_input = torch.cat([
                static_repeated,
                last_repeated,
                baseline_scaled[:, :, horizon_idx].unsqueeze(-1),
                time_repeated[:, :, horizon_idx, :],
            ], dim=-1)
            outputs.append(head(head_input).squeeze(-1))
        return torch.stack(outputs, dim=-1)


def build_model_from_prepared(prepared: dict, **overrides) -> GRUResidualModel:
    settings = {
        "observed_dim": prepared["observed"].shape[-1],
        "weather_dim": prepared["future_weather"].shape[-1],
        "static_dim": prepared["static"].shape[-1],
        "last_state_dim": prepared["last_state"].shape[-1],
    }
    settings.update(overrides)
    return GRUResidualModel(**settings)


def build_mlp_from_prepared(prepared: dict, **overrides) -> MLPResidualModel:
    settings = {
        "observed_dim": prepared["observed"].shape[-1],
        "weather_dim": prepared["future_weather"].shape[-1],
        "static_dim": prepared["static"].shape[-1],
        "last_state_dim": prepared["last_state"].shape[-1],
    }
    settings.update(overrides)
    return MLPResidualModel(**settings)
