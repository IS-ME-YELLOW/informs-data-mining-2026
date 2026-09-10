"""v3.2 序列加载、目标时刻对齐和折内预处理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    HORIZONS,
    LAST_STATE_COLUMNS,
    LOG1P_CHANNELS,
    N_HORIZONS,
    N_TARGET_HOURS,
    OBSERVED_CHANNELS,
    STATIC_NUMERIC_COLUMNS,
    WEATHER_CHANNELS,
)


@dataclass
class SequenceBundle:
    county_fips: np.ndarray
    timestamps: np.ndarray
    observed: np.ndarray
    future_weather: np.ndarray
    static_numeric: np.ndarray
    static_category: np.ndarray
    last_state: np.ndarray
    row_index: np.ndarray
    targets: np.ndarray


@dataclass
class TimelineBundle:
    county_fips: np.ndarray
    target_hours: np.ndarray
    target_timestamps: np.ndarray
    unique_targets: np.ndarray
    target_to_origin: np.ndarray
    candidate_mask: np.ndarray
    origin_horizon_to_target: np.ndarray
    row_index: np.ndarray


def load_sequence(path: Path) -> SequenceBundle:
    with np.load(path, allow_pickle=False) as data:
        return SequenceBundle(**{key: data[key].copy() for key in data.files})


def load_timeline(path: Path) -> TimelineBundle:
    with np.load(path, allow_pickle=False) as data:
        return TimelineBundle(**{key: data[key].copy() for key in data.files})


def rows_to_county_matrix(row_values: np.ndarray, row_index: np.ndarray) -> np.ndarray:
    values = np.asarray(row_values)
    if values.ndim != 2 or values.shape[1] != N_HORIZONS:
        raise ValueError("row_values 必须为 [rows,4]")
    return values[row_index]


def county_matrix_to_rows(county_values: np.ndarray, row_index: np.ndarray) -> np.ndarray:
    values = np.asarray(county_values)
    if values.shape != (*row_index.shape, N_HORIZONS):
        raise ValueError("county_values 必须为 [county,144,4]")
    output = np.full((int(row_index.max()) + 1, N_HORIZONS), np.nan, dtype=float)
    for county_idx in range(len(row_index)):
        output[row_index[county_idx]] = values[county_idx]
    return output


def align_candidates(baseline: np.ndarray, timeline: TimelineBundle) -> np.ndarray:
    """把 [county,origin,4] 原始预测对齐为 [county,target,4]。"""
    values = np.asarray(baseline, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (144, N_HORIZONS):
        raise ValueError("baseline 必须为 [county,144,4]")
    aligned = np.full((values.shape[0], N_TARGET_HOURS, N_HORIZONS), np.nan, dtype=np.float32)
    for target_idx in range(N_TARGET_HOURS):
        for horizon_idx in range(N_HORIZONS):
            origin_idx = int(timeline.target_to_origin[target_idx, horizon_idx])
            if origin_idx >= 0:
                candidate = values[:, origin_idx, horizon_idx]
                if not np.isfinite(candidate).all():
                    raise ValueError(
                        f"合法候选出现缺失: target={target_idx}, horizon={HORIZONS[horizon_idx]}"
                    )
                aligned[:, target_idx, horizon_idx] = candidate.astype(np.float32)
    expected = np.broadcast_to(timeline.candidate_mask[None, :, :], aligned.shape)
    if not np.array_equal(np.isfinite(aligned), expected):
        raise ValueError("候选有效结构与目标时刻映射不一致")
    return aligned


def equal_mean(candidates: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        result = np.nanmean(np.asarray(candidates, dtype=float), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("至少一个目标小时没有可用候选")
    return result.astype(np.float32)


def broadcast_unique(unique_values: np.ndarray, timeline: TimelineBundle) -> np.ndarray:
    """把唯一目标预测映射回 [county,origin,4]，相同目标复用同一数组值。"""
    values = np.asarray(unique_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != N_TARGET_HOURS:
        raise ValueError("unique_values 必须为 [county,143]")
    output = np.full((values.shape[0], 144, N_HORIZONS), np.nan, dtype=np.float32)
    for origin_idx in range(144):
        for horizon_idx in range(N_HORIZONS):
            target_idx = int(timeline.origin_horizon_to_target[origin_idx, horizon_idx])
            if target_idx >= 0:
                output[:, origin_idx, horizon_idx] = values[:, target_idx].astype(np.float32)
    return output


def _log_transform(values: np.ndarray, columns: list[str]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for idx, column in enumerate(columns):
        if column in LOG1P_CHANNELS:
            result[..., idx] = np.log1p(np.clip(result[..., idx], 0, None))
    return result


def _fit_stats(values: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=axes)
    std = np.nanstd(values, axis=axes)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean.astype(np.float64), std.astype(np.float64)


def _normalize_with_mask(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(values)
    filled = np.where(missing, mean, values)
    normalized = (filled - mean) / std
    return np.concatenate([normalized, missing.astype(float)], axis=-1).astype(np.float32)


def horizon_duplicate_weights() -> np.ndarray:
    """唯一目标轴上与四 horizon 等权 MSE 等价的固定权重。"""
    lengths = np.asarray([143, 138, 120, 96], dtype=np.float64)
    starts = np.asarray([0, 5, 23, 47], dtype=int)
    weights = np.zeros(N_TARGET_HOURS, dtype=np.float64)
    for start, length in zip(starts, lengths):
        weights[start:] += 1.0 / length
    weights /= weights.mean()
    return weights.astype(np.float32)


@dataclass
class FusionPreprocessor:
    observed_mean: np.ndarray
    observed_std: np.ndarray
    future_mean: np.ndarray
    future_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    last_mean: np.ndarray
    last_std: np.ndarray
    candidate_mean: np.ndarray
    candidate_std: np.ndarray
    summary_center: float
    summary_scale: float
    rural_categories: np.ndarray

    @classmethod
    def fit(
        cls,
        sequence: SequenceBundle,
        candidates: np.ndarray,
        county_indices: np.ndarray,
    ) -> "FusionPreprocessor":
        indices = np.asarray(county_indices, dtype=int)
        observed = _log_transform(sequence.observed[indices], OBSERVED_CHANNELS)
        future = _log_transform(sequence.future_weather[indices], WEATHER_CHANNELS)
        observed_mean, observed_std = _fit_stats(observed, (0, 1))
        future_mean, future_std = _fit_stats(future, (0, 1))
        static_mean, static_std = _fit_stats(sequence.static_numeric[indices], (0,))
        last_mean, last_std = _fit_stats(sequence.last_state[indices], (0,))

        subset = np.asarray(candidates[indices], dtype=np.float64)
        candidate_mean = np.zeros(N_HORIZONS, dtype=np.float64)
        candidate_std = np.ones(N_HORIZONS, dtype=np.float64)
        for horizon_idx in range(N_HORIZONS):
            valid = np.isfinite(subset[:, :, horizon_idx])
            values = subset[:, :, horizon_idx][valid]
            if values.size == 0:
                raise ValueError(f"候选 {HORIZONS[horizon_idx]} 没有有效训练值")
            candidate_mean[horizon_idx] = float(values.mean())
            candidate_std[horizon_idx] = max(float(values.std()), 1e-8)
        means = equal_mean(subset)
        summary_center = float(means.mean())
        summary_scale = max(float(means.std()), 1e-8)
        rural = np.asarray(sequence.static_category[indices], dtype=float)
        rural_categories = np.unique(rural[np.isfinite(rural)]).astype(np.float64)
        return cls(
            observed_mean, observed_std, future_mean, future_std,
            static_mean, static_std, last_mean, last_std,
            candidate_mean, candidate_std, summary_center, summary_scale,
            rural_categories,
        )

    def transform(
        self,
        sequence: SequenceBundle,
        timeline: TimelineBundle,
        candidates: np.ndarray,
    ) -> dict[str, np.ndarray]:
        observed = _normalize_with_mask(
            _log_transform(sequence.observed, OBSERVED_CHANNELS),
            self.observed_mean,
            self.observed_std,
        )
        future_weather = _normalize_with_mask(
            _log_transform(sequence.future_weather, WEATHER_CHANNELS),
            self.future_mean,
            self.future_std,
        )
        static_numeric = _normalize_with_mask(
            sequence.static_numeric, self.static_mean, self.static_std
        )
        last_state = _normalize_with_mask(sequence.last_state, self.last_mean, self.last_std)

        rural = np.asarray(sequence.static_category, dtype=float)
        rural_one_hot = np.zeros((len(rural), len(self.rural_categories) + 1), dtype=np.float32)
        matched = np.zeros(len(rural), dtype=bool)
        for category_idx, category in enumerate(self.rural_categories):
            hit = np.isfinite(rural) & (rural == category)
            rural_one_hot[hit, category_idx] = 1.0
            matched |= hit
        rural_one_hot[~matched, -1] = 1.0
        static = np.concatenate([static_numeric, rural_one_hot], axis=1).astype(np.float32)

        raw = np.asarray(candidates, dtype=np.float64)
        mask = np.isfinite(raw)
        expected_mask = np.broadcast_to(timeline.candidate_mask[None, :, :], mask.shape)
        if not np.array_equal(mask, expected_mask):
            raise ValueError("折内候选 mask 与时间轴契约不一致")
        scaled = np.where(
            mask,
            (raw - self.candidate_mean.reshape(1, 1, -1))
            / self.candidate_std.reshape(1, 1, -1),
            0.0,
        )
        mean_raw = equal_mean(raw).astype(np.float64)
        safe = np.where(mask, raw, np.nan)
        std_raw = np.nanstd(safe, axis=-1)
        range_raw = np.nanmax(safe, axis=-1) - np.nanmin(safe, axis=-1)
        count = mask.sum(axis=-1).astype(np.float64)
        summary = np.stack(
            [
                (mean_raw - self.summary_center) / self.summary_scale,
                std_raw / self.summary_scale,
                range_raw / self.summary_scale,
                count / N_HORIZONS,
            ],
            axis=-1,
        ).astype(np.float32)

        extra = np.zeros((len(raw), 144, N_HORIZONS * 2 + 4), dtype=np.float32)
        extra[:, 1:, :N_HORIZONS] = scaled.astype(np.float32)
        extra[:, 1:, N_HORIZONS:2 * N_HORIZONS] = mask.astype(np.float32)
        extra[:, 1:, 2 * N_HORIZONS:] = summary
        return {
            "observed": observed,
            "future_weather": future_weather,
            "sequence_extra": extra,
            "static": static,
            "last_state": last_state,
            "candidate_raw": np.where(mask, raw, 0.0).astype(np.float32),
            "candidate_mask": mask,
            "equal_raw": mean_raw.astype(np.float32),
            "targets": timeline.unique_targets.astype(np.float32),
            "loss_weights": horizon_duplicate_weights(),
        }

    def to_dict(self) -> dict:
        return {
            "observed_mean": self.observed_mean.tolist(),
            "observed_std": self.observed_std.tolist(),
            "future_mean": self.future_mean.tolist(),
            "future_std": self.future_std.tolist(),
            "static_mean": self.static_mean.tolist(),
            "static_std": self.static_std.tolist(),
            "last_mean": self.last_mean.tolist(),
            "last_std": self.last_std.tolist(),
            "candidate_mean": self.candidate_mean.tolist(),
            "candidate_std": self.candidate_std.tolist(),
            "summary_center": self.summary_center,
            "summary_scale": self.summary_scale,
            "rural_categories": self.rural_categories.tolist(),
            "schema": {
                "observed_channels": OBSERVED_CHANNELS,
                "future_weather_channels": WEATHER_CHANNELS,
                "static_numeric_columns": STATIC_NUMERIC_COLUMNS,
                "last_state_columns": LAST_STATE_COLUMNS,
            },
        }

    @classmethod
    def from_dict(cls, values: dict) -> "FusionPreprocessor":
        array_keys = [
            "observed_mean", "observed_std", "future_mean", "future_std",
            "static_mean", "static_std", "last_mean", "last_std",
            "candidate_mean", "candidate_std", "rural_categories",
        ]
        kwargs = {key: np.asarray(values[key], dtype=np.float64) for key in array_keys}
        kwargs["summary_center"] = float(values["summary_center"])
        kwargs["summary_scale"] = float(values["summary_scale"])
        return cls(**kwargs)
