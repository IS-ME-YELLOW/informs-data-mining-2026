"""v3.1 序列缓存加载、折内预处理和县级 batch 工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    HORIZONS,
    LAST_STATE_COLUMNS,
    LOG1P_CHANNELS,
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


def load_bundle(path: Path) -> SequenceBundle:
    with np.load(path, allow_pickle=False) as data:
        return SequenceBundle(**{key: data[key].copy() for key in data.files})


def rows_to_county_matrix(row_values: np.ndarray, row_index: np.ndarray) -> np.ndarray:
    values = np.asarray(row_values)
    if values.ndim != 2 or values.shape[1] != len(HORIZONS):
        raise ValueError("row_values 必须为 [rows, 4]")
    return values[row_index]


def county_matrix_to_rows(county_values: np.ndarray, row_index: np.ndarray) -> np.ndarray:
    values = np.asarray(county_values)
    if values.shape[:2] != row_index.shape or values.shape[2] != len(HORIZONS):
        raise ValueError("county_values 必须为 [county, 144, 4]")
    output = np.full((int(row_index.max()) + 1, len(HORIZONS)), np.nan, dtype=float)
    for county_idx in range(len(row_index)):
        output[row_index[county_idx]] = values[county_idx]
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


@dataclass
class FoldPreprocessor:
    observed_mean: np.ndarray
    observed_std: np.ndarray
    future_mean: np.ndarray
    future_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    last_mean: np.ndarray
    last_std: np.ndarray
    baseline_mean: np.ndarray
    baseline_std: np.ndarray
    residual_scale: np.ndarray
    rural_categories: np.ndarray

    @classmethod
    def fit(
        cls,
        bundle: SequenceBundle,
        baseline_raw: np.ndarray,
        county_indices: np.ndarray,
    ) -> "FoldPreprocessor":
        indices = np.asarray(county_indices, dtype=int)
        observed = _log_transform(bundle.observed[indices], OBSERVED_CHANNELS)
        future = _log_transform(bundle.future_weather[indices], WEATHER_CHANNELS)
        static = np.asarray(bundle.static_numeric[indices], dtype=np.float64)
        last = np.asarray(bundle.last_state[indices], dtype=np.float64)

        observed_mean, observed_std = _fit_stats(observed, (0, 1))
        future_mean, future_std = _fit_stats(future, (0, 1))
        static_mean, static_std = _fit_stats(static, (0,))
        last_mean, last_std = _fit_stats(last, (0,))

        baseline_subset = np.asarray(baseline_raw[indices], dtype=np.float64)
        target_subset = np.asarray(bundle.targets[indices], dtype=np.float64)
        baseline_mean = np.zeros(len(HORIZONS), dtype=np.float64)
        baseline_std = np.ones(len(HORIZONS), dtype=np.float64)
        residual_scale = np.ones(len(HORIZONS), dtype=np.float64)
        for horizon_idx in range(len(HORIZONS)):
            mask = np.isfinite(target_subset[:, :, horizon_idx]) & np.isfinite(
                baseline_subset[:, :, horizon_idx]
            )
            values = baseline_subset[:, :, horizon_idx][mask]
            residual = (
                target_subset[:, :, horizon_idx][mask]
                - baseline_subset[:, :, horizon_idx][mask]
            )
            if values.size == 0:
                raise ValueError(f"horizon {HORIZONS[horizon_idx]} 没有合法基线预测")
            baseline_mean[horizon_idx] = float(values.mean())
            baseline_std[horizon_idx] = max(float(values.std()), 1e-8)
            residual_scale[horizon_idx] = max(float(residual.std()), 1e-8)

        rural = np.asarray(bundle.static_category[indices], dtype=float)
        rural_categories = np.unique(rural[np.isfinite(rural)]).astype(np.float64)
        return cls(
            observed_mean=observed_mean,
            observed_std=observed_std,
            future_mean=future_mean,
            future_std=future_std,
            static_mean=static_mean,
            static_std=static_std,
            last_mean=last_mean,
            last_std=last_std,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            residual_scale=residual_scale,
            rural_categories=rural_categories,
        )

    def transform(self, bundle: SequenceBundle, baseline_raw: np.ndarray) -> dict[str, np.ndarray]:
        observed = _log_transform(bundle.observed, OBSERVED_CHANNELS)
        future = _log_transform(bundle.future_weather, WEATHER_CHANNELS)
        observed = _normalize_with_mask(observed, self.observed_mean, self.observed_std)
        future = _normalize_with_mask(future, self.future_mean, self.future_std)
        static_numeric = _normalize_with_mask(
            bundle.static_numeric, self.static_mean, self.static_std
        )
        last_state = _normalize_with_mask(bundle.last_state, self.last_mean, self.last_std)

        rural = np.asarray(bundle.static_category, dtype=float)
        rural_one_hot = np.zeros(
            (len(rural), len(self.rural_categories) + 1), dtype=np.float32
        )
        matched = np.zeros(len(rural), dtype=bool)
        for category_idx, category in enumerate(self.rural_categories):
            hit = np.isfinite(rural) & (rural == category)
            rural_one_hot[hit, category_idx] = 1.0
            matched |= hit
        rural_one_hot[~matched, -1] = 1.0
        static = np.concatenate([static_numeric, rural_one_hot], axis=1).astype(np.float32)

        baseline_raw = np.asarray(baseline_raw, dtype=np.float64)
        baseline_finite = np.isfinite(baseline_raw)
        baseline_scaled = np.where(
            baseline_finite,
            (baseline_raw - self.baseline_mean.reshape(1, 1, -1))
            / self.baseline_std.reshape(1, 1, -1),
            0.0,
        ).astype(np.float32)

        targets = np.asarray(bundle.targets, dtype=np.float64)
        valid_mask = np.isfinite(targets) & baseline_finite
        residual = np.where(valid_mask, targets - baseline_raw, 0.0)
        residual_std = (
            residual / self.residual_scale.reshape(1, 1, -1)
        ).astype(np.float32)

        return {
            "observed": observed,
            "future_weather": future,
            "static": static,
            "last_state": last_state,
            "baseline_scaled": baseline_scaled,
            "baseline_raw": baseline_raw.astype(np.float32),
            "targets": targets.astype(np.float32),
            "valid_mask": valid_mask,
            "residual_std": residual_std,
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
            "baseline_mean": self.baseline_mean.tolist(),
            "baseline_std": self.baseline_std.tolist(),
            "residual_scale": self.residual_scale.tolist(),
            "rural_categories": self.rural_categories.tolist(),
            "schema": {
                "observed_channels": OBSERVED_CHANNELS,
                "future_weather_channels": WEATHER_CHANNELS,
                "static_numeric_columns": STATIC_NUMERIC_COLUMNS,
                "last_state_columns": LAST_STATE_COLUMNS,
            },
        }

    @classmethod
    def from_dict(cls, values: dict) -> "FoldPreprocessor":
        keys = [
            "observed_mean", "observed_std", "future_mean", "future_std",
            "static_mean", "static_std", "last_mean", "last_std",
            "baseline_mean", "baseline_std", "residual_scale", "rural_categories",
        ]
        return cls(**{key: np.asarray(values[key], dtype=np.float64) for key in keys})

