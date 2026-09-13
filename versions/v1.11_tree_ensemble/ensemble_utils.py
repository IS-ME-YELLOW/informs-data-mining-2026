"""Shared deterministic utilities for the v1.11/v2.6 tree ensembles."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

HORIZONS = ("osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h")
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}
SEED = 42
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 2000
KEYS = ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fipsCode"] = (
        result["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    result["timestamp_et"] = pd.to_datetime(result["timestamp_et"])
    result["hour_idx"] = pd.to_numeric(result["hour_idx"], errors="raise").astype(int)
    result["stateAbbr"] = result["stateAbbr"].astype(str)
    if result[KEYS[:2]].duplicated().any():
        raise ValueError("Duplicate county/timestamp keys found.")
    return result


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return normalize_frame(pd.read_parquet(path))
    return normalize_frame(pd.read_csv(path, dtype={"fipsCode": str}))


def assert_aligned(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"{label}: row count differs ({len(other)} != {len(reference)}).")
    for column in KEYS:
        if column == "timestamp_et":
            left = reference[column].to_numpy(dtype="datetime64[ns]")
            right = other[column].to_numpy(dtype="datetime64[ns]")
            equal = np.array_equal(left, right)
        else:
            equal = np.array_equal(reference[column].to_numpy(), other[column].to_numpy())
        if not equal:
            raise ValueError(f"{label}: key column {column} is not exactly row-aligned.")


def assert_values_equal(
    reference: np.ndarray, other: np.ndarray, label: str, atol: float = 1e-12
) -> None:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(other, dtype=float)
    if not np.array_equal(np.isfinite(left), np.isfinite(right)):
        raise ValueError(f"{label}: finite/missing masks differ.")
    valid = np.isfinite(left)
    if valid.any() and not np.allclose(left[valid], right[valid], atol=atol, rtol=0.0):
        maximum = float(np.max(np.abs(left[valid] - right[valid])))
        raise ValueError(f"{label}: values differ; max absolute difference={maximum}.")


def load_row_folds(project_root: Path, frame: pd.DataFrame) -> np.ndarray:
    path = project_root / "cv" / "cv_assignments_balanced_v1_seed42.csv"
    assignment = pd.read_csv(path, dtype={"fipsCode": str})
    assignment["fipsCode"] = assignment["fipsCode"].astype(str).str.zfill(5)
    if assignment["fipsCode"].duplicated().any():
        raise ValueError("CV assignment contains duplicate counties.")
    mapping = assignment.set_index("fipsCode")["fold"]
    folds = frame["fipsCode"].map(mapping)
    if folds.isna().any() or set(folds.unique()) != set(range(5)):
        raise ValueError("OOF counties and balanced_v1 fold assignment do not match.")
    county_fold_counts = pd.DataFrame({"fipsCode": frame["fipsCode"], "fold": folds}).drop_duplicates()
    if county_fold_counts["fipsCode"].duplicated().any():
        raise ValueError("A county appears in more than one fold.")
    return folds.to_numpy(dtype=int)


def post_process(values: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=float), 0.0, 0.65)
    return np.where(result < 0.001, 0.0, result)


def clip_component(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def compose_osi(component_predictions: dict[str, np.ndarray]) -> np.ndarray:
    result = sum(
        COMPONENT_WEIGHTS[component] * np.clip(component_predictions[component], 0.0, 1.0)
        for component in COMPONENTS
    )
    return post_process(np.maximum(result, 0.0))


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    error = pred[valid] - truth[valid]
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "n": int(valid.sum()),
    }


def fit_simplex_least_squares(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact small-p non-negative least squares with sum(weights)=1.

    Every non-empty active set is enumerated. There are at most 63 sets in v1.11
    and 7 in v2.6, so this is deterministic and avoids an optimizer dependency.
    """
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(target, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x = x[valid]
    y = y[valid]
    p = x.shape[1]
    best_weights: np.ndarray | None = None
    best_loss = np.inf
    for size in range(1, p + 1):
        for active in itertools.combinations(range(p), size):
            xa = x[:, active]
            gram = xa.T @ xa
            rhs = xa.T @ y
            kkt = np.block(
                [[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]]
            )
            solution = np.linalg.lstsq(kkt, np.r_[rhs, 1.0], rcond=None)[0][:-1]
            if np.min(solution) < -1e-10:
                continue
            solution = np.maximum(solution, 0.0)
            solution /= solution.sum()
            weights = np.zeros(p, dtype=float)
            weights[list(active)] = solution
            loss = float(np.mean((x @ weights - y) ** 2))
            if loss < best_loss - 1e-18:
                best_loss = loss
                best_weights = weights
    if best_weights is None:
        raise RuntimeError("No feasible simplex solution found.")
    return best_weights


def fit_safe_alpha(target: np.ndarray, baseline: np.ndarray, ensemble: np.ndarray) -> float:
    y = np.asarray(target, dtype=float)
    base = np.asarray(baseline, dtype=float)
    blend = np.asarray(ensemble, dtype=float)
    valid = np.isfinite(y) & np.isfinite(base) & np.isfinite(blend)
    direction = blend[valid] - base[valid]
    denominator = float(direction @ direction)
    if denominator <= 1e-30:
        return 0.0
    alpha = float(direction @ (y[valid] - base[valid]) / denominator)
    return float(np.clip(alpha, 0.0, 1.0))


def nested_predictions(
    y: np.ndarray,
    matrix: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    processor: Callable[[np.ndarray], np.ndarray] = post_process,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    n, p = matrix.shape
    output = {
        "simple_average": np.full(n, np.nan),
        "static_convex": np.full(n, np.nan),
        "safe_convex": np.full(n, np.nan),
    }
    output["simple_average"] = processor(np.mean(matrix, axis=1))
    models: list[dict[str, object]] = []
    for fold in range(5):
        train = (folds != fold) & np.isfinite(y) & np.isfinite(matrix).all(axis=1)
        valid = (folds == fold) & np.isfinite(y) & np.isfinite(matrix).all(axis=1)
        weights = fit_simplex_least_squares(matrix[train], y[train])
        inner_static = processor(matrix[train] @ weights)
        alpha = fit_safe_alpha(y[train], baseline[train], inner_static)
        outer_static = processor(matrix[valid] @ weights)
        outer_safe = processor(baseline[valid] + alpha * (outer_static - baseline[valid]))
        output["static_convex"][valid] = outer_static
        output["safe_convex"][valid] = outer_safe
        models.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
                "weights": weights.tolist(),
                "safe_alpha": alpha,
            }
        )
    return output, models


def final_model(
    y: np.ndarray,
    matrix: np.ndarray,
    baseline: np.ndarray,
    processor: Callable[[np.ndarray], np.ndarray] = post_process,
) -> dict[str, object]:
    valid = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
    weights = fit_simplex_least_squares(matrix[valid], y[valid])
    static = processor(matrix[valid] @ weights)
    alpha = fit_safe_alpha(y[valid], baseline[valid], static)
    return {"train_rows": int(valid.sum()), "weights": weights.tolist(), "safe_alpha": alpha}


def metric_rows(
    frame: pd.DataFrame,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    horizon: str,
    group_column: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    groups: Iterable[tuple[object, np.ndarray]]
    if group_column is None:
        groups = [("all", np.ones(len(frame), dtype=bool))]
    else:
        groups = ((name, frame[group_column].to_numpy() == name) for name in sorted(frame[group_column].unique()))
    for group, mask in groups:
        for model, prediction in predictions.items():
            values = metric_values(truth[mask], prediction[mask])
            row: dict[str, object] = {"horizon": horizon, "model": model, **values}
            if group_column is not None:
                row[group_column] = group
            rows.append(row)
    return rows


def paired_county_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    valid = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(candidate)
    work = pd.DataFrame(
        {
            "county": frame.loc[valid, "fipsCode"].to_numpy(),
            "base_sq": (baseline[valid] - truth[valid]) ** 2,
            "cand_sq": (candidate[valid] - truth[valid]) ** 2,
            "base_abs": np.abs(baseline[valid] - truth[valid]),
            "cand_abs": np.abs(candidate[valid] - truth[valid]),
        }
    )
    aggregate = work.groupby("county", sort=True).agg(["sum", "count"])
    counties = aggregate.index.to_numpy()
    base_sq = aggregate[("base_sq", "sum")].to_numpy()
    cand_sq = aggregate[("cand_sq", "sum")].to_numpy()
    base_abs = aggregate[("base_abs", "sum")].to_numpy()
    cand_abs = aggregate[("cand_abs", "sum")].to_numpy()
    counts = aggregate[("base_sq", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    rmse_delta = np.empty(replicates)
    mae_delta = np.empty(replicates)
    for index in range(replicates):
        sampled = rng.integers(0, len(counties), len(counties))
        total_n = counts[sampled].sum()
        rmse_delta[index] = np.sqrt(cand_sq[sampled].sum() / total_n) - np.sqrt(
            base_sq[sampled].sum() / total_n
        )
        mae_delta[index] = (cand_abs[sampled].sum() - base_abs[sampled].sum()) / total_n
    point_base = metric_values(truth[valid], baseline[valid])
    point_candidate = metric_values(truth[valid], candidate[valid])
    return {
        "counties": int(len(counties)),
        "replicates": int(replicates),
        "rmse_delta": float(point_candidate["rmse"] - point_base["rmse"]),
        "rmse_ci_low": float(np.quantile(rmse_delta, 0.025)),
        "rmse_ci_high": float(np.quantile(rmse_delta, 0.975)),
        "rmse_probability_improved": float(np.mean(rmse_delta < 0.0)),
        "mae_delta": float(point_candidate["mae"] - point_base["mae"]),
        "mae_ci_low": float(np.quantile(mae_delta, 0.025)),
        "mae_ci_high": float(np.quantile(mae_delta, 0.975)),
        "mae_probability_improved": float(np.mean(mae_delta < 0.0)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_metadata() -> dict[str, object]:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
    }


def hash_tree(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in sorted(paths)}
