"""补充 M4 关键模型之间的县级配对 bootstrap。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import numpy as np
import pandas as pd

from config import HORIZONS, M4_DIR, M4_METADATA, M4_PREDICTIONS, SEED, TRAIN_SEQUENCE_CACHE
from dataset import load_bundle
from evaluate import postprocess


OUTPUT = M4_DIR / "paired_bootstrap_extended.csv"
REPLICATES = 2000
COMPARISONS = [
    ("B1_target_aligned", "B0_lightgbm", "目标时刻对齐相对基线"),
    ("B3_gru_huber", "B0_lightgbm", "主 GRU 相对基线"),
    ("B3_gru_huber", "B1_target_aligned", "主 GRU 相对目标时刻对齐"),
    ("B3_gru_huber", "B2_mlp_huber", "GRU 时序相对非时序 MLP"),
    ("B3_gru_huber", "B4_constant_mean", "主 GRU 相对常数校正"),
    ("B3_gru_huber", "B5_gru_no_future", "加入未来气象相对无未来气象"),
    ("B5_gru_no_future", "B0_lightgbm", "无未来气象 GRU 相对基线"),
    ("B6_gru_mse", "B3_gru_huber", "MSE 相对 Huber"),
    ("B7_gru_weighted_mse", "B6_gru_mse", "高 OSI 加权相对普通 MSE"),
    ("B7_gru_weighted_mse", "B3_gru_huber", "高 OSI 加权相对主 GRU"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def county_statistics(target: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.isfinite(target) & np.isfinite(prediction)
    error = np.where(mask, prediction - target, 0.0)
    return {
        "sse": np.sum(error ** 2, axis=1),
        "sae": np.sum(np.abs(error), axis=1),
        "n": np.sum(mask, axis=1),
    }


def metric_values(statistics: dict[str, np.ndarray], sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = statistics["n"][sampled].sum(axis=1)
    rmse = np.sqrt(statistics["sse"][sampled].sum(axis=1) / n)
    mae = statistics["sae"][sampled].sum(axis=1) / n
    return rmse, mae


def point_metrics(statistics: dict[str, np.ndarray]) -> tuple[float, float]:
    n = statistics["n"].sum()
    return (
        float(np.sqrt(statistics["sse"].sum() / n)),
        float(statistics["sae"].sum() / n),
    )


def main() -> None:
    bundle = load_bundle(TRAIN_SEQUENCE_CACHE)
    with np.load(M4_PREDICTIONS, allow_pickle=False) as data:
        names = data["model_names"].astype(str).tolist()
        predictions = {
            name: postprocess(data["predictions"][idx])
            for idx, name in enumerate(names)
        }
    rng = np.random.default_rng(SEED + 4100)
    sampled = rng.integers(
        0, len(bundle.county_fips), size=(REPLICATES, len(bundle.county_fips))
    )
    rows = []
    for horizon_idx, horizon in enumerate(HORIZONS):
        target = bundle.targets[:, :, horizon_idx]
        stats = {
            name: county_statistics(target, values[:, :, horizon_idx])
            for name, values in predictions.items()
        }
        for model, reference, purpose in COMPARISONS:
            model_rmse, model_mae = metric_values(stats[model], sampled)
            ref_rmse, ref_mae = metric_values(stats[reference], sampled)
            model_point = point_metrics(stats[model])
            ref_point = point_metrics(stats[reference])
            for metric_idx, (metric, difference) in enumerate([
                ("rmse", model_rmse - ref_rmse),
                ("mae", model_mae - ref_mae),
            ]):
                low, high = np.quantile(difference, [0.025, 0.975])
                rows.append({
                    "model": model,
                    "reference": reference,
                    "purpose": purpose,
                    "horizon": horizon,
                    "metric": metric,
                    "difference": model_point[metric_idx] - ref_point[metric_idx],
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "bootstrap_improvement_probability": float(np.mean(difference < 0)),
                    "replicates": REPLICATES,
                    "resampling_unit": "county_full_trajectory",
                    "evaluation_status": "strict_nested_cv",
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    metadata = json.loads(M4_METADATA.read_text(encoding="utf-8"))
    metadata.setdefault("supplemental_outputs", {})[
        str(OUTPUT.relative_to(ROOT))
    ] = {
        "sha256": sha256_file(OUTPUT),
        "description": "M4 关键模型之间 2000 次县级配对 bootstrap",
        "comparisons": len(COMPARISONS),
    }
    M4_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(f"已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
