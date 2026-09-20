"""Compare completed comparison runs using only their frozen outer OOF rows.

This utility never chooses a variant and never reads test labels.  It is a
reporting step: each run must already have passed ``verify_artifacts.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import COMPARE_VARIANTS, HORIZONS, PROTOCOL
from metrics import post_process_osi


KEY = ["fipsCode", "timestamp_et", "horizon"]


def _load_run(path: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"not a comparison run: {path}")
    if manifest.get("variant") not in COMPARE_VARIANTS:
        raise ValueError(f"invalid or missing variant in {path}")
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"run is not independently verified: {path}")
    frame = pd.read_parquet(path / "outer_oof.parquet", engine="pyarrow")
    frame["fipsCode"] = frame["fipsCode"].astype(str).str.zfill(5)
    if frame.duplicated(KEY).any():
        raise ValueError(f"duplicate outer key in {path}")
    if len(frame) != 34416 * len(HORIZONS):
        raise ValueError(f"unexpected outer row count in {path}")
    return manifest, frame


def _metrics(frame: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, counties = [], []
    for horizon, group in frame.groupby("horizon", sort=True):
        valid = group["is_scoreable"].astype(bool).to_numpy()
        y = group.loc[valid, "y_true"].to_numpy(dtype=float)
        base = post_process_osi(group.loc[valid, "base_prediction"].to_numpy(dtype=float))
        pred = group.loc[valid, "prediction"].to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(base).all() and np.isfinite(pred).all()):
            raise FloatingPointError(f"non-finite outer values for {variant}/{horizon}")
        be, pe = base - y, pred - y
        rows.extend([
            {"variant": variant, "horizon": horizon, "model": "base", "n": len(y),
             "sse": float(np.sum(be * be, dtype=np.float64)),
             "rmse": float(np.sqrt(np.mean(be * be))),
             "mae": float(np.mean(np.abs(be)))},
            {"variant": variant, "horizon": horizon, "model": "gat", "n": len(y),
             "sse": float(np.sum(pe * pe, dtype=np.float64)),
             "rmse": float(np.sqrt(np.mean(pe * pe))),
             "mae": float(np.mean(np.abs(pe)))},
        ])
        valid_frame = group.loc[valid].copy()
        valid_frame["base_error"] = valid_frame["base_prediction"].to_numpy(dtype=float)
        valid_frame["base_error"] = post_process_osi(valid_frame["base_error"]) - valid_frame["y_true"]
        valid_frame["gat_error"] = valid_frame["prediction"] - valid_frame["y_true"]
        for fips, county in valid_frame.groupby("fipsCode", sort=True):
            bs = float(np.sum(county.base_error.to_numpy() ** 2, dtype=np.float64))
            gs = float(np.sum(county.gat_error.to_numpy() ** 2, dtype=np.float64))
            counties.append({"variant": variant, "horizon": horizon, "fipsCode": fips,
                             "outer_fold": int(county.outer_fold.iloc[0]), "n": len(county),
                             "base_sse": bs, "gat_sse": gs, "delta_sse": gs - bs,
                             "base_rmse": float(np.sqrt(bs / len(county))),
                             "gat_rmse": float(np.sqrt(gs / len(county)))})
    return pd.DataFrame(rows), pd.DataFrame(counties)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="verified comparison run directories")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests, frames = [], []
    seen = set()
    for raw in args.run_dirs:
        manifest, frame = _load_run(Path(raw).resolve())
        variant = manifest["variant"]
        if variant in seen:
            raise ValueError(f"variant supplied more than once: {variant}")
        seen.add(variant)
        manifests.append(manifest)
        frames.append((variant, frame))
    if len(frames) < 2:
        raise ValueError("provide at least two completed variants")
    reference_keys = frames[0][1][KEY]
    summary, county = [], []
    for variant, frame in frames:
        if not frame[KEY].reset_index(drop=True).equals(reference_keys.reset_index(drop=True)):
            raise ValueError("variants do not have identical outer OOF key/order")
        reference_base = frames[0][1]["base_prediction"].to_numpy(dtype=float)
        candidate_base = frame["base_prediction"].to_numpy(dtype=float)
        if not np.allclose(candidate_base, reference_base, rtol=0, atol=1e-12):
            raise ValueError("variants do not share the same LightGBM base predictions")
        s, c = _metrics(frame, variant)
        summary.append(s)
        county.append(c)
    summary_frame = pd.concat(summary, ignore_index=True)
    county_frame = pd.concat(county, ignore_index=True)
    summary_frame.to_csv(output / "compare_summary.csv", index=False)
    county_frame.to_csv(output / "compare_county_metrics.csv", index=False)
    worst = county_frame.sort_values(["horizon", "delta_sse"], ascending=[True, False]).groupby(
        "horizon", sort=True, as_index=False
    ).head(10)
    worst.to_csv(output / "compare_worst_counties.csv", index=False)
    (output / "compare_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL, "variants": sorted(seen),
        "runs": [{"run_id": m["run_id"], "variant": m["variant"]} for m in manifests],
        "source": "verified outer_oof.parquet only",
    }, indent=2), encoding="utf-8")
    print(f"WROTE={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
