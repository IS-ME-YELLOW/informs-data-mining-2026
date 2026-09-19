"""Rebuild all v1.5.6 input columns in memory and perturb future outage data.

No cache/model writes. --shard permits separate read-only workers; only their
new audit CSV/JSON files in this review directory are written.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import sys
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / "versions/xyy/v1.5.6"
sys.path.insert(0, str(ROOT / "code_phase1"))
from data_loader import preprocess
from features import build_feature_matrix

NAMES = json.loads((FROZEN / "feature_names_v1.5.6.json").read_text())
tree = ast.parse((ROOT / "code_phase1/feature_dataset_v156.py").read_text())
body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_add_window_quality_features"]
namespace = {"np": np, "WINDOW_HOURS": [1, 6, 24, 48], "frozen_schema": lambda: NAMES}
exec(compile(ast.Module(body=body, type_ignores=[]), "feature_dataset_v156.py", "exec"), namespace)


def reconstruct(raw, external, is_train):
    prepared = preprocess(raw.copy())
    original_observed = raw._hour_idx < 72
    np.testing.assert_array_equal(prepared.loc[original_observed, ["N_t", "R_t"]], raw.loc[original_observed, ["N_t", "R_t"]])
    x, _, meta = build_feature_matrix(prepared, is_train=is_train)
    for column, value in external.items():
        x[column] = float(value)
    x["forest_x_customers"] = x.pct_forest * x.log_customers
    return namespace["_add_window_quality_features"](x, meta), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    ext = pd.read_csv(ROOT / "data/county_features_v1.5.6.csv", dtype={"fipsCode": str}).set_index("fipsCode")
    jobs = []
    for split, source in [("train", "DM_Train.csv"), ("test", "DM_Test.csv")]:
        raw = pd.read_csv(ROOT / "data" / source, dtype={"fipsCode": str})
        raw["_dt"] = pd.to_datetime(raw.timestamp_et)
        raw = raw.sort_values(["fipsCode", "_dt"]).reset_index(drop=True)
        raw["_hour_idx"] = raw.groupby("fipsCode").cumcount()
        x = pd.read_parquet(FROZEN / f"features_{split}_v1.5.6.parquet")
        meta = pd.read_parquet(FROZEN / f"meta_{split}_v1.5.6.parquet")
        meta["fipsCode"] = meta.fipsCode.astype(str).str.zfill(5)
        for fips, group in raw.groupby("fipsCode", sort=True):
            mask = meta.fipsCode == fips
            jobs.append((split, fips, group.reset_index(drop=True), x.loc[mask].reset_index(drop=True), meta.loc[mask].reset_index(drop=True)))
    selected = jobs[args.shard::args.shards]
    rows, column_errors = [], np.zeros(len(NAMES))
    start = time.monotonic()
    for index, (split, fips, raw, frozen, frozen_meta) in enumerate(selected):
        baseline, meta = reconstruct(raw, ext.loc[fips].to_dict(), split == "train")
        assert np.array_equal(meta.hour_idx, frozen_meta.hour_idx)
        assert np.array_equal(pd.to_datetime(meta.timestamp_et), pd.to_datetime(frozen_meta.timestamp_et))
        a, b = baseline.to_numpy(dtype=float), frozen.to_numpy(dtype=float)
        equal = np.isclose(a, b, atol=1e-10, rtol=1e-12, equal_nan=True)
        nan_equal = np.array_equal(np.isnan(a), np.isnan(b))
        errors = np.abs(np.nan_to_num(a - b, nan=0.0))
        column_errors = np.maximum(column_errors, errors.max(axis=0))
        poison_columns = [c for c in raw.columns if c.startswith(("outage", "osi")) or c in
                          {"P_t", "N_t", "D_t", "R_t", "peak_pct", "peak_customers", "time_to_restore_h", "event_duration_h"}]
        mutation_ok = {}
        for label, value in [("future_nan", np.nan), ("future_large_constant", 987654.321)]:
            poisoned = raw.copy()
            # Explicit float conversion avoids pandas integer-column assignment ambiguity.
            for column in poison_columns:
                poisoned[column] = poisoned[column].astype(float)
                poisoned.loc[poisoned._hour_idx >= 72, column] = value
            rebuilt, _ = reconstruct(poisoned, ext.loc[fips].to_dict(), split == "train")
            result = rebuilt.to_numpy(dtype=float)
            mutation_ok[label] = bool(np.array_equal(a, result, equal_nan=True))
        positive_control = None
        if index == 0:
            changed = raw.copy()
            changed.loc[changed._hour_idx == 90, "gust"] += 5.0
            altered, _ = reconstruct(changed, ext.loc[fips].to_dict(), split == "train")
            positive_control = not np.array_equal(a, altered.to_numpy(dtype=float), equal_nan=True)
            assert positive_control
        rows.append({"split": split, "fipsCode": fips, "rows": len(a), "columns": len(NAMES),
                     "frozen_equal_at_tolerance": bool(equal.all()), "nan_mask_equal": nan_equal,
                     "max_abs_difference": float(errors.max()), "mismatching_cells": int((~equal).sum()),
                     **mutation_ok, "weather_positive_control_changes_features": positive_control})
        if not equal.all():
            detail = np.argwhere(~equal)[:10]
            print("FROZEN_MISMATCH", fips, [(int(i), NAMES[j], float(a[i,j]), float(b[i,j])) for i,j in detail], flush=True)
        if (index + 1) % 10 == 0 or index + 1 == len(selected):
            print(f"shard={args.shard} checked={index+1}/{len(selected)} seconds={time.monotonic()-start:.1f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"feature_causality_shard{args.shard}.csv", index=False)
    passed = bool(frame.frozen_equal_at_tolerance.all() and frame.nan_mask_equal.all() and
                  frame.future_nan.all() and frame.future_large_constant.all())
    result = {"shard": args.shard, "shards": args.shards, "counties": len(frame), "passed": passed,
              "training_performed": False, "frozen_comparison_atol": 1e-10, "frozen_comparison_rtol": 1e-12,
              "mutation_comparison": "exact equality including NaN mask",
              "rows": int(frame.rows.sum()), "columns": len(NAMES),
              "max_abs_difference": float(frame.max_abs_difference.max()),
              "column_max_abs_differences": dict(zip(NAMES, map(float, column_errors))),
              "poisoned_field_rule": "hour>=72 outage/osi/lag/target/delta and P/N/D/R/event-outcome fields; weather and customersTracked retained",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [ROOT/'code_phase1/features.py', ROOT/'code_phase1/data_loader.py', ROOT/'code_phase1/feature_dataset_v156.py']},
              "elapsed_seconds": time.monotonic()-start}
    (OUT / f"feature_causality_shard{args.shard}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    assert passed, "Feature causality audit failed; inspect the new review artifacts"


if __name__ == "__main__":
    main()
