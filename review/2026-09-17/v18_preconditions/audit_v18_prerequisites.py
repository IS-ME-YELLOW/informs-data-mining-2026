"""Read-only v1.8 prerequisite audit; writes only new review evidence.

Never calls training, existing --validate-only, or the existing verifier because
those entrypoints can write historical artifacts. No changes to source files.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.metadata
import json
import runpy
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V18 = ROOT / "versions/xyy/v1.5.8"
FEATURES = ROOT / "versions/xyy/v1.5.6"
HOURS = (1, 6, 24, 48)
HORIZONS = tuple(f"osi_target_t{h:02d}h" for h in HOURS)
COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
SEEDS = (42, 20260917, 20260918)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(frame):
    frame = frame.copy()
    frame["fipsCode"] = frame.fipsCode.astype(str).str.zfill(5)
    if "timestamp_et" in frame:
        frame["time"] = pd.to_datetime(frame.timestamp_et)
    return frame


def protected_files():
    files = list((ROOT / "code_phase1").glob("*.py")) + list(V18.glob("*.py"))
    files += list(FEATURES.glob("*.parquet")) + list(FEATURES.glob("feature_names*.json"))
    files += list(V18.glob("*.parquet")) + list(V18.glob("*.csv")) + list(V18.glob("*.json"))
    files += list((V18 / "models").rglob("*.txt"))
    files += [ROOT / "data/DM_Train.csv", ROOT / "data/DM_Test.csv", ROOT / "data/sample_submission.csv",
              ROOT / "data/county_features_v1.5.6.csv", ROOT / "cv/cv_assignments_balanced_v1_seed42.csv"]
    return sorted(set(files))


def same_values(a, b, atol=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    valid = np.isfinite(a)
    assert np.array_equal(valid, np.isfinite(b))
    delta = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0
    assert delta <= atol, (delta, atol)
    return delta


def main():
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
    results = {"audit_date": "2026-09-17", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.executable, "training_performed": False,
               "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn", "scipy"]}}
    cfg = runpy.run_path(str(V18 / "config.py"))
    results["configured_paths"] = {key: {"path": str(cfg[key]), "exists": cfg[key].exists()}
                                  for key in ["PROJECT_ROOT", "CACHE_DIR", "DATA_DIR", "CV_FILE", "RAW_TRAIN_FILE", "SUBMISSION_TEMPLATE"]}
    protocol_ast = ast.parse((V18 / "protocol.py").read_text())
    selected = [node for node in protocol_ast.body if isinstance(node, ast.FunctionDef) and node.name in {"load_data", "metrics"}]
    ns = dict(cfg, np=np, pd=pd, json=json, ExperimentData=object)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "v18_protocol_readonly_extract", "exec"), ns)
    try:
        ns["load_data"]()
    except FileNotFoundError as exc:
        results["original_loader_failure"] = str(exc)
    else:
        raise AssertionError("Expected path issue changed; re-review before reporting it")
    results["missing_prediction_metric_probe"] = ns["metrics"]([0., 1.], [0., np.nan])
    raw = {}
    bundles = {}
    checks = {}
    for split, source, count in [("train", "DM_Train.csv", 239), ("test", "DM_Test.csv", 63)]:
        source_frame = pd.read_csv(ROOT / "data" / source, dtype={"fipsCode": str})
        original_shape = list(source_frame.shape)
        frame = normalized(source_frame).sort_values(["fipsCode", "time"]).reset_index(drop=True)
        frame["hour"] = ((frame.time - pd.Timestamp("2026-03-11")) / pd.Timedelta(hours=1)).astype(int)
        assert not frame.duplicated(["fipsCode", "hour"]).any()
        assert frame.groupby("fipsCode").hour.apply(lambda s: np.array_equal(s.to_numpy(), np.arange(216))).all()
        assert frame.fipsCode.nunique() == count
        raw[split] = frame
        x = pd.read_parquet(FEATURES / f"features_{split}_v1.5.6.parquet")
        meta = normalized(pd.read_parquet(FEATURES / f"meta_{split}_v1.5.6.parquet"))
        expected_meta = frame[frame.hour >= 72].reset_index(drop=True)
        assert np.array_equal(meta.fipsCode, expected_meta.fipsCode)
        assert np.array_equal(meta.time, expected_meta.time)
        assert np.array_equal(meta.hour_idx, expected_meta.hour)
        names = json.loads((FEATURES / "feature_names_v1.5.6.json").read_text())
        assert list(x) == names and x.shape == (count * 144, 163)
        assert not np.isinf(x.to_numpy(dtype=float)).any()
        assert set(x).isdisjoint({"fipsCode", "stateAbbr", "stateName", "countyName", "timestamp_et", "severity_tier", "split", "P_t", "N_t", "D_t", "R_t", "osi", *HORIZONS})
        expected_nan = {f"{name}_at_t{h}h": meta.hour_idx + h > 215 for h in HOURS for name in ["gust", "wind_speed", "t2m", "tp"]}
        expected_nan["hours_since_peak"] = x.osi_max_72h == 0
        for column in x:
            assert np.array_equal(x[column].isna(), expected_nan.get(column, np.zeros(len(x), dtype=bool))), column
        nr_max, d_max = {"N_t": 0.0, "R_t": 0.0}, 0.0
        nr_rows = 0
        for fips, group in frame.groupby("fipsCode"):
            permitted = group if split == "train" else group[group.hour < 72]
            change = permitted.outageCount.diff()
            flows = {"N_t": change.clip(lower=0) / permitted.customersTracked,
                     "R_t": (-change).clip(lower=0) / permitted.customersTracked}
            for comp, rate in flows.items():
                reconstructed = rate.rolling(3, center=True, min_periods=3).mean()
                mask = reconstructed.notna()
                delta = float((reconstructed[mask] - permitted.loc[mask, comp]).abs().max())
                nr_max[comp] = max(nr_max[comp], delta)
                if comp == "N_t":
                    nr_rows += int(mask.sum())
            d_max = max(d_max, float((permitted.P_t.rolling(6, min_periods=1).mean() - permitted.D_t).abs().max()))
        assert max(nr_max.values()) < 1e-6 and d_max < 1e-6
        last = frame[frame.hour == 71].set_index("fipsCode")
        for name in COMPONENTS:
            same_values(x[f"last_{name}"], meta.fipsCode.map(last[name]))
        assert np.isfinite(last[list(COMPONENTS)]).all().all()
        if split == "test":
            withheld = [c for c in frame if c.startswith(("outage", "osi")) or c in COMPONENTS]
            assert frame.loc[frame.hour >= 72, withheld].isna().all().all()
        checks[split] = {"raw_shape": original_shape, "counties": count, "feature_shape": list(x.shape),
                         "feature_nan_cells": int(x.isna().sum().sum()), "county_time_keys_exact": True,
                         "approved_nan_masks_match": True, "last_components_equal_official_hour71": True,
                         "NR_centered_complete_windows_compared": nr_rows, "NR_max_abs_difference": nr_max,
                         "D_trailing6_max_abs_difference": d_max}
        bundles[split] = (x, meta)
    assert set(raw["train"].fipsCode).isdisjoint(raw["test"].fipsCode)
    results["data_checks"] = checks
    y = pd.read_parquet(FEATURES / "targets_train_v1.5.6.parquet")
    components = normalized(pd.read_parquet(V18 / "component_targets_v1.8.parquet"))
    meta = bundles["train"][1]
    assert np.array_equal(meta.fipsCode, components.fipsCode)
    assert np.array_equal(meta.time, components.time) and np.array_equal(meta.hour_idx, components.hour_idx)
    source = raw["train"].set_index(["fipsCode", "hour"])
    label_rows = []
    for h, horizon in zip(HOURS, HORIZONS):
        expected = meta.hour_idx.to_numpy() + h <= 215
        origin_keys = pd.MultiIndex.from_arrays([meta.fipsCode, meta.hour_idx])
        target_keys = pd.MultiIndex.from_arrays([meta.fipsCode, meta.hour_idx + h])
        same_values(y[horizon], source.loc[origin_keys, horizon])
        same_values(y[horizon], source.osi.reindex(target_keys))
        assert np.array_equal(y[horizon].notna(), expected)
        composed = np.zeros(len(meta))
        for comp, weight in zip(COMPONENTS, [.4, .35, .25, -.1]):
            name = f"{comp}_target_t{h:02d}h"
            same_values(components[name], source[comp].reindex(target_keys))
            assert np.array_equal(components[name].notna(), expected)
            composed += weight * components[name].to_numpy()
        maximum = same_values(np.maximum(composed, 0), y[horizon], atol=5.1e-5)
        label_rows.append({"horizon": horizon, "valid_rows": int(expected.sum()), "nan_rows": int((~expected).sum()),
                           "official_and_shifted_osi_exact": True, "four_component_targets_exact": True,
                           "composed_vs_official_max_abs_difference": maximum})
    results["labels"] = label_rows
    # Assignments have already been frozen by the separate outcome-blind generator.
    registry = json.loads((ROOT / "cv/repeated_cv_manifest_2026-09-17.json").read_text())
    tail, focus = [], []
    for record in registry["assignments"]:
        path = ROOT / record["file"]
        assert sha(path) == record["sha256"]
        assignment = normalized(pd.read_csv(path))
        assert len(assignment) == 239 and set(assignment.fipsCode) == set(meta.fipsCode)
        county_meta = meta[["fipsCode", "stateAbbr", "severity_tier"]].drop_duplicates().set_index("fipsCode").sort_index()
        actual_meta = assignment.set_index("fipsCode")[["stateAbbr", "severity_tier"]].sort_index()
        pd.testing.assert_frame_equal(county_meta, actual_meta, check_dtype=False)
        row_fold = meta.fipsCode.map(assignment.set_index("fipsCode").fold).to_numpy()
        for h, horizon in zip(HOURS, HORIZONS):
            target = y[horizon].to_numpy()
            valid = np.isfinite(target)
            for fold in range(5):
                val = valid & (row_fold == fold)
                train = valid & (row_fold != fold)
                tail.append({"split_seed": record["seed"], "horizon": horizon, "fold": fold,
                             "validation_counties": int((assignment.fold == fold).sum()),
                             "valid_rows": int(val.sum()), "validation_max_osi": float(target[val].max()),
                             "validation_rows_gt015": int((target[val] > .15).sum()),
                             "validation_counties_gt015": int(meta.loc[val & (target > .15), "fipsCode"].nunique()),
                             "training_max_osi": float(target[train].max()),
                             "training_rows_gt015": int((target[train] > .15).sum())})
        for fips in ["42053", "39117", "18013", "54015", "54013", "54007", "39115"]:
            row = assignment[assignment.fipsCode == fips].iloc[0]
            focus.append({"split_seed": record["seed"], "fipsCode": fips, "stateAbbr": row.stateAbbr,
                          "severity_tier": int(row.severity_tier), "fold": int(row.fold)})
    pd.DataFrame(tail).to_csv(OUT / "cv_tail_coverage_diagnostics.csv", index=False)
    pd.DataFrame(focus).to_csv(OUT / "focus_county_fold_assignments.csv", index=False)
    results["cv"] = {"registry_sha256": sha(ROOT / "cv/repeated_cv_manifest_2026-09-17.json"),
                     "all_three_splits_metadata_and_coverage_pass": True,
                     "tail_diagnostics_used_to_change_splits": False,
                     "assignments": registry["assignments"]}
    results["protected_sha256"] = before
    changed = [name for name, digest in before.items() if sha(ROOT/name) != digest]
    assert not changed, changed
    results["protected_files_unchanged"] = True
    results["formal_training_ready"] = False
    results["reasons_not_ready"] = ["v1.8 relocated paths and hard-coded CV selection",
                                    "outer validation labels used for early stopping",
                                    "run/resume identity does not isolate split hashes",
                                    "strict valid-row checks and read-only preflight need strengthening"]
    (OUT / "prerequisite_audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"data_checks": checks, "labels": label_rows,
                      "original_loader_failure": results["original_loader_failure"],
                      "protected_files_unchanged": True, "formal_training_ready": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
