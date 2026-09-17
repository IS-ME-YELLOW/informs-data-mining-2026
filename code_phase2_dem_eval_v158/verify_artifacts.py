"""Independent artifact verifier for dem_v158_nested_v2.

This script never fits a model.  It validates row coverage, frozen CV
evidence, checkpoint identities, submission hard checks, and—when dependencies
are available—reloads the final stack in a fresh Python process and reproduces
test predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import EXPERIMENT_VERSION, FEATURE_VERSION, HORIZON_HOURS, HORIZONS, OFFICIAL_SCOREABLE_ROWS, PRED_END, PRED_START, PROTOCOL, SUBMISSION_FILE
from metrics import post_process_osi


def _fail(message):
    raise AssertionError(message)


def _check_run(run_dir: Path):
    required = [
        "run_manifest.json", "environment.json", "inputs_manifest.json", "feature_schema.json",
        "graph.npz", "folds.csv", "base_fit_manifest.parquet", "inner_oof.parquet",
        "outer_oof.parquet", "alpha_selection.parquet", "cv_summary.csv", "fold_metrics.csv",
        "county_metrics.csv", "county_bootstrap.csv", "cv_manifest.json",
        "CV_COMPLETE", "final_graph_base.parquet", "test_predictions.parquet",
        "alpha_selection_final.parquet", "submission_phase2_dem_gat.csv", "submission_audit.json",
        "verification.json", "REPORT.md", "FINAL_READY",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        _fail(f"missing required artifact(s): {missing}")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        _fail("run protocol mismatch")
    if manifest.get("experiment_version") != EXPERIMENT_VERSION or manifest.get("feature_version") != FEATURE_VERSION:
        _fail("run version identity mismatch")
    mode = manifest.get("base_mode")
    if mode not in {"direct", "component_v158"}:
        _fail("invalid base mode")
    input_manifest = json.loads((run_dir / "inputs_manifest.json").read_text(encoding="utf-8"))
    outer = pd.read_parquet(run_dir / "outer_oof.parquet", engine="pyarrow")
    key = ["fipsCode", "timestamp_et", "horizon"]
    if len(outer) != 34416 * 4 or outer.duplicated(key).any():
        _fail("outer_oof row count or key uniqueness failed")
    for horizon in HORIZONS:
        subset = outer[outer["horizon"] == horizon]
        if len(subset) != 34416:
            _fail(f"outer_oof row count failed for {horizon}")
        expected = subset["is_scoreable"].to_numpy(dtype=bool)
        if int(expected.sum()) != OFFICIAL_SCOREABLE_ROWS[horizon]:
            _fail(f"outer_oof scoreable count failed for {horizon}")
        for column in ("base_prediction", "correction_osi"):
            if not np.isfinite(subset[column].to_numpy(dtype=float)).all():
                _fail(f"non-finite outer {column} for {horizon}")
        alpha_values = subset["alpha"].to_numpy(dtype=float)
        if not np.isfinite(alpha_values[expected]).all():
            _fail(f"non-finite outer alpha for {horizon}")
        for column in ("y_true", "prediction"):
            values = subset[column].to_numpy(dtype=float)
            if not np.isfinite(values[expected]).all():
                _fail(f"non-finite expected outer {column} for {horizon}")
            if np.isfinite(values[~expected]).any():
                _fail(f"invalid tail value in outer {column} for {horizon}")
    county = pd.read_csv(run_dir / "county_metrics.csv")
    if county.duplicated(["horizon", "fipsCode"]).any():
        _fail("county_metrics key uniqueness failed")
    expected_counties = outer.loc[outer["is_scoreable"].astype(bool), ["horizon", "fipsCode"]].drop_duplicates()
    if len(county) != len(expected_counties) or not expected_counties.merge(
        county[["horizon", "fipsCode"]], on=["horizon", "fipsCode"], how="left", indicator=True
    )["_merge"].eq("both").all():
        _fail("county_metrics coverage does not match scoreable outer counties")
    bootstrap = pd.read_csv(run_dir / "county_bootstrap.csv")
    if set(bootstrap["horizon"]) != set(HORIZONS) or len(bootstrap) != len(HORIZONS):
        _fail("county_bootstrap horizon coverage failed")
    inner = pd.read_parquet(run_dir / "inner_oof.parquet", engine="pyarrow")
    inner_key = ["outer_fold", "inner_fold", "fipsCode", "timestamp_et", "horizon"]
    if inner.duplicated(inner_key).any():
        _fail("inner_oof key uniqueness failed")
    alpha = pd.read_parquet(run_dir / "alpha_selection.parquet", engine="pyarrow")
    alpha_outer = alpha[alpha["selection_type"] == "outer_train_inner_cv"]
    if len(alpha_outer) != 5 * 4 * 8 * 4:
        _fail("outer alpha candidate coverage is not 5*4*8 horizons")
    for keys, group in alpha_outer.groupby(["outer_fold", "horizon"], sort=False):
        if len(group) != 8 or group["n"].nunique() != 1 or int(group["selected"].sum()) != 1:
            _fail(f"alpha candidate coverage failed for {keys}")
    alpha_final = pd.read_parquet(run_dir / "alpha_selection_final.parquet", engine="pyarrow")
    alpha_candidates = alpha_final[alpha_final["selection_type"] == "full_train_cv_for_final"]
    if len(alpha_candidates) != 4 * 8:
        _fail("final alpha candidate coverage is not 4*8")
    for horizon, group in alpha_candidates.groupby("horizon", sort=False):
        if len(group) != 8 or group["n"].nunique() != 1 or int(group["selected"].sum()) != 1:
            _fail(f"final alpha candidate coverage failed for {horizon}")
    final_selected = alpha_final[alpha_final["selection_type"] == "final_selected"]
    if len(final_selected) != 4 or final_selected["horizon"].nunique() != 4:
        _fail("final alpha selected-row coverage failed")
    graph = pd.read_parquet(run_dir / "final_graph_base.parquet", engine="pyarrow")
    if len(graph) != 302 * 144 * 4 or graph.duplicated(["fipsCode", "hour_idx", "horizon"]).any():
        _fail("final_graph_base coverage failed")
    test = pd.read_parquet(run_dir / "test_predictions.parquet", engine="pyarrow")
    if len(test) != 9072 * 4 or test.duplicated(["fipsCode", "timestamp_et", "horizon"]).any():
        _fail("test_predictions coverage failed")
    submission = pd.read_csv(run_dir / "submission_phase2_dem_gat.csv")
    template = pd.read_csv(SUBMISSION_FILE)
    if len(submission) != len(template) or list(submission.columns) != list(template.columns):
        _fail("submission structure differs from the official template")
    if not submission[["fipsCode", "timestamp_et"]].astype(str).reset_index(drop=True).equals(
        template[["fipsCode", "timestamp_et"]].astype(str).reset_index(drop=True)
    ):
        _fail("submission identifier order differs from the official template")
    for horizon in HORIZONS:
        values = submission[horizon].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if int(finite.sum()) != OFFICIAL_SCOREABLE_ROWS[horizon]:
            _fail(f"submission finite count failed for {horizon}")
        if not np.isfinite(values[finite]).all() or not ((values[finite] >= 0) & (values[finite] <= 0.65)).all():
            _fail(f"submission value range failed for {horizon}")
    schema = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))
    graph_names = schema.get("graph_schema", [])
    if schema.get("graph_input_dim") != 205 or len(schema.get("ordered_phase1_features", [])) != 163 or len(graph_names) != 205:
        _fail("feature schema artifact is not the 205-dimensional protocol")
    graph_schema_hash = hashlib.sha256(json.dumps(graph_names, separators=(",", ":")).encode()).hexdigest()
    base_files = sorted((run_dir / "models" / "base").glob("*.txt"))
    expected_base = 26 * 4 * (1 if mode == "direct" else 4)
    if len(base_files) != expected_base:
        _fail(f"base model count is {len(base_files)}, expected {expected_base}")
    base_manifest = pd.read_parquet(run_dir / "base_fit_manifest.parquet", engine="pyarrow")
    if len(base_manifest) != expected_base:
        _fail("base_fit_manifest count does not match the required scope models")
    gat_files = sorted((run_dir / "models" / "gat").glob("*.pt"))
    expected_gat = 16 * 4
    if len(gat_files) != expected_gat:
        _fail(f"GAT checkpoint count is {len(gat_files)}, expected {expected_gat}")
    import torch
    for path in gat_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("protocol") != PROTOCOL or payload.get("architecture", {}).get("in_dim") != 205:
            _fail(f"invalid GAT checkpoint identity: {path.name}")
        if payload.get("graph_hash") != input_manifest.get("graph_hash"):
            _fail(f"GAT graph identity mismatch: {path.name}")
        if payload.get("feature_package_hash") != input_manifest.get("feature_package_hash"):
            _fail(f"GAT feature package identity mismatch: {path.name}")
        if payload.get("feature_schema_hash") != graph_schema_hash or payload.get("feature_names") != graph_names:
            _fail(f"GAT feature schema identity mismatch: {path.name}")
        weight = payload.get("state_dict", {}).get("gat1.linear.weight")
        if weight is None or tuple(weight.shape) != (96, 205):
            _fail(f"GAT first linear layer is not [96,205]: {path.name}")
    cv_manifest = json.loads((run_dir / "cv_manifest.json").read_text(encoding="utf-8"))
    if (run_dir / "CV_COMPLETE").read_text(encoding="utf-8") != cv_manifest.get("cv_hash"):
        _fail("CV_COMPLETE marker does not match cv_manifest")
    for name, digest in cv_manifest.get("artifact_hashes", {}).items():
        actual = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        if actual != digest:
            _fail(f"frozen CV artifact hash mismatch: {name}")
    return manifest


def _independent_reload(run_dir: Path, manifest: dict):
    """Recreate final test predictions without fitting any model."""

    import main as runner

    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    params = manifest["parameters"]
    args = SimpleNamespace(
        stage="final", run_id=manifest["run_id"], resume=True,
        feature_dir=manifest["inputs"]["feature_dir"],
        parquet_engine=manifest["inputs"]["parquet_engine"],
        base_mode=manifest["base_mode"], seed=int(manifest["seed"]),
        device=environment["device"], epochs=int(params["epochs"]),
        patience=int(params["patience"]), time_stride=int(params["time_stride"]),
        k=int(params["k"]), gat_loss=params["gat_loss"],
        expected_feature_package_hash=manifest["inputs"]["feature_package_hash"],
        expected_feature_side_hash=manifest["inputs"]["feature_side_hash"],
    )
    preflight = runner._preflight(args, load_supervision_data=False)
    ctx = runner._make_context(
        args, preflight, run_dir, resume=True, strict_resume=True, inference_only=True
    )
    saved = pd.read_parquet(run_dir / "test_predictions.parquet", engine="pyarrow")
    final_alpha = pd.read_parquet(run_dir / "alpha_selection_final.parquet", engine="pyarrow")
    final_alpha = final_alpha[final_alpha["selection_type"] == "final_selected"]
    if len(final_alpha) != len(HORIZONS) or final_alpha["horizon"].nunique() != len(HORIZONS):
        _fail("final alpha artifact does not contain exactly one selected alpha per horizon")
    alpha_by_horizon = {
        str(row.horizon): float(row.alpha)
        for row in final_alpha.itertuples(index=False)
    }
    for horizon in HORIZONS:
        stack = ctx.fit_stack(tuple(range(5)), horizon, args.base_mode)
        alpha = alpha_by_horizon[horizon]
        if not np.isfinite(alpha):
            _fail(f"non-finite final alpha for {horizon}")
        values = []
        for row_id, row in preflight["bundle"].meta_test.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            node = stack.inputs.all_fips.index(row["fips_str"])
            scoreable = int(row["hour_idx"]) + HORIZON_HOURS[horizon] <= PRED_END - 1
            value = float(post_process_osi(stack.inputs.base_grid[t, node] + alpha * stack.correction_grid[t, node])) if scoreable else np.nan
            values.append(value)
        recorded = saved.loc[saved["horizon"] == horizon].sort_values(["fipsCode", "hour_idx"])["prediction"].to_numpy(dtype=float)
        actual = np.asarray(values, dtype=float)
        if not np.allclose(actual, recorded, rtol=0.0, atol=1e-7, equal_nan=True):
            _fail(f"independent reload mismatch for {horizon}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if (run_dir / "COMPLETE").exists():
        _fail("COMPLETE already exists; verified runs are immutable")
    manifest = _check_run(run_dir)
    _independent_reload(run_dir, manifest)
    verification = {
        "protocol": PROTOCOL,
        "run_id": manifest["run_id"],
        "artifact_checks": True,
        "independent_reload": True,
        "passed": True,
    }
    verification_tmp = run_dir / "verification.json.tmp"
    verification_tmp.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    verification_tmp.replace(run_dir / "verification.json")
    complete_tmp = run_dir / "COMPLETE.tmp"
    complete_tmp.write_text("verified", encoding="utf-8")
    complete_tmp.replace(run_dir / "COMPLETE")
    print(json.dumps(verification, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
