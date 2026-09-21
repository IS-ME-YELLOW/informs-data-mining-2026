"""Strict artifact and independent test-prediction verification for code_GAT."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import main as runner
from config import (
    COMPONENTS, HORIZONS, HORIZON_HOURS, N_FOLDS, OFFICIAL_SCOREABLE_ROWS,
    PRED_END, PROTOCOL, SUBMISSION_FILE,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_models(run_dir, manifest):
    import torch
    model_dir = run_dir / "models" / "gat"
    expected = set()
    for size in (4, 5):
        for scope in itertools.combinations(range(N_FOLDS), size):
            for h in HORIZONS:
                for c in COMPONENTS:
                    expected.add(f"gat_direct_{c}_{h.replace('osi_target_', '')}_S{'-'.join(map(str, scope))}.pt")
    actual = {p.name for p in model_dir.glob("*.pt")}
    _require(actual == expected, f"direct GAT checkpoint inventory mismatch: {len(actual)} vs {len(expected)}")
    for name in sorted(expected):
        payload = torch.load(model_dir / name, map_location="cpu", weights_only=False)
        _require(payload.get("protocol") == PROTOCOL, f"protocol mismatch: {name}")
        _require(payload.get("architecture", {}).get("in_dim") == 205, f"input dimension mismatch: {name}")
        _require(payload.get("architecture", {}).get("model") == "DirectGAT", f"model architecture mismatch: {name}")
        _require(np.asarray(payload["feature_mean"]).shape == (205,), f"scaler shape mismatch: {name}")
        _require(np.asarray(payload["feature_std"]).shape == (205,), f"scaler shape mismatch: {name}")
        _require(np.isfinite(payload["target_scale"]) and payload["target_scale"] >= 0.003, f"target scale mismatch: {name}")
        for value in payload["state_dict"].values():
            _require(bool(torch.isfinite(value).all()), f"non-finite model parameter: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    _require((run_dir / "FINAL_READY").is_file(), "FINAL_READY is missing")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    _require(manifest.get("protocol") == PROTOCOL, "run protocol mismatch")
    _check_models(run_dir, manifest)
    outer = pd.read_parquet(run_dir / "outer_oof.parquet", engine="pyarrow")
    _require(len(outer) == 34416 * len(HORIZONS), "outer OOF row count mismatch")
    _require(not outer.duplicated(["fipsCode", "timestamp_et", "horizon"]).any(), "outer OOF key duplicate")
    for h in HORIZONS:
        part = outer[outer.horizon == h]
        expected = part.hour_idx.to_numpy(dtype=int) + HORIZON_HOURS[h] <= PRED_END - 1
        _require(int(expected.sum()) == {"osi_target_t01h": 34177, "osi_target_t06h": 32982, "osi_target_t24h": 28680, "osi_target_t48h": 22944}[h], f"OOF count mismatch: {h}")
        _require(np.isfinite(part.loc[expected, "y_true"]).all(), f"OOF target non-finite: {h}")
        _require(np.isfinite(part.loc[expected, "prediction"]).all(), f"OOF prediction non-finite: {h}")
        for c in COMPONENTS:
            _require(np.isfinite(part.loc[:, f"prediction_{c}"]).all(), f"OOF component non-finite: {h}/{c}")
    reload_args = SimpleNamespace(
        seed=int(manifest["seed"]), device=manifest["parameters"]["device"],
        feature_dir=manifest["inputs"]["feature_dir"], parquet_engine=manifest["inputs"]["parquet_engine"],
        epochs=int(manifest["parameters"]["epochs"]), patience=int(manifest["parameters"]["patience"]),
        time_stride=1, k=8, gat_loss=manifest["parameters"]["gat_loss"],
        resume=True, expected_run_identity=manifest["identity"], run_id=manifest["run_id"],
    )
    runner._seed_process(reload_args.seed, reload_args.device)
    preflight = runner._preflight(reload_args, load_supervision_data=False)
    ctx = runner.DirectGATContext(reload_args, preflight, run_dir, inference_only=True)
    saved = pd.read_parquet(run_dir / "test_predictions.parquet", engine="pyarrow")
    actual = []
    for h in HORIZONS:
        models = {c: ctx.fit_one(tuple(range(N_FOLDS)), h, c) for c in COMPONENTS}
        grids = {c: models[c]["values"] for c in COMPONENTS}
        for _, row in ctx.bundle.meta_test.iterrows():
            t = int(row.hour_idx) - 72
            node = ctx.all_fips.index(str(row.fips_str))
            parts = {c: float(grids[c][t, node]) for c in COMPONENTS}
            scoreable = int(row.hour_idx) + HORIZON_HOURS[h] <= PRED_END - 1
            prediction = float(runner.post_process_osi(runner.component_weighted_osi(parts))) if scoreable else np.nan
            actual.append({"fipsCode": str(row.fips_str), "timestamp_et": str(row.timestamp_et), "horizon": h, "prediction": prediction})
    actual = pd.DataFrame(actual).sort_values(["horizon", "fipsCode", "timestamp_et"]).reset_index(drop=True)
    expected = saved[["fipsCode", "timestamp_et", "horizon", "prediction"]].copy()
    expected["fipsCode"] = expected["fipsCode"].astype(str).str.zfill(5)
    expected = expected.sort_values(["horizon", "fipsCode", "timestamp_et"]).reset_index(drop=True)
    _require(actual[["fipsCode", "timestamp_et", "horizon"]].equals(expected[["fipsCode", "timestamp_et", "horizon"]]), "reloaded test keys mismatch")
    _require(np.allclose(actual.prediction.to_numpy(), expected.prediction.to_numpy(), equal_nan=True, rtol=0, atol=1e-7), "reloaded test predictions mismatch")
    submission = pd.read_csv(run_dir / "submission_phase2_dem_gat.csv", dtype={"fipsCode": str})
    template = pd.read_csv(SUBMISSION_FILE, dtype={"fipsCode": str})
    _require(list(submission.columns) == list(template.columns) and len(submission) == 9072, "submission schema mismatch")
    identifiers = [c for c in template.columns if c not in HORIZONS]
    _require(submission[identifiers].reset_index(drop=True).equals(template[identifiers].reset_index(drop=True)), "submission identifiers/order mismatch")
    for h in HORIZONS:
        hours = pd.to_datetime(template.timestamp_et) - pd.Timestamp("2026-03-11")
        hour_idx = (hours / pd.Timedelta(hours=1)).astype(int).to_numpy()
        expected = hour_idx + HORIZON_HOURS[h] < PRED_END
        values = submission[h].to_numpy(dtype=float)
        _require(int(np.isfinite(values).sum()) == OFFICIAL_SCOREABLE_ROWS[h], f"submission count mismatch: {h}")
        _require(np.array_equal(np.isfinite(values), expected), f"submission missing mask mismatch: {h}")
        _require(np.isfinite(values[expected]).all() and ((values[expected] >= 0) & (values[expected] <= 0.65)).all(), f"submission range mismatch: {h}")
    verification = {"protocol": PROTOCOL, "run_id": manifest["run_id"], "models_checked": True,
                    "outer_oof_checked": True, "independent_reload": True, "passed": True}
    (run_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    (run_dir / "COMPLETE").write_text("verified", encoding="utf-8")
    print(json.dumps(verification))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
