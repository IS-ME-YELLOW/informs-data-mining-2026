"""Read-only reload of saved v1.8 models, with explicit relocated-path mapping.

Does not edit historical manifests or run the existing writing verifier. A
numeric reproduction pass does not certify the original CV selection protocol.
"""
from pathlib import Path
import ast
import hashlib
import json
import runpy
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
V18 = ROOT / "versions/xyy/v1.5.8"
FROZEN = ROOT / "versions/xyy/v1.5.6"


def functions_from(path, wanted, namespace):
    body = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(body) == len(wanted)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def maximum_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b))
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    mask = np.isfinite(a)
    return float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0


def main():
    start = time.monotonic()
    cfg = runpy.run_path(str(V18 / "config.py"))
    modes = ("C0_direct_osi", "C1_component_osi", "C2_equal_blend", "C3_aligned_component")
    ns = dict(cfg, pd=pd, np=np, MODELS=modes)
    functions_from(V18 / "protocol.py", {"post_process", "clip_component", "compose_osi", "align_same_target"}, ns)
    functions_from(V18 / "train_v18.py", {"_build_controls"}, ns)
    frames = {}
    for split in ["train", "test"]:
        x = pd.read_parquet(FROZEN / f"features_{split}_v1.5.6.parquet")
        meta = pd.read_parquet(FROZEN / f"meta_{split}_v1.5.6.parquet")
        meta["fipsCode"] = meta.fipsCode.astype(str).str.zfill(5)
        meta["timestamp_et"] = pd.to_datetime(meta.timestamp_et)
        frames[split] = (x, meta)
    y = pd.read_parquet(FROZEN / "targets_train_v1.5.6.parquet")
    targets = pd.read_parquet(V18 / "component_targets_v1.8.parquet")
    folds = pd.read_csv(ROOT / "cv/cv_assignments_balanced_v1_seed42.csv", dtype={"fipsCode": str}).set_index("fipsCode").fold
    row_folds = frames["train"][1].fipsCode.map(folds).to_numpy()
    H, C = cfg["HORIZONS"], cfg["COMPONENTS"]
    direct = {s: {h: np.full(len(frames[s][0]), np.nan) for h in H} for s in frames}
    comp = {s: {h: {c: np.full(len(frames[s][0]), np.nan) for c in C} for h in H} for s in frames}
    manifest = pd.read_csv(V18 / "model_manifest.csv")
    assert len(manifest) == 120
    audits = []
    for index, row in manifest.iterrows():
        assert row.path.startswith("versions/v1.8/models/")
        path = V18 / "models" / row.path.split("models/", 1)[1]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        model = lgb.Booster(model_file=str(path))
        assert model.num_feature() == 163 and model.current_iteration() == int(row.rounds)
        split = "train" if row.role == "cv" else "test"
        x, meta = frames[split]
        if split == "train":
            truth = y[row.target] if row.target_kind == "direct_osi" else targets[row.target]
            ix = np.flatnonzero((row_folds == int(row.fold)) & truth.notna().to_numpy())
        else:
            ix = np.arange(len(x))
        pred = model.predict(x.iloc[ix], num_iteration=int(row.rounds), num_threads=1)
        assert np.isfinite(pred).all()
        if row.target_kind == "direct_osi":
            direct[split][row.target][ix] = pred
        else:
            component, suffix = row.target.split("_target_")
            comp[split][f"osi_target_{suffix}"][component][ix] = pred
        audits.append({"role": row.role, "target": row.target, "fold": row.fold,
                       "historical_path": row.path, "actual_path": str(path.relative_to(ROOT)),
                       "historical_sha256": row.sha256, "current_sha256": digest,
                       "historical_hash_matches": digest == row.sha256,
                       "line_ending_conversion_matches_historical_hash": any(hashlib.sha256(b).hexdigest() == row.sha256 for b in
                           [payload.replace(b'\r\n', b'\n'), payload.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')]),
                       "features": model.num_feature(), "rounds": int(row.rounds), "predicted_rows": len(ix)})
        if (index+1) % 20 == 0:
            print(f"Read-only model reload {index+1}/120", flush=True)
    stored = {
        "train": pd.read_parquet(V18 / "oof_predictions.parquet"),
        "test": pd.read_parquet(V18 / "test_control_predictions.parquet"),
    }
    stored_comp = {
        "train": pd.read_parquet(V18 / "oof_component_predictions.parquet"),
        "test": pd.read_parquet(V18 / "test_component_predictions.parquet"),
    }
    differences, metric_rows = [], []
    controls = {}
    for split, (_, meta) in frames.items():
        controls[split], auxiliary = ns["_build_controls"](meta, direct[split], comp[split])
        for h in H:
            for c in C:
                name = f"pred_{c}_target_{h.rsplit('_',1)[-1]}"
                differences.append(maximum_difference(ns["clip_component"](comp[split][h][c]), stored_comp[split][name]))
            for mode in modes:
                pred = controls[split][mode][h]
                differences.append(maximum_difference(pred, stored[split][f"pred_{mode}_{h}"]))
                if split == "train":
                    target = y[h].to_numpy(dtype=float)
                    expected = np.isfinite(target)
                    assert np.array_equal(np.isfinite(pred), expected)
                    error = pred[expected] - target[expected]
                    metric_rows.append({"model": mode, "horizon": h, "n": int(expected.sum()),
                                        "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))})
        for component, unique in auxiliary["unique_components"].groupby("component"):
            assert unique.candidate_count.value_counts().to_dict() == {k:v*meta.fipsCode.nunique() for k,v in {1:5,2:18,3:24,4:96}.items()}
    template = pd.read_csv(ROOT / "data/sample_submission.csv", dtype={"fipsCode": str}, float_precision="round_trip")
    meta_test = frames["test"][1]
    identifiers = [column for column in template if column not in H]
    submission_counts = {}
    for mode in modes:
        submission = pd.read_csv(V18 / f"submission_v1.8_{mode}_balanced_v1.csv", dtype={"fipsCode": str}, float_precision="round_trip")
        pd.testing.assert_frame_equal(submission[identifiers], template[identifiers])
        key = pd.MultiIndex.from_arrays([submission.fipsCode, pd.to_datetime(submission.timestamp_et)])
        mapped = pd.Series(np.arange(len(meta_test)), index=pd.MultiIndex.from_frame(meta_test[["fipsCode", "timestamp_et"]])).loc[key].to_numpy()
        submission_counts[mode] = {}
        for h in H:
            expected = meta_test.hour_idx.to_numpy()+cfg["HORIZON_HOURS"][h] <= 215
            pred = controls["test"][mode][h].copy()
            pred[~expected] = np.nan
            differences.append(maximum_difference(pred[mapped], submission[h]))
            assert np.array_equal(submission[h].notna(), expected[mapped])
            assert submission.loc[expected[mapped], h].between(0,.65).all()
            submission_counts[mode][h] = int(expected.sum())
    maximum = max(differences)
    assert maximum < 1e-12, maximum
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(OUT / "saved_model_reload_audit.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(OUT / "recomputed_v18_metrics.csv", index=False)
    result = {"numeric_reproduction_passed": True, "training_performed": False,
              "models_reloaded": len(audits), "cv_models": int((audit_frame.role=='cv').sum()),
              "final_models": int((audit_frame.role=='final').sum()),
              "historical_model_hashes_matching": int(audit_frame.historical_hash_matches.sum()),
              "historical_hashes_matching_after_LF_CRLF_conversion": int(audit_frame.line_ending_conversion_matches_historical_hash.sum()),
              "maximum_prediction_difference": maximum, "all_submission_identifiers_and_masks_match": True,
              "submission_valid_counts": submission_counts,
              "path_policy": "Explicit read-only alias versions/v1.8/models -> versions/xyy/v1.5.8/models; no historical files rewritten",
              "caveat": "Current files reproduce saved numbers, but differing historical hashes remain a provenance issue; original early stopping is not strict nested CV.",
              "elapsed_seconds": time.monotonic()-start}
    (OUT / "saved_model_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
