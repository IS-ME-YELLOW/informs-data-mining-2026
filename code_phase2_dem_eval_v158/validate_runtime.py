"""Explicit diagnostic-only small-model acceptance, in fresh worker processes.

Default backend is real and requires Torch. --backend mock performs arithmetic
stand-in checks only and can never mark real-model acceptance as passed.
Nothing here writes formal outputs/runs or changes frozen data/CV files.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd

import base_model as base
import config
import diagnostic_support as support
import model_records
import run_identity
import stacking


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backend", choices=("real", "mock"), default="real")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--modes", default="direct,component_v158")
    p.add_argument("--horizons", default="1,6,24,48")
    p.add_argument("--outer-folds", default="0,1,2,3,4")
    p.add_argument("--counties-per-fold", type=int, default=2)
    p.add_argument("--test-counties", type=int, default=2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--base-rounds", type=int, default=12)
    p.add_argument("--base-patience", type=int, default=3)
    p.add_argument("--worker", choices=("outer", "outer_changed", "inner", "inner_changed", "crossfit",
                                       "crossfit_changed", "positive", "prepare", "resume", "reload", "rounds"))
    a = p.parse_args(argv)
    a.mode_list = a.modes.split(",")
    a.horizon_list = [f"osi_target_t{int(h):02d}h" for h in a.horizons.split(",")]
    a.fold_list = [int(k) for k in a.outer_folds.split(",")]
    if (not a.mode_list or set(a.mode_list) - {"direct", "component_v158"}
            or set(a.horizon_list) - set(config.HORIZONS) or set(a.fold_list) - set(range(5))
            or len(set(a.mode_list)) != len(a.mode_list) or len(set(a.horizon_list)) != len(a.horizon_list)
            or len(set(a.fold_list)) != len(a.fold_list)):
        p.error("invalid/duplicate mode, horizon or fold")
    if min(a.counties_per_fold, a.test_counties, a.epochs, a.base_rounds, a.base_patience) < 1:
        p.error("all budgets/counts must be positive")
    if a.base_rounds < 2:
        p.error("base-rounds must be >=2 to test shortened boosters")
    return a


def save_json(path, value):
    model_records.atomic_json(path, value)


def receipts(root):
    result = {}
    for path in Path(root).rglob("*.complete.json"):
        record = json.loads(path.read_text())
        result[record["model_id"]] = {"receipt": str(path.relative_to(root)), "details": record["details"],
                                     "receipt_hash": model_records.file_hash(path), "model_hash": record["sha256"]}
    return result


def round_boundary(a, root):
    X = pd.DataFrame({"constant": np.ones(100)})
    y = pd.DataFrame({h: np.zeros(100) for h in config.HORIZONS})
    m = pd.DataFrame({"hour_idx": np.tile(np.arange(72, 92), 5)})
    f = np.repeat(np.arange(5), 20)
    folds = [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(5)]
    def store():
        return base.BaseModelStore(X, X.iloc[:5], m, folds, y_train=y, model_dir=root/"models/base",
                    run_identity="diagnostic-round-boundary-v2", reuse_existing=True,
                    fit_options={"round_limit": a.base_rounds, "patience": a.base_patience})
    # Fix probe recommendations to exercise a requested budget larger than the
    # constant-feature refit's actual tree count. This is a serialization unit
    # test, not a claimed nested score or new tree-count selection policy.
    with patch.object(base, "_fit_probe", return_value=(None, a.base_rounds)):
        if a.backend == "mock":
            original = base._fit_fixed
            def shortened(X, y, rounds, seed):
                return support.ArithmeticBooster(mean=0., rounds=1, columns=list(X.columns))
            with patch.object(base, "_fit_fixed", side_effect=shortened):
                fitted = store().get((0, 1), config.HORIZONS[0])
        else:
            fitted = store().get((0, 1), config.HORIZONS[0])
    if not 1 <= fitted.actual_rounds < fitted.final_rounds:
        raise AssertionError("constant-feature test did not exercise actual < requested")
    loaded = store()
    with patch.object(base, "fit_base", side_effect=AssertionError("reload must not fit")):
        loaded.load_from_run(root)
        other = loaded.get((0, 1), config.HORIZONS[0])
    np.testing.assert_array_equal(fitted.predict(X), other.predict(X))
    path = loaded._model_path((0, 1), config.HORIZONS[0], "osi")
    receipt = model_records.record_path(path)
    original_receipt = receipt.read_bytes()
    bad = json.loads(original_receipt)
    bad["details"]["actual_rounds"] += 1
    save_json(receipt, bad)
    try:
        store().load_from_run(root)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect actual-round metadata was accepted")
    receipt.write_bytes(original_receipt)
    original_bytes = path.read_bytes()
    path.write_bytes(original_bytes[:len(original_bytes)//2])
    try:
        store().load_from_run(root)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated booster was accepted")
    path.write_bytes(original_bytes)
    return {"requested_rounds": fitted.final_rounds, "actual_rounds": fitted.actual_rounds,
            "reload_equal": True, "tamper_rejected": True}


def worker(a):
    root = Path(a.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode, h, k = a.mode_list[0], a.horizon_list[0], a.fold_list[0]
    if a.worker == "rounds":
        result = round_boundary(a, root)
        save_json(root / "rounds.json", result)
        return
    inference = a.worker == "reload"
    guard = nullcontext()
    if inference:
        from verify_artifacts import _inference_guard
        guard = _inference_guard({"inputs": {"feature_dir": str(config.DEFAULT_FEATURE_DIR)}})
    with guard:
        fixture = support.small_fixture(a.counties_per_fold, a.test_counties, include_supervision=not inference)
        A = tuple(i for i in range(5) if i != k)
        j, r = (k+1)%5, (k+2)%5
        B = tuple(i for i in A if i != j)
        mutate = {"outer_changed": k, "inner_changed": j, "crossfit_changed": r, "positive": r}.get(a.worker)
        persistence = root / ("resumed" if a.worker in {"prepare", "resume"} else "continuous")
        store_output = persistence if a.worker in {"outer", "prepare", "resume", "reload"} else None
        ctx = support.context(fixture, mode, device=a.device, epochs=a.epochs, base_rounds=a.base_rounds,
                              base_patience=a.base_patience, mutate_fold=mutate, output=store_output,
                              resume=a.worker in {"resume", "reload"}, inference=inference)
        if a.worker == "prepare":
            ctx.fit_stack(B, h, mode)
            committed = receipts(persistence)
            component = "osi" if mode == "direct" else "P_t"
            original_atomic = model_records.atomic_json
            def interrupted(path, payload):
                if str(path).endswith(".txt.complete.json"):
                    raise InterruptedError("diagnostic interruption after model write, before completion record")
                return original_atomic(path, payload)
            try:
                with patch.object(model_records, "atomic_json", side_effect=interrupted):
                    ctx.base_store.get(A, h, component)
            except InterruptedError:
                pass
            else:
                raise AssertionError("interruption was not injected")
            orphan = ctx.base_store._model_path(A, h, component)
            if not orphan.exists() or model_records.record_path(orphan).exists():
                raise AssertionError("interruption did not leave an uncommitted model")
            save_json(root / "prepared.json", committed)
            return
        fit_calls = []
        with ExitStack() as instrument:
            if a.worker == "resume":
                committed = json.loads((root / "prepared.json").read_text())
                original_base, original_gat = base.fit_base, stacking.fit_gat
                def checked_base(*args, **kwargs):
                    component = kwargs.get("component", "osi")
                    name = f"base:{component}:{args[5]}:S{','.join(map(str,args[4]))}"
                    if name in committed: raise AssertionError("completed base was refitted")
                    fit_calls.append(name)
                    return original_base(*args, **kwargs)
                seeds = {v["details"]["seed"] for name, v in committed.items() if name.startswith("stack:")}
                def checked_gat(*args, **kwargs):
                    if kwargs["seed"] in seeds: raise AssertionError("completed GAT was refitted")
                    fit_calls.append(f"gat-seed:{kwargs['seed']}")
                    return original_gat(*args, **kwargs)
                instrument.enter_context(patch.object(base, "fit_base", side_effect=checked_base))
                instrument.enter_context(patch.object(stacking, "fit_gat", side_effect=checked_gat))
            kind = "inner" if a.worker.startswith("inner") else "crossfit" if a.worker.startswith("crossfit") else "outer"
            override = None
            if inference:
                with np.load(root / "outer.npz") as snapshot: override = float(snapshot["alpha"][0])
            values = support.snapshot(ctx, h, kind=kind, outer_fold=k, alpha_override=override)
        if a.worker == "resume":
            current = receipts(persistence)
            for name, original in committed.items():
                if current[name] != original: raise AssertionError("completed receipt/model bytes changed on resume")
            save_json(root / "resume_calls.json", fit_calls)
        np.savez_compressed(root / f"{a.worker}.npz", **values)


def driver(a):
    root = Path(a.output_dir).resolve()
    if root.exists(): raise FileExistsError("diagnostic output already exists; choose a fresh directory")
    if (config.OUTPUT_ROOT / "runs") in root.parents or root == config.OUTPUT_ROOT / "runs":
        raise ValueError("diagnostics cannot write inside formal outputs/runs")
    root.mkdir(parents=True)
    summary = {"diagnostic_only": True, "backend": a.backend, "device": a.device,
               "modes": a.mode_list, "horizons": a.horizon_list, "outer_folds": a.fold_list,
               "budget": {"epochs": a.epochs, "base_rounds": a.base_rounds, "base_patience": a.base_patience},
               "source_hashes": run_identity.source_hashes(), "environment": None,
               "diagnostic_source_hashes": {
                   name: model_records.file_hash(Path(__file__).resolve().parent/name)
                   for name in ("diagnostic_support.py", "validate_runtime.py")},
               "input_hashes": {},
               "passed": False, "real_acceptance_passed": False, "cases": []}
    try:
        summary["input_hashes"] = {
            str(path.relative_to(config.PROJECT_ROOT)): model_records.file_hash(path)
            for path in (config.CV_FILE, config.COMPONENT_TARGETS_FILE, config.GEO_DBF,
                         config.GEO_DBF.with_suffix(".shp"), config.TERRAIN_FILE,
                         config.DEFAULT_FEATURE_DIR/"manifest.json")}
        summary["environment"] = run_identity.environment(a.device)
        if a.backend == "real":
            import torch
            if a.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        for mode in a.mode_list:
            for h in a.horizon_list:
                for k in a.fold_list:
                    folder = root / mode / h / f"outer{k}"
                    folder.mkdir(parents=True)
                    for case in ["outer", "outer_changed", "inner", "inner_changed", "crossfit", "crossfit_changed",
                                 "positive", "prepare", "resume", "reload"]:
                        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", case,
                            "--output-dir", str(folder), "--backend", a.backend, "--device", a.device,
                            "--modes", mode, "--horizons", str(config.HORIZON_HOURS[h]), "--outer-folds", str(k),
                            "--counties-per-fold", str(a.counties_per_fold), "--test-counties", str(a.test_counties),
                            "--epochs", str(a.epochs), "--base-rounds", str(a.base_rounds), "--base-patience", str(a.base_patience)]
                        print(f"[{mode}/{h}/outer{k}] {case}", flush=True)
                        with (folder / f"{case}.log").open("w", encoding="utf-8") as log:
                            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                        if process.returncode != 0: raise RuntimeError(f"worker failed: {folder / (case+'.log')}")
                    def snapshot(name):
                        with np.load(folder / f"{name}.npz") as values: return dict(values)
                    for left, right in [("outer", "outer_changed"), ("inner", "inner_changed"),
                                        ("crossfit", "crossfit_changed"), ("outer", "resume"), ("outer", "reload")]:
                        support.compare_snapshots(snapshot(left), snapshot(right))
                    try:
                        support.compare_snapshots(snapshot("outer"), snapshot("positive"))
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("positive control did not detect changed permitted labels")
                    summary["cases"].append({"mode": mode, "horizon": h, "outer_fold": k, "passed": True})
                    save_json(root / "diagnostic_report.json", summary)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "rounds", "--output-dir", str(root/"rounds"),
                   "--backend", a.backend, "--device", a.device, "--base-rounds", str(a.base_rounds)]
        with (root / "rounds.log").open("w", encoding="utf-8") as log:
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"round-boundary worker failed: {root/'rounds.log'}")
        summary["passed"] = True
        summary["full_coverage"] = (set(a.mode_list)=={"direct","component_v158"}
            and set(a.horizon_list)==set(config.HORIZONS) and set(a.fold_list)==set(range(5)))
        summary["real_acceptance_passed"] = a.backend == "real" and summary["full_coverage"]
    except Exception:
        summary["error"] = traceback.format_exc()
        raise
    finally:
        save_json(root / "diagnostic_report.json", summary)
    print(json.dumps({k: summary[k] for k in ["passed", "full_coverage", "real_acceptance_passed"]}), flush=True)


def main(argv=None):
    a = arguments(argv)
    run_identity.configure_determinism()
    root = Path(a.output_dir).resolve()
    if (config.OUTPUT_ROOT / "runs") in root.parents or root == config.OUTPUT_ROOT / "runs":
        raise ValueError("diagnostics cannot write inside formal outputs/runs")
    if a.worker:
        with support.arithmetic_backend() if a.backend == "mock" else nullcontext():
            worker(a)
    else:
        driver(a)


if __name__ == "__main__":
    main()
