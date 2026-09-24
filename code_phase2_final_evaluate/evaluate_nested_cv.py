"""Nested outer-CV evaluation for the final I3 + m2_robust_input GAT.

This evaluator deliberately reuses ``code_phase2_final.final_model``.  It does
not reimplement the GAT or the I3 reader, and it never uses the outer-fold
labels while fitting an outer-fold model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "code_phase2_final"
PACKAGE_DEFAULT = FINAL_DIR / "I3_prediction_package_v1"
COMPARE_CV_DEFAULT = ROOT / "code_phase2_compare" / "cv" / "cv_assignments_balanced_v1_seed42.csv"
PACKAGE_CV_DEFAULT = PACKAGE_DEFAULT / "inputs" / "cv_seed42.csv"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "outputs" / "runs"

# This is the frozen hash recorded by code_phase2_compare and its completed
# seed-42 run manifest.  The I3 package's inputs/cv_seed42.csv has the same
# bytes.  The check prevents accidentally evaluating with another county split.
COMPARE_CV_SHA256 = "2ee47b74590051cec9d0d1ed0ac7d7a640250c8370433b1dfdac9b281ffe4021"
BOOTSTRAP_SEED = 20260910

sys.path.insert(0, str(FINAL_DIR))
from final_model import (  # noqa: E402
    GAT_EPOCHS,
    GAT_PATIENCE,
    HORIZONS,
    HORIZON_HOURS,
    N_FOLDS,
    PRED_END,
    FinalI3M2Runner,
    INPUT_POLICY,
    VARIANT,
    _grid_to_rows,
    _post_process_osi,
    _resolve_device,
    _sha256,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if y_true.shape != prediction.shape or y_true.size == 0:
        raise ValueError("metric arrays have incompatible shapes or are empty")
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("metric arrays contain non-finite values")
    error = y_true - prediction
    sse = float(np.sum(error * error, dtype=np.float64))
    mse = sse / len(y_true)
    total = float(np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))
    return {
        "n": int(len(y_true)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "medae": float(np.median(np.abs(error))),
        "bias": float(np.mean(prediction - y_true)),
        "error_std": float(np.std(error)),
        "r2": float(1.0 - sse / total) if total > 0.0 else np.nan,
        "max_abs_error": float(np.max(np.abs(error))),
    }


def _validate_cv_file(path: Path, runner: FinalI3M2Runner) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing comparison CV file: {path}")
    actual_hash = _sha256(path).lower()
    if actual_hash != COMPARE_CV_SHA256:
        raise ValueError(
            f"CV hash differs from code_phase2_compare seed-42 split: "
            f"{actual_hash} != {COMPARE_CV_SHA256}"
        )
    assignment = pd.read_csv(path, dtype={"fipsCode": str})
    expected_columns = ["fipsCode", "stateAbbr", "severity_tier", "fold"]
    if list(assignment.columns) != expected_columns:
        raise ValueError(f"CV columns differ from frozen compare contract: {assignment.columns.tolist()}")
    assignment["fips_str"] = (
        assignment["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    if assignment["fips_str"].duplicated().any():
        raise ValueError("CV assignment contains duplicate counties")
    if set(assignment["fold"]) != set(range(N_FOLDS)):
        raise ValueError("CV assignment does not contain exactly folds 0..4")
    counts = assignment["fold"].value_counts().sort_index().to_dict()
    if counts != {0: 48, 1: 47, 2: 48, 3: 48, 4: 48}:
        raise ValueError(f"CV county counts differ from compare contract: {counts}")

    meta_fips = runner.meta_train["fips_str"].astype(str).to_numpy()
    mapping = assignment.set_index("fips_str")["fold"]
    expected = pd.Series(meta_fips).map(mapping)
    if expected.isna().any() or not np.array_equal(expected.to_numpy(dtype=int), runner.row_fold_train):
        raise ValueError("I3 package folds_42 do not match code_phase2_compare CV assignment")


def _select_cv_file(package_dir: Path, explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        return explicit.resolve(), "explicit"
    compare_path = COMPARE_CV_DEFAULT
    if compare_path.is_file():
        return compare_path.resolve(), "code_phase2_compare/cv/cv_assignments_balanced_v1_seed42.csv"
    package_path = package_dir / "inputs" / "cv_seed42.csv"
    return package_path.resolve(), "I3 package inputs/cv_seed42.csv (same frozen compare hash)"


def _summary_rows(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scoreable = outer[outer["is_scoreable"].astype(bool)]
    for horizon, group in scoreable.groupby("horizon", sort=True):
        y = group["y_true"].to_numpy(dtype=float)
        base = group["baseline_pred"].to_numpy(dtype=float)
        gat = group["gat_pred"].to_numpy(dtype=float)
        base_metrics = _metrics(y, base)
        gat_metrics = _metrics(y, gat)
        for model, values in (("baseline", base_metrics), ("gat", gat_metrics)):
            row = {"horizon": horizon, "model": model, **values}
            row["rmse_improvement_vs_baseline"] = (
                0.0 if model == "baseline" else base_metrics["rmse"] - values["rmse"]
            )
            row["mae_improvement_vs_baseline"] = (
                0.0 if model == "baseline" else base_metrics["mae"] - values["mae"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _fold_rows(outer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scoreable = outer[outer["is_scoreable"].astype(bool)]
    for (horizon, outer_fold), group in scoreable.groupby(["horizon", "outer_fold"], sort=True):
        y = group["y_true"].to_numpy(dtype=float)
        base = group["baseline_pred"].to_numpy(dtype=float)
        gat = group["gat_pred"].to_numpy(dtype=float)
        base_metrics = _metrics(y, base)
        gat_metrics = _metrics(y, gat)
        alpha_values = group["alpha"].unique()
        if len(alpha_values) != 1:
            raise ValueError("alpha is not constant within an outer fold")
        for model, values in (("baseline", base_metrics), ("gat", gat_metrics)):
            row = {
                "horizon": horizon,
                "outer_fold": int(outer_fold),
                "model": model,
                "alpha": 0.0 if model == "baseline" else float(alpha_values[0]),
                **values,
            }
            row["rmse_improvement_vs_baseline"] = (
                0.0 if model == "baseline" else base_metrics["rmse"] - values["rmse"]
            )
            row["mae_improvement_vs_baseline"] = (
                0.0 if model == "baseline" else base_metrics["mae"] - values["mae"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _county_and_bootstrap(outer: pd.DataFrame, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    county_rows = []
    scoreable = outer[outer["is_scoreable"].astype(bool)]
    for (horizon, fips), group in scoreable.groupby(["horizon", "fipsCode"], sort=True):
        y = group["y_true"].to_numpy(dtype=float)
        base_error = group["baseline_pred"].to_numpy(dtype=float) - y
        gat_error = group["gat_pred"].to_numpy(dtype=float) - y
        base_sse = float(np.sum(base_error * base_error, dtype=np.float64))
        gat_sse = float(np.sum(gat_error * gat_error, dtype=np.float64))
        county_rows.append({
            "horizon": horizon,
            "fipsCode": str(fips),
            "outer_fold": int(group["outer_fold"].iloc[0]),
            "n": int(len(group)),
            "base_sse": base_sse,
            "gat_sse": gat_sse,
            "base_rmse": float(np.sqrt(base_sse / len(group))),
            "gat_rmse": float(np.sqrt(gat_sse / len(group))),
            "gat_minus_base_sse": gat_sse - base_sse,
            "gat_better": bool(gat_sse < base_sse),
        })
    county = pd.DataFrame(county_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_rows = []
    for horizon, group in county.groupby("horizon", sort=True):
        base_sse = group["base_sse"].to_numpy(dtype=float)
        gat_sse = group["gat_sse"].to_numpy(dtype=float)
        n = group["n"].to_numpy(dtype=float)
        draws = rng.integers(0, len(group), size=(replicates, len(group)))
        total_n = n[draws].sum(axis=1)
        delta = np.sqrt(gat_sse[draws].sum(axis=1) / total_n) - np.sqrt(
            base_sse[draws].sum(axis=1) / total_n
        )
        base_rmse = float(np.sqrt(base_sse.sum() / n.sum()))
        gat_rmse = float(np.sqrt(gat_sse.sum() / n.sum()))
        bootstrap_rows.append({
            "horizon": horizon,
            "replicates": int(replicates),
            "seed": BOOTSTRAP_SEED,
            "n_counties": int(len(group)),
            "base_rmse": base_rmse,
            "gat_rmse": gat_rmse,
            "delta_rmse_gat_minus_base": gat_rmse - base_rmse,
            "delta_ci_low": float(np.quantile(delta, 0.025)),
            "delta_ci_high": float(np.quantile(delta, 0.975)),
            "gat_better_probability": float(np.mean(delta < 0.0)),
        })
    return county, pd.DataFrame(bootstrap_rows)


class NestedEvaluator:
    def __init__(
        self,
        package_dir: Path,
        output_dir: Path,
        run_id: str,
        seed: int,
        device: str,
        epochs: int,
        patience: int,
        cv_file: Path | None,
        bootstrap_replicates: int,
    ):
        if seed != 42:
            raise ValueError("This evaluator is pinned to code_phase2_compare's CV seed 42")
        self.package_dir = package_dir.resolve()
        self.run_dir = (output_dir / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.device = device
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.bootstrap_replicates = int(bootstrap_replicates)
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        self.runner = FinalI3M2Runner(
            self.package_dir,
            self.seed,
            self.device,
            self.epochs,
            self.patience,
            self.run_dir,
        )
        self.cv_path, self.cv_source = _select_cv_file(self.package_dir, cv_file)
        _validate_cv_file(self.cv_path, self.runner)
        self.all_folds = tuple(range(N_FOLDS))

    def run(self) -> Path:
        outer_records = []
        alpha_records = []
        for horizon in HORIZONS:
            horizon_hours = HORIZON_HOURS[horizon]
            for outer_fold in self.all_folds:
                train_scope = tuple(fold for fold in self.all_folds if fold != outer_fold)
                print(
                    f"[EVAL] horizon={horizon} outer_fold={outer_fold} "
                    f"train_scope={train_scope}: inner alpha selection",
                    flush=True,
                )
                selection = self.runner._select_alpha(train_scope, horizon)
                alpha = float(selection["selected_alpha"])
                for candidate in selection["candidates"]:
                    alpha_records.append({
                        "horizon": horizon,
                        "outer_fold": outer_fold,
                        "selection_type": "outer_train_inner_cv",
                        "allowed_folds": json.dumps(list(train_scope)),
                        "selected_alpha": alpha,
                        **candidate,
                    })

                stack = self.runner._fit_scope(train_scope, horizon)
                inputs = stack["inputs"]
                base_rows, base_coverage = _grid_to_rows(
                    inputs["base_grid"], inputs["train_rows"], self.runner.n_train
                )
                correction_rows, correction_coverage = _grid_to_rows(
                    stack["correction"], inputs["train_rows"], self.runner.n_train
                )
                if not np.all(base_coverage == 1) or not np.all(correction_coverage == 1):
                    raise ValueError("outer graph does not cover every training row exactly once")

                validation_rows = np.flatnonzero(self.runner.row_fold_train == outer_fold)
                scoreable = (
                    self.runner.meta_train.iloc[validation_rows]["hour_idx"].to_numpy(dtype=int)
                    + horizon_hours
                    <= PRED_END - 1
                )
                global_rows = self.runner.train_positions[validation_rows]
                score_global_rows = global_rows[scoreable]
                y_true = np.full(len(validation_rows), np.nan, dtype=float)
                y_true[scoreable] = self.runner.package.labels_for_rows(
                    self.seed,
                    self.all_folds,
                    horizon_hours,
                    score_global_rows,
                )
                base_raw = base_rows[validation_rows]
                gat_raw = base_raw + alpha * correction_rows[validation_rows]
                baseline_pred = _post_process_osi(base_raw)
                gat_pred = _post_process_osi(gat_raw)
                baseline_pred[~scoreable] = np.nan
                gat_pred[~scoreable] = np.nan
                gat_raw[~scoreable] = np.nan
                base_raw[~scoreable] = np.nan

                meta = self.runner.meta_train.iloc[validation_rows].reset_index(drop=True)
                for i in range(len(validation_rows)):
                    outer_records.append({
                        "fipsCode": str(meta.loc[i, "fips_str"]),
                        "timestamp_et": meta.loc[i, "timestamp_et"],
                        "hour_idx": int(meta.loc[i, "hour_idx"]),
                        "outer_fold": int(outer_fold),
                        "horizon": horizon,
                        "is_scoreable": bool(scoreable[i]),
                        "y_true": y_true[i],
                        "baseline_pred": baseline_pred[i],
                        "gat_pred": gat_pred[i],
                        "baseline_pred_raw": base_raw[i],
                        "gat_pred_raw": gat_raw[i],
                        "correction_osi": (
                            float(correction_rows[validation_rows[i]]) if scoreable[i] else np.nan
                        ),
                        "alpha": alpha if scoreable[i] else np.nan,
                        "train_scope": json.dumps(list(train_scope)),
                    })

        outer = pd.DataFrame(outer_records)
        expected_rows = self.runner.n_train * len(HORIZONS)
        if len(outer) != expected_rows:
            raise ValueError(f"outer row count {len(outer)} != {expected_rows}")
        if outer.duplicated(["fipsCode", "timestamp_et", "horizon"]).any():
            raise ValueError("outer OOF keys are duplicated")
        for horizon in HORIZONS:
            subset = outer[outer["horizon"] == horizon]
            expected = subset["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[horizon] <= PRED_END - 1
            if not np.array_equal(subset["is_scoreable"].to_numpy(dtype=bool), expected):
                raise ValueError(f"scoreable mask mismatch for {horizon}")
            for column in ["y_true", "baseline_pred", "gat_pred"]:
                values = subset[column].to_numpy(dtype=float)
                if not np.isfinite(values[expected]).all() or np.isfinite(values[~expected]).any():
                    raise ValueError(f"invalid finite-value pattern in {horizon}/{column}")

        summary = _summary_rows(outer)
        folds = _fold_rows(outer)
        county, bootstrap = _county_and_bootstrap(outer, self.bootstrap_replicates)
        alpha_frame = pd.DataFrame(alpha_records)

        outer.to_parquet(self.run_dir / "outer_oof.parquet", index=False)
        alpha_frame.to_csv(self.run_dir / "alpha_selection.csv", index=False)
        folds.to_csv(self.run_dir / "fold_metrics.csv", index=False)
        summary.to_csv(self.run_dir / "cv_summary.csv", index=False)
        county.to_csv(self.run_dir / "county_metrics.csv", index=False)
        bootstrap.to_csv(self.run_dir / "county_bootstrap.csv", index=False)

        artifact_names = [
            "outer_oof.parquet",
            "alpha_selection.csv",
            "fold_metrics.csv",
            "cv_summary.csv",
            "county_metrics.csv",
            "county_bootstrap.csv",
        ]
        manifest = {
            "protocol": "i3_base_m2_robust_input_nested_outer_cv_v1",
            "variant": VARIANT,
            "input_policy": INPUT_POLICY,
            "seed": self.seed,
            "cv_file": str(self.cv_path),
            "cv_source": self.cv_source,
            "cv_sha256": _sha256(self.cv_path),
            "code_phase2_compare_cv_sha256": COMPARE_CV_SHA256,
            "package_complete_sha256": _sha256(self.package_dir / "PACKAGE_COMPLETE.json"),
            "device": self.device,
            "epochs": self.epochs,
            "patience": self.patience,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": self.bootstrap_replicates,
            "outer_folds": list(self.all_folds),
            "horizons": list(HORIZONS),
            "outer_rows": int(len(outer)),
            "scoreable_rows": {
                h: int(outer.loc[outer["horizon"] == h, "is_scoreable"].sum()) for h in HORIZONS
            },
            "artifact_hashes": {
                name: _sha256(self.run_dir / name) for name in artifact_names
            },
        }
        _write_json(self.run_dir / "run_manifest.json", manifest)
        (self.run_dir / "CV_COMPLETE").write_text("PASS\n", encoding="utf-8")
        print(f"EVAL_READY={self.run_dir}", flush=True)
        return self.run_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DEFAULT)
    parser.add_argument("--cv-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--run-id", default="i3_m2_nested_cv_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    parser.add_argument("--patience", type=int, default=GAT_PATIENCE)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.seed != 42:
        raise ValueError("--seed must be 42 to use code_phase2_compare's fixed split")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    evaluator = NestedEvaluator(
        args.package_dir,
        args.output_dir,
        args.run_id,
        args.seed,
        device,
        args.epochs,
        args.patience,
        args.cv_file,
        args.bootstrap_replicates,
    )
    evaluator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
