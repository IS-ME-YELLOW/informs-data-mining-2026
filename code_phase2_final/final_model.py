"""Final I3-base + m2_robust_input residual-GAT submission runner.

The I3 package is the only source of the base prediction.  The GAT
implementation is copied verbatim from code_phase2_compare/gat_model.py and
is used with the m2_robust_input policy:

    residual = official OSI - I3_OSI
    input    = scope-fitted standardisation followed by fixed z clipping [-5, 5]
    model    = two-layer edge-aware residual GAT with the raw-input skip path

This runner performs the I3 scope-aware inner alpha selection and final
five-fold fit, then writes the official submission template without changing
identifier columns or row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent / "I3_prediction_package_v1"
M2_SOURCE_DIR = Path(__file__).resolve().parent / "m2_source"
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(M2_SOURCE_DIR))

from data import add_neighbor_feature_aggregates, load_terrain_features, make_county_time_view
from prediction_reader import PredictionPackage
from spatial import build_spatial_graph


HORIZONS = ("osi_target_t01h", "osi_target_t06h", "osi_target_t24h", "osi_target_t48h")
HORIZON_HOURS = {
    "osi_target_t01h": 1,
    "osi_target_t06h": 6,
    "osi_target_t24h": 24,
    "osi_target_t48h": 48,
}
PRED_START = 72
PRED_END = 216
N_FOLDS = 5
GRAPH_K = 8
GAT_HIDDEN = 24
GAT_HEADS = 4
GAT_DROPOUT = 0.12
GAT_LR = 0.003
GAT_WEIGHT_DECAY = 2e-4
GAT_EPOCHS = 220
GAT_PATIENCE = 35
GAT_TIME_STRIDE = 1
GAT_ALPHA_GRID = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
VARIANT = "m2_robust_input"
INPUT_POLICY = {"skip": True, "input_clip_z": 5.0, "correction_limit": None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scoped_seed(global_seed: int, scope, horizon: str) -> int:
    """Same stable scope seed convention used by the comparison runner."""

    scope = tuple(sorted(int(v) for v in scope))
    payload = json.dumps(
        [int(global_seed), "gat", scope, str(horizon), "fit"],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _post_process_osi(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("cannot post-process non-finite OSI values")
    return np.where(np.clip(values, 0.0, 0.65) < 0.001, 0.0, np.clip(values, 0.0, 0.65))


def _strict_standardize(features, train_node_mask, valid_time):
    """Exact m2 scope-fitted standardisation semantics."""

    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise FloatingPointError("raw graph features are non-finite before standardization")
    mask = np.asarray(valid_time, dtype=bool)[:, None] & np.asarray(train_node_mask, dtype=bool)[None, :]
    if not mask.any():
        raise ValueError("standardization has no allowed training cells")
    flat = features.reshape(-1, features.shape[-1])
    train_values = flat[mask.reshape(-1)]
    mean = np.mean(train_values, axis=0, dtype=np.float64)
    std = np.std(train_values, axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("feature standardization statistics are non-finite")
    std[std < 1e-6] = 1.0
    scaled = ((features.astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("standardized graph features are non-finite")
    return scaled, mean, std


def _apply_m2_input_policy(scaled):
    values = np.clip(np.asarray(scaled, dtype=np.float32), -5.0, 5.0).astype(np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("m2 input policy produced non-finite features")
    return values


def _residual_scale(residual_grid, mask) -> float:
    values = np.asarray(residual_grid, dtype=float)[np.asarray(mask, dtype=bool)]
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("residual scale has no finite allowed supervision")
    return float(max(np.std(values, ddof=0), 0.003))


def _grid_to_rows(grid, rows, n_rows):
    values = np.full(n_rows, np.nan, dtype=float)
    coverage = np.zeros(n_rows, dtype=int)
    for t in range(rows.shape[0]):
        for node in range(rows.shape[1]):
            row_id = int(rows[t, node])
            if row_id >= 0:
                if coverage[row_id] != 0:
                    raise ValueError(f"duplicate grid coverage for row {row_id}")
                values[row_id] = grid[t, node]
                coverage[row_id] = 1
    return values, coverage


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _save_checkpoint(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(payload)
    state["state_dict"] = {
        name: tensor.detach().cpu() for name, tensor in payload["state_dict"].items()
    }
    torch.save(state, path)


class FinalI3M2Runner:
    def __init__(self, package_dir: Path, seed: int, device: str, epochs: int, patience: int, output_dir: Path):
        if seed not in (42, 20260917, 20260918):
            raise ValueError("seed must be one of the frozen I3 seeds: 42, 20260917, 20260918")
        self.package_dir = package_dir.resolve()
        self.seed = int(seed)
        self.device = device
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.output_dir = output_dir.resolve()
        self.package = PredictionPackage(self.package_dir)
        self.meta = self.package.meta.copy()
        self.features = self.package.model_features("base")
        if len(self.meta) != 43488 or self.features.shape != (43488, 163):
            raise ValueError("I3 package base metadata/features do not have the frozen shape")
        if not self.meta.index.equals(pd.RangeIndex(len(self.meta))):
            raise ValueError("I3 package metadata must use a RangeIndex")
        self.meta["fips_str"] = self.meta["fipsCode"].astype(str).str.zfill(5)
        self.train_positions = np.flatnonzero(self.meta["split"].to_numpy() == "train")
        self.test_positions = np.flatnonzero(self.meta["split"].to_numpy() == "test")
        self.n_train = len(self.train_positions)
        self.n_test = len(self.test_positions)
        if (self.n_train, self.n_test) != (34416, 9072):
            raise ValueError("I3 package train/test row counts differ from the frozen contract")
        self.meta_train = self.meta.iloc[self.train_positions].reset_index(drop=True)
        self.meta_test = self.meta.iloc[self.test_positions].reset_index(drop=True)
        self.X_train = self.features.iloc[self.train_positions].reset_index(drop=True)
        self.X_test = self.features.iloc[self.test_positions].reset_index(drop=True)
        self.row_fold_all = self.package.folds(self.seed)
        self.row_fold_train = self.row_fold_all[self.train_positions]
        self.all_fips = sorted(set(self.meta_train["fips_str"]) | set(self.meta_test["fips_str"]))
        if len(self.all_fips) != 302:
            raise ValueError("expected 302 graph counties")
        self.terrain = load_terrain_features()
        geo_dbf = ROOT / "data" / "geo" / "c_16ap26.dbf"
        geo_shp = geo_dbf.with_suffix(".shp")
        if not geo_dbf.exists() or not geo_shp.exists():
            raise FileNotFoundError("missing data/geo/c_16ap26.dbf or matching .shp")
        self.coords, self.edge_index, self.edge_attr = build_spatial_graph(
            self.all_fips, geo_dbf, geo_shp, k=GRAPH_K
        )
        if self.edge_index.shape != (2, 3028) or self.edge_attr.shape != (3028, 4):
            raise ValueError(f"unexpected frozen graph shape: {self.edge_index.shape}, {self.edge_attr.shape}")
        self.train_fips_to_fold = {}
        for fips, fold in zip(self.meta_train["fips_str"], self.row_fold_train):
            previous = self.train_fips_to_fold.setdefault(str(fips), int(fold))
            if previous != int(fold):
                raise ValueError("a training county spans multiple folds")
        self.graph_hash = hashlib.sha256(
            json.dumps(self.all_fips, separators=(",", ":")).encode()
            + np.asarray(self.edge_index, dtype=np.int64).tobytes()
            + np.asarray(self.edge_attr, dtype=np.float32).tobytes()
        ).hexdigest()
        self.package_hash = _sha256(self.package_dir / "PACKAGE_COMPLETE.json")
        self.cache = {}

    def _node_mask(self, scope):
        allowed = set(int(v) for v in scope)
        return np.asarray(
            [self.train_fips_to_fold.get(fips, -1) in allowed for fips in self.all_fips],
            dtype=bool,
        )

    def _build_inputs(self, scope, horizon):
        scope = tuple(sorted(int(v) for v in scope))
        h = HORIZON_HOURS[horizon]
        context = self.package.graph_context(self.seed, scope, h)
        base_all = context.base_graph_input.to_numpy(dtype=float)
        base_train = base_all[self.train_positions]
        base_test = base_all[self.test_positions]
        raw, base_grid, train_rows, test_rows, all_fips, names = make_county_time_view(
            self.X_train, self.X_test, self.meta_train, self.meta_test,
            base_train, base_test, horizon, self.coords, self.terrain,
            tuple(self.X_train.columns),
        )
        features, extra_names = add_neighbor_feature_aggregates(
            raw, names, self.edge_index, horizon
        )
        graph_names = tuple(
            ["base", *names, "latitude", "longitude", *list(self.terrain.columns), *extra_names]
        )
        if features.shape != (144, 302, 205) or not np.isfinite(features).all():
            raise ValueError(f"invalid 205-dimensional graph input: {features.shape}")
        if not np.allclose(features[:, :, 0], base_grid, rtol=0.0, atol=0.0):
            raise ValueError("graph base channel differs from I3 base_graph_input")
        residual_all, supervised_all = self.package.residual_supervision(self.seed, scope, h)
        residual_train = residual_all[self.train_positions]
        supervised_train = supervised_all[self.train_positions]
        residual_grid = np.zeros(base_grid.shape, dtype=float)
        supervised_grid = np.zeros(base_grid.shape, dtype=bool)
        for t in range(train_rows.shape[0]):
            for node in range(train_rows.shape[1]):
                row_id = int(train_rows[t, node])
                if row_id >= 0:
                    residual_grid[t, node] = residual_train[row_id]
                    supervised_grid[t, node] = bool(supervised_train[row_id])
        if not np.isfinite(residual_grid[supervised_grid]).all() or not supervised_grid.any():
            raise FloatingPointError("I3 residual supervision is invalid or empty")
        valid_time = np.arange(PRED_START, PRED_END) + h <= PRED_END - 1
        node_mask = self._node_mask(scope)
        scaled, mean, std = _strict_standardize(features, node_mask, valid_time)
        scaled = _apply_m2_input_policy(scaled)
        scale = _residual_scale(residual_grid, supervised_grid)
        return {
            "scope": scope,
            "horizon": horizon,
            "context": context,
            "features": features,
            "scaled_features": scaled,
            "residual_grid": residual_grid,
            "supervised_mask": supervised_grid,
            "base_grid": base_grid,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "all_fips": all_fips,
            "base_train": base_train,
            "base_test": base_test,
            "feature_mean": mean,
            "feature_std": std,
            "residual_scale": scale,
            "feature_names": graph_names,
        }

    def _fit_scope(self, scope, horizon):
        from gat_model import fit_gat, predict_gat

        key = (tuple(sorted(scope)), horizon)
        if key in self.cache:
            return self.cache[key]
        inputs = self._build_inputs(key[0], horizon)
        fit_times = np.arange(0, inputs["features"].shape[0], GAT_TIME_STRIDE, dtype=int)
        if fit_times[-1] != inputs["features"].shape[0] - 1:
            fit_times = np.unique(np.r_[fit_times, inputs["features"].shape[0] - 1])
        gat_seed = _scoped_seed(self.seed, key[0], horizon)
        model, _, fit_info = fit_gat(
            inputs["scaled_features"][fit_times],
            (inputs["residual_grid"] / inputs["residual_scale"])[fit_times],
            inputs["supervised_mask"][fit_times],
            self.edge_index,
            epochs=self.epochs,
            patience=self.patience,
            hidden=GAT_HIDDEN,
            heads=GAT_HEADS,
            dropout=GAT_DROPOUT,
            lr=GAT_LR,
            weight_decay=GAT_WEIGHT_DECAY,
            seed=gat_seed,
            device=self.device,
            edge_attr=self.edge_attr,
            loss_mode="mse",
            skip_enabled=True,
            output_limit=None,
        )
        correction = predict_gat(
            model, inputs["scaled_features"], self.edge_index, self.device, self.edge_attr
        ) * inputs["residual_scale"]
        if not np.isfinite(correction).all():
            raise FloatingPointError("GAT correction is non-finite")
        result = {"inputs": inputs, "model": model, "correction": correction, "fit_info": fit_info}
        self.cache[key] = result
        return result

    def _select_alpha(self, scope, horizon):
        scope = tuple(sorted(scope))
        inner_base = np.full(self.n_train, np.nan, dtype=float)
        inner_correction = np.full(self.n_train, np.nan, dtype=float)
        coverage = np.zeros(self.n_train, dtype=int)
        for valid_fold in scope:
            inner_scope = tuple(fold for fold in scope if fold != valid_fold)
            stack = self._fit_scope(inner_scope, horizon)
            inputs = stack["inputs"]
            base_rows, base_coverage = _grid_to_rows(
                inputs["base_grid"], inputs["train_rows"], self.n_train
            )
            correction_rows, correction_coverage = _grid_to_rows(
                stack["correction"], inputs["train_rows"], self.n_train
            )
            if not np.array_equal(base_coverage, correction_coverage):
                raise ValueError("inner base/correction row coverage differs")
            valid_rows = np.flatnonzero(self.row_fold_train == valid_fold)
            if np.any(coverage[valid_rows] != 0):
                raise ValueError("inner OOF row was generated more than once")
            inner_base[valid_rows] = base_rows[valid_rows]
            inner_correction[valid_rows] = correction_rows[valid_rows]
            coverage[valid_rows] = correction_coverage[valid_rows]
        expected = np.isin(self.row_fold_train, scope)
        for horizon_hours in [HORIZON_HOURS[horizon]]:
            expected &= self.meta_train["hour_idx"].to_numpy(dtype=int) + horizon_hours <= PRED_END - 1
        if not expected.any() or not np.all(coverage[expected] == 1):
            raise ValueError("inner OOF coverage is incomplete")
        rows = np.flatnonzero(expected)
        labels = self.package.labels_for_rows(self.seed, scope, HORIZON_HOURS[horizon], rows)
        if not np.isfinite(labels).all() or not np.isfinite(inner_base[rows]).all() or not np.isfinite(inner_correction[rows]).all():
            raise FloatingPointError("inner alpha inputs are non-finite")
        candidates = []
        for alpha in GAT_ALPHA_GRID:
            prediction = _post_process_osi(inner_base[rows] + float(alpha) * inner_correction[rows])
            error = labels - prediction
            sse = float(np.sum(error * error, dtype=np.float64))
            candidates.append({
                "alpha": float(alpha),
                "n": int(len(rows)),
                "rmse": float(np.sqrt(sse / len(rows))),
                "mae": float(np.mean(np.abs(error))),
                "sse": sse,
            })
        selected = min(candidates, key=lambda item: (item["rmse"], item["alpha"]))
        return {
            "scope": list(scope),
            "horizon": horizon,
            "selected_alpha": float(selected["alpha"]),
            "candidates": candidates,
            "inner_n": int(len(rows)),
        }

    def run(self, submission_template: Path, run_id: str):
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        final_scope = tuple(range(N_FOLDS))
        alpha_rows = {}
        final_predictions = {}
        baseline_scores = {}
        checkpoint_records = []
        for horizon in HORIZONS:
            print(f"[FINAL] horizon={horizon} selecting alpha with I3 scope-aware inner OOF", flush=True)
            alpha = self._select_alpha(final_scope, horizon)
            alpha_rows[horizon] = alpha
            print(f"[FINAL] horizon={horizon} selected_alpha={alpha['selected_alpha']}", flush=True)
            stack = self._fit_scope(final_scope, horizon)
            inputs = stack["inputs"]
            corr_test, corr_coverage = _grid_to_rows(
                stack["correction"], inputs["test_rows"], self.n_test
            )
            base_test, base_coverage = _grid_to_rows(
                inputs["base_grid"], inputs["test_rows"], self.n_test
            )
            if not np.all(corr_coverage == 1) or not np.all(base_coverage == 1):
                raise ValueError("final graph does not cover every test county/hour exactly once")
            valid_test = self.meta_test["hour_idx"].to_numpy(dtype=int) + HORIZON_HOURS[horizon] <= PRED_END - 1
            predictions = np.full(self.n_test, np.nan, dtype=float)
            predictions[valid_test] = _post_process_osi(
                base_test[valid_test] + alpha["selected_alpha"] * corr_test[valid_test]
            )
            final_predictions[horizon] = predictions
            context_all = self.package.graph_context(self.seed, final_scope, HORIZON_HOURS[horizon])
            valid_train = context_all.supervised.to_numpy(dtype=bool)
            train_rows = np.flatnonzero(valid_train & (self.meta["split"].to_numpy() == "train"))
            labels = self.package.labels_for_rows(
                self.seed, final_scope, HORIZON_HOURS[horizon], train_rows
            )
            base = context_all.base_prediction.to_numpy(dtype=float)[train_rows]
            err = labels - base
            baseline_scores[horizon] = {
                "n": int(len(labels)),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "mae": float(np.mean(np.abs(err))),
                "frozen_package_score_file": str(self.package_dir / "provenance" / "frozen_I3_scores.csv"),
            }
            payload = {
                "protocol": "i3_base_m2_robust_input_final_v1",
                "variant": VARIANT,
                "seed": self.seed,
                "scope": list(final_scope),
                "horizon": horizon,
                "package_complete_sha256": self.package_hash,
                "graph_hash": self.graph_hash,
                "architecture": {
                    "in_dim": 205,
                    "hidden": GAT_HIDDEN,
                    "heads": GAT_HEADS,
                    "dropout": GAT_DROPOUT,
                    "edge_dim": 4,
                    "skip_enabled": True,
                    "output_limit_normalized": None,
                },
                "feature_names": inputs["feature_names"],
                "feature_mean": inputs["feature_mean"],
                "feature_std": inputs["feature_std"],
                "residual_scale": inputs["residual_scale"],
                "input_policy": INPUT_POLICY,
                "training_config": {
                    "epochs": self.epochs,
                    "patience": self.patience,
                    "time_stride": GAT_TIME_STRIDE,
                    "loss_mode": "mse",
                    "lr": GAT_LR,
                    "weight_decay": GAT_WEIGHT_DECAY,
                },
                "fit_info": stack["fit_info"],
                "state_dict": stack["model"].state_dict(),
                "edge_index": self.edge_index,
                "edge_attr": self.edge_attr,
            }
            checkpoint = run_dir / "checkpoints" / f"gat_{horizon.replace('osi_target_', '')}_S01234.pt"
            _save_checkpoint(checkpoint, payload)
            checkpoint_records.append({
                "horizon": horizon,
                "path": str(checkpoint.relative_to(run_dir)),
                "sha256": _sha256(checkpoint),
                "selected_alpha": alpha["selected_alpha"],
            })
        template = pd.read_csv(submission_template, dtype={"fipsCode": str})

        required = ["fipsCode", "timestamp_et", *HORIZONS]
        allowed_templates = [
            required,
            ["fipsCode", "countyName", "stateAbbr", "timestamp_et", *HORIZONS],
        ]

        if list(template.columns) not in allowed_templates:
            raise ValueError("submission template columns differ from the official template")
        
        template_keys = pd.MultiIndex.from_arrays([
            template["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5),
            pd.to_datetime(template["timestamp_et"]),
        ])
        source_keys = pd.MultiIndex.from_frame(self.meta_test[["fips_str", "timestamp_et"]])
        order = source_keys.get_indexer(template_keys)
        if (order < 0).any() or template_keys.has_duplicates or source_keys.has_duplicates:
            raise ValueError("submission template keys do not exactly match I3 test metadata")
        for horizon in HORIZONS:
            template[horizon] = final_predictions[horizon][order]
        submission_path = run_dir / "submission_i3_m2_gat.csv"
        template.to_csv(submission_path, index=False)
        _save_json(run_dir / "alpha_selection.json", alpha_rows)
        _save_json(run_dir / "baseline_scores.json", baseline_scores)
        _save_json(run_dir / "run_manifest.json", {
            "protocol": "i3_base_m2_robust_input_final_v1",
            "variant": VARIANT,
            "seed": self.seed,
            "device": self.device,
            "epochs": self.epochs,
            "patience": self.patience,
            "package_dir": str(self.package_dir),
            "package_complete_sha256": self.package_hash,
            "graph_hash": self.graph_hash,
            "feature_shape": [144, 302, 205],
            "submission": str(submission_path),
            "checkpoints": checkpoint_records,
            "prediction_finite_counts": {
                h: int(np.isfinite(final_predictions[h]).sum()) for h in HORIZONS
            },
        })
        expected = {"osi_target_t01h": 9009, "osi_target_t06h": 8694, "osi_target_t24h": 7560, "osi_target_t48h": 6048}
        for horizon, count in expected.items():
            actual = int(np.isfinite(template[horizon]).sum())
            if actual != count:
                raise ValueError(f"{horizon} finite count {actual} != official {count}")
            values = template[horizon].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not ((values[finite] >= 0.0) & (values[finite] <= 0.65)).all():
                raise ValueError(f"{horizon} contains a value outside [0, 0.65]")
        print(f"FINAL_READY={run_dir}", flush=True)
        return run_dir


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--submission-template", type=Path, default=ROOT / "data" / "sample_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "runs")
    parser.add_argument("--run-id", default="i3_m2_robust_input_final_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    parser.add_argument("--patience", type=int, default=GAT_PATIENCE)
    parser.add_argument("--preflight", action="store_true", help="load and validate I3/geo/graph inputs without training")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = _resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    runner = FinalI3M2Runner(
        args.package_dir, args.seed, device, args.epochs, args.patience, args.output_dir
    )
    if args.preflight:
        print(f"PREFLIGHT_OK device={device} package={runner.package_dir} graph={runner.edge_index.shape}", flush=True)
        return 0
    import torch

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    runner.run(args.submission_template, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
