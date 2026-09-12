"""Phase 2 runner: LightGBM base + spatial GAT residual correction.

Example (from the project root):
    D:\\app\\anacnda1\\envs\\myenv\\python.exe code_phase2/main.py

The script writes a competition-format submission and CV diagnostics under
code_phase2/outputs. It never reads outage variables after the observed
window; all modelling inputs come from the clean Phase-1 feature caches.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from base_model import load_component_v21_base, load_cv_folds, train_base_models
from config import (DATA_DIR, GAT_ALPHA_GRID, GAT_DROPOUT, GAT_EPOCHS,
                    GAT_HEADS, GAT_HIDDEN, GAT_LR, GAT_PATIENCE,
                    GAT_WEIGHT_DECAY, HORIZON_HOURS, HORIZONS, MODEL_DIR,
                    OUTPUT_DIR, PRED_END, PRED_START, SEED, SUBMISSION_FILE,
                    GEO_DBF)
from data import (add_neighbor_feature_aggregates, load_cached_data,
                  load_terrain_features, make_county_time_view, make_state_map)
from gat_model import fit_gat, predict_gat
from metrics import clip_osi, joint_score, metrics
from spatial import build_spatial_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    p.add_argument("--patience", type=int, default=GAT_PATIENCE)
    p.add_argument("--folds", type=int, default=5, help="GAT evaluation folds; default 5")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--time-stride", type=int, default=1,
                   help="Train GAT on every Nth forecast time; inference remains hourly")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="Torch device; auto selects CUDA when available")
    p.add_argument("--base-mode", default="direct", choices=("direct", "component_v21"),
                   help="LightGBM base: self-contained direct OSI or frozen v2.1 P/N/D/R")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def standardize_features(features, train_node_mask):
    """Scale on training county-time cells only, preserving all graph nodes."""
    flat = features.reshape(-1, features.shape[-1])
    mask = np.broadcast_to(train_node_mask[None, :], features.shape[:2]).reshape(-1)
    train_values = flat[mask]
    mean = np.nanmean(train_values, axis=0)
    std = np.nanstd(train_values, axis=0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    out = (np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return out.astype(np.float32), mean, std


def alpha_select(y, base, correction, val_mask):
    yv = y[val_mask]
    bv = base[val_mask]
    cv = correction[val_mask]
    rows = []
    for alpha in GAT_ALPHA_GRID:
        pred = clip_osi(bv + alpha * cv)
        m = metrics(yv, pred)
        rows.append({"alpha": float(alpha), **m, "joint": joint_score(yv, pred)})
    best = min(rows, key=lambda r: r["joint"])
    return float(best["alpha"]), rows


def residual_scale(residual, mask):
    """Put small OSI residuals on a numerically useful training scale."""
    values = np.asarray(residual)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.01
    return float(max(np.std(values), 0.003))


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # Use more host threads when CUDA is unavailable; this does not change
    # training scale and keeps the full graph/time-snapshot workload intact.
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")
    else:
        device = args.device
    print(f"torch_device={device}; time_stride={args.time_stride}; gat_epochs={args.epochs}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 2: LightGBM + spatial GAT residual ===")
    X_train, X_test, meta_train, meta_test, y = load_cached_data()
    terrain = load_terrain_features()
    folds = load_cv_folds(meta_train)
    if args.folds != 5:
        folds = folds[:args.folds]
    train_fips = sorted(meta_train["fips_str"].unique())
    test_fips = sorted(meta_test["fips_str"].unique())
    all_fips = sorted(set(train_fips + test_fips))
    shp_path = GEO_DBF.with_suffix(".shp")
    coords, edge_index, edge_attr = build_spatial_graph(all_fips, GEO_DBF, shp_path, k=args.k)
    state_map = make_state_map(meta_train, meta_test)
    train_node_mask = np.asarray([f in set(train_fips) for f in all_fips], dtype=bool)
    print(f"counties={len(all_fips)} train={train_node_mask.sum()} test={(~train_node_mask).sum()} edges={edge_index.shape[1]}")

    if args.base_mode == "component_v21":
        base = load_component_v21_base(y, meta_train, meta_test, folds)
    else:
        base = train_base_models(X_train, y, meta_train, X_test, folds)
    summary_rows = []
    alpha_rows = []
    submission_predictions = {}
    for horizon in HORIZONS:
        print(f"\n[GAT residual] {horizon}")
        result = base[horizon]
        y_arr = y[horizon].to_numpy(dtype=float)
        # View using OOF LightGBM predictions for every train county.
        features, _, train_rows, test_rows, _, _ = make_county_time_view(
            X_train, X_test, meta_train, meta_test, result["oof"], result["test"],
            horizon, coords, state_map, terrain)
        features, spatial_feature_names = add_neighbor_feature_aggregates(
            features, _, edge_index, horizon
        )
        # Align the OOF target and base prediction to the same time/county grid.
        n_time, n_nodes = features.shape[:2]
        y_grid = np.full((n_time, n_nodes), np.nan, dtype=np.float32)
        base_grid = features[:, :, 0].copy()
        for t in range(n_time):
            for node in range(n_nodes):
                idx = train_rows[t, node]
                if idx >= 0:
                    y_grid[t, node] = y_arr[idx]
        residual_grid = np.nan_to_num(y_grid - base_grid, nan=0.0)
        valid_time = np.arange(PRED_START, PRED_END) + HORIZON_HOURS[horizon] <= PRED_END - 1
        # For final prediction, all train county residuals are supervised. A
        # fold's validation counties are held out from the GAT loss.
        fold_oof_corr = np.full(len(X_train), np.nan, dtype=float)
        fold_records = []
        selected_alphas = []
        for fold_id, (_, val_full) in enumerate(folds):
            val_fips = set(meta_train.iloc[val_full]["fips_str"])
            gat_train_nodes = train_node_mask.copy()
            for node, fips in enumerate(all_fips):
                if fips in val_fips:
                    gat_train_nodes[node] = False
            mask = np.broadcast_to(gat_train_nodes[None, :], (n_time, n_nodes)).copy()
            mask &= valid_time[:, None]
            # Only finite labels may supervise GAT; the last h rows are NaN.
            mask &= np.isfinite(y_grid)
            # Fold-specific train/test base predictions prevent validation
            # leakage in both the residual targets and GAT input features.
            fold_train_base = result["train_by_fold"][fold_id]
            fold_base = result["test_by_fold"][fold_id]
            fold_features, _, _, _, _, _ = make_county_time_view(
                X_train, X_test, meta_train, meta_test, fold_train_base, fold_base,
                horizon, coords, state_map, terrain)
            fold_features, _ = add_neighbor_feature_aggregates(
                fold_features, _, edge_index, horizon
            )
            fold_base_grid = fold_features[:, :, 0].copy()
            fold_residual_grid = np.nan_to_num(y_grid - fold_base_grid, nan=0.0)
            fold_scale = residual_scale(fold_residual_grid, mask)
            train_scaled, _, _ = standardize_features(fold_features, gat_train_nodes)
            fit_times = np.arange(0, n_time, max(1, args.time_stride), dtype=int)
            fit_times = np.unique(np.r_[fit_times, n_time - 1])
            gat_model, _, fit_info = fit_gat(
                train_scaled[fit_times], (fold_residual_grid / fold_scale)[fit_times],
                mask[fit_times], edge_index,
                epochs=args.epochs, patience=args.patience,
                hidden=GAT_HIDDEN, heads=GAT_HEADS, dropout=GAT_DROPOUT,
                lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY, seed=args.seed + fold_id,
                device=device,
                edge_attr=edge_attr,
            )
            corr_grid = predict_gat(gat_model, train_scaled, edge_index, device=device,
                                    edge_attr=edge_attr) * fold_scale
            val_mask_rows = np.zeros(len(X_train), dtype=bool)
            for t in range(n_time):
                for node in range(n_nodes):
                    idx = train_rows[t, node]
                    if idx >= 0 and meta_train.iloc[idx]["fips_str"] in val_fips:
                        val_mask_rows[idx] = True
                        fold_oof_corr[idx] = corr_grid[t, node]
            val_mask_rows &= np.isfinite(y_arr)
            alpha, alpha_table = alpha_select(y_arr, result["oof"], fold_oof_corr, val_mask_rows)
            selected_alphas.append(alpha)
            chosen = next(r for r in alpha_table if r["alpha"] == alpha)
            fold_records.append({"fold": fold_id, "alpha": alpha, "epochs": fit_info["epochs"],
                                 "base_rmse": metrics(y_arr[val_mask_rows], result["oof"][val_mask_rows])["rmse"],
                                 "gat_rmse": chosen["rmse"], "gat_mae": chosen["mae"]})
            alpha_rows.extend([{**r, "horizon": horizon, "fold": fold_id} for r in alpha_table])
            print(f"  fold={fold_id} alpha={alpha:.2f} val_base_rmse={fold_records[-1]['base_rmse']:.6f} "
                  f"gat_rmse={chosen['rmse']:.6f} epochs={fit_info['epochs']}")

        # A single final GAT sees all training counties' OOF residuals. Its
        # inference graph includes test counties but no test labels.
        final_mask = np.broadcast_to(train_node_mask[None, :], (n_time, n_nodes)).copy()
        final_mask &= valid_time[:, None]
        final_mask &= np.isfinite(y_grid)
        final_scaled, mean_x, std_x = standardize_features(features, train_node_mask)
        final_scale = residual_scale(residual_grid, final_mask)
        fit_times = np.arange(0, n_time, max(1, args.time_stride), dtype=int)
        fit_times = np.unique(np.r_[fit_times, n_time - 1])
        final_model, _, final_info = fit_gat(
            final_scaled[fit_times], (residual_grid / final_scale)[fit_times],
            final_mask[fit_times], edge_index,
            epochs=args.epochs, patience=args.patience,
            hidden=GAT_HIDDEN, heads=GAT_HEADS, dropout=GAT_DROPOUT,
            lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY, seed=args.seed + 100,
            device=device,
            edge_attr=edge_attr,
        )
        final_corr = predict_gat(final_model, final_scaled, edge_index, device=device,
                                 edge_attr=edge_attr) * final_scale
        # Select one global blend weight from all held-out county rows. This
        # is more stable than taking a median of fold-wise choices.
        global_valid = np.isfinite(y_arr) & np.isfinite(fold_oof_corr)
        if global_valid.any():
            alpha, global_alpha_table = alpha_select(
                y_arr, result["oof"], fold_oof_corr, global_valid
            )
            alpha_rows.extend([{**r, "horizon": horizon, "fold": "global"}
                               for r in global_alpha_table])
        else:
            alpha = 0.0
        test_pred = result["test"].copy()
        for row_idx, row in meta_test.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            node = all_fips.index(row["fips_str"])
            if 0 <= t < n_time:
                test_pred[row_idx] = clip_osi(test_pred[row_idx] + alpha * final_corr[t, node])
        submission_predictions[horizon] = test_pred
        base_summary = result["summary"]
        # GAT CV OOF correction is assembled from fold-held-out predictions.
        valid_oof = np.isfinite(y_arr) & np.isfinite(fold_oof_corr)
        gat_oof = clip_osi(result["oof"][valid_oof] + alpha * fold_oof_corr[valid_oof])
        gat_summary = metrics(y_arr[valid_oof], gat_oof)
        summary_rows.extend([
            {"horizon": horizon, "model": "lightgbm_base", **base_summary},
            {"horizon": horizon, "model": "gat_residual", **gat_summary},
        ])
        print(f"  final alpha={alpha:.2f}; base OOF RMSE={base_summary['rmse']:.6f}, "
              f"GAT OOF RMSE={gat_summary['rmse']:.6f}, base MAE={base_summary['mae']:.6f}, "
              f"GAT MAE={gat_summary['mae']:.6f}")
        torch.save({"model": final_model.state_dict(), "feature_mean": mean_x,
                    "feature_std": std_x, "edge_index": edge_index, "alpha": alpha,
                    "residual_scale": final_scale, "edge_attr": edge_attr,
                    "fit_info": final_info}, MODEL_DIR / f"gat_residual_{horizon}.pt")

    # Fill the official template by its stable identifier key and preserve NaN
    # for horizons whose target time is outside March 11-19.
    submission = pd.read_csv(SUBMISSION_FILE)
    key_to_idx = {(str(r.fipsCode), r.timestamp_et): i for i, r in submission.iterrows()}
    for horizon in HORIZONS:
        for i, row in meta_test.iterrows():
            key = (str(row.fipsCode), row.timestamp_et)
            if int(row.hour_idx) + HORIZON_HOURS[horizon] > PRED_END - 1:
                submission.loc[key_to_idx[key], horizon] = np.nan
            else:
                submission.loc[key_to_idx[key], horizon] = submission_predictions[horizon][i]
    out_file = OUTPUT_DIR / "submission_phase2_dem_gat.csv"
    submission.to_csv(out_file, index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "cv_summary.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUTPUT_DIR / "alpha_selection.csv", index=False)
    metadata = {"seed": args.seed, "epochs": args.epochs, "patience": args.patience,
                "device": device, "time_stride": args.time_stride,
                "graph_k": args.k, "edge_count": int(edge_index.shape[1]),
                "submission": str(out_file), "feature_cache": "v1.5.7",
                "terrain_file": str(DATA_DIR / "geo" / "county_terrain.csv"),
                "terrain_features": ["elevation_mean_m", "elevation_std_m",
                                     "elevation_min_m", "elevation_max_m",
                                     "slope_mean_deg", "slope_std_deg",
                                     "terrain_ruggedness"],
                "base_mode": args.base_mode,
                "gat_feature_count": int(features.shape[-1]),
                "residual_standardized_per_fold": True,
                "residual_loss": "weighted_mse",
                "causal_rule": "outage inputs only from hour_idx<72; weather full horizon"}
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved submission: {out_file}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
