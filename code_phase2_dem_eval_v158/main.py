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

from base_model import (
    load_cv_folds,
    load_component_v21_base,
    train_base_models,
    train_component_v158_base,
)
from config import (DATA_DIR, GAT_ALPHA_GRID, GAT_DROPOUT, GAT_EPOCHS,
                    GAT_HEADS, GAT_HIDDEN, GAT_LR, GAT_PATIENCE,
                    GAT_WEIGHT_DECAY, CV_VERSION, HORIZON_HOURS, HORIZONS, MODEL_DIR,
                    OUTPUT_DIR, PRED_END, PRED_START, SEED, SUBMISSION_FILE,
                    GEO_DBF, GAT_LOSS, OFFICIAL_METRIC, OFFICIAL_SCOREABLE_ROWS)
from data import (add_neighbor_feature_aggregates, load_cached_data,
                  load_terrain_features, make_county_time_view,
                  validate_feature_columns)
from gat_model import fit_gat, predict_gat
from metrics import metrics, post_process_osi
from spatial import build_spatial_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=GAT_EPOCHS)
    p.add_argument("--patience", type=int, default=GAT_PATIENCE)
    p.add_argument("--folds", type=int, default=5,
                   help="Must be 5: v1.5.8-compatible outer county folds")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--time-stride", type=int, default=1,
                   help="Train GAT on every Nth forecast time; inference remains hourly")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="Torch device; auto selects CUDA when available")
    p.add_argument(
        "--base-mode", default="direct",
        choices=("direct", "component_v158", "component_v21"),
        help=("direct=C0 v1.5.8-compatible direct OSI; "
              "component_v158=C1 v1.5.8-compatible component route; "
              "component_v21=legacy alias to strict component_v158 retraining"),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--gat-loss", default=GAT_LOSS,
                   choices=("mse", "weighted_mse", "huber"),
                   help="GAT residual loss; mse matches official pooled RMSE")
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
        pred = post_process_osi(bv + alpha * cv)
        m = metrics(yv, pred)
        rows.append({"alpha": float(alpha), **m,
                     "official_score_rmse": m["rmse"]})
    # The competition ranks by RMSE per horizon. MAE is reported for
    # diagnostics, but must not influence alpha selection.
    best = min(rows, key=lambda r: (r["official_score_rmse"], r["alpha"]))
    return float(best["alpha"]), rows


def nested_inner_folds(folds, outer_fold, outer_train_indices):
    """Use the v1.5.8 fixed county folds as inner folds inside one outer fold."""
    outer_train_indices = np.asarray(outer_train_indices, dtype=int)
    for inner_fold, (_, inner_valid_full) in enumerate(folds):
        if inner_fold == outer_fold:
            continue
        inner_valid = np.intersect1d(outer_train_indices, inner_valid_full)
        inner_train = np.setdiff1d(outer_train_indices, inner_valid, assume_unique=False)
        if len(inner_train) == 0 or len(inner_valid) == 0:
            raise ValueError(
                f"Nested CV produced an empty split: outer={outer_fold}, inner={inner_fold}"
            )
        yield inner_fold, inner_train, inner_valid


def aggregate_nested_alphas(alphas):
    """Choose a deterministic alpha for the final all-county fit.

    This aggregation uses only alpha values selected inside outer-training
    folds. It never reads outer-fold labels or the pooled outer OOF score.
    Ties are resolved toward the smaller correction for conservative blending.
    """
    if not alphas:
        return 0.0
    counts = {float(alpha): alphas.count(alpha) for alpha in set(alphas)}
    best_count = max(counts.values())
    return float(min(alpha for alpha, count in counts.items() if count == best_count))


def grid_to_rows(grid, train_rows, n_rows):
    """Map a [time, county] grid back to the original training row order."""
    values = np.full(n_rows, np.nan, dtype=float)
    for t in range(train_rows.shape[0]):
        for node in range(train_rows.shape[1]):
            row_idx = train_rows[t, node]
            if row_idx >= 0:
                values[row_idx] = grid[t, node]
    return values


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
    print("=== Phase 2: v1.5.6 LightGBM + DEM spatial GAT residual ===")
    X_train, X_test, meta_train, meta_test, y = load_cached_data()
    terrain = load_terrain_features()
    folds = load_cv_folds(meta_train)
    if args.folds != 5:
        raise ValueError(
            "Strict nested alpha selection requires all 5 v1.5.8-compatible county folds"
        )
    train_fips = sorted(meta_train["fips_str"].unique())
    test_fips = sorted(meta_test["fips_str"].unique())
    all_fips = sorted(set(train_fips + test_fips))
    shp_path = GEO_DBF.with_suffix(".shp")
    coords, edge_index, edge_attr = build_spatial_graph(all_fips, GEO_DBF, shp_path, k=args.k)
    train_node_mask = np.asarray([f in set(train_fips) for f in all_fips], dtype=bool)
    print(f"counties={len(all_fips)} train={train_node_mask.sum()} test={(~train_node_mask).sum()} edges={edge_index.shape[1]}")

    if args.base_mode == "direct":
        base = train_base_models(X_train, y, meta_train, X_test, folds)
        effective_base_mode = "direct_v1.5.8_C0"
    elif args.base_mode == "component_v158":
        base = train_component_v158_base(X_train, y, meta_train, X_test, folds)
        effective_base_mode = "component_v1.5.8_C1"
    else:
        base = load_component_v21_base(
            y, meta_train, meta_test, folds, X=X_train, X_test=X_test
        )
        effective_base_mode = "component_v1.5.8_C1_legacy_alias"
    summary_rows = []
    alpha_rows = []
    submission_predictions = {}
    for horizon in HORIZONS:
        print(f"\n[GAT residual] {horizon}")
        result = base[horizon]
        y_arr = y[horizon].to_numpy(dtype=float)
        # View using OOF LightGBM predictions for every train county.
        features, _, train_rows, test_rows, _, phase1_feature_names = make_county_time_view(
            X_train, X_test, meta_train, meta_test, result["oof"], result["test"],
            horizon, coords, terrain)
        features, spatial_feature_names = add_neighbor_feature_aggregates(
            features, phase1_feature_names, edge_index, horizon
        )
        validate_feature_columns(
            ["base_prediction", *phase1_feature_names, "latitude", "longitude",
             *list(terrain.columns), *spatial_feature_names],
            "GAT tensor features",
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
        alpha_by_row = np.full(len(X_train), np.nan, dtype=float)
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
            fold_features, _, _, _, _, fold_phase1_feature_names = make_county_time_view(
                X_train, X_test, meta_train, meta_test, fold_train_base, fold_base,
                horizon, coords, terrain)
            fold_features, _ = add_neighbor_feature_aggregates(
                fold_features, fold_phase1_feature_names, edge_index, horizon
            )
            fold_base_grid = fold_features[:, :, 0].copy()
            fold_base_rows = grid_to_rows(fold_base_grid, train_rows, len(X_train))
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
                edge_attr=edge_attr, loss_mode=args.gat_loss,
            )
            corr_grid = predict_gat(gat_model, train_scaled, edge_index, device=device,
                                    edge_attr=edge_attr) * fold_scale

            # Strict nested alpha selection. The outer validation counties
            # below are never passed to alpha_select. Inner validation is the
            # other four fixed v1.5.8 county folds inside this outer train set.
            inner_oof_corr = np.full(len(X_train), np.nan, dtype=float)
            inner_fold_ids = []
            for inner_fold, inner_train_full, inner_valid_full in nested_inner_folds(
                    folds, fold_id, folds[fold_id][0]):
                inner_fold_ids.append(inner_fold)
                inner_train_fips = set(meta_train.iloc[inner_train_full]["fips_str"])
                inner_train_nodes = np.asarray(
                    [f in inner_train_fips for f in all_fips], dtype=bool
                )
                inner_mask = np.broadcast_to(
                    inner_train_nodes[None, :], (n_time, n_nodes)
                ).copy()
                inner_mask &= valid_time[:, None]
                inner_mask &= np.isfinite(y_grid)
                inner_scale = residual_scale(fold_residual_grid, inner_mask)
                inner_scaled, _, _ = standardize_features(fold_features, inner_train_nodes)
                inner_model, _, inner_info = fit_gat(
                    inner_scaled[fit_times],
                    (fold_residual_grid / inner_scale)[fit_times],
                    inner_mask[fit_times], edge_index,
                    epochs=args.epochs, patience=args.patience,
                    hidden=GAT_HIDDEN, heads=GAT_HEADS, dropout=GAT_DROPOUT,
                    lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY,
                    seed=args.seed + 1000 + fold_id * 10 + inner_fold,
                    device=device, edge_attr=edge_attr, loss_mode=args.gat_loss,
                )
                inner_corr_grid = predict_gat(
                    inner_model, inner_scaled, edge_index, device=device,
                    edge_attr=edge_attr,
                ) * inner_scale
                inner_corr_rows = grid_to_rows(
                    inner_corr_grid, train_rows, len(X_train)
                )
                inner_oof_corr[inner_valid_full] = inner_corr_rows[inner_valid_full]
                alpha_rows.append({
                    "horizon": horizon,
                    "scope": "nested_inner_fit",
                    "outer_fold": fold_id,
                    "inner_fold": inner_fold,
                    "alpha": np.nan,
                    "inner_train_counties": int(len(inner_train_fips)),
                    "inner_valid_counties": int(
                        meta_train.iloc[inner_valid_full]["fips_str"].nunique()
                    ),
                    "epochs": inner_info["epochs"],
                })

            nested_valid = np.isfinite(y_arr) & np.isfinite(inner_oof_corr)
            alpha, alpha_table = alpha_select(
                y_arr, fold_base_rows, inner_oof_corr, nested_valid
            )
            alpha_rows.extend([
                {**row, "horizon": horizon, "scope": "nested_inner_selection",
                 "outer_fold": fold_id, "inner_fold": "pooled"}
                for row in alpha_table
            ])
            outer_corr_rows = grid_to_rows(corr_grid, train_rows, len(X_train))
            fold_oof_corr[val_full] = outer_corr_rows[val_full]
            val_mask_rows = np.zeros(len(X_train), dtype=bool)
            val_mask_rows[val_full] = True
            val_mask_rows &= np.isfinite(y_arr) & np.isfinite(fold_oof_corr)
            alpha_by_row[val_mask_rows] = alpha
            selected_alphas.append(alpha)
            outer_base_metrics = metrics(y_arr[val_mask_rows], fold_base_rows[val_mask_rows])
            outer_pred = post_process_osi(
                fold_base_rows[val_mask_rows] + alpha * fold_oof_corr[val_mask_rows]
            )
            outer_gat_metrics = metrics(y_arr[val_mask_rows], outer_pred)
            fold_records.append({
                "fold": fold_id, "alpha": alpha, "epochs": fit_info["epochs"],
                "base_rmse": outer_base_metrics["rmse"],
                "gat_rmse": outer_gat_metrics["rmse"],
                "gat_mae": outer_gat_metrics["mae"],
            })
            alpha_rows.append({
                "horizon": horizon, "scope": "outer_evaluation", "outer_fold": fold_id,
                "inner_fold": "not_used_for_selection", "alpha": alpha,
                "n": outer_gat_metrics["n"], "rmse": outer_gat_metrics["rmse"],
                "mae": outer_gat_metrics["mae"],
            })
            print(f"  fold={fold_id} alpha={alpha:.2f} val_base_rmse={fold_records[-1]['base_rmse']:.6f} "
                  f"gat_rmse={fold_records[-1]['gat_rmse']:.6f} epochs={fit_info['epochs']} "
                  f"inner_folds={inner_fold_ids}")

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
            edge_attr=edge_attr, loss_mode=args.gat_loss,
        )
        final_corr = predict_gat(final_model, final_scaled, edge_index, device=device,
                                 edge_attr=edge_attr) * final_scale
        # The final all-county GAT uses an alpha aggregated from nested
        # selections only. Outer OOF labels are never used to choose it.
        alpha = aggregate_nested_alphas(selected_alphas)
        alpha_rows.append({
            "horizon": horizon, "scope": "final_alpha", "outer_fold": "all",
            "inner_fold": "nested_mode", "alpha": alpha,
            "selected_outer_alphas": json.dumps(selected_alphas),
        })
        test_pred = result["test"].copy()
        for row_idx, row in meta_test.iterrows():
            t = int(row["hour_idx"]) - PRED_START
            node = all_fips.index(row["fips_str"])
            if 0 <= t < n_time:
                test_pred[row_idx] = post_process_osi(test_pred[row_idx] + alpha * final_corr[t, node])
        submission_predictions[horizon] = test_pred
        base_summary = result["summary"]
        # GAT CV OOF correction is assembled from fold-held-out predictions.
        valid_oof = np.isfinite(y_arr) & np.isfinite(fold_oof_corr) & np.isfinite(alpha_by_row)
        gat_oof = post_process_osi(
            result["oof"][valid_oof] + alpha_by_row[valid_oof] * fold_oof_corr[valid_oof]
        )
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
    audit_rows = []
    for horizon in HORIZONS:
        values = pd.to_numeric(submission[horizon], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        audit_rows.append({
            "horizon": horizon,
            "submission_rows": int(len(values)),
            "non_nan_rows": int(finite.sum()),
            "official_scoreable_rows": int(OFFICIAL_SCOREABLE_ROWS[horizon]),
            "row_count_match": bool(finite.sum() == OFFICIAL_SCOREABLE_ROWS[horizon]),
            "prediction_min": float(np.nanmin(values)) if finite.any() else np.nan,
            "prediction_max": float(np.nanmax(values)) if finite.any() else np.nan,
            "prediction_nan": int((~finite).sum()),
        })
    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "submission_audit.csv", index=False)
    metadata = {"seed": args.seed, "epochs": args.epochs, "patience": args.patience,
                "device": device, "time_stride": args.time_stride,
                "graph_k": args.k, "edge_count": int(edge_index.shape[1]),
                "cv_version": CV_VERSION,
                "cv_protocol": "v1.5.8-compatible balanced_v1 fixed county-level 5-fold; seed=42",
                "alpha_selection": "nested inner county CV only; outer validation labels used once for final scoring",
                "alpha_outer_oof_selection": False,
                "submission": str(out_file), "feature_cache": "v1.5.6",
                "terrain_file": str(DATA_DIR / "geo" / "county_terrain.csv"),
                "terrain_features": ["elevation_mean_m", "elevation_std_m",
                                     "elevation_min_m", "elevation_max_m",
                                     "slope_mean_deg", "slope_std_deg",
                                     "terrain_ruggedness"],
                "base_mode": effective_base_mode,
                "base_protocol": "v1.5.8; nested county cross-fit for outer GAT evaluation",
                "base_final_source": result.get("final_source", "retrained v1.5.8-compatible booster"),
                "base_nested_cross_fit": True,
                "gat_feature_count": int(features.shape[-1]),
                "residual_standardized_per_fold": True,
                "residual_loss": args.gat_loss,
                "official_metric": OFFICIAL_METRIC,
                "official_ranking": "average rank across t+1h, t+6h, t+24h, t+48h",
                "official_tie_break": "t+1h RMSE",
                "official_scoreable_rows": OFFICIAL_SCOREABLE_ROWS,
                "causal_rule": "outage inputs only from hour_idx<72; weather full horizon"}
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved submission: {out_file}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
