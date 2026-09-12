"""LightGBM phase-1-compatible base model and county-grouped OOF predictions."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (CV_FILE, DATA_DIR, HORIZONS, LGBM_EARLY_STOPPING,
                    LGBM_PARAMS, LGBM_ROUNDS, MODEL_DIR, OSI_MAX, PROJECT_ROOT)
from metrics import clip_osi, metrics


def load_cv_folds(meta, n_folds=5):
    assignment = pd.read_csv(CV_FILE, dtype={"fipsCode": str})
    assignment["fipsCode"] = assignment["fipsCode"].astype(str).str.zfill(5)
    fips = meta["fipsCode"].astype(str).str.zfill(5)
    fold_map = assignment.set_index("fipsCode")["fold"]
    row_fold = fips.map(fold_map)
    if row_fold.isna().any():
        raise ValueError("CV assignment does not cover every training county")
    row_fold = row_fold.to_numpy(dtype=int)
    return [(np.flatnonzero(row_fold != i), np.flatnonzero(row_fold == i)) for i in range(n_folds)]


def train_base_models(X, y, meta, X_test, folds):
    """Train four direct OSI models and return OOF/test predictions.

    test_by_fold is retained so GAT fold evaluation never gets a base
    prediction made by a LightGBM model trained on the validation counties.
    """
    results = {}
    for horizon in HORIZONS:
        print(f"[LightGBM] {horizon}")
        target = y[horizon].to_numpy(dtype=float)
        valid = np.isfinite(target)
        valid_indices = np.flatnonzero(valid)
        remap = np.full(len(X), -1, dtype=int)
        remap[valid_indices] = np.arange(valid.sum())
        xv = X.iloc[valid_indices].reset_index(drop=True)
        yv = target[valid]
        oof = np.full(len(X), np.nan, dtype=float)
        test_by_fold = []
        train_by_fold = []
        fold_metrics = []
        best_iters = []
        for fold_id, (tr_full, va_full) in enumerate(folds):
            tr = remap[tr_full]; tr = tr[tr >= 0]
            va = remap[va_full]; va = va[va >= 0]
            train_set = lgb.Dataset(xv.iloc[tr], label=yv[tr])
            val_set = lgb.Dataset(xv.iloc[va], label=yv[va], reference=train_set)
            model = lgb.train(
                dict(LGBM_PARAMS), train_set, num_boost_round=LGBM_ROUNDS,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
            )
            val_pred = clip_osi(model.predict(xv.iloc[va]))
            oof[valid_indices[va]] = val_pred
            fold_train_pred = np.full(len(X), np.nan, dtype=float)
            fold_train_pred[valid_indices] = clip_osi(model.predict(X.iloc[valid_indices]))
            train_by_fold.append(fold_train_pred)
            test_pred = clip_osi(model.predict(X_test))
            test_by_fold.append(test_pred)
            m = metrics(yv[va], val_pred)
            m["fold"] = fold_id
            m["best_iter"] = int(model.best_iteration)
            fold_metrics.append(m)
            best_iters.append(int(model.best_iteration))
            print(f"  fold={fold_id} rmse={m['rmse']:.6f} mae={m['mae']:.6f} iter={m['best_iter']}")

        pooled = metrics(target, oof)
        best_iter = max(1, int(round(float(np.mean(best_iters)))))
        full_set = lgb.Dataset(X.iloc[valid_indices], label=yv)
        final = lgb.train(dict(LGBM_PARAMS), full_set, num_boost_round=best_iter,
                          callbacks=[lgb.log_evaluation(0)])
        test_pred = clip_osi(final.predict(X_test))
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        final.save_model(str(MODEL_DIR / f"lightgbm_{horizon}.txt"))
        results[horizon] = {
            "oof": oof, "test": test_pred,
            "train_by_fold": np.asarray(train_by_fold), "test_by_fold": np.asarray(test_by_fold),
            "fold_metrics": fold_metrics, "summary": pooled,
        }
        print(f"  pooled OOF rmse={pooled['rmse']:.6f} mae={pooled['mae']:.6f}")
    return results


def load_component_v21_base(y, meta_train, meta_test, folds):
    """Load the repository's frozen v2.1 component LightGBM artifact.

    v2.1 predicts P/N/D/R independently and composes OSI with the official
    weights. Its OOF file is keyed and therefore aligned explicitly rather
    than relying on row order. This mode is an optional stronger base for the
    spatial residual experiment; the default remains the self-contained
    direct LightGBM retraining above.
    """
    artifact_dir = PROJECT_ROOT / "versions" / "v2" / "v2.1"
    oof_path = artifact_dir / "oof_osi_predictions.csv"
    test_path = artifact_dir / "test_component_predictions.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "component_v21 mode requires versions/v2/v2.1/oof_osi_predictions.csv "
            "and test_component_predictions.csv"
        )
    oof_frame = pd.read_csv(oof_path, dtype={"fipsCode": str})
    test_frame = pd.read_csv(test_path, dtype={"fipsCode": str})
    for frame in (oof_frame, test_frame, meta_train, meta_test):
        frame["_key"] = (
            frame["fipsCode"].astype(str).str.zfill(5) + "|" +
            frame["timestamp_et"].astype(str)
        )
    oof_lookup = oof_frame.set_index("_key")
    test_lookup = test_frame.set_index("_key")
    results = {}
    weights = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": 0.10}
    for horizon in HORIZONS:
        pred_col = f"pred_{horizon}"
        if pred_col not in oof_lookup:
            raise ValueError(f"Missing {pred_col} in component OOF artifact")
        train_keys = meta_train["_key"]
        test_keys = meta_test["_key"]
        oof = oof_lookup.loc[train_keys, pred_col].to_numpy(dtype=float)
        component_values = {}
        for component, weight in weights.items():
            col = f"pred_{component}target_{horizon.replace('osi_target_', '')}"
            # v2.1 names are pred_P_t_target_t01h, etc.
            col = f"pred_{component}_target_{horizon.replace('osi_target_', '')}"
            if col not in test_lookup:
                raise ValueError(f"Missing {col} in component test artifact")
            component_values[component] = test_lookup.loc[test_keys, col].to_numpy(dtype=float)
        test_pred = (
            weights["P_t"] * component_values["P_t"] +
            weights["N_t"] * component_values["N_t"] +
            weights["D_t"] * component_values["D_t"] -
            weights["R_t"] * component_values["R_t"]
        )
        test_pred = np.clip(test_pred, 0.0, OSI_MAX)
        test_pred = np.where(test_pred < 0.001, 0.0, test_pred)
        target = y[horizon].to_numpy(dtype=float)
        pooled = metrics(target, oof)
        results[horizon] = {
            "oof": oof, "test": test_pred,
            # The frozen artifact only exposes official OOF predictions, not
            # its internal fold models. Replicating the same OOF base across
            # GAT folds keeps this optional mode explicit and separate from
            # the strict direct-base evaluation.
            "train_by_fold": np.tile(oof[None, :], (len(folds), 1)),
            "test_by_fold": np.tile(test_pred[None, :], (len(folds), 1)),
            "fold_metrics": [], "summary": pooled,
        }
    return results
