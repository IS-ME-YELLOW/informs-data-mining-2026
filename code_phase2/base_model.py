"""LightGBM phase-1-compatible base model and county-grouped OOF predictions."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (CV_FILE, HORIZONS, LGBM_EARLY_STOPPING, LGBM_PARAMS,
                    LGBM_ROUNDS, MODEL_DIR, OSI_MAX)
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
