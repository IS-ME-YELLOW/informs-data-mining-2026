"""Leakage-safe v1.5.8 LightGBM bases for the Phase-2 spatial model."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (
    CV_FILE, CV_VERSION, FORBIDDEN_INPUT_COLUMNS, HORIZONS,
    LGBM_EARLY_STOPPING, LGBM_PARAMS, LGBM_ROUNDS, MODEL_DIR,
    PROJECT_ROOT,
)
from metrics import metrics, post_process_osi


COMPONENTS = ("P_t", "N_t", "D_t", "R_t")
COMPONENT_WEIGHTS = {"P_t": 0.40, "N_t": 0.35, "D_t": 0.25, "R_t": -0.10}
COMPONENT_TARGETS_FILE = (
    PROJECT_ROOT / "versions" / "xyy" / "v1.5.8" / "component_targets_v1.8.parquet"
)


def load_cv_folds(meta, n_folds=5):
    assignment = pd.read_csv(CV_FILE, dtype={"fipsCode": str})
    required = {"fipsCode", "stateAbbr", "severity_tier", "fold"}
    if set(assignment.columns) != required:
        raise ValueError(f"{CV_VERSION} CV file must contain exactly {sorted(required)}")
    assignment["fipsCode"] = assignment["fipsCode"].astype(str).str.zfill(5)
    assignment["stateAbbr"] = assignment["stateAbbr"].astype(str)
    assignment["severity_tier"] = pd.to_numeric(
        assignment["severity_tier"], errors="raise"
    ).astype(int)
    assignment["fold"] = pd.to_numeric(assignment["fold"], errors="raise").astype(int)
    if assignment["fipsCode"].duplicated().any():
        raise ValueError(f"{CV_VERSION} CV file contains duplicate counties")
    if set(assignment["fold"]) != set(range(n_folds)):
        raise ValueError(f"{CV_VERSION} CV file must contain folds 0..{n_folds - 1}")
    fips = meta["fipsCode"].astype(str).str.zfill(5)
    fold_map = assignment.set_index("fipsCode")["fold"]
    county_meta = meta[["fipsCode", "stateAbbr", "severity_tier"]].copy()
    county_meta["fipsCode"] = county_meta["fipsCode"].astype(str).str.zfill(5)
    county_meta["stateAbbr"] = county_meta["stateAbbr"].astype(str)
    county_meta["severity_tier"] = pd.to_numeric(
        county_meta["severity_tier"], errors="raise"
    ).astype(int)
    county_meta = county_meta.drop_duplicates().set_index("fipsCode").sort_index()
    assigned_meta = assignment.set_index("fipsCode").sort_index()
    if not county_meta.index.equals(assigned_meta.index):
        raise ValueError(f"{CV_VERSION} CV county set does not match training metadata")
    if not county_meta[["stateAbbr", "severity_tier"]].equals(
        assigned_meta[["stateAbbr", "severity_tier"]]
    ):
        raise ValueError(f"{CV_VERSION} CV stratification fields do not match metadata")
    row_fold = fips.map(fold_map)
    if row_fold.isna().any():
        raise ValueError("CV assignment does not cover every training county")
    row_fold = row_fold.to_numpy(dtype=int)
    return [
        (np.flatnonzero(row_fold != i), np.flatnonzero(row_fold == i))
        for i in range(n_folds)
    ]


def _fit_cv_booster(X_train, y_train, X_valid, y_valid):
    """Fit one v1.5.8-compatible LightGBM CV model."""
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    model = lgb.train(
        dict(LGBM_PARAMS), train_set, num_boost_round=LGBM_ROUNDS,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model, int(model.best_iteration)


def _fit_final_booster(X_train, y_train, rounds):
    return lgb.train(
        dict(LGBM_PARAMS),
        lgb.Dataset(X_train, label=y_train),
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(0)],
    )


def _post_target(prediction, target_kind):
    if target_kind == "component":
        return np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    if target_kind == "direct_osi":
        return post_process_osi(prediction)
    raise ValueError(f"Unknown target_kind={target_kind!r}")


def _nested_target_predictions(X, target, X_test, folds, target_kind, target_name):
    """Create outer OOF and inner-cross-fitted outer-training predictions.

    For outer fold k, validation/test predictions come from a model trained
    only on the other four county folds. Every outer-training row is predicted
    by an inner model that did not train on that row's county. The resulting
    ``train_by_fold[k]`` is safe to use as the complete GAT graph input.
    """
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(target)
    oof = np.full(len(X), np.nan, dtype=float)
    train_by_fold, test_by_fold = [], []
    fold_metrics, outer_best_iters = [], []

    for outer_fold, (outer_train_full, outer_valid_full) in enumerate(folds):
        outer_train = outer_train_full[valid[outer_train_full]]
        outer_valid = outer_valid_full[valid[outer_valid_full]]
        if len(outer_train) == 0 or len(outer_valid) == 0:
            raise ValueError(f"Empty outer split for {target_name}, fold={outer_fold}")

        nested_train_pred = np.full(len(X), np.nan, dtype=float)
        inner_rounds = []
        for inner_fold, (inner_train_full, inner_valid_full) in enumerate(folds):
            if inner_fold == outer_fold:
                continue
            inner_train = np.intersect1d(outer_train, inner_train_full)
            inner_valid = np.intersect1d(outer_train, inner_valid_full)
            inner_train = inner_train[valid[inner_train]]
            inner_valid = inner_valid[valid[inner_valid]]
            if len(inner_train) == 0 or len(inner_valid) == 0:
                raise ValueError(
                    f"Empty nested split for {target_name}, outer={outer_fold}, "
                    f"inner={inner_fold}"
                )
            inner_model, inner_round = _fit_cv_booster(
                X.iloc[inner_train], target[inner_train],
                X.iloc[inner_valid], target[inner_valid],
            )
            nested_train_pred[inner_valid] = _post_target(
                inner_model.predict(X.iloc[inner_valid], num_iteration=inner_round),
                target_kind,
            )
            inner_rounds.append(inner_round)

        # Early stopping for the outer model is determined only inside the
        # outer-training counties. The outer validation labels are not passed
        # to LightGBM callbacks and are used exactly once for final scoring.
        outer_rounds = max(1, int(np.mean(inner_rounds)))
        outer_model = _fit_final_booster(
            X.iloc[outer_train], target[outer_train], outer_rounds
        )
        outer_valid_pred = _post_target(
            outer_model.predict(X.iloc[outer_valid], num_iteration=outer_rounds),
            target_kind,
        )
        oof[outer_valid] = outer_valid_pred
        outer_test_pred = _post_target(
            outer_model.predict(X_test, num_iteration=outer_rounds), target_kind
        )
        nested_train_pred[outer_valid] = outer_valid_pred

        if np.isfinite(nested_train_pred[outer_train]).sum() != len(outer_train):
            raise ValueError(f"Nested cross-fit coverage incomplete for {target_name}")
        train_by_fold.append(nested_train_pred)
        test_by_fold.append(outer_test_pred)
        score = metrics(target[outer_valid], outer_valid_pred)
        score.update({
            "fold": outer_fold,
            "best_iter": outer_rounds,
            "inner_best_iters": inner_rounds,
        })
        fold_metrics.append(score)
        outer_best_iters.append(outer_rounds)
        print(
            f"  {target_name} outer_fold={outer_fold} "
            f"rmse={score['rmse']:.6f} mae={score['mae']:.6f} "
            f"iter={outer_rounds} inner_cross_fit=4"
        )

    if np.isnan(oof[valid]).any() or np.isfinite(oof[~valid]).any():
        raise ValueError(f"Invalid OOF coverage for {target_name}")
    final_rounds = max(1, int(np.mean(outer_best_iters)))
    final_model = _fit_final_booster(X.loc[valid], target[valid], final_rounds)
    final_test = _post_target(
        final_model.predict(X_test, num_iteration=final_rounds), target_kind
    )
    return {
        "oof": oof,
        "test": final_test,
        "train_by_fold": np.asarray(train_by_fold),
        "test_by_fold": np.asarray(test_by_fold),
        "fold_metrics": fold_metrics,
        "summary": metrics(target, oof),
        "final_rounds": final_rounds,
        "final_model": final_model,
        "final_source": "retrained v1.5.8-compatible booster",
    }


def _validate_model_inputs(X, X_test):
    forbidden_train = sorted(set(map(str, X.columns)) & FORBIDDEN_INPUT_COLUMNS)
    forbidden_test = sorted(set(map(str, X_test.columns)) & FORBIDDEN_INPUT_COLUMNS)
    if forbidden_train or forbidden_test:
        raise ValueError(
            "LightGBM received competition-forbidden model input columns: "
            f"train={forbidden_train}, test={forbidden_test}"
        )


def train_base_models(X, y, meta, X_test, folds):
    """Train v1.5.8 C0 direct-OSI LightGBM with nested county cross-fitting."""
    _validate_model_inputs(X, X_test)
    results = {}
    for horizon in HORIZONS:
        print(f"[LightGBM] {horizon} (v1.5.8 C0 direct, nested cross-fit)")
        result = _nested_target_predictions(
            X, y[horizon].to_numpy(dtype=float), X_test, folds,
            "direct_osi", horizon,
        )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        result["final_model"].save_model(
            str(MODEL_DIR / f"lightgbm_{horizon}.txt"),
            num_iteration=result["final_rounds"],
        )
        results[horizon] = result
        print(
            f"  pooled OOF rmse={result['summary']['rmse']:.6f} "
            f"mae={result['summary']['mae']:.6f}; "
            f"final_rounds={result['final_rounds']}"
        )
    return results


def _key_frame(frame):
    return (
        frame["fipsCode"].astype(str).str.zfill(5) + "|" +
        pd.to_datetime(frame["timestamp_et"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    )


def _load_v158_component_targets(meta_train):
    """Load the shifted P/N/D/R target columns generated by v1.5.8."""
    if not COMPONENT_TARGETS_FILE.exists():
        raise FileNotFoundError(f"Missing v1.5.8 component targets: {COMPONENT_TARGETS_FILE}")
    frame = pd.read_parquet(COMPONENT_TARGETS_FILE)
    lookup = frame.set_index(_key_frame(frame))
    meta_keys = _key_frame(meta_train)
    if not meta_keys.isin(lookup.index).all():
        raise ValueError("v1.5.8 component targets do not cover all cached rows")
    return {
        horizon: {
            component: lookup.loc[
                meta_keys,
                f"{component}_target_{horizon.replace('osi_target_', '')}",
            ].to_numpy(dtype=float)
            for component in COMPONENTS
        }
        for horizon in HORIZONS
    }


def _compose(predictions):
    output = np.zeros_like(np.asarray(next(iter(predictions.values())), dtype=float))
    for component, weight in COMPONENT_WEIGHTS.items():
        output += weight * np.asarray(predictions[component], dtype=float)
    return np.maximum(output, 0.0)


def train_component_v158_base(X, y, meta_train, X_test, folds):
    """Train v1.5.8 C1 component OSI with strict nested cross-fitting."""
    _validate_model_inputs(X, X_test)
    targets = _load_v158_component_targets(meta_train)
    component_results = {horizon: {} for horizon in HORIZONS}
    for horizon in HORIZONS:
        for component in COMPONENTS:
            target_name = f"{component}_target_{horizon.replace('osi_target_', '')}"
            print(f"[LightGBM component] {target_name}")
            component_results[horizon][component] = _nested_target_predictions(
                X, targets[horizon][component], X_test, folds,
                "component", target_name,
            )

    results = {}
    for horizon in HORIZONS:
        target = y[horizon].to_numpy(dtype=float)
        oof = np.full(len(X), np.nan, dtype=float)
        train_by_fold, test_by_fold = [], []
        for fold_id in range(len(folds)):
            train_parts = {
                component: component_results[horizon][component]["train_by_fold"][fold_id]
                for component in COMPONENTS
            }
            test_parts = {
                component: component_results[horizon][component]["test_by_fold"][fold_id]
                for component in COMPONENTS
            }
            train_pred = post_process_osi(_compose(train_parts))
            test_pred = post_process_osi(_compose(test_parts))
            train_by_fold.append(train_pred)
            test_by_fold.append(test_pred)
            _, outer_valid_full = folds[fold_id]
            valid_rows = outer_valid_full[np.isfinite(target[outer_valid_full])]
            oof[valid_rows] = train_pred[valid_rows]
        final_parts = {
            component: component_results[horizon][component]["test"]
            for component in COMPONENTS
        }
        final_test = post_process_osi(_compose(final_parts))
        if np.isnan(oof[np.isfinite(target)]).any():
            raise ValueError(f"Invalid component OOF coverage for {horizon}")
        results[horizon] = {
            "oof": oof,
            "test": final_test,
            "train_by_fold": np.asarray(train_by_fold),
            "test_by_fold": np.asarray(test_by_fold),
            "fold_metrics": [],
            "summary": metrics(target, oof),
            "final_source": "retrained v1.5.8-compatible component boosters",
            "component_results": component_results[horizon],
        }
        print(
            f"  pooled component-composed OOF rmse={results[horizon]['summary']['rmse']:.6f} "
            f"mae={results[horizon]['summary']['mae']:.6f}"
        )
    return results


def load_component_v21_base(y, meta_train, meta_test, folds, X=None, X_test=None):
    """Compatibility alias that no longer loads leakage-prone frozen artifacts."""
    if X is None or X_test is None:
        raise ValueError("component_v21 compatibility mode requires feature frames")
    print("[component_v21] frozen artifact disabled; using strict v1.5.8 component retraining")
    return train_component_v158_base(X, y, meta_train, X_test, folds)
