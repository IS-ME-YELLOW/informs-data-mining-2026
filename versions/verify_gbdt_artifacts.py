"""Verify v1.6/v1.7 metrics, serialized models, and submissions end to end."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


VERSIONS_DIR = Path(__file__).resolve().parent
if str(VERSIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VERSIONS_DIR))

from gbdt_experiment import (  # noqa: E402
    HORIZONS,
    HORIZON_HOURS,
    PROJECT_ROOT,
    compute_metrics,
    load_experiment_data,
    post_process,
)


def _normalized_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    fips = frame["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    timestamps = pd.to_datetime(frame["timestamp_et"])
    return pd.MultiIndex.from_arrays([fips, timestamps], names=["fipsCode", "timestamp_et"])


def _load_model(family: str, path: Path):
    if family == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor()
    elif family == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
    else:
        raise ValueError(f"Unknown family: {family}")
    model.load_model(path)
    return model


def verify_version(
    version: str, family: str, extension: str, data, template: pd.DataFrame
) -> None:
    version_dir = PROJECT_ROOT / "versions" / version
    summary = pd.read_csv(version_dir / "summary_metrics.csv").set_index("horizon")
    oof = pd.read_csv(version_dir / "oof_predictions.csv")
    submission = pd.read_csv(
        version_dir / f"submission_{version}_balanced_v1.csv",
        dtype={"fipsCode": str},
    )

    identifier_columns = ["fipsCode", "countyName", "stateAbbr", "timestamp_et"]
    if not submission[identifier_columns].equals(template[identifier_columns]):
        raise AssertionError(f"{version}: submission identifiers differ from the template")
    if not _normalized_keys(submission).equals(_normalized_keys(template)):
        raise AssertionError(f"{version}: normalized submission keys differ from the template")

    submission_by_key = submission.copy()
    submission_by_key.index = _normalized_keys(submission_by_key)
    test_meta = data.meta_test.copy()
    test_meta.index = _normalized_keys(test_meta)
    if not test_meta.index.is_unique or not submission_by_key.index.is_unique:
        raise AssertionError(f"{version}: duplicate test/submission keys")

    for horizon in HORIZONS:
        actual = oof[f"actual_{horizon}"].to_numpy(dtype=float)
        predicted = oof[f"pred_{horizon}"].to_numpy(dtype=float)
        recomputed = compute_metrics(actual, predicted)
        for metric in ("rmse", "mae"):
            expected = float(summary.loc[horizon, metric])
            if not np.isclose(recomputed[metric], expected, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"{version} {horizon}: {metric} mismatch "
                    f"({recomputed[metric]} != {expected})"
                )

        model_path = version_dir / (
            f"{family}_{horizon}_{version}_balanced_v1{extension}"
        )
        model = _load_model(family, model_path)
        model_predictions = post_process(model.predict(data.X_test))
        aligned_submission = submission_by_key.reindex(test_meta.index)[horizon].to_numpy()
        scored = test_meta["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] <= 215
        if np.isnan(aligned_submission[scored]).any():
            raise AssertionError(f"{version} {horizon}: NaN found on scored submission rows")
        if np.isfinite(aligned_submission[~scored]).any():
            raise AssertionError(f"{version} {horizon}: tail rows should be NaN")
        if not np.allclose(
            model_predictions[scored], aligned_submission[scored], rtol=0, atol=1e-12
        ):
            raise AssertionError(f"{version} {horizon}: serialized model and submission differ")
        if np.any((model_predictions < 0) | (model_predictions > 0.65)):
            raise AssertionError(f"{version} {horizon}: prediction outside [0, 0.65]")

    print(f"{version}: metrics, models, identifiers, and submission values verified")


def main() -> None:
    data = load_experiment_data()
    template = pd.read_csv(
        PROJECT_ROOT / "data" / "sample_submission.csv", dtype={"fipsCode": str}
    )
    verify_version("v1.6", "xgboost", ".json", data, template)
    verify_version("v1.7", "catboost", ".cbm", data, template)


if __name__ == "__main__":
    main()
