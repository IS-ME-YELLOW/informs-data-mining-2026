"""Verify versioned direct-OSI GBDT metrics, models, and submissions end to end."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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


SPECS = {
    "v1.6": ("xgboost", ".json", "v1.5.2", "v1.6"),
    "v1.7": ("catboost", ".cbm", "v1.5.2", "v1.7"),
    "v1.9_xgboost": ("xgboost", ".json", "v1.5.6", "v1.9"),
    "v1.10": ("catboost", ".cbm", "v1.5.6", "v1.10"),
}


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
    version_dir_name: str, family: str, extension: str, feature_version: str,
    artifact_version: str,
    data, template: pd.DataFrame
) -> None:
    version_dir = PROJECT_ROOT / "versions" / version_dir_name
    summary = pd.read_csv(version_dir / "summary_metrics.csv").set_index("horizon")
    oof = pd.read_csv(version_dir / "oof_predictions.csv")
    submission = pd.read_csv(
        version_dir / f"submission_{artifact_version}_balanced_v1.csv",
        dtype={"fipsCode": str},
    )

    identifier_columns = ["fipsCode", "countyName", "stateAbbr", "timestamp_et"]
    if not submission[identifier_columns].equals(template[identifier_columns]):
        raise AssertionError(f"{version_dir_name}: submission identifiers differ from the template")
    if not _normalized_keys(submission).equals(_normalized_keys(template)):
        raise AssertionError(f"{version_dir_name}: normalized submission keys differ from the template")

    submission_by_key = submission.copy()
    submission_by_key.index = _normalized_keys(submission_by_key)
    test_meta = data.meta_test.copy()
    test_meta.index = _normalized_keys(test_meta)
    if not test_meta.index.is_unique or not submission_by_key.index.is_unique:
        raise AssertionError(f"{version_dir_name}: duplicate test/submission keys")

    for horizon in HORIZONS:
        actual = oof[f"actual_{horizon}"].to_numpy(dtype=float)
        predicted = oof[f"pred_{horizon}"].to_numpy(dtype=float)
        recomputed = compute_metrics(actual, predicted)
        for metric in ("rmse", "mae"):
            expected = float(summary.loc[horizon, metric])
            if not np.isclose(recomputed[metric], expected, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"{version_dir_name} {horizon}: {metric} mismatch "
                    f"({recomputed[metric]} != {expected})"
                )

        model_path = version_dir / (
            f"{family}_{horizon}_{artifact_version}_balanced_v1{extension}"
        )
        model = _load_model(family, model_path)
        model_predictions = post_process(model.predict(data.X_test))
        aligned_submission = submission_by_key.reindex(test_meta.index)[horizon].to_numpy()
        scored = test_meta["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] <= 215
        if np.isnan(aligned_submission[scored]).any():
            raise AssertionError(f"{version_dir_name} {horizon}: NaN found on scored submission rows")
        if np.isfinite(aligned_submission[~scored]).any():
            raise AssertionError(f"{version_dir_name} {horizon}: tail rows should be NaN")
        if not np.allclose(
            model_predictions[scored], aligned_submission[scored], rtol=0, atol=1e-12
        ):
            raise AssertionError(f"{version_dir_name} {horizon}: serialized model and submission differ")
        if np.any((model_predictions < 0) | (model_predictions > 0.65)):
            raise AssertionError(f"{version_dir_name} {horizon}: prediction outside [0, 0.65]")

    verification = {
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "version": artifact_version,
        "version_directory": version_dir_name,
        "feature_version": feature_version,
        "checks": [
            "OOF RMSE and MAE recomputed",
            "four serialized final models reloaded",
            "submission identifiers and keys match template",
            "reloaded-model predictions match submission at atol=1e-12",
            "tail NaN masks and [0, 0.65] prediction bounds verified",
        ],
    }
    with (version_dir / "verification.json").open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, ensure_ascii=False, indent=2)
    print(f"{version_dir_name}: metrics, models, identifiers, and submission values verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("versions", nargs="*", choices=sorted(SPECS))
    args = parser.parse_args()
    selected = args.versions or list(SPECS)
    template = pd.read_csv(
        PROJECT_ROOT / "data" / "sample_submission.csv", dtype={"fipsCode": str}
    )
    data_by_feature = {}
    for version in selected:
        family, extension, feature_version, artifact_version = SPECS[version]
        if feature_version not in data_by_feature:
            data_by_feature[feature_version] = load_experiment_data(feature_version)
        data = data_by_feature[feature_version]
        verify_version(
            version, family, extension, feature_version, artifact_version, data, template
        )


if __name__ == "__main__":
    main()
