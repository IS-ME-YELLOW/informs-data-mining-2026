"""End-to-end verification of all v2 component-first experiment artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V2_ROOT.parents[1]
VERSIONS_ROOT = PROJECT_ROOT / "versions"
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))
if str(VERSIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSIONS_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gbdt_experiment as direct_protocol  # noqa: E402
from component_experiment import (  # noqa: E402
    COMPONENTS,
    clip_component,
    component_target_name,
    compose_osi,
    prepare_component_targets,
)


SPECS = {
    "v2.1": ("lightgbm", ".txt"),
    "v2.2": ("xgboost", ".json"),
    "v2.3": ("catboost", ".cbm"),
}


def _normalized_keys(frame: pd.DataFrame) -> pd.MultiIndex:
    fips = frame["fipsCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    timestamp = pd.to_datetime(frame["timestamp_et"])
    return pd.MultiIndex.from_arrays([fips, timestamp], names=["fipsCode", "timestamp_et"])


def _load_model(family: str, path: Path):
    if family == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if family == "xgboost":
        import xgboost as xgb

        model = xgb.XGBRegressor()
        model.load_model(path)
        return model
    if family == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(path)
        return model
    raise ValueError(f"Unknown family: {family}")


def verify_version(version: str, family: str, extension: str, data, targets, template) -> None:
    output_dir = V2_ROOT / version
    component_summary = pd.read_csv(
        output_dir / "component_summary_metrics.csv"
    ).set_index(["horizon", "component"])
    osi_summary = pd.read_csv(output_dir / "osi_summary_metrics.csv").set_index("horizon")
    component_oof = pd.read_csv(output_dir / "oof_component_predictions.csv")
    osi_oof = pd.read_csv(output_dir / "oof_osi_predictions.csv")
    test_components = pd.read_csv(output_dir / "test_component_predictions.csv")
    submission = pd.read_csv(
        output_dir / f"submission_{version}_balanced_v1.csv", dtype={"fipsCode": str}
    )

    identifier_columns = ["fipsCode", "countyName", "stateAbbr", "timestamp_et"]
    if not submission[identifier_columns].equals(template[identifier_columns]):
        raise AssertionError(f"{version}: submission identifiers differ from template")
    if not _normalized_keys(component_oof).equals(_normalized_keys(data.meta_train)):
        raise AssertionError(f"{version}: component OOF row keys are misaligned")
    if not _normalized_keys(osi_oof).equals(_normalized_keys(data.meta_train)):
        raise AssertionError(f"{version}: OSI OOF row keys are misaligned")
    if not _normalized_keys(test_components).equals(_normalized_keys(data.meta_test)):
        raise AssertionError(f"{version}: test component row keys are misaligned")

    submission_by_key = submission.copy()
    submission_by_key.index = _normalized_keys(submission_by_key)
    test_meta = data.meta_test.copy()
    test_meta.index = _normalized_keys(test_meta)

    for horizon in direct_protocol.HORIZONS:
        oof_by_component = {}
        test_by_component = {}
        for component in COMPONENTS:
            target_column = component_target_name(component, horizon)
            actual = component_oof[f"actual_{target_column}"].to_numpy(dtype=float)
            expected_actual = targets[target_column].to_numpy(dtype=float)
            if not np.allclose(actual, expected_actual, rtol=0, atol=0, equal_nan=True):
                raise AssertionError(f"{version}: actual target mismatch for {target_column}")
            predicted = component_oof[f"pred_{target_column}"].to_numpy(dtype=float)
            score = direct_protocol.compute_metrics(actual, predicted)
            for metric in ("rmse", "mae"):
                expected = float(component_summary.loc[(horizon, component), metric])
                if not np.isclose(score[metric], expected, rtol=0, atol=1e-12):
                    raise AssertionError(
                        f"{version} {target_column}: {metric} mismatch"
                    )
            oof_by_component[component] = predicted

            model_path = output_dir / (
                f"{family}_{component}_{horizon}_{version}_balanced_v1{extension}"
            )
            model = _load_model(family, model_path)
            reloaded_prediction = clip_component(model.predict(data.X_test))
            saved_prediction = test_components[
                f"pred_{target_column}"
            ].to_numpy(dtype=float)
            if not np.allclose(
                reloaded_prediction, saved_prediction, rtol=0, atol=1e-12
            ):
                raise AssertionError(
                    f"{version}: reloaded model differs for {target_column}"
                )
            if np.any((reloaded_prediction < 0) | (reloaded_prediction > 1)):
                raise AssertionError(f"{version}: component outside [0, 1]")
            test_by_component[component] = reloaded_prediction

        recomposed_oof = direct_protocol.post_process(compose_osi(oof_by_component))
        saved_oof = osi_oof[f"pred_{horizon}"].to_numpy(dtype=float)
        if not np.allclose(recomposed_oof, saved_oof, rtol=0, atol=1e-12, equal_nan=True):
            raise AssertionError(f"{version}: recomposed OOF mismatch for {horizon}")
        official = osi_oof[f"actual_{horizon}"].to_numpy(dtype=float)
        score = direct_protocol.compute_metrics(official, saved_oof)
        for metric in ("rmse", "mae"):
            expected = float(osi_summary.loc[horizon, metric])
            if not np.isclose(score[metric], expected, rtol=0, atol=1e-12):
                raise AssertionError(f"{version} {horizon}: composed {metric} mismatch")

        test_osi = direct_protocol.post_process(compose_osi(test_by_component))
        submitted = submission_by_key.reindex(test_meta.index)[horizon].to_numpy(dtype=float)
        scored = test_meta["hour_idx"].to_numpy() + direct_protocol.HORIZON_HOURS[horizon] <= 215
        if not np.allclose(test_osi[scored], submitted[scored], rtol=0, atol=1e-12):
            raise AssertionError(f"{version}: submission values differ for {horizon}")
        if np.isnan(submitted[scored]).any() or np.isfinite(submitted[~scored]).any():
            raise AssertionError(f"{version}: submission NaN mask is wrong for {horizon}")

    print(f"{version}: 16 models, component OOF, composed OSI, and submission verified")


def main() -> None:
    data = direct_protocol.load_experiment_data()
    targets, _ = prepare_component_targets()
    template = pd.read_csv(
        PROJECT_ROOT / "data" / "sample_submission.csv", dtype={"fipsCode": str}
    )
    for version, (family, extension) in SPECS.items():
        verify_version(version, family, extension, data, targets, template)


if __name__ == "__main__":
    main()
