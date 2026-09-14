"""独立重载 v1.8 的 120 个模型，并复核 OOF、测试预测、组合与提交文件。"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import COMPONENTS, CV_VERSION, EXPERIMENT_VERSION, HORIZON_HOURS, HORIZONS, PROJECT_ROOT, V18_DIR
from protocol import (
    build_component_targets,
    clip_component,
    component_target_name,
    fill_submission,
    load_data,
    sha256_file,
)
from train_v18 import MODELS, _build_controls


ATOL = 1e-12


def _transcript_duration(path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"Start time: (\d{14})", text)
    end = re.search(r"End time: (\d{14})", text)
    if not start or not end:
        return None
    fmt = "%Y%m%d%H%M%S"
    return (datetime.strptime(end.group(1), fmt) - datetime.strptime(start.group(1), fmt)).total_seconds()


def _max_diff(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        raise AssertionError("数组形状或 NaN 掩码不一致")
    valid = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else 0.0


def main() -> None:
    started = time.perf_counter()
    print("[1/5] 加载 v1.8 数据、目标和模型清单...", flush=True)
    data = load_data()
    targets, _ = build_component_targets(data)
    manifest = pd.read_csv(V18_DIR / "model_manifest.csv")
    if len(manifest) != 120:
        raise AssertionError(f"模型清单应为120行，实际{len(manifest)}")
    for _, row in manifest.iterrows():
        path = PROJECT_ROOT / row["path"]
        if not path.exists() or sha256_file(path) != row["sha256"]:
            raise AssertionError(f"模型文件缺失或哈希错误: {path}")

    saved_oof = pd.read_parquet(V18_DIR / "oof_predictions.parquet")
    saved_component_oof = pd.read_parquet(V18_DIR / "oof_component_predictions.parquet")
    saved_component_test = pd.read_parquet(V18_DIR / "test_component_predictions.parquet")
    saved_test_controls = pd.read_parquet(V18_DIR / "test_control_predictions.parquet")

    direct_oof = {h: np.full(len(data.X_train), np.nan) for h in HORIZONS}
    direct_test = {}
    component_oof = {h: {c: np.full(len(data.X_train), np.nan) for c in COMPONENTS} for h in HORIZONS}
    component_test = {h: {} for h in HORIZONS}
    discrepancies = []

    print("[2/5] 重载 100 个折模型并复算 OOF...", flush=True)
    for target_kind, target in [
        *(('direct_osi', h) for h in HORIZONS),
        *(('component', component_target_name(c, h)) for h in HORIZONS for c in COMPONENTS),
    ]:
        y = data.y_train[target] if target_kind == "direct_osi" else targets[target]
        valid_target = y.notna().to_numpy()
        for fold in range(5):
            row = manifest[(manifest["role"] == "cv") & (manifest["target"] == target) & (manifest["fold"] == fold)]
            if len(row) != 1:
                raise AssertionError(f"折模型清单异常: {target}/fold{fold}")
            booster = lgb.Booster(model_file=str(PROJECT_ROOT / row.iloc[0]["path"]))
            idx = np.flatnonzero((data.row_folds == fold) & valid_target)
            raw = booster.predict(data.X_train.iloc[idx], num_iteration=int(row.iloc[0]["rounds"]))
            if target_kind == "direct_osi":
                direct_oof[target][idx] = raw
            else:
                component = target.split("_target_")[0]
                horizon = f"osi_target_{target.split('_target_')[1]}"
                component_oof[horizon][component][idx] = raw

    print("[3/5] 重载 20 个全量模型并复算测试预测...", flush=True)
    for target_kind, target in [
        *(('direct_osi', h) for h in HORIZONS),
        *(('component', component_target_name(c, h)) for h in HORIZONS for c in COMPONENTS),
    ]:
        row = manifest[(manifest["role"] == "final") & (manifest["target"] == target)]
        if len(row) != 1:
            raise AssertionError(f"全量模型清单异常: {target}")
        booster = lgb.Booster(model_file=str(PROJECT_ROOT / row.iloc[0]["path"]))
        raw = booster.predict(data.X_test, num_iteration=int(row.iloc[0]["rounds"]))
        if target_kind == "direct_osi":
            direct_test[target] = raw
        else:
            component = target.split("_target_")[0]
            horizon = f"osi_target_{target.split('_target_')[1]}"
            component_test[horizon][component] = raw

    print("[4/5] 复核预测、组合公式、目标小时一致性和提交...", flush=True)
    oof_controls, _ = _build_controls(data.meta_train, direct_oof, component_oof)
    test_controls, _ = _build_controls(data.meta_test, direct_test, component_test)
    for horizon in HORIZONS:
        for component in COMPONENTS:
            name = component_target_name(component, horizon)
            discrepancies.append(_max_diff(
                clip_component(component_oof[horizon][component]),
                saved_component_oof[f"pred_{name}"],
            ))
            discrepancies.append(_max_diff(
                clip_component(component_test[horizon][component]),
                saved_component_test[f"pred_{name}"],
            ))
        for model in MODELS:
            discrepancies.append(_max_diff(
                oof_controls[model][horizon], saved_oof[f"pred_{model}_{horizon}"]
            ))
            discrepancies.append(_max_diff(
                test_controls[model][horizon], saved_test_controls[f"pred_{model}_{horizon}"]
            ))

    # C3 对同一县、同一目标时刻必须逐值相同。
    c3_long = []
    for horizon in HORIZONS:
        valid = data.meta_test["hour_idx"].to_numpy() + HORIZON_HOURS[horizon] <= 215
        c3_long.append(pd.DataFrame({
            "fipsCode": data.meta_test.loc[valid, "fipsCode"].to_numpy(),
            "target_timestamp": data.meta_test.loc[valid, "timestamp_et"].to_numpy()
                + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h"),
            "prediction": test_controls["C3_aligned_component"][horizon][valid],
        }))
    c3_long = pd.concat(c3_long, ignore_index=True)
    spreads = c3_long.groupby(["fipsCode", "target_timestamp"])["prediction"].agg(lambda x: x.max() - x.min())
    c3_max_spread = float(spreads.max())
    if c3_max_spread > ATOL:
        raise AssertionError(f"C3 同目标小时预测不一致: {c3_max_spread}")

    for model in MODELS:
        recomputed = fill_submission(data.meta_test, test_controls[model])
        path = V18_DIR / f"submission_{EXPERIMENT_VERSION}_{model}_{CV_VERSION}.csv"
        saved = pd.read_csv(path, dtype={"fipsCode": str})
        for key in ["fipsCode", "countyName", "stateAbbr", "timestamp_et"]:
            if not saved[key].astype(str).equals(recomputed[key].astype(str)):
                raise AssertionError(f"{model} 提交标识列 {key} 被改变")
        for horizon in HORIZONS:
            discrepancies.append(_max_diff(saved[horizon], recomputed[horizon]))

    max_discrepancy = max(discrepancies)
    if max_discrepancy > ATOL:
        raise AssertionError(f"保存预测复算误差过大: {max_discrepancy}")

    result = {
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "passed",
        "models_reloaded": 120,
        "cv_models": 100,
        "final_models": 20,
        "model_hashes_verified": 120,
        "max_prediction_discrepancy": max_discrepancy,
        "c3_same_target_max_spread": c3_max_spread,
        "train_counties": int(data.meta_train["fipsCode"].nunique()),
        "test_counties": int(data.meta_test["fipsCode"].nunique()),
        "submission_nan_counts": {
            model: {
                horizon: int(pd.read_csv(
                    V18_DIR / f"submission_{EXPERIMENT_VERSION}_{model}_{CV_VERSION}.csv"
                )[horizon].isna().sum())
                for horizon in HORIZONS
            }
            for model in MODELS
        },
    }
    result["verification_runtime_seconds"] = float(time.perf_counter() - started)
    (V18_DIR / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_path = V18_DIR / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifact_completion_seconds = float(
        metadata.get("timing", {}).get(
            "resume_and_artifact_completion_seconds", metadata.get("runtime_seconds", 0.0)
        )
    )
    training_attempt_seconds = _transcript_duration(V18_DIR / "run_console.log")
    timing = {
        "training_and_initial_processing_seconds": training_attempt_seconds,
        "resume_and_artifact_completion_seconds": artifact_completion_seconds,
        "independent_verification_seconds": result["verification_runtime_seconds"],
    }
    timing["total_observed_pipeline_seconds"] = float(sum(
        value for value in timing.values() if value is not None
    ))
    metadata["timing"] = timing
    metadata["runtime_seconds"] = timing["total_observed_pipeline_seconds"]
    metadata["verification"] = result
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
