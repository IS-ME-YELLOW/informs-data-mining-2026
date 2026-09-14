"""评估 v1.9 分量候选、逐分量替换、组合、对齐和县级不确定性。"""

from __future__ import annotations

import itertools
import json
from datetime import datetime

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COMPONENT_WEIGHTS,
    HORIZONS,
    HORIZON_HOURS,
    RESPONSE_FEATURES,
    V18_DIR,
    V19_DIR,
)
from protocol import (
    align_same_target,
    build_component_targets,
    clip_component,
    component_target_name,
    compose_osi,
    fill_submission,
    load_data,
    metrics,
    post_process,
)
from train_v19 import COMPONENTS, REGRESSION_CANDIDATES, candidate_key


def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    result = metrics(y_true, y_pred)
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    variance = float(np.var(truth[valid]))
    result["r2"] = float(1.0 - np.mean((pred[valid] - truth[valid]) ** 2) / variance) if variance > 0 else np.nan
    result["prediction_mean"] = float(np.mean(pred[valid]))
    result["prediction_max"] = float(np.max(pred[valid]))
    result["prediction_zero_pct"] = float(100 * np.mean(pred[valid] == 0))
    return result


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / ranked.sum())


def config_prediction(
    horizon: str,
    config: dict[str, str],
    source: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = {component: clip_component(source[config[component]]) for component in COMPONENTS}
    return post_process(compose_osi(components)), components


def evaluate_config(
    name: str,
    horizon: str,
    config: dict[str, str],
    source: dict[str, np.ndarray],
    truth: np.ndarray,
) -> dict:
    pred, _ = config_prediction(horizon, config, source)
    return {"model": name, "horizon": horizon, **extended_metrics(truth, pred)}


def paired_bootstrap(
    meta: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    frame = pd.DataFrame({
        "fipsCode": meta["fipsCode"].astype(str),
        "truth": truth,
        "candidate": candidate,
        "reference": reference,
    }).dropna()
    rows = []
    for _, group in frame.groupby("fipsCode", sort=True):
        rows.append((
            float(np.sum((group["candidate"] - group["truth"]) ** 2)),
            float(np.sum((group["reference"] - group["truth"]) ** 2)),
            int(len(group)),
        ))
    values = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(replicates, dtype=float)
    n_counties = len(values)
    for rep in range(replicates):
        pick = rng.integers(0, n_counties, size=n_counties)
        selected = values[pick]
        count = selected[:, 2].sum()
        diffs[rep] = np.sqrt(selected[:, 0].sum() / count) - np.sqrt(selected[:, 1].sum() / count)
    point = float(
        np.sqrt(values[:, 0].sum() / values[:, 2].sum())
        - np.sqrt(values[:, 1].sum() / values[:, 2].sum())
    )
    return {
        "rmse_diff": point,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "improvement_probability": float(np.mean(diffs < 0)),
        "replicates": replicates,
        "county_count": n_counties,
    }


def main() -> None:
    data = load_data()
    targets, _ = build_component_targets(data)
    oof_frame = pd.read_parquet(V19_DIR / "oof_component_candidates.parquet")
    test_frame = pd.read_parquet(V19_DIR / "test_component_candidates.parquet")
    v18_oof = pd.read_parquet(V18_DIR / "oof_predictions.parquet")
    v18_test = pd.read_parquet(V18_DIR / "test_control_predictions.parquet")
    hurdle = pd.read_parquet(V19_DIR / "hurdle_oof_diagnostics.parquet")
    source_oof = {column.removeprefix("pred__"): oof_frame[column].to_numpy(dtype=float)
                  for column in oof_frame if column.startswith("pred__")}
    source_test = {column.removeprefix("pred__"): test_frame[column].to_numpy(dtype=float)
                   for column in test_frame if column.startswith("pred__")}

    # E0 必须逐值等于 v1.8 C1。
    for horizon in HORIZONS:
        config = {c: candidate_key("base", c, "huber", horizon) for c in COMPONENTS}
        pred, _ = config_prediction(horizon, config, source_oof)
        saved = v18_oof[f"pred_C1_component_osi_{horizon}"].to_numpy(dtype=float)
        valid = np.isfinite(saved)
        if np.max(np.abs(pred[valid] - saved[valid])) > 1e-12:
            raise AssertionError(f"E0 与 v1.8 C1 不一致: {horizon}")

    component_rows = []
    for feature_set in ("base", "profile"):
        for component in COMPONENTS:
            for candidate in REGRESSION_CANDIDATES[component]:
                if feature_set == "base" and candidate == "huber":
                    pass
                for horizon in HORIZONS:
                    key = candidate_key(feature_set, component, candidate, horizon)
                    if key not in source_oof:
                        continue
                    truth = targets[component_target_name(component, horizon)].to_numpy(dtype=float)
                    component_rows.append({
                        "feature_set": feature_set,
                        "component": component,
                        "candidate": candidate,
                        "horizon": horizon,
                        **extended_metrics(truth, source_oof[key]),
                    })
    component_metrics = pd.DataFrame(component_rows)
    component_metrics.to_csv(V19_DIR / "component_metrics.csv", index=False)

    # Hurdle 概率和正值幅度诊断。
    hurdle_rows = []
    eps = 1e-12
    for feature_set in ("base", "profile"):
        for component in ("N_t", "R_t"):
            for horizon in HORIZONS:
                key = candidate_key(feature_set, component, "hurdle", horizon)
                prob = hurdle[f"prob__{key}"].to_numpy(dtype=float)
                magnitude = hurdle[f"magnitude__{key}"].to_numpy(dtype=float)
                truth = targets[component_target_name(component, horizon)].to_numpy(dtype=float)
                valid = np.isfinite(truth) & np.isfinite(prob)
                labels = (truth[valid] > 0).astype(int)
                clipped_prob = np.clip(prob[valid], eps, 1 - eps)
                positive = valid & (truth > 0) & np.isfinite(magnitude)
                hurdle_rows.append({
                    "feature_set": feature_set,
                    "component": component,
                    "horizon": horizon,
                    "positive_rate": float(labels.mean()),
                    "average_precision": average_precision(labels, prob[valid]),
                    "binary_logloss": float(-np.mean(labels * np.log(clipped_prob) + (1-labels) * np.log(1-clipped_prob))),
                    "brier": float(np.mean((prob[valid] - labels) ** 2)),
                    "mean_probability": float(np.mean(prob[valid])),
                    "positive_magnitude_rmse": metrics(truth[positive], magnitude[positive])["rmse"],
                    "positive_magnitude_mae": metrics(truth[positive], magnitude[positive])["mae"],
                })
    pd.DataFrame(hurdle_rows).to_csv(V19_DIR / "hurdle_metrics.csv", index=False)

    official = {h: data.y_train[h].to_numpy(dtype=float) for h in HORIZONS}
    baseline_config = {
        h: {c: candidate_key("base", c, "huber", h) for c in COMPONENTS}
        for h in HORIZONS
    }

    # 每个候选只替换一个分量，直接评估最终 OSI。
    replacement_rows = []
    for horizon in HORIZONS:
        base = baseline_config[horizon]
        base_pred, _ = config_prediction(horizon, base, source_oof)
        for feature_set in ("base", "profile"):
            for component in COMPONENTS:
                for candidate in REGRESSION_CANDIDATES[component]:
                    key = candidate_key(feature_set, component, candidate, horizon)
                    if key not in source_oof:
                        continue
                    config = dict(base)
                    config[component] = key
                    pred, _ = config_prediction(horizon, config, source_oof)
                    score = extended_metrics(official[horizon], pred)
                    base_score = metrics(official[horizon], base_pred)
                    replacement_rows.append({
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "component": component,
                        "candidate": candidate,
                        "key": key,
                        **score,
                        "rmse_change_vs_E0_pct": 100 * (score["rmse"] / base_score["rmse"] - 1),
                    })
    replacements = pd.DataFrame(replacement_rows)
    replacements.to_csv(V19_DIR / "single_component_replacement_metrics.csv", index=False)

    # E1/E3：每个分量先从专用候选中选一个，再枚举是否替换的16种组合。
    specialized = {
        "P_t": ("l2", "weighted_l2"),
        "D_t": ("l2", "weighted_l2"),
        "N_t": ("tweedie", "hurdle"),
        "R_t": ("tweedie", "hurdle"),
    }
    selected_new: dict[str, dict[str, dict[str, str]]] = {fs: {} for fs in ("base", "profile")}
    combo_rows = []
    combo_configs: dict[str, dict[str, dict[str, str]]] = {}
    for feature_set in ("base", "profile"):
        for horizon in HORIZONS:
            selected_new[feature_set][horizon] = {}
            for component in COMPONENTS:
                subset = replacements[
                    (replacements["feature_set"] == feature_set)
                    & (replacements["horizon"] == horizon)
                    & (replacements["component"] == component)
                    & (replacements["candidate"].isin(specialized[component]))
                ].sort_values(["rmse", "candidate"])
                selected_new[feature_set][horizon][component] = str(subset.iloc[0]["key"])

            family = "E1" if feature_set == "base" else "E3"
            for bits in itertools.product((0, 1), repeat=4):
                config = dict(baseline_config[horizon])
                used = []
                for component, bit in zip(COMPONENTS, bits):
                    if bit:
                        config[component] = selected_new[feature_set][horizon][component]
                        used.append(component)
                name = f"{family}_subset_{''.join(map(str,bits))}"
                row = evaluate_config(name, horizon, config, source_oof, official[horizon])
                row.update({"family": family, "mask": "".join(map(str, bits)), "components_replaced": "+".join(used) or "none"})
                combo_rows.append(row)
                combo_configs.setdefault(name, {})[horizon] = config

    # E2 为全部画像 Huber。
    for horizon in HORIZONS:
        config = {c: candidate_key("profile", c, "huber", horizon) for c in COMPONENTS}
        row = evaluate_config("E2_profile_huber_all", horizon, config, source_oof, official[horizon])
        row.update({"family": "E2", "mask": "1111", "components_replaced": "P_t+N_t+D_t+R_t"})
        combo_rows.append(row)
        combo_configs.setdefault("E2_profile_huber_all", {})[horizon] = config

    combinations = pd.DataFrame(combo_rows)
    combinations.to_csv(V19_DIR / "combination_metrics.csv", index=False)

    # 每个 family/horizon 的最佳子集，及 E0/E2/当前 v1.8 对照。
    chosen: dict[str, dict[str, str]] = {"E1": {}, "E3": {}}
    named_predictions: dict[str, dict[str, np.ndarray]] = {
        "V18_C0": {}, "V18_C1_E0": {}, "V18_C3": {},
        "E2_profile_huber_all": {}, "E1_best_subset": {}, "E3_best_subset": {},
    }
    named_test_predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in named_predictions}
    named_component_configs: dict[str, dict[str, dict[str, str]]] = {
        "V18_C1_E0": {}, "E2_profile_huber_all": {}, "E1_best_subset": {}, "E3_best_subset": {},
    }
    summary_rows = []

    for horizon in HORIZONS:
        named_predictions["V18_C0"][horizon] = v18_oof[f"pred_C0_direct_osi_{horizon}"].to_numpy(dtype=float)
        named_predictions["V18_C1_E0"][horizon] = v18_oof[f"pred_C1_component_osi_{horizon}"].to_numpy(dtype=float)
        named_predictions["V18_C3"][horizon] = v18_oof[f"pred_C3_aligned_component_{horizon}"].to_numpy(dtype=float)
        named_test_predictions["V18_C0"][horizon] = v18_test[f"pred_C0_direct_osi_{horizon}"].to_numpy(dtype=float)
        named_test_predictions["V18_C1_E0"][horizon] = v18_test[f"pred_C1_component_osi_{horizon}"].to_numpy(dtype=float)
        named_test_predictions["V18_C3"][horizon] = v18_test[f"pred_C3_aligned_component_{horizon}"].to_numpy(dtype=float)
        named_component_configs["V18_C1_E0"][horizon] = baseline_config[horizon]

        e2_config = combo_configs["E2_profile_huber_all"][horizon]
        named_predictions["E2_profile_huber_all"][horizon], _ = config_prediction(horizon, e2_config, source_oof)
        named_test_predictions["E2_profile_huber_all"][horizon], _ = config_prediction(horizon, e2_config, source_test)
        named_component_configs["E2_profile_huber_all"][horizon] = e2_config

        for family in ("E1", "E3"):
            best = combinations[
                (combinations["family"] == family) & (combinations["horizon"] == horizon)
            ].sort_values(["rmse", "mask"]).iloc[0]
            chosen[family][horizon] = str(best["model"])
            config = combo_configs[str(best["model"])][horizon]
            out_name = f"{family}_best_subset"
            named_predictions[out_name][horizon], _ = config_prediction(horizon, config, source_oof)
            named_test_predictions[out_name][horizon], _ = config_prediction(horizon, config, source_test)
            named_component_configs[out_name][horizon] = config

    # 对 E1/E3 最佳原生分量分别做同目标小时对齐。
    for family in ("E1", "E3"):
        native_name = f"{family}_best_subset"
        aligned_name = f"{family}_aligned"
        named_predictions[aligned_name] = {}
        named_test_predictions[aligned_name] = {}
        aligned_train_components = {}
        aligned_test_components = {}
        for component in COMPONENTS:
            by_horizon_train = {
                h: source_oof[named_component_configs[native_name][h][component]] for h in HORIZONS
            }
            by_horizon_test = {
                h: source_test[named_component_configs[native_name][h][component]] for h in HORIZONS
            }
            aligned_train_components[component], _ = align_same_target(data.meta_train, by_horizon_train)
            aligned_test_components[component], _ = align_same_target(data.meta_test, by_horizon_test)
        for horizon in HORIZONS:
            named_predictions[aligned_name][horizon] = post_process(compose_osi({
                c: clip_component(aligned_train_components[c][horizon]) for c in COMPONENTS
            }))
            named_test_predictions[aligned_name][horizon] = post_process(compose_osi({
                c: clip_component(aligned_test_components[c][horizon]) for c in COMPONENTS
            }))

    # 当前稳健规则：短期用 v1.8 C3，长期用 v1.8 C1。
    named_predictions["V18_current_rule"] = {}
    named_test_predictions["V18_current_rule"] = {}
    for horizon in HORIZONS:
        source_name = "V18_C3" if HORIZON_HOURS[horizon] in (1, 6) else "V18_C1_E0"
        named_predictions["V18_current_rule"][horizon] = named_predictions[source_name][horizon]
        named_test_predictions["V18_current_rule"][horizon] = named_test_predictions[source_name][horizon]

    # v1.9 的逐 horizon 最佳只在预定义 E1/E2/E3 原生/对齐模型中选择；记录同 OOF 选择性质。
    v19_pool = ["E1_best_subset", "E2_profile_huber_all", "E3_best_subset", "E1_aligned", "E3_aligned"]
    named_predictions["V19_best_by_horizon"] = {}
    named_test_predictions["V19_best_by_horizon"] = {}
    v19_choice = {}
    for horizon in HORIZONS:
        scored = [(metrics(official[horizon], named_predictions[name][horizon])["rmse"], name) for name in v19_pool]
        _, best_name = min(scored)
        v19_choice[horizon] = best_name
        named_predictions["V19_best_by_horizon"][horizon] = named_predictions[best_name][horizon]
        named_test_predictions["V19_best_by_horizon"][horizon] = named_test_predictions[best_name][horizon]

    for model, by_horizon in named_predictions.items():
        for horizon, pred in by_horizon.items():
            summary_rows.append({"model": model, "horizon": horizon, **extended_metrics(official[horizon], pred)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(V19_DIR / "summary_metrics.csv", index=False)

    # 逐折。
    fold_rows = []
    for model, by_horizon in named_predictions.items():
        for horizon, pred in by_horizon.items():
            truth = official[horizon]
            for fold in range(5):
                hit = data.row_folds == fold
                fold_rows.append({"model": model, "horizon": horizon, "fold": fold, **extended_metrics(truth[hit], pred[hit])})
    pd.DataFrame(fold_rows).to_csv(V19_DIR / "fold_metrics.csv", index=False)

    # 县级指标与误差贡献。
    county_rows = []
    for model, by_horizon in named_predictions.items():
        for horizon, pred in by_horizon.items():
            truth = official[horizon]
            total_sse = np.nansum((pred - truth) ** 2)
            for fips, idx in data.meta_train.groupby("fipsCode", sort=True).groups.items():
                idx = np.asarray(list(idx), dtype=int)
                score = extended_metrics(truth[idx], pred[idx])
                county_sse = float(np.nansum((pred[idx] - truth[idx]) ** 2))
                county_rows.append({
                    "model": model, "horizon": horizon, "fipsCode": fips,
                    **score,
                    "sse_contribution_pct": float(100 * county_sse / total_sse) if total_sse > 0 else np.nan,
                })
    pd.DataFrame(county_rows).to_csv(V19_DIR / "county_metrics.csv", index=False)

    # 真实强度与目标日期诊断。
    severity_rows = []
    day_rows = []
    severity_bins = [(-np.inf, 0.001, "lt_0.001"), (0.001, 0.01, "0.001_0.01"),
                     (0.01, 0.05, "0.01_0.05"), (0.05, np.inf, "ge_0.05")]
    for model, by_horizon in named_predictions.items():
        for horizon, pred in by_horizon.items():
            truth = official[horizon]
            for lower, upper, label in severity_bins:
                hit = np.isfinite(truth) & (truth >= lower) & (truth < upper)
                if hit.any():
                    severity_rows.append({
                        "model": model, "horizon": horizon, "severity": label,
                        **extended_metrics(truth[hit], pred[hit]),
                    })
            target_day = (
                data.meta_train["timestamp_et"]
                + pd.to_timedelta(HORIZON_HOURS[horizon], unit="h")
            ).dt.strftime("%Y-%m-%d")
            for day in sorted(target_day[np.isfinite(truth)].unique()):
                hit = np.isfinite(truth) & target_day.eq(day).to_numpy()
                day_rows.append({
                    "model": model, "horizon": horizon, "target_day": day,
                    **extended_metrics(truth[hit], pred[hit]),
                })
    pd.DataFrame(severity_rows).to_csv(V19_DIR / "severity_metrics.csv", index=False)
    pd.DataFrame(day_rows).to_csv(V19_DIR / "target_day_metrics.csv", index=False)

    # 原生分量配置的加权误差贡献及其与最终 OSI 误差的相关性。
    contribution_rows = []
    for model in ("V18_C1_E0", "E1_best_subset", "E2_profile_huber_all", "E3_best_subset"):
        for horizon in HORIZONS:
            config = named_component_configs[model][horizon]
            final_error = named_predictions[model][horizon] - official[horizon]
            for component in COMPONENTS:
                truth_component = targets[component_target_name(component, horizon)].to_numpy(dtype=float)
                pred_component = source_oof[config[component]]
                valid = np.isfinite(truth_component) & np.isfinite(pred_component) & np.isfinite(final_error)
                raw_error = pred_component[valid] - truth_component[valid]
                weighted_error = COMPONENT_WEIGHTS[component] * raw_error
                contribution_rows.append({
                    "model": model, "horizon": horizon, "component": component,
                    "component_key": config[component],
                    "raw_rmse": float(np.sqrt(np.mean(raw_error ** 2))),
                    "weighted_error_rmse": float(np.sqrt(np.mean(weighted_error ** 2))),
                    "weighted_error_correlation_with_osi_error": float(
                        np.corrcoef(weighted_error, final_error[valid])[0, 1]
                    ),
                    "n": int(valid.sum()),
                })
    pd.DataFrame(contribution_rows).to_csv(V19_DIR / "component_contribution.csv", index=False)

    # 与 E0 及当前 v1.8 规则做成对县级 bootstrap。
    bootstrap_rows = []
    compare_models = [
        "E1_best_subset", "E2_profile_huber_all", "E3_best_subset",
        "E1_aligned", "E3_aligned", "V19_best_by_horizon",
    ]
    for horizon in HORIZONS:
        for candidate_name in compare_models:
            for reference_name in ("V18_C1_E0", "V18_current_rule"):
                result = paired_bootstrap(
                    data.meta_train, official[horizon], named_predictions[candidate_name][horizon],
                    named_predictions[reference_name][horizon], BOOTSTRAP_REPLICATES,
                    BOOTSTRAP_SEED + HORIZON_HOURS[horizon],
                )
                bootstrap_rows.append({
                    "candidate": candidate_name, "reference": reference_name,
                    "horizon": horizon, **result,
                })
    pd.DataFrame(bootstrap_rows).to_csv(V19_DIR / "paired_county_bootstrap.csv", index=False)

    # 新画像特征的最终模型 gain 占比。
    importance = pd.read_csv(V19_DIR / "feature_importance.csv")
    importance["is_response_profile"] = importance["feature"].isin(RESPONSE_FEATURES)
    profile_gain = importance.groupby(
        ["feature_set", "component", "candidate", "horizon", "stage"], as_index=False
    ).agg(profile_gain_fraction=("gain_fraction", lambda x: float(x[importance.loc[x.index, "is_response_profile"]].sum())))
    profile_gain.to_csv(V19_DIR / "profile_gain_summary.csv", index=False)

    # 保存选择配置、全套命名 OOF/测试预测和候选提交。
    prediction_frame = data.meta_train[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    test_prediction_frame = data.meta_test[["fipsCode", "timestamp_et", "hour_idx", "stateAbbr"]].copy()
    for model, by_horizon in named_predictions.items():
        for horizon, pred in by_horizon.items():
            prediction_frame[f"pred__{model}__{horizon}"] = pred
            test_prediction_frame[f"pred__{model}__{horizon}"] = named_test_predictions[model][horizon]
    for horizon in HORIZONS:
        prediction_frame[f"actual__{horizon}"] = official[horizon]
    prediction_frame.to_parquet(V19_DIR / "oof_predictions.parquet", index=False)
    test_prediction_frame.to_parquet(V19_DIR / "test_predictions.parquet", index=False)

    submission_models = [
        "E1_best_subset", "E2_profile_huber_all", "E3_best_subset",
        "E1_aligned", "E3_aligned", "V19_best_by_horizon",
    ]
    for model in submission_models:
        submission = fill_submission(data.meta_test, named_test_predictions[model])
        submission.to_csv(V19_DIR / f"diagnostic_submission_v1.9_{model}_{'balanced_v1'}.csv", index=False)

    selection = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selected_specialized_candidate_by_single_replacement": selected_new,
        "best_subset_model_name": chosen,
        "v19_best_by_horizon_source": v19_choice,
        "selection_warning": "候选在同一五折 OOF 上筛选；若形成晋级收益，需严格嵌套确认。",
    }
    (V19_DIR / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.pivot(index="model", columns="horizon", values="rmse").to_string())
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
