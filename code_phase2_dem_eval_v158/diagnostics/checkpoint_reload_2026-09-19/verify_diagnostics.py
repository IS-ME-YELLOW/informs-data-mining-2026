"""Independently check saved replay, intervention arithmetic and source protection.

This reads the inference artifacts; it does not fit or re-run the models.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
PKG = ROOT.parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv(name):
    return pd.read_csv(TABLES / name, float_precision="round_trip")


def post(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 0.65)
    return np.where(value < 0.001, 0.0, value)


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def main():
    config = read("diagnostic_config.json")
    run = Path(config["source_run"])
    replay = read("replay_verification.json")
    assert replay["stage"] == "all" and replay["checkpoints_replayed"] == 64
    assert replay["base_models_predicted"] == 416
    assert replay["training_performed"] is False
    assert replay["formal_original_CUDA_acceptance_claimed"] is False

    metrics = csv("CPU_replay_metrics.csv")
    expected = {(f"osi_target_t{h:02d}h", ",".join(map(str, scope)))
                for size in (3, 4, 5)
                for scope in itertools.combinations(range(5), size)
                for h in (1, 6, 24, 48)}
    assert len(metrics) == 64
    assert set(zip(metrics.horizon, metrics.scope)) == expected
    assert metrics.groupby("evidence").size().to_dict() == {"inner": 40, "outer": 20, "test": 4}
    assert metrics.base_max_absolute_error.max() == 0
    assert metrics.correction_max_absolute_error.max() <= config["tolerances"]["CPU_vs_CUDA_raw_correction_absolute"]
    assert metrics.feature_mean_max_absolute_error.max() <= 1e-10
    assert metrics.feature_std_max_absolute_error.max() <= 1e-10
    assert metrics.residual_scale_error.max() <= 1e-10
    assert metrics.zero_threshold_flips.sum() == 0

    details = pd.read_parquet(TABLES / "CPU_replayed_outer_and_test.parquet")
    outer = details.loc[details.outer_fold.notna()].copy()
    test = details.loc[details.outer_fold.isna()].copy()
    assert len(outer) == 137664 and len(test) == 36288
    keys = ["fipsCode", "hour_idx", "horizon"]
    for frame, source in ((outer, "outer_oof.parquet"), (test, "test_predictions.parquet")):
        saved = pd.read_parquet(run / source)
        assert not frame.duplicated(keys).any()
        pd.testing.assert_frame_equal(
            frame.set_index(keys)[list(c for c in saved if c not in keys)].sort_index(),
            saved.set_index(keys).sort_index(), check_dtype=False, check_exact=True,
        )
        np.testing.assert_array_equal(frame.CPU_base, frame.base_prediction)
        np.testing.assert_allclose(frame.CPU_correction, frame.correction_osi, rtol=0, atol=1e-5)
        valid = frame.is_scoreable.to_numpy(dtype=bool)
        assert frame.CPU_prediction.isna().equals(~frame.is_scoreable)
        np.testing.assert_array_equal(
            frame.CPU_prediction.to_numpy()[valid],
            post(frame.CPU_base + frame.alpha * frame.CPU_correction)[valid],
        )

    outer_rows = []
    for horizon, group in outer.loc[outer.is_scoreable].groupby("horizon"):
        base = rmse(group.y_true, group.CPU_base)
        saved = rmse(group.y_true, group.prediction)
        cpu = rmse(group.y_true, group.CPU_prediction)
        outer_rows.append(dict(horizon=horizon, scoreable_rows=len(group),
                               base_RMSE=base, saved_CUDA_GAT_RMSE=saved,
                               reloaded_CPU_GAT_RMSE=cpu, CPU_minus_saved_RMSE=cpu-saved,
                               reloaded_GAT_relative_change_percent=(cpu/base-1)*100))
    pd.DataFrame(outer_rows).to_csv(TABLES / "outer_horizon_replay.csv", index=False)

    submission = pd.read_csv(run / "submission_phase2_dem_gat.csv", dtype={"fipsCode": str}, float_precision="round_trip")
    submission_keys = ["fipsCode", "timestamp_et"]
    assert len(submission) == 9072 and not submission.duplicated(submission_keys).any()
    test_rows = []
    for horizon, group in test.groupby("horizon"):
        joined = group.merge(submission[submission_keys + [horizon]], on=submission_keys,
                             validate="one_to_one", how="outer", indicator=True)
        assert joined._merge.eq("both").all()
        assert joined[horizon].isna().equals(~joined.is_scoreable)
        np.testing.assert_array_equal(joined.loc[joined.is_scoreable, horizon],
                                      joined.loc[joined.is_scoreable, "prediction"])
        valid = joined.loc[joined.is_scoreable]
        delta = float(np.max(np.abs(valid.CPU_prediction - valid[horizon])))
        assert delta <= 1e-5
        test_rows.append(dict(horizon=horizon, scoreable_rows=len(valid), alpha=float(valid.alpha.iloc[0]),
                              CPU_submission_max_absolute_error=delta,
                              CPU_different_from_base_rows=int((valid.CPU_prediction != valid.CPU_base).sum())))
    assert test_rows[0]["horizon"] == "osi_target_t01h"
    assert test_rows[0]["alpha"] == 0 and test_rows[0]["CPU_different_from_base_rows"] == 0
    pd.DataFrame(test_rows).to_csv(TABLES / "test_submission_replay.csv", index=False)

    interventions = csv("Newton_interventions.csv")
    timeline = csv("Newton_intervention_timeline.csv")
    decomposition = csv("Newton_path_decomposition.csv")
    valid = timeline.is_scoreable.to_numpy(dtype=bool)
    assert len(timeline) == 144 and valid.sum() == 143
    for row in interventions.itertuples():
        correction = timeline[row.intervention + "_correction"]
        prediction = timeline[row.intervention + "_prediction"]
        assert prediction.isna().equals(~timeline.is_scoreable)
        np.testing.assert_allclose(prediction[valid], post(timeline.base_prediction + timeline.alpha*correction)[valid], rtol=0, atol=1e-15)
        np.testing.assert_allclose(correction[valid].mean(), row.Newton_mean_correction, rtol=0, atol=1e-14)
        np.testing.assert_allclose(prediction[valid].mean(), row.Newton_mean_prediction, rtol=0, atol=1e-14)
        np.testing.assert_allclose(rmse(timeline.y_true[valid], prediction[valid]), row.Newton_1h_RMSE_diagnostic_only, rtol=0, atol=1e-14)
    full = timeline.original_correction
    message = timeline.zero_skip_path_correction
    skip = timeline.zero_message_path_correction
    zero = timeline.head_at_zero_hidden_correction
    expected_skip = ((skip-zero)+(full-message))/2
    expected_message = ((message-zero)+(full-skip))/2
    for column, expected_values in [("skip_path_two_player_contribution", expected_skip),
                                    ("message_path_two_player_contribution", expected_message)]:
        np.testing.assert_allclose(decomposition[column], expected_values, rtol=0, atol=1e-12)
    removed = timeline.Newton_pre_event_to_train_mean_both_paths_correction
    only_message = timeline.Newton_pre_event_removed_only_from_skip_correction
    only_skip = timeline.Newton_pre_event_removed_only_from_message_correction
    np.testing.assert_allclose(decomposition.pre_event_via_skip_two_player_contribution,
                               ((only_skip-removed)+(full-only_message))/2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(decomposition.pre_event_via_message_two_player_contribution,
                               ((only_message-removed)+(full-only_skip))/2, rtol=0, atol=1e-12)
    source_newton = outer.loc[(outer.fipsCode == "18111") & (outer.horizon == "osi_target_t01h")].sort_values("hour_idx")
    np.testing.assert_array_equal(source_newton.CPU_correction, full)
    newton = source_newton.loc[source_newton.is_scoreable]
    newton_sse_change = float(np.sum((newton.CPU_prediction-newton.y_true)**2-(newton.CPU_base-newton.y_true)**2))
    one_hour = outer.loc[outer.is_scoreable & outer.horizon.eq("osi_target_t01h")]
    total_sse_change = float(np.sum((one_hour.CPU_prediction-one_hour.y_true)**2-(one_hour.CPU_base-one_hour.y_true)**2))

    before = read("environment_before_install.json")["packages"]
    after = read("environment_after_install.json")["packages"]
    assert all(after[name] == version for name, version in before.items())
    assert after["torch"] == "2.7.1+cpu"

    protected = read("source_snapshot.json")["sha256"]
    missing = [name for name in protected if not Path(name).is_file()]
    changed = [name for name, digest in protected.items() if Path(name).is_file() and sha(name) != digest]
    assert not missing and not changed, (missing, changed)
    current_paths = {str(p) for p in PKG.rglob("*") if p.is_file()
                     and not p.is_relative_to(ROOT.parent) and "__pycache__" not in p.parts}
    assert current_paths == set(protected), sorted(current_paths.symmetric_difference(protected))
    old_manifest = json.loads((ROOT.parent / "diagnostic_manifest.json").read_text())
    old_changed = [name for name, digest in old_manifest["artifacts"].items()
                   if sha(ROOT.parent / name) != digest]
    assert not old_changed, old_changed
    assert not (run / "COMPLETE").exists()

    report = {
        "status": "PASS", "kind": "inference_replay_and_fixed_weight_sensitivity",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_identity": config["source_identity"],
        "replay_contexts": 64, "base_models_used_by_replay": 416,
        "compared_rows_by_context": metrics.groupby("evidence").compared_rows.sum().astype(int).to_dict(),
        "max_saved_training_loss_vs_CPU_difference": float(metrics.training_mask_MSE_difference.abs().max()),
        "Newton_reloaded_SSE_increase": newton_sse_change,
        "total_1h_reloaded_SSE_increase": total_sse_change,
        "Newton_fraction_of_net_SSE_increase": newton_sse_change/total_sse_change,
        "intervention_variants_verified": len(interventions),
        "two_player_decompositions_verified": True,
        "protected_existing_files_unchanged": len(protected),
        "prior_diagnostic_artifacts_unchanged": len(old_manifest["artifacts"]),
        "preexisting_environment_packages_unchanged": len(before),
        "training_performed": False, "original_COMPLETE_written": False,
        "formal_original_CUDA_acceptance_claimed": False,
    }
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
