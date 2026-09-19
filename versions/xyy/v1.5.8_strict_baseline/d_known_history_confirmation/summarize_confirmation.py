"""Summarize fixed within-split comparisons without pooling repeated county OOF."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PRIMARY = "v18_rule"
H1 = "osi_target_t01h"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(path):
    return pd.read_csv(path, dtype={"fipsCode":str}, float_precision="round_trip")


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"] + ["| "+" | ".join(map(str, row))+" |" for row in rows])


def main():
    plan = json.loads((HERE/"confirmation_plan.json").read_text())
    entries = [{"split_seed":42, "baseline_dir":plan["anchor_run"], "projection_dir":plan["anchor_projection"]}, *plan["splits"]]
    names = ["metrics_comparison", "fold_metrics", "county_metrics", "paired_county_bootstrap", "D_component_metrics", "window_metrics", "primary_county_influence", "changed_rows"]
    collected = {name:[] for name in names}
    inputs = {}
    identity_rows, diagnostics, focus_rows = [], [], []
    for entry in entries:
        seed, run, projection = entry["split_seed"], Path(entry["baseline_dir"]), Path(entry["projection_dir"])
        baseline = json.loads((run/"run_manifest.json").read_text())
        marker = json.loads((run/"CV_COMPLETE").read_text())
        assert marker["identity_hash"] == baseline["identity_hash"]
        assert baseline["identity"]["settings"] == plan["settings"]
        assert baseline["identity"]["code_sha256"] == plan["baseline_code_sha256"]
        for name, digest in marker["files"].items(): assert sha(run/name) == digest
        manifest = json.loads((projection/"manifest.json").read_text())
        verification = json.loads((projection/"verification.json").read_text())
        assert verification["status"] == "PASS"
        assert verification["manifest_sha256"] == sha(projection/"manifest.json")
        assert manifest["baseline_identity_hash"] == baseline["identity_hash"]
        assert manifest["input_sha256"]["projector"] == plan["fixed_projection_source_sha256"]
        for name, digest in manifest["output_sha256"].items(): assert sha(projection/name) == digest
        if seed != 42:
            completion = json.loads((projection.parent/"completion.json").read_text())
            assert completion["status"] == "COMPLETE"
            for name, digest in completion["files"].items(): assert sha(projection.parent/name) == digest
        identity_rows.append({"split_seed":seed, "role":"development" if seed==42 else "confirmation",
                              "baseline_dir":str(run), "projection_dir":str(projection),
                              "baseline_identity_hash":baseline["identity_hash"], "candidate_identity_hash":manifest["candidate_identity_hash"],
                              "cv_sha256":baseline["identity"]["inputs"]["cv"]})
        inputs[str(run/"CV_COMPLETE")] = sha(run/"CV_COMPLETE")
        inputs[str(projection/"manifest.json")] = sha(projection/"manifest.json")
        inputs[str(projection/"verification.json")] = sha(projection/"verification.json")
        local = {}
        for name in names:
            frame = csv(projection/f"{name}.csv")
            frame.insert(0, "split_seed", seed)
            local[name] = frame
            collected[name].append(frame)
            inputs[str(projection/f"{name}.csv")] = sha(projection/f"{name}.csv")
        county = local["primary_county_influence"]
        morrow = county.loc[county.fipsCode == "39117"].iloc[0]
        d = local["D_component_metrics"].query("horizon == @H1").iloc[0]
        main_metric = local["metrics_comparison"].query("horizon == @H1 and model == @PRIMARY").iloc[0]
        boot = local["paired_county_bootstrap"].query("horizon == @H1 and model == @PRIMARY and population == 'all_counties'").iloc[0]
        less = local["paired_county_bootstrap"].query("horizon == @H1 and model == @PRIMARY and population == 'exclude_Morrow_diagnostic_only'").iloc[0]
        changes = local["changed_rows"]
        diagnostics.append({"split_seed":seed, "D_projected_rows":manifest["exact_D_projection_rows"],
                            "D_projected_counties":manifest["projected_counties"], "D_rmse_delta":d.rmse_delta,
                            "primary_changed_rows":main_metric.changed_predictions,
                            "primary_changed_counties":int((county.changed_predictions>0).sum()),
                            "improved_counties":int((county.sse_reduction>0).sum()),
                            "worsened_counties":int((county.sse_reduction<0).sum()),
                            "unchanged_counties":int((county.sse_reduction==0).sum()),
                            "D_improved_rows":int((changes.D_sse_reduction>0).sum()),
                            "D_worsened_rows":int((changes.D_sse_reduction<0).sum()),
                            "primary_improved_rows":int((changes.sse_reduction_v18_rule>0).sum()),
                            "primary_worsened_rows":int((changes.sse_reduction_v18_rule<0).sum()),
                            "total_sse_reduction":main_metric.sse_reduction,
                            "morrow_sse_reduction":morrow.sse_reduction,
                            "morrow_fraction_of_net_sse_reduction":morrow.sse_reduction/main_metric.sse_reduction if main_metric.sse_reduction else np.nan,
                            "primary_rmse_delta":main_metric.rmse_delta, "primary_rmse_change_pct":main_metric.rmse_change_pct,
                            "primary_ci_low":boot.ci_low, "primary_ci_high":boot.ci_high,
                            "exclude_morrow_rmse_change_pct":less.rmse_change_pct,
                            "exclude_morrow_ci_low":less.ci_low, "exclude_morrow_ci_high":less.ci_high})
        oof = pd.read_parquet(projection/"candidate_oof_predictions.parquet")
        parts = pd.read_parquet(projection/"projected_component_predictions.parquet")
        source = pd.read_parquet(run/"oof_component_predictions.parquet")
        # Prespecified Morrow illustration, including early rows even if no projection fired.
        indices = np.flatnonzero((oof.fipsCode == "39117") & oof.hour_idx.between(72,75))
        for i in indices:
            focus_rows.append({"split_seed":seed, "fipsCode":"39117", "fold":int(oof.fold.iloc[i]),
                               "target_hour":int(oof.hour_idx.iloc[i])+1,
                               "known_D":parts.known_D_lower_bound_osi_target_t01h.iloc[i],
                               "baseline_D":parts.baseline_pred_D_t_target_t01h.iloc[i],
                               "projected_D":parts.projected_pred_D_t_target_t01h.iloc[i],
                               "actual_D":source.actual_D_t_target_t01h.iloc[i],
                               "baseline_P":parts.baseline_pred_P_t_target_t01h.iloc[i],
                               "actual_P":source.actual_P_t_target_t01h.iloc[i],
                               "baseline_osi":oof.baseline_v18_rule_osi_target_t01h.iloc[i],
                               "candidate_osi":oof.candidate_v18_rule_osi_target_t01h.iloc[i],
                               "actual_osi":oof.actual_osi_target_t01h.iloc[i]})
    all_tables = {name:pd.concat(frames, ignore_index=True) for name, frames in collected.items()}
    out = HERE/"summary"; out.mkdir(exist_ok=True)
    outputs = {}
    def save(name, frame):
        frame.to_csv(out/name, index=False)
        outputs[name] = sha(out/name)
    for name, frame in all_tables.items(): save(name+"_all_splits.csv", frame)
    primary = all_tables["metrics_comparison"].query("model == @PRIMARY")
    save("primary_metrics_all_splits.csv", primary)
    diagnostics = pd.DataFrame(diagnostics)
    save("diagnostic_summary.csv", diagnostics)
    save("run_index.csv", pd.DataFrame(identity_rows))
    save("morrow_first_four_targets.csv", pd.DataFrame(focus_rows))
    counties = all_tables["primary_county_influence"]
    stability = counties.pivot(index=["fipsCode","countyName"], columns="split_seed", values="sse_reduction")
    stability.columns = [f"sse_reduction_split{s}" for s in stability.columns]
    stability["confirmation_improved_splits"] = (stability[[f"sse_reduction_split{s}" for s in (20260917,20260918)]]>0).sum(axis=1)
    stability["confirmation_worsened_splits"] = (stability[[f"sse_reduction_split{s}" for s in (20260917,20260918)]]<0).sum(axis=1)
    save("county_effect_stability.csv", stability.reset_index())
    descriptive = []
    for h, group in primary.query("split_seed != 42").groupby("horizon", sort=True):
        descriptive.append({"horizon":h, "confirmation_splits":2,
                            "mean_within_split_rmse_delta":group.rmse_delta.mean(),
                            "min_within_split_rmse_delta":group.rmse_delta.min(),
                            "max_within_split_rmse_delta":group.rmse_delta.max(),
                            "mean_within_split_rmse_change_pct":group.rmse_change_pct.mean()})
    save("confirmation_descriptive_summary.csv", pd.DataFrame(descriptive))
    # All pooled counts and fixed invariants must match independently verified source OOF.
    assert set(primary.split_seed) == {42,20260917,20260918}
    assert primary.groupby("horizon").n.nunique().eq(1).all()
    assert (all_tables["metrics_comparison"].query("horizon != @H1").changed_predictions == 0).all()
    assert (all_tables["metrics_comparison"].query("model == 'C0_direct_osi'").changed_predictions == 0).all()
    assert (diagnostics.improved_counties+diagnostics.worsened_counties+diagnostics.unchanged_counties == 239).all()
    for seed in (42,20260917,20260918):
        close = np.testing.assert_allclose
        rows = counties.query("split_seed == @seed")
        target = primary.query("split_seed == @seed and horizon == @H1").iloc[0]
        close(rows.baseline_sse.sum(), target.baseline_sse, atol=1e-12, rtol=0)
        close(rows.candidate_sse.sum(), target.candidate_sse, atol=1e-12, rtol=0)
    for path, digest in inputs.items(): assert sha(path) == digest
    (out/"summary_manifest.json").write_text(json.dumps({"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"PASS", "source_sha256":sha(__file__), "input_sha256":inputs, "output_sha256":outputs,
        "split_roles":{"42":"development_reference","20260917":"confirmation","20260918":"confirmation"},
        "no_pooled_repeated_oof_score":True, "same_fixed_projection_source":True}, indent=2)+"\n")
    print(primary[["split_seed","horizon","baseline_rmse","candidate_rmse","rmse_change_pct"]].to_string(index=False))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
