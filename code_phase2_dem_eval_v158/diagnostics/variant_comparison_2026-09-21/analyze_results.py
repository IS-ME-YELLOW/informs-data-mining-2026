"""Saved-artifact analysis only: no model loading, fitting or source modification."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
EXPERIMENT = REPO / "code_phase2_experiment"
CODE = REPO / "code_phase2_compare"
LIMIT = REPO / "code_phase2_dem_eval_v158_limit"
TABLES = OUT / "tables"
HOURS = {f"osi_target_t{h:02d}h": h for h in (1, 6, 24, 48)}
GRID = (0., .05, .1, .2, .35, .5, .75, 1.)
COMP = {"P_t": .4, "N_t": .35, "D_t": .25, "R_t": -.1}
KEY = ["fipsCode", "hour_idx"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def table(name, value):
    pd.DataFrame(value).to_csv(TABLES / name, index=False)


def post(value):
    value = np.clip(np.asarray(value, dtype=float), 0, .65)
    return np.where(value < .001, 0., value)


def near(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol, equal_nan=True)


def source_match(path, digest):
    if not path.exists():
        return "missing"
    b = path.read_bytes()
    for label, data in [("exact", b), ("LF", b.replace(b"\r\n", b"\n")),
                        ("CRLF", b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))]:
        if hashlib.sha256(data).hexdigest() == digest:
            return label
    return "mismatch"


def model_id(kind, variant, h, scope, component="P_t"):
    middle = component if kind == "base" else f"{variant}:component_v158"
    return f"{kind}:{middle}:{h}:S{','.join(map(str,scope))}"


def choose(group, candidate, variant, horizon, scope):
    candidate = candidate.sort_values("alpha")
    assert candidate.alpha.tolist() == list(GRID)
    assert candidate.selection_id.eq(model_id("alpha", variant, horizon, scope)).all()
    assert candidate.allowed_folds.map(lambda s: tuple(json.loads(s)) == scope).all()
    results = []
    for a in GRID:
        err = post(group.base_prediction + a*group.correction_osi) - group.y_true
        results.append((len(group), float(np.sum(err**2)), float(np.sqrt(np.mean(err**2))), float(np.mean(abs(err)))))
    for j, name in enumerate(("n", "sse", "rmse", "mae")):
        near(candidate[name], [r[j] for r in results])
    best = min(range(len(GRID)), key=lambda k: (results[k][2], GRID[k]))
    assert candidate.selected.tolist() == [k == best for k in range(len(GRID))]
    return GRID[best]


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    # Every write is below OUT. Original model bytes are only hashed, never deserialized.
    originals = [p for root in (EXPERIMENT, CODE, LIMIT) for p in root.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]
    before = {str(p): sha(p) for p in originals}
    dump("source_snapshot.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": before})
    roots = sorted(p.parent for p in EXPERIMENT.rglob("run_manifest.json"))
    assert len(roots) == 4
    cache = REPO / "versions/xyy/v1.5.6"
    meta = pd.read_parquet(cache / "meta_train_v1.5.6.parquet")
    meta.fipsCode = meta.fipsCode.astype(str).str.zfill(5)
    target = pd.read_parquet(cache / "targets_train_v1.5.6.parquet")
    truth = pd.concat([meta[KEY], target], axis=1).set_index(KEY)
    names = pd.read_csv(OUT.parent / "tables/county_effects.csv", dtype={"fipsCode": str})
    names = names[["fipsCode", "countyName"]].drop_duplicates().set_index("fipsCode").countyName.to_dict()
    metrics, counties, folds, tests, bootstraps, audit, source_rows, verifier_checks = [], [], [], [], [], [], [], []
    reference, reference_inputs, base_hashes = None, None, None
    sys.path.insert(0, str(CODE))
    from artifact_checks import check_alpha_group
    from base_model import scoped_seed

    for root in roots:
        manifest = read(root / "run_manifest.json")
        v = manifest["variant"]
        print("CHECK", v, flush=True)
        inputs = manifest["inputs"]
        if reference_inputs is None:
            reference_inputs = inputs
        else:
            assert inputs == reference_inputs
        for name, entry in inputs["package_manifest"]["files"].items():
            assert sha(cache / name) == entry["sha256"].lower()
        for name, digest in manifest["identity"]["definition"]["sources"].items():
            match = source_match(CODE/name, digest)
            assert match in ("exact", "LF", "CRLF")
            source_rows.append(dict(variant=v, source=name, compare_match=match, limit_match=source_match(LIMIT/name, digest)))
        cv = read(root / "cv_manifest.json")
        assert cv["variant"] == v
        cv_digest = hashlib.sha256(json.dumps(cv["artifact_hashes"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert cv_digest == cv["cv_hash"] == (root/"CV_COMPLETE").read_text().strip()
        assert sha(root/"run_manifest.json") == cv["run_manifest_sha256"]
        assert all(sha(root/name) == digest for name, digest in cv["artifact_hashes"].items())

        fold_table = pd.read_csv(root/"folds.csv", dtype={"fipsCode": str}, float_precision="round_trip").sort_values("row_id").reset_index(drop=True)
        assert fold_table.row_id.tolist() == list(range(len(meta)))
        assert fold_table.fipsCode.tolist() == meta.fipsCode.tolist()
        assert fold_table.groupby("fipsCode").fold.nunique().eq(1).all()
        fold_of = fold_table.groupby("fipsCode").fold.first()
        base = pd.read_parquet(root/"base_fit_manifest_final.parquet")
        expected_base = {model_id("base", v, h, s, c) for size in range(2,6)
                         for s in itertools.combinations(range(5), size) for h in HOURS for c in COMP}
        assert len(base) == 416 and set(base.model_id) == expected_base
        cur_hashes = base.set_index("model_id").model_sha256.to_dict()
        if base_hashes is None:
            base_hashes = cur_hashes
        else:
            assert cur_hashes == base_hashes
        path_mismatches = 0
        for row in base.to_dict("records"):
            s = tuple(json.loads(row["allowed_folds"]))
            h, c = row["horizon"], row["component"]
            path = root/row["model_path"].replace("\\", "/")
            receipt = read(str(path)+".complete.json")
            assert sha(path) == receipt["sha256"] == row["model_sha256"]
            assert row["run_identity"] == receipt["run_identity"] == manifest["identity"]["digest"]
            assert receipt["model_id"] == row["model_id"]
            for field, value in receipt["details"].items():
                if field == "model_path":
                    path_mismatches += int(row[field] != value)
                    assert row[field].replace("\\", "/") == value.replace("\\", "/")
                else:
                    assert row.get(field) == value, (v, field)
            its, probes = json.loads(row["best_iterations"]), json.loads(row["probe_records"])
            assert len(its) == len(probes) == len(s)
            assert row["final_rounds"] == max(1, int(np.mean(its)))
            assert 1 <= row["actual_rounds"] <= row["final_rounds"]
            valid_time = meta.hour_idx.to_numpy()+HOURS[h] < 216
            rowfold = fold_table.fold.to_numpy()
            assert row["valid_rows"] == int((valid_time & np.isin(rowfold, s)).sum())
            for q, n, probe in zip(s, its, probes):
                assert probe["probe_fold"] == q and probe["valid_folds"] == [q]
                assert probe["train_folds"] == [k for k in s if k != q] and probe["best_iteration"] == n
                assert probe["train_rows"] == int((valid_time & np.isin(rowfold, [k for k in s if k != q])).sum())
                assert probe["valid_rows"] == int((valid_time & (rowfold == q)).sum())
                assert probe["seed"] == scoped_seed(42, "base_probe", s, h, c, q)
            assert row["seed"] == scoped_seed(42, "base_refit", s, h, c, "refit")
        gat_receipts = list((root/"models/gat").glob("*.complete.json"))
        assert len(gat_receipts) == 64
        for rec in gat_receipts:
            receipt = read(rec)
            assert receipt["run_identity"] == manifest["identity"]["digest"]
            assert receipt["details"]["variant"] == v
            assert sha(Path(str(rec).removesuffix(".complete.json"))) == receipt["sha256"]
            s = tuple(receipt["details"]["scope"]); h = receipt["details"]["horizon"]
            deps = {model_id("base", v, h, t, c): cur_hashes[model_id("base", v, h, t, c)]
                    for t in [s]+[tuple(k for k in s if k!=q) for q in s] for c in COMP}
            assert receipt["details"]["base_dependencies"] == deps

        outer = pd.read_parquet(root/"outer_oof.parquet")
        inner = pd.read_parquet(root/"inner_oof.parquet")
        test = pd.read_parquet(root/"test_predictions.parquet")
        alpha = pd.read_parquet(root/"alpha_selection.parquet")
        final = pd.read_parquet(root/"alpha_selection_final.parquet")
        assert len(outer) == 137664 and len(inner) == 475132 and len(test) == 36288
        assert len(alpha) == 160 and len(final) == 36
        assert not outer.duplicated(KEY+["horizon"]).any()
        assert not inner.duplicated(KEY+["horizon", "outer_fold"]).any()
        assert not test.duplicated(KEY+["horizon"]).any()
        common = [*KEY, "horizon", "outer_fold", "y_true", "is_scoreable", "base_prediction"]
        if reference is None:
            reference = outer[common]
        else:
            pd.testing.assert_frame_equal(reference, outer[common], check_exact=True)
        rng = np.random.default_rng(20260910)
        for h, hours in HOURS.items():
            all_out = outer[outer.horizon==h]
            assert len(all_out)==34416
            assert all_out.outer_fold.tolist() == all_out.fipsCode.map(fold_of).tolist()
            near(all_out.y_true, truth.loc[pd.MultiIndex.from_frame(all_out[KEY]), h], 0)
            for frame in (all_out, test[test.horizon==h]):
                valid = frame.hour_idx.to_numpy()+hours < 216
                assert np.array_equal(valid, frame.is_scoreable)
                assert np.isfinite(frame[["base_prediction", "correction_osi", "alpha"]]).all().all()
                near(frame.prediction_before_postprocess, np.where(valid, frame.base_prediction+frame.alpha*frame.correction_osi, np.nan))
                near(frame.prediction, np.where(valid, post(frame.base_prediction+frame.alpha*frame.correction_osi), np.nan))
            score = all_out[all_out.is_scoreable]
            assert len(score) == 239*(144-hours)
            for k in range(5):
                s=tuple(q for q in range(5) if q!=k)
                part=inner[(inner.horizon==h)&(inner.outer_fold==k)]
                refpart=score[score.outer_fold!=k]
                assert set(map(tuple,part[KEY].to_numpy())) == set(map(tuple,refpart[KEY].to_numpy()))
                assert part.is_scoreable.all()
                assert part.inner_fold.tolist()==part.fipsCode.map(fold_of).tolist()
                near(part.y_true,truth.loc[pd.MultiIndex.from_frame(part[KEY]),h],0)
                for j in s:
                    allowed=tuple(q for q in s if q!=j); g=part[part.inner_fold==j]
                    assert g.allowed_folds.map(lambda val:tuple(json.loads(val))==allowed).all()
                    assert g.stack_id.eq(model_id("stack",v,h,allowed)).all()
                    assert g.base_source_id.eq(model_id("base",v,h,allowed)).all()
                candidates=alpha[(alpha.horizon==h)&(alpha.outer_fold==k)]
                a=choose(part,candidates,v,h,s)
                g=score[score.outer_fold==k]
                assert g.alpha.eq(a).all() and g.stack_id.eq(model_id("stack",v,h,s)).all()
                for c in COMP:
                    assert g[f"base_source_{c}"].eq(model_id("base",v,h,s,c)).all()
                be=(g.base_prediction-g.y_true)**2; pe=(g.prediction-g.y_true)**2
                folds.append(dict(variant=v,horizon=h,outer_fold=k,alpha=a,n=len(g),base_rmse=float(np.sqrt(be.mean())),gat_rmse=float(np.sqrt(pe.mean())),delta_sse=float(pe.sum()-be.sum())))
                if h=="osi_target_t01h" and k==0:
                    try:
                        check_alpha_group(candidates,part.y_true,part.base_prediction,part.correction_osi,S=s,h=h,mode="component_v158",selection_type="outer_train_inner_cv",outer_fold=k)
                        verifier_checks.append(dict(variant=v,check="original_alpha_identity_check",status="PASS"))
                    except AssertionError as err:
                        verifier_checks.append(dict(variant=v,check="original_alpha_identity_check",status="FAIL",error=str(err)))
            final_a=choose(score,final[(final.horizon==h)&(final.selection_type=="full_train_cv_for_final")],v,h,tuple(range(5)))
            assert final[(final.horizon==h)&(final.selection_type=="final_selected")].alpha.tolist()==[final_a]
            test_h=test[(test.horizon==h)&test.is_scoreable]
            assert test_h.alpha.eq(final_a).all()
            tests.append(dict(variant=v,horizon=h,alpha=final_a,scoreable_rows=len(test_h),changed_from_base=int((test_h.prediction!=test_h.base_prediction).sum())))
            local=[]
            for f,g in score.groupby("fipsCode",sort=True):
                bs=float(np.sum((g.base_prediction-g.y_true)**2)); ps=float(np.sum((g.prediction-g.y_true)**2))
                row=dict(variant=v,horizon=h,fipsCode=f,countyName=names.get(f,f),outer_fold=int(g.outer_fold.iloc[0]),alpha=float(g.alpha.iloc[0]),n=len(g),base_sse=bs,gat_sse=ps,delta_sse=ps-bs,base_rmse=float(np.sqrt(bs/len(g))),gat_rmse=float(np.sqrt(ps/len(g))),correction_mean=float(g.correction_osi.mean()),correction_min=float(g.correction_osi.min()),correction_max=float(g.correction_osi.max()),prediction_mean=float(g.prediction.mean()),actual_mean=float(g.y_true.mean()))
                local.append(row);counties.append(row)
            county=pd.DataFrame(local)
            bs,ps=float(county.base_sse.sum()),float(county.gat_sse.sum()); b=np.sqrt(bs/len(score));p=np.sqrt(ps/len(score))
            metrics.append(dict(variant=v,horizon=h,n=len(score),base_rmse=b,gat_rmse=p,delta_rmse=p-b,relative_percent=100*(p/b-1),delta_sse=ps-bs,counties_better=int((county.delta_sse < -1e-12).sum()),counties_worse=int((county.delta_sse>1e-12).sum()),counties_equal=int((abs(county.delta_sse)<=1e-12).sum())))
            saved_summary=pd.read_csv(root/"cv_summary.csv",float_precision="round_trip")
            near([b,p],saved_summary[(saved_summary.horizon==h)].set_index("model").loc[["base","gat"],"rmse"])
            draws=rng.integers(0,239,size=(2000,239));counts=county.n.to_numpy()[draws].sum(axis=1)
            delta=np.sqrt(county.gat_sse.to_numpy()[draws].sum(axis=1)/counts)-np.sqrt(county.base_sse.to_numpy()[draws].sum(axis=1)/counts)
            low,high=np.quantile(delta,[.025,.975]);fraction=float(np.mean(delta<0))
            saved_boot=pd.read_csv(root/"county_bootstrap.csv",float_precision="round_trip").set_index("horizon").loc[h]
            near([low,high,fraction],[saved_boot.delta_ci_low,saved_boot.delta_ci_high,saved_boot.gat_better_probability])
            bootstraps.append(dict(variant=v,horizon=h,delta_rmse=p-b,ci_low=low,ci_high=high,resample_fraction_better=fraction))
        if v=="m3_bounded":
            assert max(outer.correction_osi.abs().max(),inner.correction_osi.abs().max(),test.correction_osi.abs().max())<=.250001
        audit.append(dict(variant=v,run_root=str(root),source_hashes_matched=13,base_models_hashed=416,GAT_files_hashed=64,base_path_separator_mismatches=path_mismatches,full_reload=read(root/"verification.json")["independent_reload"],COMPLETE_present=(root/"COMPLETE").exists(),saved_numerical_and_scope_checks="PASS"))

    table("overall_metrics.csv",metrics);table("fold_metrics.csv",folds);table("county_metrics.csv",counties)
    table("final_alpha.csv",tests);table("bootstrap.csv",bootstraps);table("source_code_match.csv",source_rows)
    table("verifier_findings.csv",verifier_checks)
    c=pd.DataFrame(counties)
    table("Newton.csv",c[c.fipsCode=="18111"])
    table("largest_harm.csv",c.sort_values("delta_sse",ascending=False).groupby(["variant","horizon"]).head(5))
    table("largest_benefit.csv",c.sort_values("delta_sse").groupby(["variant","horizon"]).head(5))
    without=[]
    for (v,h),g in c[c.fipsCode!="18111"].groupby(["variant","horizon"]):
        b=float(np.sqrt(g.base_sse.sum()/g.n.sum()));p=float(np.sqrt(g.gat_sse.sum()/g.n.sum()))
        without.append(dict(variant=v,horizon=h,base_rmse=b,gat_rmse=p,relative_percent=100*(p/b-1),delta_sse=float(g.delta_sse.sum()),diagnostic_exclusion_only=True))
    table("excluding_Newton_diagnostic_only.csv",without)
    assert all(sha(path)==digest for path,digest in before.items())
    dump("audit.json",dict(status="SAVED_ARTIFACT_ANALYSIS_PASS_WITH_VERIFIER_DEFECTS",runs=audit,original_files_unchanged=len(before),training_performed=False,model_reload_performed=False,alpha_candidates_recomputed=4*(160+32),bootstrap_replicates_per_horizon=2000,original_verifier_smoke_checks=verifier_checks,source_root_for_results=str(CODE),requested_limit_source_is_different_173_dim_variant=True))
    print(pd.DataFrame(metrics).pivot(index="variant",columns="horizon",values="relative_percent").to_string())
    print("OUTPUT",OUT)


if __name__=="__main__":
    main()
