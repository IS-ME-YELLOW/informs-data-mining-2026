"""Frozen confirmation-split U target, source identities, and reconstruction."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/"experiment_config.json").read_text())
ROOT = Path(CONFIG["project_root"]).resolve()
BASE = Path(CONFIG["baseline_root"]).resolve()
assert HERE == Path(CONFIG["experiment_root"]).resolve() == BASE/"d_unknown_contribution"/"confirmation"/f"split{CONFIG['split_seed']}"
assert BASE.is_relative_to(ROOT) and (ROOT/"data/DM_Train.csv").is_file()
sys.path.insert(0, str(BASE))
from config import HORIZONS, HORIZON_HOURS, COMPONENTS, make_lgbm_params
from protocol import load_data, expected_mask, build_controls, MODES, TARGETS, sha256_file
from run_artifacts import data_identity, digest_object, write_json, write_frame, check_stage, environment
from scoped_training import ScopedData, fit_scoped_base, provenance, model_spec, load_fit
from known_history import observed_history, known_bounds, project_components

H1 = "osi_target_t01h"
DT = "D_t_target_t01h"
UT = "U_D_t_target_t01h"
KEYS = ["fipsCode", "timestamp_et", "hour_idx", "stateAbbr", "fold", "split_id"]
CODE_FILES = ("unknown_d_protocol.py", "known_history.py", "metric_helpers.py", "independent_numerics.py",
              "unknown_d_evaluation.py", "train_unknown_d.py", "verify_unknown_d.py")


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE) or path == HERE:
        raise ValueError("Output path escapes this experiment")
    return path


def json_read(path):
    return json.loads(Path(path).read_text())


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def source_files():
    run = Path(CONFIG["baseline_run"])
    marker = json_read(run/"CV_COMPLETE")
    manifest = json_read(run/"run_manifest.json")
    if manifest["identity_hash"] != CONFIG["baseline_identity_hash"]:
        raise ValueError("Baseline identity changed")
    check_stage(run, "CV_COMPLETE", manifest["identity_hash"])
    paths = {str(run/name):digest for name,digest in marker["files"].items()}
    paths[str(run/"CV_COMPLETE")] = sha256_file(run/"CV_COMPLETE")
    paths.update({str(BASE/name):digest for name,digest in manifest["identity"]["code_sha256"].items()})
    for name, digest in manifest["identity"]["inputs"].items():
        path = Path(manifest["input_paths"][name])
        paths[str(path if path.is_absolute() else ROOT/path)] = digest
    bdir = Path(CONFIG["projection_reference"])
    bmanifest = json_read(bdir/"manifest.json")
    if bmanifest["baseline_identity_hash"] != manifest["identity_hash"]:
        raise ValueError("Projection B is not from this baseline")
    bcheck = json_read(bdir/"verification.json")
    if bcheck["status"] != "PASS" or bcheck["manifest_sha256"] != sha256_file(bdir/"manifest.json"):
        raise ValueError("Projection B lacks matching independent verification")
    paths.update({str(bdir/name):digest for name,digest in bmanifest["output_sha256"].items()})
    paths.update({bmanifest["input_paths"][name]:digest for name,digest in bmanifest["input_sha256"].items()})
    paths[str(bdir/"manifest.json")] = sha256_file(bdir/"manifest.json")
    paths[str(bdir/"verification.json")] = sha256_file(bdir/"verification.json")
    for path,digest in paths.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Changed numerical source: {path}")
    return paths


def load_context(split_seed=None):
    split_seed = CONFIG["split_seed"] if split_seed is None else split_seed
    if split_seed != CONFIG["split_seed"] or CONFIG["authorized_split_seeds"] != [split_seed] or split_seed not in (20260917,20260918):
        raise ValueError("Split does not match this frozen confirmation directory")
    sources = source_files()
    data = load_data(ROOT/"versions/xyy/v1.5.6", ROOT/f"cv/cv_assignments_balanced_v1_seed{split_seed}.csv")
    baseline_identity = data_identity(data, CONFIG["settings"])
    if digest_object(baseline_identity) != CONFIG["baseline_identity_hash"]:
        raise ValueError("Feature/data/code/settings/environment differ from baseline")
    run = Path(CONFIG["baseline_run"])
    oof = pd.read_parquet(run/"oof_predictions.parquet")
    component = pd.read_parquet(run/"oof_component_predictions.parquet")
    raw = pd.read_parquet(run/"base_predictions_cv.parquet")
    meta = oof[KEYS].copy()
    pd.testing.assert_frame_equal(meta[["fipsCode","timestamp_et","hour_idx","stateAbbr"]],
                                  data.meta_train[["fipsCode","timestamp_et","hour_idx","stateAbbr"]])
    np.testing.assert_array_equal(meta.fold, data.row_folds)
    assert meta.split_id.eq(data.loaded_hashes["cv"]).all()
    for frame in (component, raw): pd.testing.assert_frame_equal(frame[KEYS],meta)
    raw_history = pd.read_csv(ROOT/"data/DM_Train.csv", usecols=["fipsCode","timestamp_et","P_t"], dtype={"fipsCode":str})
    history = observed_history(raw_history)
    bounds = known_bounds(meta,history)
    a = {h:{c:component[f"pred_{c}_target_{h.rsplit('_',1)[-1]}"].to_numpy(copy=True) for c in COMPONENTS} for h in HORIZONS}
    b = project_components(a,bounds)
    direct = {h:raw[f"raw_{h}"].to_numpy(copy=True) for h in HORIZONS}
    saved_b = pd.read_parquet(Path(CONFIG["projection_reference"])/"candidate_oof_predictions.parquet")
    pd.testing.assert_frame_equal(saved_b[KEYS],meta)
    controls = {}
    for label,parts in (("A",a),("B",b)):
        controls[label],_ = build_controls(meta,direct,parts)
        for mode in MODES:
            for h in HORIZONS:
                reference = oof[f"pred_{mode}_{h}"] if label=="A" else saved_b[f"candidate_{mode}_{h}"]
                np.testing.assert_array_equal(controls[label][mode][h],reference)
    actual = data.component_targets[DT].to_numpy(copy=True)
    np.testing.assert_array_equal(actual,component[f"actual_{DT}"])
    valid = expected_mask(meta,H1)
    unknown_count = np.minimum(6,meta.hour_idx.to_numpy()+1-71)
    unknown = actual-bounds[H1]
    validate_targets(unknown[valid],unknown_count[valid])
    return SimpleNamespace(data=data,meta=meta,raw_history=raw_history,history=history,bounds=bounds,
                           actual_d=actual,valid=valid,unknown_count=unknown_count,raw=raw,oof=oof,
                           component=component,parts={"A":a,"B":b},controls=controls,direct=direct,sources=sources)


def validate_targets(values,unknown_count):
    values = np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or (values < -CONFIG["epsilon"]).any() or (values > np.asarray(unknown_count)/6+CONFIG["epsilon"]).any():
        raise ValueError("U label outside fixed physical/rounding tolerance")


def make_scoped(ctx,scope):
    scope = tuple(sorted(scope))
    if len(scope)!=4 or len(set(scope))!=4 or not set(scope).issubset(range(5)):
        raise ValueError("U CV fitting requires exactly four allowed folds")
    rows = np.flatnonzero(np.isin(ctx.data.row_folds,scope)&ctx.valid)
    # Select allowed rows BEFORE accessing official future labels; no target audit input.
    d = ctx.data.component_targets[DT].iloc[rows].to_numpy(dtype=float,copy=True)
    u = d-ctx.bounds[H1][rows]
    validate_targets(u,ctx.unknown_count[rows])
    return ScopedData(ctx.data.X_train.iloc[rows].reset_index(drop=True),u,
                      ctx.data.meta_train.iloc[rows].reset_index(drop=True),ctx.data.row_folds[rows].copy(),scope,UT)


def reconstructed_d(raw_u,k):
    raw_u,k = np.asarray(raw_u,dtype=float),np.asarray(k,dtype=float)
    valid = np.isfinite(k)
    if raw_u.shape!=k.shape or np.isinf(k).any() or not np.isfinite(raw_u[valid]).all() or not np.isnan(raw_u[~valid]).all():
        raise ValueError("Invalid U prediction shape/mask")
    if not ((k[valid]>=0)&(k[valid]<=1)).all():
        raise ValueError("Invalid K")
    return np.clip(k+np.maximum(raw_u,0),0,1)


def candidate(ctx,raw_u):
    c = {h:{comp:values.copy() for comp,values in parts.items()} for h,parts in ctx.parts["A"].items()}
    c[H1]["D_t"] = reconstructed_d(raw_u,ctx.bounds[H1])
    controls,aux = build_controls(ctx.meta,ctx.direct,c)
    for h in HORIZONS:
        for comp in COMPONENTS:
            if (h,comp)!=(H1,"D_t"):np.testing.assert_array_equal(c[h][comp],ctx.parts["A"][h][comp])
        np.testing.assert_array_equal(controls["C0_direct_osi"][h],ctx.controls["A"]["C0_direct_osi"][h])
        if h!=H1:
            for mode in ("C1_component_osi","C2_equal_blend"):
                np.testing.assert_array_equal(controls[mode][h],ctx.controls["A"][mode][h])
        if HORIZON_HOURS[h]>=24:
            np.testing.assert_array_equal(controls["v18_rule"][h],ctx.controls["A"]["v18_rule"][h])
    return c,controls,aux


def experiment_identity(ctx):
    return {"protocol":CONFIG["protocol"],"split_seed":CONFIG["split_seed"],"config_sha256":sha256_file(HERE/"experiment_config.json"),
            "settings":CONFIG["settings"],"environment":environment(),"baseline_identity_hash":CONFIG["baseline_identity_hash"],
            "source_sha256":ctx.sources,"code_sha256":{name:sha256_file(HERE/name) for name in CODE_FILES},
            "target":UT,"definition":CONFIG["definition"],"reconstruction":CONFIG["reconstruction"],
            "raw_U_valid_array_sha256":array_sha((ctx.actual_d-ctx.bounds[H1])[ctx.valid]),
            "feature_names":list(ctx.data.X_train.columns),"epsilon":CONFIG["epsilon"]}


def make_source_manifest(ctx):
    records = json_read(Path(CONFIG["baseline_run"])/"model_manifest_cv.json")
    reused = [r for r in records if r["spec"]["target"]!=DT]
    controls = [r for r in records if r["spec"]["target"]==DT]
    assert len(reused)==95 and len(controls)==5
    return {"baseline_run":CONFIG["baseline_run"],"baseline_identity_hash":CONFIG["baseline_identity_hash"],
            "projection_reference":CONFIG["projection_reference"],"source_sha256":ctx.sources,
            "reused_95_model_records":reused,"old_D1h_5_control_only_records":controls}


def preflight_checks(ctx):
    from independent_numerics import brute_bounds,exact,close
    brute = brute_bounds(ctx.raw_history,ctx.meta)
    for h in HORIZONS:close(brute[HORIZON_HOURS[h]],ctx.bounds[h])
    future = pd.to_datetime(ctx.raw_history.timestamp_et)>=pd.Timestamp("2026-03-14")
    for replacement in (np.nan,999.,np.random.default_rng(17).normal(size=int(future.sum()))):
        modified = ctx.raw_history.copy();modified.loc[future,"P_t"] = replacement
        pd.testing.assert_frame_equal(observed_history(modified),ctx.history)
    pd.testing.assert_frame_equal(observed_history(ctx.raw_history.loc[~future]),ctx.history)
    order = ctx.meta.sample(frac=1,random_state=17)
    reordered = known_bounds(order,ctx.history.sample(frac=1,random_state=19))
    for h in HORIZONS:exact(reordered[h],ctx.bounds[h][order.index])
    def rejects(action):
        try:action()
        except ValueError:return
        raise AssertionError("Invalid input accepted")
    rejects(lambda:observed_history(ctx.raw_history.drop(ctx.raw_history.index[0])))
    rejects(lambda:known_bounds(pd.concat([ctx.meta.iloc[:1]]*2),ctx.history))
    bad = ctx.meta.copy();bad.loc[0,"timestamp_et"]+=pd.Timedelta(hours=1)
    rejects(lambda:known_bounds(bad,ctx.history))
    for outer in range(5):
        allowed = tuple(f for f in range(5) if f!=outer)
        before = make_scoped(ctx,allowed)
        data_copy = SimpleNamespace(**vars(ctx.data))
        data_copy.component_targets = ctx.data.component_targets.copy()
        data_copy.component_targets.loc[ctx.data.row_folds==outer,DT] = 999.
        test_ctx = SimpleNamespace(**vars(ctx));test_ctx.data=data_copy
        after = make_scoped(test_ctx,allowed)
        pd.testing.assert_frame_equal(before.X,after.X);exact(before.y,after.y)
        exact(before.row_folds,after.row_folds);assert provenance(before)==provenance(after)
        excluded = set(ctx.meta.loc[ctx.meta.fold==outer,"fipsCode"])
        for probe in provenance(before)["probes"]:
            tr,va = set(probe["train"]["counties"]),set(probe["early_stop"]["counties"])
            assert not (tr&va) and not ((tr|va)&excluded)
    zeros = np.where(ctx.valid,0.,np.nan)
    candidate(ctx,zeros)  # exercise C3 and fixed long-horizon invariants without fitting.
    reconstructed_d(zeros,ctx.bounds[H1])
    rejects(lambda:reconstructed_d(np.zeros(len(zeros)),ctx.bounds[H1]))
    rejects(lambda:validate_targets(np.array([-2e-6]),np.array([2])))
    rejects(lambda:validate_targets(np.array([.34]),np.array([2])))
    unknown = (ctx.actual_d-ctx.bounds[H1])[ctx.valid]
    raw_test = pd.read_csv(ROOT/"data/DM_Test.csv",usecols=["fipsCode","timestamp_et","P_t"],dtype={"fipsCode":str})
    test_k = known_bounds(ctx.data.meta_test,observed_history(raw_test))
    for h,value in brute_bounds(raw_test,ctx.data.meta_test).items():close(value,test_k[f"osi_target_t{h:02d}h"])
    return {"status":"PASS","split_seed":CONFIG["split_seed"],"training_performed":False,"train_counties":239,"test_counties":63,
            "features":163,"scoreable_rows":{h:int(expected_mask(ctx.meta,h).sum()) for h in HORIZONS},
            "positive_K_rows":int((ctx.bounds[H1]>0).sum()),"negative_U_rows":int((unknown<0).sum()),
            "minimum_U":float(unknown.min()),"checks":["source and baseline identities","exact A/B replay",
            "independent scalar K train/test","future P perturbation/deletion","outer labels excluded before U transformation",
            "inner training/early-stop county isolation","row reorder and invalid key rejection","C3 propagation boundaries",
            "fixed label tolerance","63 test counties need no future labels"]}
