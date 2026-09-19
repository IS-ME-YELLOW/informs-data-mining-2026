"""Strict tree-only v1.8 runner. Default is read-only preflight, never training."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import json
import numpy as np
import pandas as pd

from config import (PROTOCOL, CACHE_DIR, CV_FILE, RUNS_DIR, SEED, MAX_BOOST_ROUNDS,
                    EARLY_STOPPING_ROUNDS, HORIZONS, HORIZON_HOURS, COMPONENTS,
                    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
from protocol import (MODES, TARGETS, load_data, expected_mask, require_predictions,
                      target_horizon, component_target_name, clip_component,
                      build_controls, fill_submission, resolve_input)
from scoped_training import get_or_fit
from run_artifacts import (prepare_run, run_lock, write_json, write_frame, check_stage,
                           freeze_stage, data_identity, digest_object)
from evaluation import _evaluation_tables, _paired_bootstrap

MODELS = MODES
_build_controls = build_controls  # pure helper retained for research consumers

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight","cv","final","all"), default="preflight")
    parser.add_argument("--validate-only", action="store_true", help="Read-only alias for preflight")
    parser.add_argument("--feature-dir", default=str(CACHE_DIR))
    parser.add_argument("--cv-file", default=str(CV_FILE))
    parser.add_argument("--model-seed", type=int, default=SEED)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    parser.add_argument("--diagnostic", action="store_true", help="Explicitly mark a reduced-budget diagnostic run")
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.stage != "preflight":
            parser.error("--validate-only cannot be combined with a training stage")
    if args.stage == "preflight" and args.resume:
        parser.error("preflight does not resume or write runs")
    if args.stage != "preflight" and not args.run_id:
        parser.error("Training requires an explicit --run-id")
    if args.max_rounds < 1 or args.early_stopping_rounds < 1 or not 0 <= args.model_seed < 2**31:
        parser.error("Invalid rounds, patience or model seed")
    if not args.diagnostic and (args.max_rounds != MAX_BOOST_ROUNDS or args.early_stopping_rounds != EARLY_STOPPING_ROUNDS):
        parser.error("Changing the frozen budget requires --diagnostic")
    if args.stage == "final" and not args.resume:
        parser.error("final requires --resume of a verified CV run")
    return args

def settings_from_args(args):
    return {"model_seed":args.model_seed, "max_rounds":args.max_rounds,
            "early_stopping_rounds":args.early_stopping_rounds, "diagnostic":args.diagnostic,
            "bootstrap_replicates":BOOTSTRAP_REPLICATES,"bootstrap_seed":BOOTSTRAP_SEED}

def empty_predictions(data, phase):
    meta = data.meta_train if phase == "cv" else data.meta_test
    return ({t:np.full(len(meta),np.nan) for t in TARGETS},
            {t:np.full(len(meta),"",dtype=object) for t in TARGETS})

def generate_predictions(run_dir,data,settings,identity_hash,phase):
    predictions,sources = empty_predictions(data,phase)
    records = []
    outer_folds = range(5) if phase == "cv" else [None]
    for fold in outer_folds:
        scope = tuple(f for f in range(5) if f != fold)
        meta = data.meta_train if phase == "cv" else data.meta_test
        x = data.X_train if phase == "cv" else data.X_test
        for target in TARGETS:
            model,record = get_or_fit(run_dir,data,scope,target,settings,identity_hash)
            mask = expected_mask(meta,target_horizon(target))
            if phase == "cv":
                mask &= data.row_folds == fold
            rows = np.flatnonzero(mask)
            if not np.isnan(predictions[target][rows]).all() or (sources[target][rows] != "").any():
                raise ValueError("Duplicate OOF coverage")
            values = model.predict(x.iloc[rows],num_iteration=record["fit"]["requested_rounds"],num_threads=1)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite model output")
            predictions[target][rows] = values
            sources[target][rows] = record["model_id"]
            records.append(record)
        print(f"{phase}: completed {'all counties' if fold is None else 'outer fold '+str(fold)}",flush=True)
    for target in TARGETS:
        mask = expected_mask(meta,target_horizon(target))
        require_predictions(predictions[target],mask,exact_tail=True)
        if (sources[target][mask] == "").any() or (sources[target][~mask] != "").any():
            raise ValueError("Invalid source-model coverage")
    return predictions,sources,records

def output_frames(data,predictions,sources,phase,identity_hash):
    meta = (data.meta_train if phase=="cv" else data.meta_test).copy()
    meta["fold"] = data.row_folds if phase=="cv" else -1
    ids = meta[["fipsCode","timestamp_et","hour_idx","stateAbbr","fold"]].copy()
    # Same split can compare different feature/model runs; keep its identity
    # separate from the complete run identity.
    ids["split_id"] = data.loaded_hashes.get("cv", digest_object(data.assignment.to_dict(orient="records")))
    ids["run_identity_hash"] = identity_hash
    base = ids.copy()
    component_frame = ids.copy()
    components = {h:{c:predictions[component_target_name(c,h)] for c in COMPONENTS} for h in HORIZONS}
    direct = {h:predictions[h] for h in HORIZONS}
    controls,aux = build_controls(meta,direct,components)
    oof = ids.copy()
    alignment = []
    for target in TARGETS:
        base[f"raw_{target}"] = predictions[target]
        base[f"source_{target}"] = sources[target]
    for h in HORIZONS:
        mask = expected_mask(meta,h)
        oof[f"scoreable_{h}"] = mask
        if phase=="cv":
            oof[f"actual_{h}"] = data.y_train[h].to_numpy()
        for mode in MODES:
            oof[f"pred_{mode}_{h}"] = controls[mode][h]
        for c in COMPONENTS:
            target = component_target_name(c,h)
            component_frame[f"pred_{target}"] = clip_component(predictions[target])
            component_frame[f"source_{target}"] = sources[target]
            if phase=="cv":
                component_frame[f"actual_{target}"] = data.component_targets[target].to_numpy()
            frame = ids.loc[mask].copy()
            frame["component"] = c
            frame["source_horizon"] = h
            frame["target_timestamp"] = frame.timestamp_et + pd.to_timedelta(HORIZON_HOURS[h],unit="h")
            frame["source_model_id"] = sources[target][mask]
            frame["prediction"] = clip_component(predictions[target][mask])
            alignment.append(frame)
    name = "oof" if phase=="cv" else "test"
    frames = {f"{name}_predictions.parquet":oof, f"{name}_component_predictions.parquet":component_frame,
              f"base_predictions_{phase}.parquet":base, f"alignment_sources_{phase}.parquet":pd.concat(alignment,ignore_index=True),
              f"aligned_unique_components_{phase}.parquet":aux["unique_components"]}
    if phase=="cv":
        frames.update(_evaluation_tables(data,data.component_targets,controls,components))
    else:
        for mode in MODES:
            frames[f"submission_{mode}.csv"] = fill_submission(meta,controls[mode],data.input_paths["submission"])
    return frames,controls

def stage_frames(data,predictions,sources,records,phase,identity_hash,settings):
    frames,controls = output_frames(data,predictions,sources,phase,identity_hash)
    rounds = []
    for record in records:
        for probe,best in zip(record["fit"]["probes"],record["fit"]["best_iterations"]):
            rounds.append({"model_id":record["model_id"],"target":record["spec"]["target"],
                           "allowed_folds":json.dumps(record["spec"]["scope"]),"inner_fold":probe["inner_fold"],
                           "train_counties":json.dumps(probe["train"]["counties"]),
                           "early_stop_counties":json.dumps(probe["early_stop"]["counties"]),
                           "best_iteration":best,"refit_rounds":record["fit"]["requested_rounds"]})
    frames[f"round_selection_{phase}.csv"] = pd.DataFrame(rounds)
    if phase=="cv":
        frames["paired_county_bootstrap.csv"] = _paired_bootstrap(data,controls,settings["bootstrap_replicates"],settings["bootstrap_seed"])
    return frames

def run_stage(run_dir,data,manifest,phase):
    if phase not in ("cv", "final"):
        raise ValueError("Unknown stage")
    settings,identity_hash = manifest["identity"]["settings"],manifest["identity_hash"]
    if phase=="cv" and (run_dir/"CV_COMPLETE").exists():
        raise FileExistsError("CV is frozen; use --stage final --resume")
    if phase=="cv" and any((run_dir/name).exists() for name in ("FINAL_READY", "COMPLETE")):
        raise ValueError("Cannot restart CV in a final-stage run")
    if phase=="final":
        check_stage(run_dir,"CV_COMPLETE",identity_hash)
        if (run_dir/"FINAL_READY").exists() or (run_dir/"COMPLETE").exists():
            raise FileExistsError("Final predictions are frozen; run verify_artifacts.py")
    predictions,sources,records = generate_predictions(run_dir,data,settings,identity_hash,phase)
    frames = stage_frames(data,predictions,sources,records,phase,identity_hash,settings)
    for name,frame in frames.items():
        write_frame(run_dir/name,frame)
    model_manifest = f"model_manifest_{phase}.json"
    write_json(run_dir/model_manifest,records)
    from verify_artifacts import verify_loaded_run
    # Reload every model and recompute the saved outputs before freezing a stage.
    result = verify_loaded_run(run_dir,data,phase,require_marker=False)
    write_json(run_dir/f"verification_{phase}.json",result)
    files = [*frames,model_manifest,"run_manifest.json","environment.json","input_manifest.json","cv_assignments.csv"]
    files += [r[key] for r in records for key in ("model_path","receipt_path")]
    if phase=="cv":
        freeze_stage(run_dir,"CV_COMPLETE",files,identity_hash)
    else:
        freeze_stage(run_dir,"FINAL_READY",files,identity_hash)
        verify_loaded_run(run_dir,data,"cv")
        freeze_stage(run_dir,"COMPLETE",["CV_COMPLETE","FINAL_READY"],identity_hash)
    print(json.dumps(result,indent=2),flush=True)

def main(argv=None):
    args = parse_args(argv)
    data = load_data(args.feature_dir,args.cv_file)
    settings = settings_from_args(args)
    identity = data_identity(data,settings)
    if args.stage=="preflight":
        print(json.dumps({"preflight":"passed","protocol":PROTOCOL,"training_performed":False,
                          "cv_file":str(resolve_input(args.cv_file)),"cv_sha256":identity["inputs"]["cv"],
                          "train_shape":list(data.X_train.shape),"test_shape":list(data.X_test.shape),
                          "fold_counties":data.assignment.groupby("fold").size().to_dict(),
                          "scoreable_rows":{h:int(expected_mask(data.meta_train,h).sum()) for h in HORIZONS},
                          "identity_hash":digest_object(identity)},indent=2))
        return
    # Validate id before forming any path; prepare_run repeats this check.
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",args.run_id):
        raise ValueError("Invalid run id")
    run_dir = RUNS_DIR/args.run_id
    manifest = prepare_run(run_dir,args.run_id,data,settings,args.resume)
    with run_lock(run_dir):
        if args.stage in ("cv","all"):
            if args.stage=="all" and args.resume and (run_dir/"CV_COMPLETE").exists():
                check_stage(run_dir,"CV_COMPLETE",manifest["identity_hash"])
            else:
                run_stage(run_dir,data,manifest,"cv")
        if args.stage in ("final","all"):
            run_stage(run_dir,data,manifest,"final")

if __name__=="__main__":
    main()
