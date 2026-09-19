"""Reload and verify a new v18_tree_nested_v2 run; read-only by default."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import json
import numpy as np
import pandas as pd
from config import HORIZONS, PROTOCOL
from protocol import TARGETS, target_horizon, expected_mask, load_data, resolve_input
from scoped_training import scope_data, load_fit
from run_artifacts import (data_identity, digest_object, safe_artifact, check_stage,
                           write_json, freeze_stage, run_lock)

ATOL = 1e-12

def check_frame(path, expected):
    if path.suffix==".parquet":
        actual = pd.read_parquet(path,engine="pyarrow")
    else:
        actual = pd.read_csv(path,dtype={"fipsCode":str},float_precision="round_trip")
    pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),
                                  check_dtype=False,check_exact=False,atol=ATOL,rtol=0)

def verify_loaded_run(run_dir,data,phase,*,require_marker=True):
    """Shared verification core. CLI always reloads and validates real input files."""
    if phase not in ("cv", "final"):
        raise ValueError("Unknown verification phase")
    from train_v18 import empty_predictions, stage_frames
    run_dir=Path(run_dir)
    manifest=json.loads((run_dir/"run_manifest.json").read_text())
    if manifest["identity"]["protocol"]!=PROTOCOL or digest_object(manifest["identity"])!=manifest["identity_hash"]:
        raise ValueError("Not a valid new-protocol run manifest")
    settings=manifest["identity"]["settings"]
    identity_hash=manifest["identity_hash"]
    if digest_object(data_identity(data,settings))!=identity_hash:
        raise ValueError("Current inputs/code/settings/environment differ from run identity")
    marker_name="CV_COMPLETE" if phase=="cv" else "FINAL_READY"
    marker=check_stage(run_dir,marker_name,identity_hash) if require_marker else None
    if phase=="final":
        check_stage(run_dir,"CV_COMPLETE",identity_hash)
    check_frame(run_dir/"cv_assignments.csv",data.assignment)
    if json.loads((run_dir/"environment.json").read_text())!=manifest["identity"]["environment"]:
        raise ValueError("Environment artifact differs from run identity")
    input_manifest=json.loads((run_dir/"input_manifest.json").read_text())
    if input_manifest!={k:{"path":manifest["input_paths"][k],"sha256":v} for k,v in manifest["identity"]["inputs"].items()}:
        raise ValueError("Input manifest mismatch")
    records=json.loads((run_dir/f"model_manifest_{phase}.json").read_text())
    expected_keys={(target,fold) for target in TARGETS for fold in (range(5) if phase=="cv" else [-1])}
    seen=set()
    predictions,sources=empty_predictions(data,phase)
    meta=data.meta_train if phase=="cv" else data.meta_test
    x=data.X_train if phase=="cv" else data.X_test
    for receipt in records:
        target=receipt["spec"]["target"]
        scope=tuple(receipt["spec"]["scope"])
        if len(scope)!=(4 if phase=="cv" else 5) or len(set(scope))!=len(scope) or not set(scope).issubset(range(5)):
            raise ValueError("Invalid model scope")
        fold=next(iter(set(range(5))-set(scope))) if phase=="cv" else -1
        key=(target,fold)
        if key not in expected_keys or key in seen:
            raise ValueError("Duplicate, missing or unexpected model identity")
        seen.add(key)
        role=f"outer{fold}" if phase=="cv" else "final"
        if receipt["model_path"]!=f"models/{role}/{target}.txt" or receipt["receipt_path"]!=f"models/{role}/{target}.json":
            raise ValueError("Unexpected model artifact location")
        if json.loads(safe_artifact(run_dir,receipt["receipt_path"]).read_text())!=receipt:
            raise ValueError("Model manifest and individual receipt differ")
        scoped=scope_data(data,scope,target)
        model=load_fit(run_dir,receipt,scoped,identity_hash,settings)
        mask=expected_mask(meta,target_horizon(target))
        if phase=="cv":
            mask &= data.row_folds==fold
        rows=np.flatnonzero(mask)
        if not np.isnan(predictions[target][rows]).all():
            raise ValueError("Repeated prediction coverage")
        values=model.predict(x.iloc[rows],num_iteration=receipt["fit"]["requested_rounds"],num_threads=1)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite prediction after reload")
        predictions[target][rows]=values
        sources[target][rows]=receipt["model_id"]
    if seen!=expected_keys:
        raise ValueError("Incomplete model manifest")
    frames=stage_frames(data,predictions,sources,records,phase,identity_hash,settings)
    for name,frame in frames.items():
        check_frame(run_dir/name,frame)
    required={*frames,f"model_manifest_{phase}.json","run_manifest.json","environment.json","input_manifest.json","cv_assignments.csv"}
    required.update(r[k] for r in records for k in ("model_path","receipt_path"))
    if marker is not None and set(marker["files"])!=required:
        raise ValueError("Frozen stage does not cover exactly its required artifacts")
    if require_marker:
        check_stage(run_dir,marker_name,identity_hash)
    return {"status":"passed","protocol":PROTOCOL,"run_id":manifest["run_id"],"phase":phase,
            "diagnostic":settings["diagnostic"],"models_reloaded":len(records),
            "absolute_tolerance":ATOL,"all_predictions_metrics_and_lineage_recomputed":True,
            "scoreable_rows":{h:int(expected_mask(meta,h).sum()) for h in HORIZONS},
            "training_performed_by_verifier":False}

def verify_run(run_dir,write_report=False):
    run_dir=resolve_input(run_dir)
    manifest=json.loads((run_dir/"run_manifest.json").read_text())
    paths=manifest["input_paths"]
    data=load_data(Path(paths["features_train"]).parent,paths["cv"])
    phases=["cv"]+(["final"] if (run_dir/"FINAL_READY").exists() else [])
    results={phase:verify_loaded_run(run_dir,data,phase) for phase in phases}
    if (run_dir/"COMPLETE").exists():
        complete=check_stage(run_dir,"COMPLETE",manifest["identity_hash"])
        if "final" not in phases or set(complete["files"])!={"CV_COMPLETE","FINAL_READY"}:
            raise ValueError("Invalid COMPLETE marker")
    if write_report:
        with run_lock(run_dir):
            # Recheck immutable hashes after acquiring the write lock.
            for marker in ["CV_COMPLETE"]+(["FINAL_READY"] if "final" in phases else []):
                check_stage(run_dir,marker,manifest["identity_hash"])
            for phase,result in results.items():
                write_json(run_dir/f"verification_{phase}.json",result)
            if "final" in phases:
                freeze_stage(run_dir,"COMPLETE",["CV_COMPLETE","FINAL_READY"],manifest["identity_hash"])
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",help="New run directory, relative to project root or absolute")
    parser.add_argument("--write-report",action="store_true",help="Write verification reports only inside this run")
    args=parser.parse_args()
    print(json.dumps(verify_run(args.run_dir,args.write_report),indent=2))

if __name__=="__main__":
    main()
