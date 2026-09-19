"""Strict U1h training for the split fixed by this confirmation directory."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import resource
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
from unknown_d_protocol import (HERE,CONFIG,H1,UT,DT,load_context,output_path,experiment_identity,
    make_source_manifest,preflight_checks,make_scoped,fit_scoped_base,model_spec,load_fit,
    digest_object,write_json,write_frame,sha256_file,array_sha,json_read)
from run_artifacts import run_lock,atomic_write,freeze_stage
from unknown_d_evaluation import artifact_frames

RUN_ID = f"v18_ud01_nested_v1_split{CONFIG['split_seed']}_model42"


def require_same_manifest(saved,expected):
    if saved!=expected or digest_object(saved["identity"])!=saved["identity_hash"]:
        raise ValueError("Resume identity mismatch")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","cv"),default="preflight")
    parser.add_argument("--split-seed",type=int,choices=(CONFIG["split_seed"],),default=CONFIG["split_seed"])
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    if args.stage=="preflight" and args.resume:parser.error("preflight cannot resume")
    ctx=load_context(args.split_seed)
    identity=experiment_identity(ctx);identity_hash=digest_object(identity)
    manifest={"run_id":RUN_ID,"identity":identity,"identity_hash":identity_hash}
    run=output_path(HERE/"runs"/RUN_ID)
    if args.stage=="preflight":
        result=preflight_checks(ctx)
        require_same_manifest(manifest,manifest)
        altered={**manifest,"identity_hash":"invalid"}
        try:require_same_manifest(altered,manifest)
        except ValueError:pass
        else:raise AssertionError("Mismatched resume was accepted")
        try:output_path(HERE.parent/"unexpected")
        except ValueError:pass
        else:raise AssertionError("Output escaped experiment")
        result.update(identity_hash=identity_hash,code_and_config_identity=identity["code_sha256"],
                      resume_identity_rejection="PASS",output_path_rejection="PASS")
        write_json(output_path(HERE/f"preflight/seed{CONFIG['split_seed']}.json"),result)
        write_json(output_path(HERE/"reference/source_manifest.json"),make_source_manifest(ctx))
        print(json.dumps(result,indent=2));return
    preflight=json_read(HERE/f"preflight/seed{CONFIG['split_seed']}.json")
    if preflight["status"]!="PASS" or preflight["identity_hash"]!=identity_hash:
        raise ValueError("Matching preflight required before training")
    if run.exists():
        if not args.resume or not (run/"run_manifest.json").exists():
            raise FileExistsError("Existing run requires identical manifested --resume")
        require_same_manifest(json_read(run/"run_manifest.json"),manifest)
        if (run/"U_CV_COMPLETE").exists():raise FileExistsError("Run is frozen; use verifier")
    else:
        if args.resume:raise FileNotFoundError("Cannot resume absent run")
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/"run_manifest.json",manifest)
        write_json(run/"base_source_manifest.json",make_source_manifest(ctx))
        write_frame(run/"cv_assignments.csv",ctx.data.assignment)
    with run_lock(run):
        attempts=json_read(run/"execution.json")["attempts"] if (run/"execution.json").exists() else []
        attempt={"started_at_utc":datetime.now(timezone.utc).isoformat(),"status":"running",
                 "command":[sys.executable,"-B",str(Path(__file__).resolve()),*sys.argv[1:]],"fits_this_attempt":0}
        attempts.append(attempt)
        def save_execution():write_json(run/"execution.json",{"attempts":attempts})
        save_execution();start=time.monotonic()
        raw_u=np.full(len(ctx.meta),np.nan);source_ids=np.full(len(ctx.meta),"",dtype=object)
        records=[];rounds=[]
        try:
            for fold in range(5):
                scope=tuple(f for f in range(5) if f!=fold)
                scoped=make_scoped(ctx,scope)
                relative=f"models/outer{fold}/{UT}.txt"
                receipt_name=f"models/outer{fold}/{UT}.json"
                if (run/receipt_name).exists():
                    receipt=json_read(run/receipt_name)
                    assert receipt["model_path"]==relative and receipt["receipt_path"]==receipt_name
                    assert receipt["U_labels_sha256"]==array_sha(scoped.y)
                    model=load_fit(run,receipt,scoped,identity_hash,CONFIG["settings"])
                else:
                    model,fit=fit_scoped_base(scoped,CONFIG["settings"])
                    atomic_write(run/relative,lambda temp:model.save_model(str(temp)))
                    spec=model_spec(scoped,identity_hash)
                    receipt={"spec":spec,"model_id":digest_object(spec),"fit":fit,
                             "model_path":relative,"receipt_path":receipt_name,"model_sha256":sha256_file(run/relative),
                             "target_semantics":CONFIG["definition"],"reconstructed_target":DT,
                             "U_labels_sha256":array_sha(scoped.y),"negative_U_labels":int((scoped.y<0).sum())}
                    write_json(run/receipt_name,receipt)
                    attempt["fits_this_attempt"]+=5;save_execution()
                take=np.flatnonzero(ctx.valid&(ctx.data.row_folds==fold))
                assert np.isnan(raw_u[take]).all()
                raw_u[take]=model.predict(ctx.data.X_train.iloc[take],num_iteration=receipt["fit"]["requested_rounds"],num_threads=1)
                source_ids[take]=receipt["model_id"]
                records.append(receipt)
                for probe,best in zip(receipt["fit"]["probes"],receipt["fit"]["best_iterations"]):
                    rounds.append({"outer_fold":fold,"model_id":receipt["model_id"],"target":UT,"inner_fold":probe["inner_fold"],
                        "train_counties":json.dumps(probe["train"]["counties"]),"early_stop_counties":json.dumps(probe["early_stop"]["counties"]),
                        "best_iteration":best,"refit_rounds":receipt["fit"]["requested_rounds"]})
                print(f"outer{fold} complete; inner={receipt['fit']['best_iterations']}; refit={receipt['fit']['requested_rounds']}",flush=True)
            assert np.isfinite(raw_u[ctx.valid]).all() and np.isnan(raw_u[~ctx.valid]).all()
            assert (source_ids[ctx.valid]!="").all() and (source_ids[~ctx.valid]=="").all()
            frames=artifact_frames(ctx,raw_u,source_ids,identity_hash)
            import pandas as pd
            frames["round_selection.csv"]=pd.DataFrame(rounds)
            for name,frame in frames.items():write_frame(output_path(run/name),frame)
            write_json(run/"new_model_manifest.json",records)
            from verify_unknown_d import verify_run
            verification=verify_run(run,require_marker=False)
            write_json(run/"verification.json",verification)
            names=[*frames,"run_manifest.json","base_source_manifest.json","new_model_manifest.json","cv_assignments.csv","verification.json"]
            names += [r[key] for r in records for key in ("model_path","receipt_path")]
            freeze_stage(run,"U_CV_COMPLETE",names,identity_hash)
            attempt.update(status="completed",cv_complete=True)
            print(json.dumps(verification,indent=2),flush=True)
        except BaseException as exc:
            attempt.update(status="failed",error=repr(exc));raise
        finally:
            usage=resource.getrusage(resource.RUSAGE_SELF)
            attempt.update(ended_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
                           cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss)
            save_execution()


if __name__=="__main__":
    main()
