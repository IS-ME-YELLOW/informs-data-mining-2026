"""Approved first-round F1/F2 experiment; only seed42, no final/test prediction."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import resource
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from information_protocol import (HERE,CONFIG,KINDS,NEW_CASES,TARGETS,load_reference,attach_features,
    experiment_identity,output_path,read_json,digest_object,write_json,write_frame,sha256_file,support)
from build_information_features import build_features,feature_inputs
from information_training import get_or_fit
from information_preflight import run_checks,reject
from evaluate_information import artifact_frames
from run_artifacts import run_lock,freeze_stage

RUN_ID='v18_pinfo_v1_split42_model42'


def require_manifest(a,b):
    if a!=b or digest_object(a['identity'])!=a['identity_hash']:raise ValueError('Resume identity mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','cv'),default='preflight')
    parser.add_argument('--split-seed',type=int,choices=(42,),default=42)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.stage=='preflight' and args.resume:parser.error('preflight cannot resume')
    ctx=load_reference(args.split_seed)
    if args.stage=='preflight':build_features(feature_inputs(ctx.data))
    ctx=attach_features(ctx);identity=experiment_identity(ctx);identity_hash=digest_object(identity)
    manifest={'run_id':RUN_ID,'identity':identity,'identity_hash':identity_hash}
    run=output_path(HERE/'runs'/RUN_ID);pre=HERE/'preflight/split42'
    if args.stage=='preflight':
        result,provenance=run_checks(ctx)
        require_manifest(manifest,manifest);reject(lambda:require_manifest({**manifest,'identity_hash':'bad'},manifest))
        result.update(identity_hash=identity_hash,split_seed=42,planned_formal_fits=100,planned_new_outer_models=20,resume_mismatch_rejection='PASS')
        write_json(pre/'checks.json',result);write_json(pre/'fit_support_and_weights.json',provenance)
        write_json(HERE/'reference/source_manifest.json',{'source_sha256':ctx.sources,'reference_E2_identity_hash':CONFIG['reference_identity_hash']})
        print(json.dumps(result,indent=2));return
    check=read_json(pre/'checks.json')
    if check['status']!='PASS' or check['identity_hash']!=identity_hash:raise ValueError('Matching successful preflight required')
    if run.exists():
        if not args.resume or not (run/'run_manifest.json').exists():raise FileExistsError('Existing run requires matching --resume')
        require_manifest(read_json(run/'run_manifest.json'),manifest)
        if (run/'INFO_CV_COMPLETE').exists():raise FileExistsError('Completed run cannot be overwritten')
    else:
        if args.resume:raise FileNotFoundError('Cannot resume absent run')
        run.mkdir(parents=True);write_json(run/'run_manifest.json',manifest)
        write_json(run/'source_manifest.json',read_json(HERE/'reference/source_manifest.json'))
        write_json(run/'environment.json',identity['environment']);write_frame(run/'cv_assignments.csv',ctx.data.assignment)
    with run_lock(run):
        attempts=read_json(run/'execution.json')['attempts'] if (run/'execution.json').exists() else []
        attempt={'start_utc':datetime.now(timezone.utc).isoformat(),'status':'running','fits_this_attempt':0,
            'command':[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]]}
        attempts.append(attempt);start=time.monotonic()
        def save():write_json(run/'execution.json',{'attempts':attempts})
        def progress(message,n):
            attempt['fits_this_attempt']+=n;save();print(f"[{attempt['fits_this_attempt']:03d}/100] {case} {message}",flush=True)
        save();raw={c:{k:np.full(len(ctx.meta),np.nan) for k in KINDS} for c in NEW_CASES}
        ids={c:{k:np.full(len(ctx.meta),'',object) for k in KINDS} for c in NEW_CASES}
        records=[];rounds=[]
        try:
            for case in NEW_CASES:
                cc=ctx.cases[case]
                for outer in range(5):
                    for kind in KINDS:
                        model,receipt=get_or_fit(cc,run,outer,kind,identity_hash,progress)
                        all_rows=ctx.valid&(ctx.data.row_folds==outer);take=np.flatnonzero(all_rows&support(ctx.p,kind))
                        raw[case][kind][take]=model.predict(cc.data.X_train.iloc[take],num_threads=1)
                        ids[case][kind][all_rows]=receipt['model_id'];records.append(receipt)
                        for probe,best in zip(receipt['spec']['probes'],receipt['fit']['best_iterations']):
                            rounds.append({'candidate':case,'outer_fold':outer,'kind':kind,'target':TARGETS[kind],
                                'inner_fold':probe['inner_fold'],'best_iteration':best,'refit_rounds':receipt['fit']['requested_rounds'],
                                'actual_trees':receipt['fit']['actual_trees'],'training_rows':probe['train']['rows'],
                                'training_counties':len(probe['train']['counties']),'validation_rows':probe['validation']['rows'],
                                'weight_mean':probe['train']['weight_mean'],'row_weight_ess':probe['train']['row_weight_ess'],'model_id':receipt['model_id']})
            frames=artifact_frames(ctx,raw,ids,identity_hash);frames['round_selection.csv']=pd.DataFrame(rounds)
            for name,frame in frames.items():write_frame(output_path(run/name),frame)
            write_json(run/'new_model_manifest.json',records);write_json(run/'fit_provenance.json',[r['spec'] for r in records])
            print('All 100 fits complete; launching independent feature/model verification.',flush=True)
            cmd=[sys.executable,'-B',str(HERE/'verify_information.py'),'--split-seed','42','--unfrozen']
            result=subprocess.run(cmd,cwd=HERE,text=True,capture_output=True)
            (run/'logs/independent_verification_process.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError('Independent verification failed: '+result.stderr[-2500:])
            verification=read_json(run/'logs/independent_verification.json')
            assert verification['status']=='PASS' and verification['identity_hash']==identity_hash
            write_json(run/'verification.json',verification)
            # Required old-file integrity is checked before the completion marker.
            snapshot=read_json(HERE/'reference/initial_workspace_snapshot.json')
            changed=[p for p,h in snapshot['sha256'].items() if not Path(p).is_file() or sha256_file(p)!=h]
            integrity={'status':'PASS' if not changed else 'FAIL','checked_files':len(snapshot['sha256']),'changed_files':changed}
            write_json(run/'historical_integrity.json',integrity)
            if changed:raise ValueError('Historical files changed during run')
            names=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name not in ('execution.json','.run.lock','INFO_CV_COMPLETE')]
            freeze_stage(run,'INFO_CV_COMPLETE',names,identity_hash)
            attempt.update(status='completed',independent_verification='PASS');print(json.dumps(verification,indent=2),flush=True)
        except BaseException as exc:
            attempt.update(status='failed',error=repr(exc));raise
        finally:
            usage=resource.getrusage(resource.RUSAGE_SELF)
            attempt.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
                cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss);save()


if __name__=='__main__':main()
