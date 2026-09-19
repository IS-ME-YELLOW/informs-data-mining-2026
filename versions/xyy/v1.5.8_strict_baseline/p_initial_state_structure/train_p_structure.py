"""Authorized seed42-only P comparison. Default stage is a no-fit preflight."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import resource
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from p_structure_protocol import (HERE,CONFIG,KINDS,KEYS,TARGETS,load_context,experiment_identity,output_path,
    read_json,digest_object,write_json,write_frame,sha256_file,support)
from run_artifacts import run_lock,freeze_stage
from weighted_scoped_training import get_or_fit
from preflight_checks import run_checks,reject
from evaluate_p_structure import artifact_frames

RUN_ID='v18_pstate_nested_v1_split42_model42'


def require_manifest(saved,expected):
    if saved!=expected or digest_object(saved['identity'])!=saved['identity_hash']:
        raise ValueError('Resume identity mismatch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','cv'),default='preflight')
    parser.add_argument('--split-seed',choices=(42,),type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='preflight' and args.resume:parser.error('preflight cannot resume')
    ctx=load_context(args.split_seed)
    identity=experiment_identity(ctx);identity_hash=digest_object(identity)
    manifest={'run_id':RUN_ID,'identity':identity,'identity_hash':identity_hash}
    run=output_path(HERE/'runs'/RUN_ID)
    if args.stage=='preflight':
        checks,provenance=run_checks(ctx)
        require_manifest(manifest,manifest)
        reject(lambda:require_manifest({**manifest,'identity_hash':'wrong'},manifest))
        checks.update(identity_hash=identity_hash,resume_identity_rejection='PASS')
        write_json(output_path(HERE/'preflight/seed42.json'),checks)
        write_json(output_path(HERE/'preflight/fit_support_and_weights.json'),provenance)
        write_json(output_path(HERE/'reference/source_manifest.json'),{'source_sha256':ctx.sources,
            'baseline_identity_hash':CONFIG['baseline_identity_hash'],'D_G_identity_hash':CONFIG['g_identity_hash']})
        print(json.dumps(checks,indent=2));return
    preflight=read_json(HERE/'preflight/seed42.json')
    if preflight['status']!='PASS' or preflight['identity_hash']!=identity_hash:
        raise ValueError('Matching successful preflight required')
    if run.exists():
        if not args.resume or not (run/'run_manifest.json').exists():
            raise FileExistsError('Existing run requires matching --resume')
        require_manifest(read_json(run/'run_manifest.json'),manifest)
        if (run/'P_CV_COMPLETE').exists():raise FileExistsError('Run already frozen; use independent verifier')
    else:
        if args.resume:raise FileNotFoundError('Cannot resume absent run')
        run.mkdir(parents=True)
        write_json(run/'run_manifest.json',manifest)
        write_json(run/'source_manifest.json',read_json(HERE/'reference/source_manifest.json'))
        write_json(run/'environment.json',identity['environment'])
        write_frame(run/'cv_assignments.csv',ctx.data.assignment)
    with run_lock(run):
        attempts=read_json(run/'execution.json')['attempts'] if (run/'execution.json').exists() else []
        attempt={'start_utc':datetime.now(timezone.utc).isoformat(),'status':'running','fits_this_attempt':0,
            'command':[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]]}
        attempts.append(attempt);start=time.monotonic()
        def save():write_json(run/'execution.json',{'attempts':attempts})
        def progress(message,n):
            attempt['fits_this_attempt']+=n;save()
            print(f"[{attempt['fits_this_attempt']:02d}/75] {message}",flush=True)
        save()
        raw={k:np.full(len(ctx.meta),np.nan) for k in KINDS}
        ids={k:np.full(len(ctx.meta),'',object) for k in KINDS}
        records=[];rounds=[]
        try:
            for outer in range(5):
                for kind in KINDS:
                    model,receipt=get_or_fit(ctx,run,outer,kind,identity_hash,progress)
                    all_rows=ctx.valid&(ctx.data.row_folds==outer)
                    take=np.flatnonzero(all_rows&support(ctx.p,kind))
                    raw[kind][take]=model.predict(ctx.data.X_train.iloc[take],num_threads=1)
                    ids[kind][all_rows]=receipt['model_id']
                    records.append(receipt)
                    for probe,best in zip(receipt['spec']['probes'],receipt['fit']['best_iterations']):
                        rounds.append({'outer_fold':outer,'kind':kind,'target':TARGETS[kind],
                            'inner_fold':probe['inner_fold'],'best_iteration':best,
                            'refit_rounds':receipt['fit']['requested_rounds'],'actual_trees':receipt['fit']['actual_trees'],
                            'training_rows':probe['train']['rows'],'training_counties':len(probe['train']['counties']),
                            'validation_rows':probe['validation']['rows'],'weight_mean':probe['train']['weight_mean'],
                            'row_weight_ess':probe['train']['row_weight_ess'],'model_id':receipt['model_id']})
            frames=artifact_frames(ctx,raw,ids,identity_hash)
            frames['round_selection.csv']=pd.DataFrame(rounds)
            for name,frame in frames.items():write_frame(output_path(run/name),frame)
            write_json(run/'new_model_manifest.json',records)
            write_json(run/'fit_provenance.json',[r['spec'] for r in records])
            print('All 75 fits complete; starting independent verification in a new process.',flush=True)
            command=[sys.executable,'-B',str(HERE/'verify_p_structure.py'),'--split-seed','42','--unfrozen']
            result=subprocess.run(command,cwd=HERE,text=True,capture_output=True)
            (run/'logs/independent_verification_process.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError('Independent verifier failed: '+result.stderr[-2500:])
            report=read_json(run/'logs/independent_verification.json')
            if report['status']!='PASS' or report['identity_hash']!=identity_hash:
                raise ValueError('Independent verification identity mismatch')
            write_json(run/'verification.json',report)
            names=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name not in ('execution.json','.run.lock','P_CV_COMPLETE')]
            freeze_stage(run,'P_CV_COMPLETE',names,identity_hash)
            attempt.update(status='completed',independent_verification='PASS')
            print(json.dumps(report,indent=2),flush=True)
        except BaseException as exc:
            attempt.update(status='failed',error=repr(exc));raise
        finally:
            usage=resource.getrusage(resource.RUSAGE_SELF)
            attempt.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
                cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,max_rss_kib=usage.ru_maxrss)
            save()


if __name__=='__main__':main()
