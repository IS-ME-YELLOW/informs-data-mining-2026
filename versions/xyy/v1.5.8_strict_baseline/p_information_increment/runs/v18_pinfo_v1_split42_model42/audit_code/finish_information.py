"""Finalize the existing 100-fit run after the versioned independent audit; no fit API."""
from datetime import datetime,timezone
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from information_protocol import HERE,read_json,write_json,sha256_file
from run_artifacts import run_lock,freeze_stage


def main():
    run=HERE/'runs/v18_pinfo_v1_split42_model42'
    if (run/'INFO_CV_COMPLETE').exists():raise FileExistsError('Run is frozen; use verify_information_exact.py')
    with run_lock(run):
        execution=read_json(run/'execution.json')
        assert sum(a['fits_this_attempt'] for a in execution['attempts'])==100
        assert len(read_json(run/'new_model_manifest.json'))==20
        # Pin every model/receipt, input and prediction before running the audit.
        frozen_files={str(p.relative_to(run)):sha256_file(p) for p in run.rglob('*') if p.is_file() and p.suffix in ('.txt','.parquet','.csv')}
        frozen_files.update({str(p.relative_to(run)):sha256_file(p) for p in (run/'models').rglob('*.json')})
        for path in (HERE/'features/v1').rglob('*'):
            if path.is_file():frozen_files[str(path)]=sha256_file(path)
        note={'phase':'independent_audit_finalization','start_utc':datetime.now(timezone.utc).isoformat(),
              'status':'running','fits_this_attempt':0,'reason':'Correct independent float64 summation order; preserve all trained artifacts'}
        execution['attempts'].append(note);write_json(run/'execution.json',execution);start=time.monotonic()
        try:
            command=[sys.executable,'-B',str(HERE/'verify_information_exact.py'),'--split-seed','42','--unfrozen']
            result=subprocess.run(command,cwd=HERE,text=True,capture_output=True)
            (run/'logs/independent_verification_exact_process.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(result.stderr[-3000:])
            verified=read_json(run/'logs/independent_verification.json')
            assert verified['status']=='PASS' and verified['feature_verification']['augmented_features_bitwise_exact']
            for name,h in frozen_files.items():
                path=Path(name) if Path(name).is_absolute() else run/name
                assert sha256_file(path)==h,name
            snapshot=read_json(HERE/'reference/initial_workspace_snapshot.json')
            changed=[p for p,h in snapshot['sha256'].items() if not Path(p).is_file() or sha256_file(p)!=h]
            if changed:raise ValueError('Historical source changes: '+str(changed))
            write_json(run/'historical_integrity.json',{'status':'PASS','checked_files':len(snapshot['sha256']),'changed_files':[]})
            write_json(run/'verification.json',verified)
            write_json(run/'audit_finalization.json',{'status':'PASS','audit_identity_hash':verified['audit_identity_hash'],
                'unchanged_fit_feature_prediction_artifacts':frozen_files,'extra_fits':0,
                'preserved_initial_failure_log':'logs/independent_verification_process.log'})
            (run/'audit_code').mkdir(exist_ok=True)
            shutil.copyfile(HERE/'verify_information_exact.py',run/'audit_code/verify_information_exact.py')
            shutil.copyfile(HERE/'finish_information.py',run/'audit_code/finish_information.py')
            names=[str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and p.name not in ('execution.json','.run.lock','INFO_CV_COMPLETE')]
            freeze_stage(run,'INFO_CV_COMPLETE',names,verified['identity_hash'])
            note.update(status='completed',independent_verification='PASS',audit_identity_hash=verified['audit_identity_hash'])
            print(json.dumps(verified,indent=2))
        except BaseException as exc:
            note.update(status='failed',error=repr(exc));raise
        finally:
            note.update(end_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
            write_json(run/'execution.json',execution)


if __name__=='__main__':main()
