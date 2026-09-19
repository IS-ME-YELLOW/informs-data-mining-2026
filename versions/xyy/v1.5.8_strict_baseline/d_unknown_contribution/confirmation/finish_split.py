"""Fresh-process verification and summary for an already completed confirmation CV."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-seed',type=int,required=True,choices=(20260917,20260918))
    args=parser.parse_args();folder=HERE/f'split{args.split_seed}'
    run=folder/'runs'/f'v18_ud01_nested_v1_split{args.split_seed}_model42'
    marker=run/'U_CV_COMPLETE'
    if not marker.exists():raise FileNotFoundError('Training and internal verification are not complete')
    if (folder/'completion.json').exists():
        saved=json.loads((folder/'completion.json').read_text())
        for name,digest in saved['files'].items():assert sha(folder/name)==digest,name
        print('Completed split remains intact:',args.split_seed);return
    (run/'logs').mkdir(exist_ok=True)
    for script,log in [('verify_unknown_d.py',run/'logs/independent_verification.json'),
                       ('analyze_unknown_d.py',folder/'logs/analysis.log')]:
        print(f'split{args.split_seed}: {script}',flush=True)
        with log.open('w') as handle:
            result=subprocess.run([sys.executable,'-B','-u',str(folder/script)],cwd=folder,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Failed step: inspect {log}')
    verified=json.loads((run/'logs/independent_verification.json').read_text())
    assert verified['status']=='PASS' and verified['new_models_reloaded']==5 and verified['split_seed']==args.split_seed
    summary=json.loads((folder/f'summary/split{args.split_seed}/summary_manifest.json').read_text())
    assert summary['status']=='PASS'
    files={str(f.relative_to(folder)):sha(f) for f in sorted(folder.rglob('*')) if f.is_file() and f!=folder/'completion.json'}
    completion={'status':'COMPLETE','split_seed':args.split_seed,'identity_hash':verified['identity_hash'],
                'completed_at_utc':datetime.now(timezone.utc).isoformat(),'formal_fits':25,'new_models':5,'files':files}
    (folder/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps({k:v for k,v in completion.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':main()
