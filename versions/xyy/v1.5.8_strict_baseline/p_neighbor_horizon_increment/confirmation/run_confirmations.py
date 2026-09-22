"""Run only P24 on the two explicitly authorized confirmation CVs."""
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def main():
    for seed in (20260917,20260918):
        run=HERE/f'split{seed}/runs/v18_pneighbor_v1_split{seed}_model42'
        if not (run/'MODEL_CV_COMPLETE').exists():
            cmd=[sys.executable,'-B',str(HERE/'train_neighbors.py'),'--stage','train','--split-seed',str(seed)]
            if (run/'run_manifest.json').exists():cmd.append('--resume')
            subprocess.run(cmd,check=True)
        if not (run/'EVALUATION_COMPLETE').exists():subprocess.run([sys.executable,'-B',str(HERE/'evaluate_neighbors.py'),'--split-seed',str(seed)],check=True)
        if not (run/'VERIFIED_COMPLETE').exists():subprocess.run([sys.executable,'-B',str(HERE/'verify_neighbors.py'),'--split-seed',str(seed)],check=True)
        subprocess.run([sys.executable,'-B',str(HERE/'analyze_confirmation.py'),'--split-seed',str(seed)],check=True)
    print('Both P24 confirmations completed',flush=True)


if __name__=='__main__':main()
