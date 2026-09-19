"""Select one authorized F2 confirmation split before importing numerical helpers."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode=True
CODE_ROOT=Path(__file__).resolve().parent
HERE=CODE_ROOT.parent
CONFIG_PATH=CODE_ROOT/'experiment_config.json'
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True)
args,_=p.parse_known_args();SEED=args.split_seed
CONFIG=json.loads(CONFIG_PATH.read_text())
assert SEED in CONFIG['authorized_split_seeds'] and CONFIG['trained_cases']==['F2']
CONFIG.update(next(s for s in CONFIG['splits'] if s['split_seed']==SEED))
assert HERE==Path(CONFIG['experiment_root']).resolve()
META_ROOT=CODE_ROOT/f'split{SEED}'
RUN_ID=f'v18_pinfo_v1_split{SEED}_model42'
# Keep confirmation modules before frozen parent modules in module resolution.
sys.path.append(str(HERE))
