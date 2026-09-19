"""Select one authorized CV split before importing the shared numerical modules."""
from pathlib import Path
import argparse
import json
import sys

CODE_ROOT=Path(__file__).resolve().parent
HERE=CODE_ROOT.parent
CONFIG_PATH=CODE_ROOT/'experiment_config.json'
_parser=argparse.ArgumentParser(add_help=False)
_parser.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True)
_args,_remaining=_parser.parse_known_args()
SEED=_args.split_seed
CONFIG=json.loads(CONFIG_PATH.read_text())
if SEED not in CONFIG['authorized_split_seeds']:
    raise ValueError('Split not authorized for this confirmation')
CONFIG.update(next(item for item in CONFIG['splits'] if item['split_seed']==SEED))
assert HERE==Path(CONFIG['experiment_root']).resolve()
assert CODE_ROOT==Path(CONFIG['confirmation_root']).resolve()
META_ROOT=CODE_ROOT/f'split{SEED}'
RUN_ID=f'v18_pstate_nested_v1_split{SEED}_model42'
# The five numerical/support modules are reused from the frozen seed42 source.
sys.path.append(str(HERE))
