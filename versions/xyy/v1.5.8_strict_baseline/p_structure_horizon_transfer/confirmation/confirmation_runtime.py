"""Explicit split selection; no default and no shared output path between CVs."""
from pathlib import Path
import argparse
import sys
sys.dont_write_bytecode=True

CONFIRMATION=Path(__file__).resolve().parent
PARENT=CONFIRMATION.parent
BASE=PARENT.parent
ROOT=BASE.parents[2]
parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--split-seed',type=int,choices=(20260917,20260918),required=True)
options,_=parser.parse_known_args()
SEED=options.split_seed
OUT=CONFIRMATION/f'split{SEED}'
CV_FILE=ROOT/f'cv/cv_assignments_balanced_v1_seed{SEED}.csv'
F2_ID={20260917:'64f8263f5356378124c5e981fcb7c1117ad808c97b46c383b64b7cdbe109484d',
       20260918:'a3f16d3463442c01b1e1315df498317eed53298a1f89ef087b016ab8bd0c9532'}[SEED]
