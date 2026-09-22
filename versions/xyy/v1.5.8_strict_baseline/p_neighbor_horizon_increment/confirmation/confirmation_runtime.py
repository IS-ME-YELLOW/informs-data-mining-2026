"""Explicit confirmation split; frozen feature package shared read-only."""
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
args,_=parser.parse_known_args()
SEED=args.split_seed
OUT=CONFIRMATION/f'split{SEED}'
CV_FILE=ROOT/f'cv/cv_assignments_balanced_v1_seed{SEED}.csv'
REFERENCE_ID={20260917:'2effe8d5a50b0f6c902d038615336cc2a51e8472412c1faf929dfd79eb15c798',
              20260918:'399da73952d32a3e333a9933a0ed1a57a1ed39eada5477cdc3fa80c9eb56f3d4'}[SEED]
