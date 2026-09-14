"""Build or verify v1.5.6 feature data: v1.5.5 plus land-cover sensitivity."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code_phase1.feature_dataset_v156 import build_dataset, load_feature_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace v1.5.6 only; older versions stay frozen.')
    parser.add_argument('--verify', action='store_true', help='Read and validate existing output; no writes.')
    args = parser.parse_args()
    if args.verify and args.overwrite:
        parser.error('--verify cannot be combined with --overwrite.')
    if args.verify:
        data = load_feature_dataset()
        print(f'Verified v1.5.6: train={data.X_train.shape}, test={data.X_test.shape}')
    else:
        build_dataset(overwrite=args.overwrite)


if __name__ == '__main__':
    main()
