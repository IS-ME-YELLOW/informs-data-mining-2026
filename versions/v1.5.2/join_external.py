"""Compatibility entry for the corrected v1.5.5 feature dataset.

The historical v1.5.2 cache stays frozen. Run the complete corrected pipeline
so a partial external join cannot be mislabeled as v1.5.5.
Usage: python versions/v1.5.2/join_external.py [--from-raw] [--overwrite]
"""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'v1.5.5/build_features.py'),
                  run_name='__main__')
