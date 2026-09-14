from pathlib import Path
import sys

VERSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VERSION_DIR.parents[0]))
from verify_bagging_tree_artifacts import verify  # noqa: E402

if __name__ == "__main__":
    verify(VERSION_DIR, "extratrees")
