from pathlib import Path
import sys

VERSION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VERSION_DIR.parents[0]))
from bagging_tree_experiment import cli  # noqa: E402

if __name__ == "__main__":
    cli(VERSION_DIR, "random_forest")
