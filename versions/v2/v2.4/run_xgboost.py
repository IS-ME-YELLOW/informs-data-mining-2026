"""Run v2.4: XGBoost components on frozen v1.5.6 features."""

import argparse
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from adapters import XGBoostAdapter  # noqa: E402
from component_experiment import run_component_experiment  # noqa: E402


FEATURE_VERSION = "v1.5.6"
MAX_BOOST_ROUNDS = 8_000
EARLY_STOPPING_ROUNDS = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run_component_experiment(
        XGBoostAdapter(),
        "v2.4",
        Path(__file__).resolve().parent,
        args.max_rounds,
        args.early_stopping_rounds,
        args.validate_only,
        feature_version=FEATURE_VERSION,
        direct_metadata_path=V2_ROOT.parent / "v1.9_xgboost" / "run_metadata.json",
    )
