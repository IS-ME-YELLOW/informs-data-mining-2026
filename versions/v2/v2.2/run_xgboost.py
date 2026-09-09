"""Run v2.2: XGBoost component prediction followed by OSI composition."""

import argparse
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from adapters import XGBoostAdapter  # noqa: E402
from component_experiment import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    run_component_experiment,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run_component_experiment(
        XGBoostAdapter(),
        "v2.2",
        Path(__file__).resolve().parent,
        args.max_rounds,
        args.early_stopping_rounds,
        args.validate_only,
    )
