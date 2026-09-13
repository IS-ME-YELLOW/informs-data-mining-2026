"""Run the v1.9 direct-OSI XGBoost baseline on frozen v1.5.6 features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parent
VERSIONS_DIR = VERSION_DIR.parent
V2_DIR = VERSIONS_DIR / "v2"
for path in (VERSIONS_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters import XGBoostAdapter as ComponentXGBoostAdapter  # noqa: E402
from gbdt_experiment import PROJECT_ROOT, run_experiment  # noqa: E402


FEATURE_VERSION = "v1.5.6"
MAX_BOOST_ROUNDS = 4_000
EARLY_STOPPING_ROUNDS = 100


class XGBoostAdapter(ComponentXGBoostAdapter):
    version = "v1.9"

    @staticmethod
    def fit_cv(model, X_train, y_train, X_valid, y_valid) -> None:
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument(
        "--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        XGBoostAdapter(),
        VERSION_DIR,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        validate_only=args.validate_only,
        feature_version=FEATURE_VERSION,
    )
