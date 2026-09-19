from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bootstrap import activate_local_packages


def main() -> None:
    parser = argparse.ArgumentParser(description="Stella v1.12 strict nested-CV experiment")
    parser.add_argument("command", choices=("preflight", "train", "assemble", "verify", "all"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", default="stella_v112_nested_v1_split42_model42")
    args = parser.parse_args()

    project_root = activate_local_packages(args.project_root)
    experiment_root = Path(__file__).resolve().parent
    if args.command == "preflight":
        from src.preflight import run_preflight

        result = run_preflight(project_root, experiment_root)
    else:
        from src.pipeline import run_command

        result = run_command(args.command, project_root, experiment_root, args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

