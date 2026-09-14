"""Command-line orchestration for the v1.20 experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data_contract import load_inputs, write_input_manifest
from distributional_config import MANIFESTS_DIR, PRIMARY_SPLIT, REPEAT_SEEDS, ensure_output_dirs
from folds import repeat_name

HERE = Path(__file__).resolve().parent


def call(script: str, *arguments: str) -> None:
    command = [sys.executable, str(HERE / script), *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train-primary")
    train.add_argument("--num-threads", type=int, default=-1)
    train.add_argument("--quick", action="store_true")
    train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate-primary")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2000)
    repeats = sub.add_parser("train-repeats")
    repeats.add_argument("--num-threads", type=int, default=-1)
    repeats.add_argument("--quick", action="store_true")
    repeats.add_argument("--resume", action="store_true")
    sub.add_parser("evaluate-repeats")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--num-threads", type=int, default=-1)
    finalize.add_argument("--quick", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--split", default=PRIMARY_SPLIT)
    verify.add_argument("--model-sample", type=int)
    verify.add_argument("--allow-missing-v112", action="store_true")
    smoke = sub.add_parser("dry-run")
    smoke.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    ensure_output_dirs()

    if args.command == "prepare":
        call("folds.py")
        inputs = load_inputs(require_v112=False)
        write_input_manifest(inputs, MANIFESTS_DIR / "input_manifest.json")
    elif args.command == "train-primary":
        options = ["--split", PRIMARY_SPLIT, "--num-threads", str(args.num_threads)]
        if args.quick: options.append("--quick")
        if args.resume: options.append("--resume")
        call("train_distributional.py", *options)
    elif args.command == "evaluate-primary":
        call("evaluate_and_promote.py", "primary", "--bootstrap-replicates", str(args.bootstrap_replicates))
    elif args.command == "train-repeats":
        for seed in REPEAT_SEEDS:
            options = ["--split", repeat_name(seed), "--num-threads", str(args.num_threads), "--primary-winner-only"]
            if args.quick: options.append("--quick")
            if args.resume: options.append("--resume")
            call("train_distributional.py", *options)
    elif args.command == "evaluate-repeats":
        call("evaluate_and_promote.py", "repeats")
    elif args.command == "finalize":
        options = ["finalize", "--num-threads", str(args.num_threads)]
        if args.quick: options.append("--quick")
        call("evaluate_and_promote.py", *options)
    elif args.command == "verify":
        options = ["--split", args.split]
        if args.model_sample is not None: options += ["--model-sample", str(args.model_sample)]
        if args.allow_missing_v112: options.append("--allow-missing-v112")
        call("verify_artifacts.py", *options)
    else:
        call("folds.py")
        inputs = load_inputs(require_v112=False)
        write_input_manifest(inputs, MANIFESTS_DIR / "input_manifest.json")
        call("train_distributional.py", "--split", PRIMARY_SPLIT, "--num-threads", str(args.num_threads), "--quick", "--dry-run")
        call("verify_artifacts.py", "--split", PRIMARY_SPLIT, "--model-sample", "0", "--allow-missing-v112")


if __name__ == "__main__":
    main()
