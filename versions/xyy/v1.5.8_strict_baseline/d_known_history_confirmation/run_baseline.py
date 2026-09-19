"""Logged CV-only launcher for exactly the two frozen confirmation splits."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", required=True, type=int, choices=(20260917, 20260918))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    plan = json.loads((HERE/"confirmation_plan.json").read_text())
    item = next(row for row in plan["splits"] if row["split_seed"] == args.split_seed)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(ROOT/item["cv_file"]) == item["cv_sha256"]
    for name, digest in plan["baseline_code_sha256"].items():
        assert sha(BASE/name) == digest, name
    run = Path(item["baseline_dir"])
    if (run/"CV_COMPLETE").exists():
        raise FileExistsError("CV already frozen; use the independent verifier")
    command = [sys.executable, "-B", "-u", str(BASE/"train_v18.py"), "--stage", "cv",
               "--cv-file", item["cv_file"], "--model-seed", "42", "--run-id", item["run_id"]]
    if args.resume:
        command.append("--resume")
    state = HERE/f"split{args.split_seed}"/"execution.json"
    attempts = json.loads(state.read_text())["attempts"] if state.exists() else []
    if attempts and not args.resume:
        raise FileExistsError("An execution attempt exists; review it and use --resume")
    attempt = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "launcher_pid": os.getpid()}
    attempts.append(attempt)
    def save():
        temp = state.with_suffix(".pending.json")
        temp.write_text(json.dumps({"split_seed": args.split_seed, "run_id": item["run_id"], "attempts": attempts}, indent=2)+"\n")
        temp.replace(state)
    save()
    start = time.monotonic()
    with (Path(item["logs_dir"])/"train_cv.log").open("a", encoding="utf-8") as log:
        log.write(f"\nStart {attempt['started_at_utc']}\n"); log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        attempt["child_pid"] = process.pid; save()
        for line in process.stdout:
            log.write(line); log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt.update(status="completed" if code == 0 else "failed", return_code=code,
                   ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-start,
                   cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                   max_rss_kib=usage.ru_maxrss, cv_complete=(run/"CV_COMPLETE").exists())
    save()
    print(json.dumps(attempt, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
