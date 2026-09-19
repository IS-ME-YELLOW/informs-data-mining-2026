"""Logged launcher for the one authorized seed42 strict CV experiment."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN_ID = "v18_tree_nested_v2_split42_model42"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-B", "-u", str(HERE / "train_v18.py"),
               "--stage", "cv", "--cv-file", "cv/cv_assignments_balanced_v1_seed42.csv",
               "--model-seed", "42", "--run-id", RUN_ID]
    if args.resume:
        command.append("--resume")
    path = HERE / "execution.json"
    attempts = json.loads(path.read_text()).get("attempts", []) if path.exists() else []
    attempt = {"command": command, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "stage": "cv"}
    attempts.append(attempt)
    def save():
        path.write_text(json.dumps({"run_id": RUN_ID, "attempts": attempts}, ensure_ascii=False, indent=2)+"\n")
    save()
    start = time.monotonic()
    with (HERE / "logs" / "train_cv.log").open("a", encoding="utf-8") as log:
        log.write(f"\nStart {attempt['started_at_utc']}\n"); log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, cwd=HERE.parents[2])
        attempt["pid"] = process.pid; save()
        for line in process.stdout:
            log.write(line); log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    attempt.update(status="completed" if code == 0 else "failed", return_code=code,
                   ended_at_utc=datetime.now(timezone.utc).isoformat(),
                   wall_seconds=time.monotonic()-start, cpu_user_seconds=usage.ru_utime,
                   cpu_system_seconds=usage.ru_stime, max_rss_kib=usage.ru_maxrss,
                   cv_complete=(HERE / "runs" / RUN_ID / "CV_COMPLETE").exists())
    save()
    print(json.dumps(attempt, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
