"""Verify a completed baseline, project D, and independently verify its candidate."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, required=True, choices=(20260917, 20260918))
    args = parser.parse_args()
    plan = json.loads((HERE/"confirmation_plan.json").read_text())
    split = next(row for row in plan["splits"] if row["split_seed"] == args.split_seed)
    run, logs, projection = [Path(split[key]) for key in ("baseline_dir", "logs_dir", "projection_dir")]
    if not (run/"CV_COMPLETE").exists():
        raise FileNotFoundError("Baseline CV is not complete")
    completion = logs.parent/"completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        for name, digest in saved["files"].items():
            assert sha(logs.parent/name) == digest, name
        print("Completed split artifacts remain intact", args.split_seed)
        return
    steps = [([str(BASE/"verify_artifacts.py"), str(run)], "baseline_verification.json"),
             ([str(HERE/"run_projection.py"), "--split-seed", str(args.split_seed)], "projection_evaluation.log"),
             ([str(HERE/"verify_projection.py"), "--split-seed", str(args.split_seed)], "projection_verification.log")]
    for command, name in steps:
        print(f"split{args.split_seed}: {name}", flush=True)
        with (logs/name).open("w") as handle:
            completed = subprocess.run([sys.executable, "-B", "-u", *command], cwd=ROOT,
                                       stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Step failed; inspect {logs/name}")
    baseline_check = json.loads((logs/"baseline_verification.json").read_text())["cv"]
    candidate_check = json.loads((projection/"verification.json").read_text())
    assert baseline_check["status"] == "passed" and baseline_check["models_reloaded"] == 100
    assert candidate_check["status"] == "PASS"
    files = [p for p in projection.iterdir() if p.is_file()] + [logs/name for _, name in steps]
    record = {"status":"COMPLETE", "split_seed":args.split_seed, "baseline_run":str(run),
              "baseline_identity_hash":json.loads((run/"run_manifest.json").read_text())["identity_hash"],
              "candidate_identity_hash":candidate_check["candidate_identity_hash"],
              "completed_at_utc":datetime.now(timezone.utc).isoformat(),
              "models_reloaded_independently":100, "projection_independent_verification":"PASS",
              "files":{str(p.relative_to(logs.parent)):sha(p) for p in files}}
    completion.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
