"""Read-only, standard-library file transfer check. Does not import/train models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def under(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise ValueError(f"Nonportable relative path: {relative}")
    result = root.joinpath(*rel.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Path leaves project root: {relative}")
    return result


def inspect(root: Path, inventory: dict, groups: set[str]) -> dict:
    errors, passed = [], Counter()
    entries = inventory["files"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate inventory paths")
    unknown = groups - {entry["group"] for entry in entries}
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")
    selected = {e["path"]: e for e in entries if e["group"] in groups}
    for relative, entry in selected.items():
        path = under(root, relative)
        if not path.is_file():
            errors.append({"path": relative, "issue": "missing"})
            continue
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            errors.append({"path": relative, "issue": "bytes_or_sha256_mismatch"})
            continue
        passed[entry["group"]] += 1

    identity_checks, baseline_markers, fixed_f2_outputs = [], [], []
    for run in inventory["reference_runs"]:
        if run["manifest"] not in selected:
            continue
        path = under(root, run["manifest"])
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(json.dumps(
                manifest["identity"], sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode()).hexdigest()
            if not digest == manifest["identity_hash"] == run["identity_hash"]:
                raise ValueError("saved identity digest differs")
            identity_checks.append(run["run_id"])
            # Full baseline run is supplied. F2 is deliberately a pinned output only.
            if run["kind"] == "fixed_F2_output":
                marker = json.loads((path.parent / "INFO_CV_COMPLETE").read_text(encoding="utf-8"))
                if marker["identity_hash"] != digest:
                    raise ValueError("F2 completion identity differs")
                if sha256(path.parent / "control_predictions.parquet") != marker["files"]["control_predictions.parquet"]:
                    raise ValueError("F2 fixed control output differs from its completion record")
                fixed_f2_outputs.append(run["run_id"])
                continue
            if run["kind"] != "strict_baseline":
                continue
            run_dir = path.parent
            marker_path = run_dir / "CV_COMPLETE"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker["identity_hash"] != digest:
                raise ValueError("baseline completion identity differs")
            for rel, expected_hash in marker["files"].items():
                target = under(run_dir, rel)
                project_rel = target.relative_to(root).as_posix()
                if project_rel not in selected:
                    raise ValueError(f"baseline marker member absent from selected inventory: {rel}")
                if not target.is_file() or sha256(target) != expected_hash:
                    raise ValueError(f"baseline marker member differs: {rel}")
            baseline_markers.append(run["run_id"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": run["manifest"], "issue": str(exc)})

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "handoff_source_bytes_and_saved_identities_only",
        "groups": sorted(groups), "selected_files": len(selected),
        "passed_files_by_group": dict(sorted(passed.items())),
        "saved_identity_digests_checked": identity_checks,
        "baseline_completion_file_hashes_checked": baseline_markers,
        "f2_control_output_completion_entry_checked": fixed_f2_outputs,
        "errors": errors,
        "training_performed": False,
        "models_reloaded": False,
        "migrated_experiment_accepted": False,
        "limitations": [
            "No model training, inference, leakage test or score recomputation.",
            "F2 is a pinned previously validated output; its complete upstream model chain is not checked.",
            "Input/code identity and fresh-run acceptance still require the new experiment verifier.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--include-confirmation", action="store_true")
    args = parser.parse_args()
    inventory = json.loads(Path(__file__).with_name("source_inventory.json").read_text(encoding="utf-8"))
    groups = set(inventory["default_groups"])
    if args.include_confirmation:
        groups.add("confirmation")
    result = inspect(args.project_root.resolve(), inventory, groups)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
