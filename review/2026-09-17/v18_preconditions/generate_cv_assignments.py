"""Freeze two repeated county CV assignments without reading outcomes or fitting.

Uses only the original assignment's county/state/severity table and the existing
pure allocation functions. Never overwrites an existing assignment with new bytes.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CV = ROOT / "cv"
SEEDS = (20260917, 20260918)  # fixed before any outcome diagnostics; never searched
REFERENCE = CV / "cv_assignments_balanced_v1_seed42.csv"
SOURCE = ROOT / "code_phase1/cv.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_cv_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    names = {"_county_table", "_validate_assignment", "_build_assignment"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(body) == len(names)
    namespace = {"np": np, "pd": pd, "REQUIRED_META_COLUMNS": ("fipsCode", "stateAbbr", "severity_tier")}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def write_once(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different frozen assignment: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(payload)


def main():
    original_bytes = REFERENCE.read_bytes()
    reference = pd.read_csv(REFERENCE, dtype={"fipsCode": str, "stateAbbr": str})
    funcs = pure_cv_functions()
    counties = funcs["_county_table"](reference)
    rebuilt = funcs["_build_assignment"](counties, 5, 42)
    pd.testing.assert_frame_equal(rebuilt, reference, check_dtype=False)
    assignments = {42: reference}
    for seed in SEEDS:
        frame = funcs["_build_assignment"](counties, 5, seed)
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        path = CV / f"cv_assignments_balanced_v1_seed{seed}.csv"
        write_once(path, payload)
        assignments[seed] = frame
    records = []
    partitions = []
    strata = []
    for seed, frame in assignments.items():
        funcs["_validate_assignment"](frame, counties, 5)
        partition = frozenset(frozenset(group.fipsCode) for _, group in frame.groupby("fold"))
        assert partition not in partitions, "Repeated partition up to fold-label permutation"
        partitions.append(partition)
        path = CV / f"cv_assignments_balanced_v1_seed{seed}.csv"
        counts = frame.groupby("fold").size()
        balance = frame.groupby(["stateAbbr", "severity_tier", "fold"]).size().unstack(fill_value=0)
        records.append({
            "seed": seed, "file": str(path.relative_to(ROOT)), "sha256": sha(path),
            "county_count": len(frame), "fold_counties": {int(k): int(v) for k, v in counts.items()},
            "max_stratum_fold_count_difference": int((balance.max(axis=1) - balance.min(axis=1)).max()),
            "role": "original_development_and_stress_reference" if seed == 42 else "frozen_confirmation_repeat",
        })
        count_table = balance.reset_index().melt(id_vars=["stateAbbr", "severity_tier"], var_name="fold", value_name="county_count")
        count_table.insert(0, "seed", seed)
        strata.append(count_table)
    comparisons = []
    for i, s1 in enumerate(assignments):
        for s2 in list(assignments)[i + 1:]:
            a = assignments[s1].set_index("fipsCode").sort_index().fold
            b = assignments[s2].set_index("fipsCode").sort_index().fold
            comparisons.append({"seed_a": s1, "seed_b": s2,
                                "adjusted_rand_index": float(adjusted_rand_score(a, b))})
    assert REFERENCE.read_bytes() == original_bytes
    registry_path = CV / "repeated_cv_manifest_2026-09-17.json"
    stable = {
        "protocol": "v18_repeated_balanced_county_cv_20260917",
        "experiment_anchor": "xyy v1.8; frozen v1.5.6 163-column features",
        "n_folds": 5, "new_seeds": list(SEEDS),
        "model_seed": 42,
        "selection_rule": "Exactly these two seeds, fixed before outcome diagnostics; no seed search, no reassignment of named counties.",
        "algorithm": "Existing code_phase1/cv.py _build_assignment; RandomState; balance state x severity and total county counts",
        "algorithm_sha256": sha(SOURCE), "generator_sha256": sha(Path(__file__)),
        "source_assignment_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_fields": ["fipsCode", "stateAbbr", "severity_tier"],
        "outcomes_used_in_generation": False,
        "severity_note": "Organizer-provided severity is label-derived and is used only for the expressly allowed stratification, not as model input.",
        "seed42_reproduced_exact_county_assignment": True,
        "assignments": records, "partition_comparisons": comparisons,
        "usage": "Keep seed42; compare identical candidate and baseline pipelines within each split. Run both frozen repeats for selected candidates and report both. Never merge old OOF into a new assignment.",
    }
    if registry_path.exists():
        previous = json.loads(registry_path.read_text())
        assert {k: v for k, v in previous.items() if k != "created_at_utc"} == stable
    else:
        payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **stable}
        write_once(registry_path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    pd.concat(strata, ignore_index=True).to_csv(OUT / "cv_stratum_counts.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "cv_partition_comparisons.csv", index=False)
    print(json.dumps({"assignments": records, "partition_comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
