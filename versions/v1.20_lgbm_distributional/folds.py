"""Generate and validate county-isolated primary and repeat folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from distributional_config import (
    FOLDS_DIR, N_FOLDS, PRIMARY_FOLD_FILE, PRIMARY_SPLIT, REPEAT_SEEDS,
    ensure_output_dirs, stable_fingerprint,
)
from data_contract import load_inputs, normalize_fips, sha256_file

REQUIRED_COLUMNS = ("fipsCode", "stateAbbr", "severity_tier", "fold")


def repeat_name(seed: int) -> str:
    return f"repeat_seed{seed}"


def repeat_path(seed: int) -> Path:
    return FOLDS_DIR / f"cv_assignments_state_severity_balanced_seed{seed}.csv"


def _county_table(meta: pd.DataFrame) -> pd.DataFrame:
    counties = meta[["fipsCode", "stateAbbr", "severity_tier"]].copy()
    counties["fipsCode"] = normalize_fips(counties["fipsCode"])
    counties["stateAbbr"] = counties["stateAbbr"].astype(str)
    counties["severity_tier"] = pd.to_numeric(counties["severity_tier"], errors="raise").astype(int)
    uniqueness = counties.groupby("fipsCode")[["stateAbbr", "severity_tier"]].nunique()
    if (uniqueness > 1).any().any():
        raise ValueError("County stratification fields vary between rows")
    return counties.drop_duplicates().sort_values("fipsCode").reset_index(drop=True)


def generate_repeat_folds(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    counties = _county_table(meta)
    rng = np.random.default_rng(seed)
    assignments: list[dict] = []
    fold_sizes = np.zeros(N_FOLDS, dtype=int)
    strata = counties.groupby(["stateAbbr", "severity_tier"], sort=True)
    for (_, _), group in strata:
        indices = rng.permutation(len(group))
        ordered = group.iloc[indices]
        # Start each stratum at a randomly chosen currently-smallest fold, then
        # greedily preserve global county balance while spreading the stratum.
        stratum_counts = np.zeros(N_FOLDS, dtype=int)
        for _, row in ordered.iterrows():
            score = np.column_stack((stratum_counts, fold_sizes, rng.random(N_FOLDS)))
            fold = int(np.lexsort((score[:, 2], score[:, 1], score[:, 0]))[0])
            assignments.append({**row.to_dict(), "fold": fold})
            stratum_counts[fold] += 1
            fold_sizes[fold] += 1
    result = pd.DataFrame(assignments)[list(REQUIRED_COLUMNS)].sort_values("fipsCode").reset_index(drop=True)
    validate_fold_frame(result, meta)
    return result


def validate_fold_frame(frame: pd.DataFrame, meta: pd.DataFrame) -> dict:
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Fold columns must be exactly {REQUIRED_COLUMNS}")
    actual = frame.copy()
    actual["fipsCode"] = normalize_fips(actual["fipsCode"])
    actual["stateAbbr"] = actual["stateAbbr"].astype(str)
    actual["severity_tier"] = pd.to_numeric(actual["severity_tier"], errors="raise").astype(int)
    actual["fold"] = pd.to_numeric(actual["fold"], errors="raise").astype(int)
    expected = _county_table(meta).set_index("fipsCode").sort_index()
    indexed = actual.set_index("fipsCode").sort_index()
    if indexed.index.has_duplicates or not indexed.index.equals(expected.index):
        raise ValueError("Fold assignment county set is duplicated or incomplete")
    if not indexed[["stateAbbr", "severity_tier"]].equals(expected[["stateAbbr", "severity_tier"]]):
        raise ValueError("Fold assignment stratification fields differ from metadata")
    if set(indexed["fold"]) != set(range(N_FOLDS)):
        raise ValueError("Fold IDs are incomplete")
    counts = indexed["fold"].value_counts().sort_index()
    if counts.max() - counts.min() > 1:
        raise ValueError(f"County fold sizes are not balanced: {counts.to_dict()}")
    row_folds = normalize_fips(meta["fipsCode"]).map(indexed["fold"])
    if row_folds.isna().any():
        raise ValueError("Training rows are missing fold assignments")
    for fold in range(N_FOLDS):
        train_counties = set(indexed.index[indexed["fold"] != fold])
        valid_counties = set(indexed.index[indexed["fold"] == fold])
        if train_counties & valid_counties:
            raise ValueError(f"County leakage in fold {fold}")
    canonical = actual.sort_values("fipsCode").to_dict(orient="records")
    return {
        "counties": len(actual),
        "fold_counts": {str(k): int(v) for k, v in counts.items()},
        "assignment_fingerprint": stable_fingerprint(canonical),
    }


def load_fold_assignments(split: str, meta: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, Path]:
    if split == PRIMARY_SPLIT:
        path = PRIMARY_FOLD_FILE
    elif split.startswith("repeat_seed"):
        seed = int(split.removeprefix("repeat_seed"))
        if seed not in REPEAT_SEEDS:
            raise ValueError(f"Unsupported repeat seed: {seed}")
        path = repeat_path(seed)
    else:
        raise ValueError(f"Unknown split: {split}")
    if not path.exists():
        raise FileNotFoundError(f"Missing fold assignment: {path}; run folds.py first")
    frame = pd.read_csv(path, dtype={"fipsCode": str, "stateAbbr": str})
    validate_fold_frame(frame, meta)
    mapping = frame.assign(fipsCode=normalize_fips(frame["fipsCode"])).set_index("fipsCode")["fold"]
    rows = normalize_fips(meta["fipsCode"]).map(mapping).to_numpy(dtype=int)
    return frame, rows, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    inputs = load_inputs(require_v112=False)
    records = []
    primary = pd.read_csv(PRIMARY_FOLD_FILE, dtype={"fipsCode": str, "stateAbbr": str})
    records.append({"split": PRIMARY_SPLIT, "path": str(PRIMARY_FOLD_FILE), **validate_fold_frame(primary, inputs.data.meta_train), "sha256": sha256_file(PRIMARY_FOLD_FILE)})
    for seed in REPEAT_SEEDS:
        path = repeat_path(seed)
        generated = generate_repeat_folds(inputs.data.meta_train, seed)
        if path.exists():
            existing = pd.read_csv(path, dtype={"fipsCode": str, "stateAbbr": str})
            if not existing.equals(generated):
                raise ValueError(f"Existing repeat fold differs from deterministic generation: {path}")
        elif not args.validate_only:
            generated.to_csv(path, index=False)
        else:
            raise FileNotFoundError(f"Repeat fold has not been generated: {path}")
        records.append({"split": repeat_name(seed), "path": str(path), **validate_fold_frame(generated, inputs.data.meta_train), "sha256": sha256_file(path) if path.exists() else None})
    manifest = FOLDS_DIR / "fold_manifest.json"
    if not args.validate_only:
        manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
