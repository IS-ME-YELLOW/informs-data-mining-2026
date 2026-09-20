from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .config import LONG_HORIZONS, target_horizon


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_lf_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n")


def canonical_lf_sha256(path: Path) -> str:
    return sha256_bytes(canonical_lf_bytes(path.read_bytes()))


def digest_object(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(encoded)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


@contextmanager
def _temporary_module(name: str, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_baseline_protocol(project_root: Path):
    base = project_root / "versions" / "xyy" / "v1.5.8_strict_baseline"
    baseline_config = _load_module("stella_v112_baseline_config", base / "config.py")
    with _temporary_module("config", baseline_config):
        protocol = _load_module("stella_v112_baseline_protocol", base / "protocol.py")
    return baseline_config, protocol


def load_data(project_root: Path):
    baseline_config, protocol = load_baseline_protocol(project_root)
    data = protocol.load_data(
        project_root / "versions" / "xyy" / "v1.5.6",
        project_root / "cv" / "cv_assignments_balanced_v1_seed42.csv",
    )
    return data, baseline_config, protocol


def values_for_target(data, target: str) -> np.ndarray:
    horizon = target_horizon(target)
    if target == horizon:
        return data.y_train[target].to_numpy(dtype=float)
    return data.component_targets[target].to_numpy(dtype=float)


def expected_mask(protocol, meta, target_or_horizon: str) -> np.ndarray:
    horizon = target_or_horizon if target_or_horizon in LONG_HORIZONS else target_horizon(target_or_horizon)
    return np.asarray(protocol.expected_mask(meta, horizon), dtype=bool)


def row_key_hash(meta, rows: np.ndarray) -> str:
    frame = meta.iloc[np.asarray(rows, dtype=int)][["fipsCode", "timestamp_et", "hour_idx"]].copy()
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def scope_record(data, rows: np.ndarray) -> dict:
    rows = np.asarray(rows, dtype=int)
    counties = sorted(data.meta_train.iloc[rows].fipsCode.astype(str).unique().tolist())
    return {
        "rows": int(len(rows)),
        "counties": counties,
        "row_keys_hash": row_key_hash(data.meta_train, rows),
    }


def verify_handoff_sources(project_root: Path) -> dict:
    """Verify the moved handoff and Git-CRLF checkout against the frozen LF inventory."""
    handoff = project_root / "versions" / "stella_v112_handoff"
    inventory = json.loads((handoff / "source_inventory.json").read_text(encoding="utf-8"))
    groups = set(inventory["default_groups"])
    records = []
    counts: dict[str, int] = {}
    for entry in inventory["files"]:
        if entry["group"] not in groups:
            continue
        relative = entry["path"]
        path = project_root / relative
        mapping = "inventory_path"
        old_prefix = "versions/xyy/v1.5.8_strict_baseline/stella_v112_handoff/"
        if relative.startswith(old_prefix):
            path = handoff / relative.removeprefix(old_prefix)
            mapping = "moved_handoff"
        if not path.is_file():
            status = "missing"
            actual = canonical = None
        else:
            raw = path.read_bytes()
            actual = sha256_bytes(raw)
            canonical = sha256_bytes(canonical_lf_bytes(raw))
            if actual == entry["sha256"]:
                status = "exact"
            elif canonical == entry["sha256"]:
                status = "crlf_checkout_canonical_match"
            else:
                status = "mismatch"
        counts[status] = counts.get(status, 0) + 1
        records.append({
            "group": entry["group"],
            "inventory_path": relative,
            "resolved_path": str(path.relative_to(project_root)),
            "path_mapping": mapping,
            "expected_sha256": entry["sha256"],
            "raw_sha256": actual,
            "canonical_lf_sha256": canonical,
            "status": status,
        })
    failed = [record for record in records if record["status"] in {"missing", "mismatch"}]
    return {
        "status": "PASS" if not failed else "FAIL",
        "interpretation": "Exact bytes or a documented Git CRLF checkout whose LF-normalized bytes match the frozen inventory.",
        "groups": sorted(groups),
        "counts": counts,
        "failures": failed,
        "records": records,
    }
