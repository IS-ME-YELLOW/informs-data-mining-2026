"""Atomic model-level completion records for interrupted runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

RECORD_VERSION = 2


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_path(path):
    path = Path(path)
    return path.with_name(path.name + ".complete.json")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_record(path, run_identity, model_id=None):
    path = Path(path)
    receipt = record_path(path)
    if not receipt.exists():
        return None
    record = json.loads(receipt.read_text(encoding="utf-8"))
    if record.get("record_version") != RECORD_VERSION or record.get("run_identity") != run_identity:
        raise ValueError(f"model completion identity mismatch: {receipt}")
    if record.get("model_file") != path.name or (model_id is not None and record.get("model_id") != model_id):
        raise ValueError(f"model completion scope/path mismatch: {receipt}")
    if not path.is_file() or record.get("sha256") != file_hash(path):
        raise ValueError(f"completed model missing or corrupt: {path}")
    return record


def publish_model(path, run_identity, model_id, details, writer, validator):
    """The completion JSON is the commit point; orphan model files are ignored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if record_path(path).exists():
        raise FileExistsError(f"completed model is immutable: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary)
    try:
        writer(temporary)
        validator(temporary)
        checksum = file_hash(temporary)
        os.replace(temporary, path)
        atomic_json(record_path(path), {"record_version": RECORD_VERSION, "run_identity": run_identity,
                    "model_id": model_id, "model_file": path.name, "sha256": checksum, "details": details})
    finally:
        if temporary.exists():
            temporary.unlink()


def completed_paths(directory):
    return [Path(str(path)[:-len(".complete.json")])
            for path in sorted(Path(directory).glob("*.complete.json"))]
