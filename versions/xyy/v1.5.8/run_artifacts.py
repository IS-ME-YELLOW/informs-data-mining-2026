"""Atomic run artifacts and immutable input/code/split identities."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
import pandas as pd
from config import PROTOCOL, V18_DIR, PROJECT_ROOT, FEATURE_VERSION, COMPONENT_WEIGHTS, ZERO_THRESHOLD, OSI_MAX
from protocol import sha256_file, json_ready, MODES

CODE_FILES = ("config.py", "protocol.py", "scoped_training.py", "evaluation.py", "run_artifacts.py", "train_v18.py", "verify_artifacts.py")

def digest_object(obj):
    return hashlib.sha256(json.dumps(json_ready(obj), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def write_json(path, value):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n"))

def write_frame(path, frame):
    path = Path(path)
    if path.suffix == ".parquet":
        atomic_write(path, lambda tmp: frame.to_parquet(tmp, index=False, engine="pyarrow"))
    else:
        atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False))

def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": {n:importlib.metadata.version(n) for n in ("numpy", "pandas", "pyarrow", "lightgbm", "scipy", "scikit-learn")}}

def data_identity(data, settings):
    current_hashes = {k:sha256_file(p) for k,p in sorted(data.input_paths.items())}
    if data.loaded_hashes and current_hashes != data.loaded_hashes:
        raise ValueError("Inputs changed after loading; do not mix old frames with new file hashes")
    return {
        "protocol": PROTOCOL, "feature_version": FEATURE_VERSION,
        "inputs": current_hashes,
        "assignment": data.assignment.to_dict(orient="records"),
        "feature_names": list(data.X_train.columns), "settings": settings,
        "code_sha256": {n:sha256_file(V18_DIR/n) for n in CODE_FILES},
        "environment": environment(), "modes": MODES,
        "component_weights": COMPONENT_WEIGHTS, "post_process": [0., OSI_MAX, ZERO_THRESHOLD],
        "rule": {"t01h":"C3", "t06h":"C3", "t24h":"C1", "t48h":"C1"},
        "round_selection": "floor(mean(inner county-fold early-stopping iterations)); refit entire allowed scope",
    }

def prepare_run(run_dir, run_id, data, settings, resume=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",run_id):
        raise ValueError("Invalid run id")
    run_dir = Path(run_dir)
    identity = data_identity(data, settings)
    def portable(p):
        p=Path(p).resolve()
        return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
    manifest = {"run_id": run_id, "identity": identity, "identity_hash": digest_object(identity),
                "input_paths": {k:portable(p) for k,p in data.input_paths.items()}}
    if run_dir.exists():
        if not resume or not (run_dir/"run_manifest.json").is_file():
            raise FileExistsError("Run exists; only an identical, manifested --resume is allowed")
        existing = json.loads((run_dir/"run_manifest.json").read_text())
        if existing["run_id"] != run_id or existing["identity_hash"] != manifest["identity_hash"]:
            raise ValueError("Run identity differs (CV/data/code/settings/environment); cannot resume")
        if existing["input_paths"] != manifest["input_paths"]:
            raise ValueError("Resume input locations differ from the saved manifest")
        if digest_object(existing["identity"]) != existing["identity_hash"]:
            raise ValueError("Invalid run manifest identity")
        return existing
    if resume:
        raise FileNotFoundError("Cannot resume an absent run")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir/"run_manifest.json", manifest)
    write_json(run_dir/"environment.json", identity["environment"])
    write_json(run_dir/"input_manifest.json", {k:{"path":manifest["input_paths"][k],"sha256":v} for k,v in identity["inputs"].items()})
    write_frame(run_dir/"cv_assignments.csv", data.assignment)
    return manifest

@contextmanager
def run_lock(run_dir):
    """OS locks release on process death; no stale PID lock blocks resume."""
    with (Path(run_dir)/".run.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def safe_artifact(run_dir, relative):
    path = (Path(run_dir)/relative).resolve()
    if not path.is_relative_to(Path(run_dir).resolve()):
        raise ValueError("Artifact path escapes run directory")
    return path

def freeze_stage(run_dir, name, files, identity_hash):
    marker = {"identity_hash":identity_hash, "files": {str(p):sha256_file(safe_artifact(run_dir,p)) for p in sorted(set(files))}}
    path = Path(run_dir)/name
    if path.exists():
        if json.loads(path.read_text()) != marker:
            raise ValueError("Cannot replace a frozen stage")
    else:
        write_json(path, marker)

def check_stage(run_dir, name, identity_hash):
    marker = json.loads((Path(run_dir)/name).read_text())
    if marker["identity_hash"] != identity_hash:
        raise ValueError("Stage identity mismatch")
    for relative, digest in marker["files"].items():
        path = safe_artifact(run_dir,relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen artifact changed or missing: {relative}")
    return marker
