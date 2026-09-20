from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def activate_local_packages(root: Path | None = None) -> Path:
    """Use the repository-pinned tree libraries without mutating any environment."""
    root = (root or project_root()).resolve()
    package_dir = root / ".python_packages"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"Missing local package directory: {package_dir}")
    package_text = str(package_dir)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    return root

