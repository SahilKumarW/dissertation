"""Hashing and environment capture for reproducibility (master spec section 34)."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

__all__ = ["sha256_file", "sha256_text", "environment_fingerprint"]

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment_fingerprint() -> dict:
    """Software versions relevant to the reproducibility of data artefacts."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("pandas", "numpy", "openpyxl", "yaml"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = "not installed"
        else:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions
