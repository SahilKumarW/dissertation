"""Central configuration loading (master spec section 32).

Production code reads settings from configs/*.yaml rather than duplicating
literals. Nothing in this module has defaults that silently substitute for a
missing configuration file: a missing key is an error, not a fallback.
"""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["repo_root", "load_yaml", "load_benchmark_config", "load_paths_config"]


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict, failing loudly if it is absent or empty."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"configuration file is not a mapping: {path}")
    return data


def load_benchmark_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or repo_root() / "configs" / "benchmark.yaml")


def load_paths_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or repo_root() / "configs" / "paths.yaml")
