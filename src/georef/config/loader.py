"""Central configuration loading (master spec section 32).

Production code reads settings from configs/*.yaml rather than duplicating
literals. Nothing in this module has defaults that silently substitute for a
missing configuration file: a missing key is an error, not a fallback.
"""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = [
    "repo_root",
    "load_yaml",
    "load_benchmark_config",
    "load_paths_config",
    "resolve_llm_profile",
]


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


def resolve_llm_profile(models_config: dict, profile: str | None = None) -> dict:
    """Return ``models_config`` with ``llm`` set to the chosen profile.

    Profiles exist so an unresponsive endpoint does not stop development work.
    They are NOT an experimental variable: the frozen benchmark runs both
    conditions on one profile, and the profile name travels with every record so
    a mixed output file is detectable rather than silent.
    """
    profiles = models_config.get("llm_profiles")
    if not profiles:
        return models_config  # older config with a single inline llm block

    name = profile or models_config.get("active_llm_profile")

    if name not in profiles:
        # Accept a provider name as shorthand, so switching endpoints on the
        # command line is "--profile gemini" rather than a profile id to
        # memorise. Ambiguous shorthand is refused rather than guessed.
        matches = sorted(k for k, v in profiles.items() if v.get("provider") == name)
        if len(matches) == 1:
            name = matches[0]
        elif len(matches) > 1:
            default = models_config.get(f"default_profile_for_{name}")
            if default in matches:
                name = default
            else:
                raise ValueError(
                    f"{name!r} matches several profiles: {matches}. Name one, or set "
                    f"default_profile_for_{name} in configs/models.yaml."
                )
        else:
            raise ValueError(
                f"unknown LLM profile {name!r}; available: {sorted(profiles)}"
            )
    resolved = dict(models_config)
    resolved["llm"] = {**profiles[name], "profile": name}
    return resolved


def load_benchmark_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or repo_root() / "configs" / "benchmark.yaml")


def load_paths_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or repo_root() / "configs" / "paths.yaml")
