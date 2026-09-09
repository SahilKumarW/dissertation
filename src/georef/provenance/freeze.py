"""Experiment freeze and manifest (master spec sections 35 and 56).

Freezing is not a ceremony. It is the point after which any change to the
configuration invalidates the results, and the manifest is what lets someone
else check that nothing changed between the freeze and the run.

Every input to a prediction is hashed: the benchmark, the ground truth, the
LINZ snapshot, every configuration file, every source module, and the prompts.
The run manifest records the freeze hash, so a prediction file can be matched
to the exact state that produced it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from georef.provenance.hashing import environment_fingerprint, sha256_file, sha256_text

__all__ = ["build_freeze_manifest", "freeze_hash", "verify_against_freeze"]

CONFIG_FILES = (
    "benchmark.yaml",
    "linz.yaml",
    "models.yaml",
    "apis.yaml",
    "retrieval.yaml",
    "disambiguation.yaml",
    "pipeline.yaml",
    "condition_c.yaml",
    "paths.yaml",
)


def _hash_tree(root: Path, pattern: str = "*.py") -> dict:
    """Hash every source file, sorted, so the digest is order-independent.

    Keys use POSIX separators regardless of platform. `str(Path)` yields
    backslashes on Windows, which made a manifest written on one platform read
    as though every file had been deleted and replaced when verified on another.
    """
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob(pattern))
        if "__pycache__" not in path.parts
    }


def build_freeze_manifest(root: Path, extra: dict | None = None) -> dict:
    """Collect everything that must not change between freeze and benchmark."""
    from georef.llm.baseline import BASELINE_PROMPT, BASELINE_PROMPT_VERSION
    from georef.llm.extraction import PROMPT_VERSION, SYSTEM_PROMPT

    data_hashes: dict[str, str] = {}
    for relative in (
        "data/benchmark/benchmark.csv",
        "data/ground_truth/ground_truth.csv",
        "data/dev/dev_ground_truth.csv",
        "data/validation/validation.csv",
        "data/validation/validation_ground_truth.csv",
    ):
        path = root / relative
        if path.is_file():
            data_hashes[relative] = sha256_file(path)

    snapshots = sorted((root / "data" / "linz" / "normalised").glob("*.jsonl"))
    if snapshots:
        data_hashes[f"data/linz/normalised/{snapshots[-1].name}"] = sha256_file(snapshots[-1])

    index_manifest = root / "data" / "chroma" / "index_manifest.json"
    chroma = json.loads(index_manifest.read_text()) if index_manifest.is_file() else {}

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_hashes": data_hashes,
        "config_hashes": {
            name: sha256_file(root / "configs" / name)
            for name in CONFIG_FILES
            if (root / "configs" / name).is_file()
        },
        "code_hashes": _hash_tree(root / "src"),
        "prompts": {
            "extraction_version": PROMPT_VERSION,
            "extraction_sha256": sha256_text(SYSTEM_PROMPT),
            "baseline_version": BASELINE_PROMPT_VERSION,
            "baseline_sha256": sha256_text(BASELINE_PROMPT),
        },
        "chroma": {
            "collection_name": chroma.get("collection_name"),
            "record_count": chroma.get("record_count"),
            "embedding_model": chroma.get("embedding_model"),
            "embedding_dimension": chroma.get("embedding_dimension"),
            "snapshot_sha256": chroma.get("snapshot_sha256"),
        },
        "environment": environment_fingerprint(),
    }
    if extra:
        manifest.update(extra)
    manifest["freeze_sha256"] = freeze_hash(manifest)
    return manifest


def freeze_hash(manifest: dict) -> str:
    """One digest over everything except the timestamp and the digest itself.

    Excluding the timestamp is deliberate: re-running the freeze on an unchanged
    repository should produce the same hash, or the hash would say only that
    time had passed.
    """
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in ("frozen_at_utc", "freeze_sha256", "environment")
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def verify_against_freeze(root: Path, frozen: dict) -> list[str]:
    """Differences between the current state and a freeze manifest.

    Returns a list of human-readable differences; empty means nothing changed.
    Used before the benchmark runs, so a drifted configuration is caught before
    it produces results rather than after.
    """
    current = build_freeze_manifest(root)
    differences: list[str] = []
    for section in ("data_hashes", "config_hashes", "code_hashes", "prompts", "chroma"):
        frozen_section = frozen.get(section, {})
        current_section = current.get(section, {})
        for key in sorted(set(frozen_section) | set(current_section)):
            before, after = frozen_section.get(key), current_section.get(key)
            if before != after:
                if before is None:
                    differences.append(f"{section}/{key}: added since freeze")
                elif after is None:
                    differences.append(f"{section}/{key}: removed since freeze")
                else:
                    differences.append(f"{section}/{key}: changed since freeze")
    return differences
