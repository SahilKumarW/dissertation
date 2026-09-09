"""LINZ ingestion: raw fetch to frozen, hashed snapshot.

    LINZ WFS -> raw pages -> validation -> normalisation -> snapshot -> hash

Everything downstream (embedding, ChromaDB, grounding, disambiguation) reads
the snapshot, never the live service. The snapshot is the authoritative
geographic source of record for a given experiment version, and its hash is
what makes "we grounded against LINZ" a checkable claim rather than an
assertion.

Rejections are counted and reported by reason. A record is never silently
dropped, and a missing field is never filled with a plausible default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from georef.config.loader import load_yaml, repo_root
from georef.linz.client import LinzApiError, LinzClient, redact
from georef.provenance.hashing import environment_fingerprint, sha256_file
from georef.spatial.bounds import containing_box

__all__ = ["IngestResult", "normalise_feature", "ingest"]

REJECTION_REASONS = (
    "MISSING_REQUIRED_FIELD",
    "NON_NUMERIC_COORDINATE",
    "OUTSIDE_BOUNDING_BOXES",
    "DUPLICATE_LINZ_ID",
)


@dataclass
class IngestResult:
    snapshot_path: Path
    manifest_path: Path
    record_count: int
    rejected: dict = field(default_factory=dict)
    snapshot_hash: str = ""


def normalise_feature(feature: dict, fields: dict, required: list[str]) -> tuple[dict | None, str | None]:
    """Map one WFS feature onto the internal record schema.

    Returns ``(record, None)`` or ``(None, reason)``. Coordinates come from the
    gazetteer's own decimal-degree columns rather than the GeoJSON geometry, so
    the value stored is the one LINZ publishes rather than one reprojected in
    transit.
    """
    properties = feature.get("properties") or {}
    record: dict = {}
    for internal_name, source_name in fields.items():
        value = properties.get(source_name)
        record[internal_name] = None if value == "" else value

    for name in required:
        if record.get(name) in (None, ""):
            return None, "MISSING_REQUIRED_FIELD"

    try:
        record["latitude"] = float(record["latitude"])
        record["longitude"] = float(record["longitude"])
    except (TypeError, ValueError):
        return None, "NON_NUMERIC_COORDINATE"

    record["linz_id"] = str(record["linz_id"])
    record["name"] = str(record["name"]).strip()
    if not record["name"]:
        return None, "MISSING_REQUIRED_FIELD"

    record["country"] = "New Zealand"
    record["source"] = "LINZ"
    return record, None


def ingest(
    config: dict,
    output_root: Path,
    client: LinzClient,
    dataset_version: str | None = None,
) -> IngestResult:
    """Fetch the layer, validate and normalise it, and write a hashed snapshot."""
    output_root = Path(output_root)
    paths = config["paths"]
    normalised_dir = output_root / paths["normalised_dir"]
    manifest_dir = output_root / paths["manifest_dir"]
    for directory in (normalised_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fields = config["fields"]
    required = config["required_fields"]
    boxes = config["bounding_boxes"]

    rejected = {reason: 0 for reason in REJECTION_REASONS}
    seen_ids: set[str] = set()
    records: list[dict] = []

    for feature in client.iter_features():
        record, reason = normalise_feature(feature, fields, required)
        if reason:
            rejected[reason] += 1
            continue
        box = containing_box(record["latitude"], record["longitude"], boxes)
        if box is None:
            rejected["OUTSIDE_BOUNDING_BOXES"] += 1
            continue
        if record["linz_id"] in seen_ids:
            rejected["DUPLICATE_LINZ_ID"] += 1
            continue
        seen_ids.add(record["linz_id"])
        record["bounding_box"] = box
        records.append(record)

    if not records:
        raise LinzApiError(
            "LINZ ingest produced zero usable records. This is a failure, not an "
            "empty dataset: check the layer id, the field mapping in "
            "configs/linz.yaml, and the --describe output."
        )

    # Sorted by identifier so the snapshot is byte-stable regardless of the
    # order the service happened to return pages in.
    records.sort(key=lambda r: r["linz_id"])

    version = dataset_version or datetime.now(timezone.utc).strftime("%Y%m%d")
    snapshot_path = normalised_dir / f"linz_place_names_{version}.jsonl"
    with open(snapshot_path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record["dataset_version"] = version
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    snapshot_hash = sha256_file(snapshot_path)

    manifest = {
        "source_authority": config["dataset"]["source_authority"],
        "dataset_name": config["dataset"]["dataset_name"],
        "layer_id": config["dataset"]["layer_id"],
        "dataset_version": version,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": redact(client.build_url(0, config["service"]["page_size"])),
        "wfs_version": config["service"]["version"],
        "srs_name": config["service"]["srs_name"],
        "sort_by": config["service"].get("sort_by"),
        "datum_note": config.get("datum_note"),
        "record_count": len(records),
        "rejected": rejected,
        "field_mapping": fields,
        "bounding_boxes": boxes,
        "snapshot_file": snapshot_path.name,
        "snapshot_sha256": snapshot_hash,
        "environment": environment_fingerprint(),
    }
    manifest_path = manifest_dir / f"linz_manifest_{version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (normalised_dir / f"linz_place_names_{version}.sha256").write_text(
        snapshot_hash + "\n", encoding="utf-8"
    )

    return IngestResult(
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        record_count=len(records),
        rejected=rejected,
        snapshot_hash=snapshot_hash,
    )


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so the key need not be exported by hand."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and freeze the LINZ snapshot.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="fetch one record and print the field names the service returns, then stop",
    )
    parser.add_argument("--dataset-version", default=None, help="defaults to today's UTC date")
    parser.add_argument("--config", default=None, help="path to linz.yaml")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    root = repo_root()
    _load_dotenv(root / ".env")
    config = load_yaml(args.config or root / "configs" / "linz.yaml")

    try:
        client = LinzClient(config)
        if args.describe:
            description = client.describe()
            print(json.dumps(description, indent=2, ensure_ascii=False))
            print(
                "\nCompare 'field_names' against the 'fields:' mapping in "
                "configs/linz.yaml before running the full ingest."
            )
            return 0

        result = ingest(
            config,
            Path(args.output_root) if args.output_root else root,
            client,
            args.dataset_version,
        )
    except LinzApiError as error:
        print(f"LINZ INGEST FAILED\n\n{error}", file=sys.stderr)
        return 2

    print("LINZ snapshot complete.\n")
    print(f"  records kept       {result.record_count}")
    for reason, count in result.rejected.items():
        print(f"  rejected {reason:<26} {count}")
    print(f"\n  snapshot   {result.snapshot_path}")
    print(f"  sha256     {result.snapshot_hash}")
    print(f"  manifest   {result.manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
