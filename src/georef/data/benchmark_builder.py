"""Construct and freeze the benchmark, ground truth and dev pool.

This implements Step 1 of the implementation plan and the procedure
pre-registered in docs/BENCHMARK_CONSTRUCTION.md.

Two supplier files are the only inputs:

* ``SampleDataNoCoords.xlsx``  -- natural-language descriptions, no coordinates.
* ``MWLRDataCHR...csv``        -- the same corpus WITH coordinates.

The CSV therefore holds the answer to every description in the corpus. It is
EVALUATION-ONLY. This module is the single place in the repository permitted to
read it, and it lives under ``georef.data`` rather than under any prediction
package so that the isolation test can enforce that boundary.

Nothing here inspects model behaviour, and no selection decision depends on any
prediction. Selection depends only on the pre-registered policy in
configs/benchmark.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from georef.config.loader import load_benchmark_config, load_paths_config, repo_root
from georef.normalise import normalise, normalisation_version
from georef.provenance.hashing import (
    environment_fingerprint,
    sha256_file,
    sha256_text,
)

__all__ = ["BuildResult", "build_benchmark", "compile_pattern", "audit_description"]

# Audit categories. The first three are eligible for the benchmark; the rest are
# reported but excluded, because they have no single defensible answer key.
SINGLE_ROW_EXACT = "SINGLE_ROW_EXACT"
DUPLICATE_ROWS_SAME_COORDINATE = "DUPLICATE_ROWS_SAME_COORDINATE"
SINGLE_ROW_NORMALIZED = "SINGLE_ROW_NORMALIZED"
UNRESOLVED_MULTI_COORD = "UNRESOLVED_MULTI_COORD"
NO_COORDINATE = "NO_COORDINATE"
NO_MATCH = "NO_MATCH"

ALL_CATEGORIES = (
    SINGLE_ROW_EXACT,
    DUPLICATE_ROWS_SAME_COORDINATE,
    SINGLE_ROW_NORMALIZED,
    UNRESOLVED_MULTI_COORD,
    NO_COORDINATE,
    NO_MATCH,
)


@dataclass
class BuildResult:
    """Paths and counts produced by a build, for reporting and testing."""

    benchmark_path: Path
    ground_truth_path: Path
    dev_pool_path: Path
    audit_report_path: Path
    manifest_path: Path
    counts: dict = field(default_factory=dict)
    audit_counts: dict = field(default_factory=dict)


def compile_pattern(pattern: str) -> re.Pattern:
    """Compile the eligibility pattern, stripping the readability whitespace.

    The pattern is stored as a folded YAML scalar, so it arrives with newlines
    already folded into spaces. Every literal whitespace character is removed;
    spaces that must be matched are written as ``\\s`` in the configuration.
    """
    collapsed = re.sub(r"\s+", "", pattern)
    return re.compile(collapsed, re.IGNORECASE)


def _in_any_box(lat: float, lon: float, boxes: dict) -> str | None:
    """Return the name of the first bounding box containing the point."""
    for name, box in boxes.items():
        if (
            box["lat_min"] <= lat <= box["lat_max"]
            and box["lon_min"] <= lon <= box["lon_max"]
        ):
            return name
    return None


def audit_description(
    raw: str,
    key: str,
    exact_index: dict,
    norm_index: dict,
    decimals: int,
) -> dict:
    """Resolve one description against the coordinate corpus.

    Exact matching on the untouched ``verbatimLocality`` string is attempted
    first; normalised matching is the fallback, and is labelled distinctly so
    that the audit shows how much work normalisation is doing.
    """
    rows = exact_index.get(raw)
    matched_by = "exact"
    if not rows:
        rows = norm_index.get(key)
        matched_by = "normalised"

    if not rows:
        return {
            "audit_category": NO_MATCH,
            "matched_by": None,
            "source_row_count": 0,
            "source_event_ids": [],
            "latitude": None,
            "longitude": None,
        }

    with_coords = [r for r in rows if r["lat"] is not None and r["lon"] is not None]
    if not with_coords:
        return {
            "audit_category": NO_COORDINATE,
            "matched_by": matched_by,
            "source_row_count": len(rows),
            "source_event_ids": [r["event_id"] for r in rows],
            "latitude": None,
            "longitude": None,
        }

    distinct = {(round(r["lat"], decimals), round(r["lon"], decimals)) for r in with_coords}
    event_ids = [r["event_id"] for r in with_coords]
    metadata: dict[str, str] = {}
    for column in (with_coords[0].get("metadata") or {}):
        values = sorted({r["metadata"].get(column, "") for r in with_coords} - {""})
        metadata[column] = "|".join(values)

    if len(distinct) > 1:
        category = UNRESOLVED_MULTI_COORD
        lat = lon = None
    else:
        lat, lon = next(iter(distinct))
        if len(with_coords) == 1:
            category = SINGLE_ROW_EXACT if matched_by == "exact" else SINGLE_ROW_NORMALIZED
        else:
            category = DUPLICATE_ROWS_SAME_COORDINATE

    return {
        "audit_category": category,
        "matched_by": matched_by,
        "source_row_count": len(with_coords),
        "source_event_ids": event_ids,
        "latitude": lat,
        "longitude": lon,
        "metadata": metadata,
    }


def _load_descriptions(source_dir: Path, paths_cfg: dict) -> pd.DataFrame:
    path = source_dir / paths_cfg["descriptions_file"]
    frame = pd.read_excel(path, sheet_name=paths_cfg["descriptions_sheet"])
    column = paths_cfg["descriptions_column"]
    if column not in frame.columns:
        raise ValueError(f"expected column {column!r} in {path}")
    frame = frame[[column]].rename(columns={column: "description"})
    frame["description"] = frame["description"].astype(str)
    frame = frame[frame["description"].str.strip() != ""]
    frame["key"] = frame["description"].map(normalise)
    frame = frame[frame["key"] != ""]
    # First occurrence in file order wins, so the retained original text is
    # deterministic for a given input file.
    frame = frame.drop_duplicates(subset="key", keep="first").reset_index(drop=True)
    return frame


def _build_coordinate_indexes(source_dir: Path, paths_cfg: dict) -> tuple[dict, dict, dict]:
    """Index the EVALUATION-ONLY coordinate corpus by raw and normalised locality."""
    path = source_dir / paths_cfg["coordinates_file"]
    metadata_columns = list(paths_cfg.get("coordinates_metadata_columns") or [])
    usecols = [
        paths_cfg["coordinates_locality_column"],
        paths_cfg["coordinates_lat_column"],
        paths_cfg["coordinates_lon_column"],
        paths_cfg["coordinates_id_column"],
        *metadata_columns,
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)

    exact_index: dict[str, list] = defaultdict(list)
    norm_index: dict[str, list] = defaultdict(list)
    missing_coords = 0

    locality_col = paths_cfg["coordinates_locality_column"]
    lat_col = paths_cfg["coordinates_lat_column"]
    lon_col = paths_cfg["coordinates_lon_column"]
    id_col = paths_cfg["coordinates_id_column"]

    for record in frame.to_dict("records"):
        locality = record[locality_col]
        if not isinstance(locality, str) or not locality.strip():
            continue
        lat = record[lat_col]
        lon = record[lon_col]
        lat = float(lat) if pd.notna(lat) else None
        lon = float(lon) if pd.notna(lon) else None
        if lat is None or lon is None:
            missing_coords += 1
        row = {"event_id": str(record[id_col]), "lat": lat, "lon": lon}
        row["metadata"] = {
            column: ("" if pd.isna(record[column]) else str(record[column]))
            for column in metadata_columns
        }
        exact_index[locality].append(row)
        norm_index[normalise(locality)].append(row)

    stats = {"corpus_rows": int(len(frame)), "rows_missing_coordinates": missing_coords}
    return dict(exact_index), dict(norm_index), stats


def build_benchmark(
    source_dir: Path,
    output_root: Path,
    benchmark_cfg: dict | None = None,
    paths_cfg: dict | None = None,
) -> BuildResult:
    """Run the full construction and write every artefact under ``output_root``."""
    benchmark_cfg = benchmark_cfg or load_benchmark_config()
    paths_cfg = paths_cfg or load_paths_config()

    source_dir = Path(source_dir)
    output_root = Path(output_root)
    benchmark_dir = output_root / paths_cfg["benchmark_dir"]
    ground_truth_dir = output_root / paths_cfg["ground_truth_dir"]
    dev_dir = output_root / paths_cfg["dev_dir"]
    for directory in (benchmark_dir, ground_truth_dir, dev_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pattern = compile_pattern(benchmark_cfg["distance_direction_pattern"])
    decimals = int(benchmark_cfg["coordinate_equality_decimals"])
    boxes = benchmark_cfg["bounding_boxes"]
    eligible_categories = set(benchmark_cfg["eligible_audit_categories"])

    descriptions = _load_descriptions(source_dir, paths_cfg)
    exact_index, norm_index, corpus_stats = _build_coordinate_indexes(source_dir, paths_cfg)

    descriptions["is_dd_candidate"] = descriptions["description"].str.contains(
        pattern, regex=True
    )
    candidates = descriptions[descriptions["is_dd_candidate"]].copy()

    audit_rows = []
    for record in candidates.to_dict("records"):
        result = audit_description(
            record["description"], record["key"], exact_index, norm_index, decimals
        )
        result["description"] = record["description"]
        result["key"] = record["key"]
        audit_rows.append(result)

    audit_counts = {name: 0 for name in ALL_CATEGORIES}
    for row in audit_rows:
        audit_counts[row["audit_category"]] += 1

    # --- Quality gate ------------------------------------------------------
    eligible, rejected_out_of_box = [], 0
    for row in audit_rows:
        if row["audit_category"] not in eligible_categories:
            continue
        if row["latitude"] is None or row["longitude"] is None:
            continue
        box = _in_any_box(row["latitude"], row["longitude"], boxes)
        if box is None:
            rejected_out_of_box += 1
            continue
        row["bounding_box"] = box
        eligible.append(row)

    removed_shared_coordinate = 0
    if bool(benchmark_cfg.get("one_record_per_coordinate", False)):
        seen: set[tuple[float, float]] = set()
        independent = []
        for row in sorted(eligible, key=lambda r: r["key"]):
            point = (round(row["latitude"], decimals), round(row["longitude"], decimals))
            if point in seen:
                removed_shared_coordinate += 1
                continue
            seen.add(point)
            independent.append(row)
        eligible = independent

    size = int(benchmark_cfg["benchmark_size"])
    if len(eligible) < size:
        raise RuntimeError(
            f"only {len(eligible)} eligible records, need {size}; "
            "adjust the pre-registered policy before rebuilding, and record why"
        )

    # --- Seeded selection --------------------------------------------------
    eligible_by_key = {row["key"]: row for row in eligible}
    ordered_keys = sorted(eligible_by_key)
    rng = random.Random(int(benchmark_cfg["random_seed"]))
    selected_keys = sorted(rng.sample(ordered_keys, size))

    prefix = benchmark_cfg["benchmark_id_prefix"]
    width = max(3, len(str(size)))
    selected = []
    for position, key in enumerate(selected_keys, start=1):
        row = dict(eligible_by_key[key])
        row["benchmark_id"] = f"{prefix}-{position:0{width}d}"
        selected.append(row)

    # --- Artefacts ---------------------------------------------------------
    line_terminator = benchmark_cfg["csv_line_terminator"]
    coord_format = f"%.{int(benchmark_cfg['coordinate_output_decimals'])}f"

    benchmark_frame = pd.DataFrame(
        [{"benchmark_id": r["benchmark_id"], "description": r["description"]} for r in selected]
    )
    benchmark_path = benchmark_dir / "benchmark.csv"
    benchmark_frame.to_csv(benchmark_path, index=False, lineterminator=line_terminator)

    ground_truth_frame = pd.DataFrame(
        [
            {
                "benchmark_id": r["benchmark_id"],
                "decimal_latitude": r["latitude"],
                "decimal_longitude": r["longitude"],
                "audit_category": r["audit_category"],
                "matched_by": r["matched_by"],
                "bounding_box": r["bounding_box"],
                "source_row_count": r["source_row_count"],
                "source_event_ids": "|".join(r["source_event_ids"]),
                **{
                    f"source_{column}": value
                    for column, value in (r.get("metadata") or {}).items()
                },
            }
            for r in selected
        ]
    )
    ground_truth_path = ground_truth_dir / "ground_truth.csv"
    ground_truth_frame.to_csv(
        ground_truth_path,
        index=False,
        lineterminator=line_terminator,
        float_format=coord_format,
    )

    selected_key_set = {r["key"] for r in selected}
    dev_frame = descriptions[~descriptions["key"].isin(selected_key_set)].copy()
    dev_frame = dev_frame.sort_values("key").reset_index(drop=True)
    dev_frame["dev_id"] = [f"DEV-{i:05d}" for i in range(1, len(dev_frame) + 1)]
    dev_path = dev_dir / "dev_pool.csv"
    dev_frame[["dev_id", "description", "is_dd_candidate"]].to_csv(
        dev_path, index=False, lineterminator=line_terminator
    )

    counts = {
        "description_rows_unique": int(len(descriptions)),
        "distance_direction_candidates": int(len(candidates)),
        "eligible_after_quality_gate": len(eligible) + removed_shared_coordinate,
        "removed_sharing_a_coordinate": removed_shared_coordinate,
        "independent_eligible_records": len(eligible),
        "rejected_outside_bounding_boxes": rejected_out_of_box,
        "benchmark_size": len(selected),
        "defensible_denominator": len(selected),
        "dev_pool_size": int(len(dev_frame)),
        **corpus_stats,
    }

    audit_report = {
        "benchmark_version": benchmark_cfg["benchmark_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_category_counts": audit_counts,
        "counts": counts,
        "eligible_audit_categories": sorted(eligible_categories),
        "selected_category_counts": {
            name: sum(1 for r in selected if r["audit_category"] == name)
            for name in sorted(eligible_categories)
        },
        "note": (
            "Every description in SampleDataNoCoords.xlsx is drawn from the "
            "coordinate corpus, so NO_MATCH is expected to be zero. It remains "
            "an explicit category so that a future corpus change is visible."
        ),
    }
    audit_report_path = ground_truth_dir / "audit_report.json"
    audit_report_path.write_text(json.dumps(audit_report, indent=2) + "\n", encoding="utf-8")

    benchmark_hash = sha256_file(benchmark_path)
    ground_truth_hash = sha256_file(ground_truth_path)
    (benchmark_dir / "benchmark_hash.txt").write_text(benchmark_hash + "\n", encoding="utf-8")
    (ground_truth_dir / "ground_truth_hash.txt").write_text(
        ground_truth_hash + "\n", encoding="utf-8"
    )

    builder_source = Path(__file__).read_text(encoding="utf-8")
    manifest = {
        "benchmark_version": benchmark_cfg["benchmark_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "distance_direction_pattern": re.sub(
                r"\s+", "", benchmark_cfg["distance_direction_pattern"]
            ),
            "coordinate_equality_decimals": decimals,
            "bounding_boxes": boxes,
            "eligible_audit_categories": sorted(eligible_categories),
            "benchmark_size": size,
            "one_record_per_coordinate": bool(benchmark_cfg.get("one_record_per_coordinate", False)),
            "random_seed": benchmark_cfg["random_seed"],
            "normalisation_version": normalisation_version(),
        },
        "counts": counts,
        "audit_category_counts": audit_counts,
        "artefact_hashes": {
            "benchmark.csv": benchmark_hash,
            "ground_truth.csv": ground_truth_hash,
            "dev_pool.csv": sha256_file(dev_path),
        },
        "source_hashes": {
            paths_cfg["descriptions_file"]: sha256_file(
                source_dir / paths_cfg["descriptions_file"]
            ),
            paths_cfg["coordinates_file"]: sha256_file(
                source_dir / paths_cfg["coordinates_file"]
            ),
        },
        "code_hashes": {
            "benchmark_builder.py": sha256_text(builder_source),
            "normalise.py": sha256_file(
                Path(__file__).resolve().parents[1] / "normalise.py"
            ),
        },
        "environment": environment_fingerprint(),
        "reproducibility_note": (
            "generated_at_utc makes this manifest non-reproducible byte-for-byte. "
            "Reproducibility is asserted over artefact_hashes, which are stable "
            "for a fixed policy, source pair and interpreter."
        ),
    }
    manifest_path = benchmark_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return BuildResult(
        benchmark_path=benchmark_path,
        ground_truth_path=ground_truth_path,
        dev_pool_path=dev_path,
        audit_report_path=audit_report_path,
        manifest_path=manifest_path,
        counts=counts,
        audit_counts=audit_counts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and freeze the benchmark.")
    parser.add_argument("--source-dir", default=None, help="directory holding the two supplier files")
    parser.add_argument("--output-root", default=None, help="repository root to write data/ under")
    args = parser.parse_args(argv)

    root = repo_root()
    paths_cfg = load_paths_config()
    source_dir = Path(args.source_dir) if args.source_dir else root / paths_cfg["source_dir"]
    output_root = Path(args.output_root) if args.output_root else root

    result = build_benchmark(source_dir, output_root, load_benchmark_config(), paths_cfg)

    print("Benchmark construction complete.\n")
    print("Audit of DISTANCE_DIRECTION candidates:")
    for name, value in result.audit_counts.items():
        print(f"  {name:<32} {value}")
    print("\nCounts:")
    for name, value in result.counts.items():
        print(f"  {name:<32} {value}")
    print("\nArtefacts:")
    for path in (
        result.benchmark_path,
        result.ground_truth_path,
        result.dev_pool_path,
        result.audit_report_path,
        result.manifest_path,
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
