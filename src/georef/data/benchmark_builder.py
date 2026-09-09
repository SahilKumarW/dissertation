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
from georef.spatial.bounds import containing_box

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
    dev_ground_truth_path: Path | None = None
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
    path = resolve_source_file(source_dir, paths_cfg["descriptions_file"])
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
    path = resolve_source_file(source_dir, paths_cfg["coordinates_file"])
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


def _loose_key(name: str) -> str:
    """Filename with separators and case removed, for tolerant matching."""
    return re.sub(r"[\s\-_]+", "", name).lower()


def resolve_source_file(source_dir: Path, configured_name: str) -> Path | None:
    """Find the configured file, tolerating separator and case differences.

    An exact match always wins. The fallback exists because the same supplier
    file circulates as 'SampleDataNoCoords.xlsx' and 'SampleData-NoCoords.xlsx';
    resolving that is a convenience, not a licence to read some other file, so
    the match must be unambiguous and the resolved name is recorded in the
    manifest.
    """
    exact = source_dir / configured_name
    if exact.is_file():
        return exact
    if not source_dir.is_dir():
        return None
    wanted = _loose_key(configured_name)
    matches = [p for p in source_dir.iterdir() if p.is_file() and _loose_key(p.name) == wanted]
    return matches[0] if len(matches) == 1 else None


def _preflight(source_dir: Path, paths_cfg: dict) -> None:
    """Fail early and legibly when a supplier file is absent.

    A missing input is a data-integrity failure with a specific cause, not a
    generic error, so it is reported as one (master spec section 28).
    """
    missing = [
        name
        for name in (paths_cfg["descriptions_file"], paths_cfg["coordinates_file"])
        if resolve_source_file(source_dir, name) is None
    ]
    if missing:
        listed = "\n".join(f"    {name}" for name in missing)
        if source_dir.is_dir():
            found = sorted(p.name for p in source_dir.iterdir() if p.is_file())
            present = (
                "\n".join(f"    {name}" for name in found)
                if found
                else "    (directory is empty)"
            )
        else:
            present = "    (directory does not exist)"
        raise FileNotFoundError(
            f"Supplier file(s) not found in {source_dir}\n\n"
            f"  expected but missing:\n{listed}\n\n"
            f"  actually present:\n{present}\n\n"
            "Filenames must match configs/paths.yaml exactly. If your copy is "
            "named differently, update descriptions_file / coordinates_file in "
            "that config rather than renaming the supplier data, so the manifest "
            "records the real filename."
        )


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
    _preflight(source_dir, paths_cfg)
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
        box = containing_box(row["latitude"], row["longitude"], boxes)
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

    # --- Stable seeded selection -------------------------------------------
    #
    # Selection orders the eligible keys by SHA-256 of "<seed>:<key>" and takes
    # the first N. This is deterministic and seed-controlled like a shuffle, but
    # each record's position depends only on itself, so adding or removing
    # OTHER records does not reshuffle the draw.
    #
    # random.sample does not have that property: correcting five ineligible
    # records changed 52 of 75 selections, because the sample is a function of
    # the whole list. A benchmark that churns wholesale on any construction fix
    # is fragile, and the churn obscures what actually changed.
    #
    # SHA-256 is also portable in a way the Mersenne Twister is not: the
    # ordering is identical on any interpreter and platform.
    eligible_by_key = {row["key"]: row for row in eligible}
    seed = benchmark_cfg["random_seed"]

    def _draw_order(key: str, draw_seed) -> str:
        return sha256_text(f"{draw_seed}:{key}")

    ordered_keys = sorted(eligible_by_key, key=lambda k: _draw_order(k, seed))
    selected_keys = sorted(ordered_keys[:size])

    # Dev ground truth: sampled from what the benchmark left behind, with its
    # own seed. The benchmark draw happens first and is unaffected by this
    # addition, so benchmark.csv keeps its published hash.
    dev_size = int(benchmark_cfg.get("dev_ground_truth_size", 0))
    dev_seed = benchmark_cfg.get("dev_random_seed", 0)
    taken = set(selected_keys)
    remaining_keys = sorted(
        (key for key in ordered_keys if key not in taken),
        key=lambda k: _draw_order(k, dev_seed),
    )
    dev_selected_keys = sorted(remaining_keys[:dev_size]) if dev_size else []

    prefix = benchmark_cfg["benchmark_id_prefix"]
    width = max(3, len(str(size)))
    selected = []
    for position, key in enumerate(selected_keys, start=1):
        row = dict(eligible_by_key[key])
        row["benchmark_id"] = f"{prefix}-{position:0{width}d}"
        selected.append(row)

    # Validation: everything the benchmark and dev set left behind, in the same
    # deterministic hash order, so it is reproducible and disjoint by
    # construction rather than by check.
    validation_setting = benchmark_cfg.get("validation_size", 0)
    taken_for_dev = set(dev_selected_keys)
    validation_pool = [k for k in remaining_keys if k not in taken_for_dev]
    if validation_setting == "all":
        validation_keys = sorted(validation_pool)
    else:
        validation_keys = sorted(validation_pool[: int(validation_setting or 0)])

    validation_selected = []
    for position, key in enumerate(validation_keys, start=1):
        row = dict(eligible_by_key[key])
        row["validation_id"] = f"{benchmark_cfg.get('validation_id_prefix', 'VAL')}-{position:03d}"
        validation_selected.append(row)

    dev_prefix = benchmark_cfg.get("dev_id_prefix", "DEVGT")
    dev_width = max(3, len(str(dev_size)))
    dev_selected = []
    for position, key in enumerate(dev_selected_keys, start=1):
        row = dict(eligible_by_key[key])
        row["dev_gt_id"] = f"{dev_prefix}-{position:0{dev_width}d}"
        dev_selected.append(row)

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

    if dev_selected:
        dev_gt_frame = pd.DataFrame(
            [
                {
                    "dev_gt_id": r["dev_gt_id"],
                    "description": r["description"],
                    "decimal_latitude": r["latitude"],
                    "decimal_longitude": r["longitude"],
                    "audit_category": r["audit_category"],
                    "bounding_box": r["bounding_box"],
                    "source_event_ids": "|".join(r["source_event_ids"]),
                }
                for r in dev_selected
            ]
        )
        # Descriptions only, so the extraction stage never opens a file that
        # contains answers. Mirrors benchmark.csv, for the same reason.
        dev_descriptions_frame = pd.DataFrame(
            [{"dev_gt_id": r["dev_gt_id"], "description": r["description"]} for r in dev_selected]
        )
        dev_descriptions_frame.to_csv(
            dev_dir / "dev_descriptions.csv", index=False, lineterminator=line_terminator
        )

        dev_gt_path = dev_dir / "dev_ground_truth.csv"
        dev_gt_frame.to_csv(
            dev_gt_path,
            index=False,
            lineterminator=line_terminator,
            float_format=coord_format,
        )
        dev_gt_hash = sha256_file(dev_gt_path)
        (dev_dir / "dev_ground_truth_hash.txt").write_text(dev_gt_hash + "\n", encoding="utf-8")
    else:
        dev_gt_path, dev_gt_hash = None, None

    validation_hash = None
    if validation_selected:
        validation_dir = output_root / "data" / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"validation_id": r["validation_id"], "description": r["description"]}
             for r in validation_selected]
        ).to_csv(validation_dir / "validation.csv", index=False, lineterminator=line_terminator)
        validation_gt = validation_dir / "validation_ground_truth.csv"
        pd.DataFrame(
            [
                {
                    "validation_id": r["validation_id"],
                    "decimal_latitude": r["latitude"],
                    "decimal_longitude": r["longitude"],
                    "audit_category": r["audit_category"],
                    "bounding_box": r["bounding_box"],
                    "source_event_ids": "|".join(r["source_event_ids"]),
                }
                for r in validation_selected
            ]
        ).to_csv(validation_gt, index=False, lineterminator=line_terminator,
                 float_format=coord_format)
        validation_hash = sha256_file(validation_gt)
        (validation_dir / "validation_ground_truth_hash.txt").write_text(
            validation_hash + "\n", encoding="utf-8"
        )

    # Selection set: dev + benchmark, the data parameters may be fitted to under
    # experiment v2. Emitted as one file so a sweep cannot accidentally fit to
    # the validation set by pointing at the wrong path.
    selection_dir = output_root / "data" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = [
        {"selection_id": r["dev_gt_id"], "description": r["description"], "origin": "dev",
         "latitude": r["latitude"], "longitude": r["longitude"]}
        for r in dev_selected
    ] + [
        {"selection_id": r["benchmark_id"], "description": r["description"], "origin": "benchmark",
         "latitude": r["latitude"], "longitude": r["longitude"]}
        for r in selected
    ]
    selection_rows.sort(key=lambda r: r["selection_id"])
    pd.DataFrame(
        [{"selection_id": r["selection_id"], "description": r["description"],
          "origin": r["origin"]} for r in selection_rows]
    ).to_csv(selection_dir / "selection_descriptions.csv", index=False,
             lineterminator=line_terminator)
    selection_gt = selection_dir / "selection_ground_truth.csv"
    pd.DataFrame(
        [{"selection_id": r["selection_id"], "decimal_latitude": r["latitude"],
          "decimal_longitude": r["longitude"], "origin": r["origin"]} for r in selection_rows]
    ).to_csv(selection_gt, index=False, lineterminator=line_terminator,
             float_format=coord_format)
    selection_hash = sha256_file(selection_gt)
    (selection_dir / "selection_ground_truth_hash.txt").write_text(
        selection_hash + "\n", encoding="utf-8"
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
        "dev_ground_truth_size": len(dev_selected),
        "validation_size": len(validation_selected),
        "selection_set_size": len(selection_rows),
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
            "selection_method": "sha256(seed:key) ordering, stable under pool changes",
            "dev_ground_truth_size": dev_size,
            "dev_random_seed": benchmark_cfg.get("dev_random_seed"),
            "normalisation_version": normalisation_version(),
        },
        "counts": counts,
        "audit_category_counts": audit_counts,
        "artefact_hashes": {
            "benchmark.csv": benchmark_hash,
            "ground_truth.csv": ground_truth_hash,
            "dev_pool.csv": sha256_file(dev_path),
            **({"dev_ground_truth.csv": dev_gt_hash} if dev_gt_hash else {}),
            **({"validation_ground_truth.csv": validation_hash} if validation_hash else {}),
            "selection_ground_truth.csv": selection_hash,
        },
        "source_hashes": {
            # Keyed by the filename actually read, not the configured one.
            resolve_source_file(source_dir, name).name: sha256_file(
                resolve_source_file(source_dir, name)
            )
            for name in (paths_cfg["descriptions_file"], paths_cfg["coordinates_file"])
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
        dev_ground_truth_path=dev_gt_path,
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

    try:
        result = build_benchmark(source_dir, output_root, load_benchmark_config(), paths_cfg)
    except FileNotFoundError as error:
        print(f"BUILD FAILED -- missing input\n\n{error}")
        return 2

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
