"""Verification of a frozen LINZ snapshot (acceptance gate 2 and gate 5 evidence).

Answers two questions before anything is embedded:

1. What is actually in the snapshot? Feature types, administrative fields,
   geometry types, bounding boxes, coordinate ranges.
2. Do the anchor entities that appear in real descriptions exist in it at all?
   A gazetteer that does not contain the anchors cannot ground them, and that
   is far cheaper to discover now than after the pipeline is built.

**The anchor extraction here is a DIAGNOSTIC, not the pipeline.** It is a crude
regex used only to estimate coverage. Semantic interpretation is the LLM's job
in step 4, and nothing in this module is imported by the prediction path.

**It reads the dev pool, never the benchmark.** Measuring coverage against the
75 benchmark descriptions would be inspecting the evaluation set before the
experiment, so the check runs on the 3,945 held-out dev descriptions instead.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from georef.normalise import match_key, normalise

__all__ = [
    "load_snapshot",
    "summarise_snapshot",
    "extract_anchor",
    "anchor_coverage",
]

# Diagnostic only. The distance/direction part is loaded from the SAME
# pre-registered pattern the benchmark builder uses, so the two cannot drift
# apart; only the anchor capture group is added here.
def _build_anchor_pattern() -> re.Pattern:
    from georef.config.loader import load_benchmark_config

    base = re.sub(r"\s+", "", load_benchmark_config()["distance_direction_pattern"])
    # Periods are allowed inside the capture so abbreviations survive
    # ("Mt. Cook", "Rte. 6"); a sentence-ending period is trimmed afterwards.
    return re.compile(base + r"\s*(?P<anchor>[^,;:\[\]()]+)", re.IGNORECASE)


_ANCHOR_PATTERN = _build_anchor_pattern()

_LEADING_NOISE = re.compile(
    r"^(?:the\s+)?(?:town|township|village|settlement|city|locality|suburb)\s+of\s+"
    r"|^(?:the|a|an|near|about|approx\.?)\s+",
    re.IGNORECASE,
)

# "Erua township" -> "Erua". Only stripped when something precedes it.
_TRAILING_NOISE = re.compile(
    r"\s+(?:township|village|settlement|town|area|district|region)$", re.IGNORECASE
)

# Abbreviations whose period is part of the token rather than a sentence end.
_ABBREVIATION = re.compile(r"(?:^|\s)[A-Za-z]{1,4}$")


def _trim_sentence_period(text: str) -> str:
    """Cut at the first period that ends a word rather than an abbreviation."""
    out = ""
    for fragment in text.split("."):
        if not out:
            out = fragment
            continue
        if _ABBREVIATION.search(out):
            out = f"{out}.{fragment}"
        else:
            break
    return out


def load_snapshot(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarise_snapshot(records: list[dict], top: int = 15) -> dict:
    latitudes = [r["latitude"] for r in records]
    longitudes = [r["longitude"] for r in records]
    return {
        "record_count": len(records),
        "distinct_names": len({normalise(r["name"]) for r in records}),
        "latitude_range": [min(latitudes), max(latitudes)],
        "longitude_range": [min(longitudes), max(longitudes)],
        "by_bounding_box": dict(Counter(r.get("bounding_box") for r in records)),
        "by_geometry_type": dict(Counter(r.get("geometry_type") for r in records)),
        "by_name_status": dict(Counter(r.get("name_status") for r in records)),
        "top_feature_types": dict(
            Counter(r.get("feature_type") for r in records).most_common(top)
        ),
        "top_land_districts": dict(
            Counter(r.get("land_district") for r in records).most_common(top)
        ),
        "land_district_populated": sum(1 for r in records if r.get("land_district")),
        "region_populated": sum(1 for r in records if r.get("region")),
        "territorial_authority_populated": sum(
            1 for r in records if r.get("territorial_authority")
        ),
    }


def extract_anchor(description: str) -> str | None:
    """Crude anchor extraction for coverage diagnostics only."""
    match = _ANCHOR_PATTERN.search(description)
    if not match:
        return None
    anchor = _trim_sentence_period(match.group("anchor").strip()).strip()
    anchor = _LEADING_NOISE.sub("", anchor).strip()
    # Drop a trailing relative clause: "Waiouru on the main highway" -> "Waiouru"
    anchor = re.split(r"\s+(?:on|at|in|near|by|to|along|between|and)\s+", anchor, maxsplit=1)[0]
    anchor = _TRAILING_NOISE.sub("", anchor.strip())
    return anchor.strip(" .") or None


def anchor_coverage(descriptions: list[str], records: list[dict], examples: int = 15) -> dict:
    """How many extracted anchors match a gazetteer name exactly after normalisation.

    This is a floor, not a measure of pipeline capability: exact name matching
    ignores everything retrieval and disambiguation exist to do. A low number
    here is a real warning; a high number is not a promise.
    """
    exact_names = {normalise(r["name"]) for r in records}
    folded_names = {match_key(r["name"]) for r in records}
    anchors = [a for a in (extract_anchor(d) for d in descriptions) if a]

    exact, folded_only, unresolved = [], [], []
    for anchor in anchors:
        if normalise(anchor) in exact_names:
            exact.append(anchor)
        elif match_key(anchor) in folded_names:
            folded_only.append(anchor)
        else:
            unresolved.append(anchor)

    matched = len(exact) + len(folded_only)
    return {
        "descriptions_scanned": len(descriptions),
        "anchors_extracted": len(anchors),
        "exact_name_match": len(exact),
        "matched_only_after_diacritic_folding": len(folded_only),
        "no_match": len(unresolved),
        "exact_match_rate": round(len(exact) / len(anchors), 4) if anchors else 0.0,
        "match_rate_with_folding": round(matched / len(anchors), 4) if anchors else 0.0,
        "folding_examples": [a for a, _ in Counter(folded_only).most_common(examples)],
        "unmatched_examples": [a for a, _ in Counter(unresolved).most_common(examples)],
    }


def main(argv: list[str] | None = None) -> int:
    import csv

    from georef.config.loader import load_paths_config, load_yaml, repo_root

    parser = argparse.ArgumentParser(description="Verify a frozen LINZ snapshot.")
    parser.add_argument("--snapshot", default=None, help="path to the .jsonl snapshot")
    args = parser.parse_args(argv)

    root = repo_root()
    linz_cfg = load_yaml(root / "configs" / "linz.yaml")

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        candidates = sorted((root / linz_cfg["paths"]["normalised_dir"]).glob("*.jsonl"))
        if not candidates:
            print("No snapshot found. Run scripts/fetch_linz.py first.")
            return 2
        snapshot_path = candidates[-1]

    records = load_snapshot(snapshot_path)
    print(f"Snapshot: {snapshot_path}\n")
    print(json.dumps(summarise_snapshot(records), indent=2, ensure_ascii=False))

    dev_path = root / load_paths_config()["dev_dir"] / "dev_pool.csv"
    if dev_path.is_file():
        with open(dev_path, "r", encoding="utf-8", newline="") as handle:
            descriptions = [row["description"] for row in csv.DictReader(handle)]
        print("\nAnchor coverage (dev pool only, diagnostic):")
        print(json.dumps(anchor_coverage(descriptions, records), indent=2, ensure_ascii=False))
    else:
        print("\nDev pool not found; skipping anchor coverage.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
