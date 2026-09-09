#!/usr/bin/env python
"""Evaluate the benchmark runs (master spec sections 36 to 41).

    python scripts/evaluate.py --baseline <run_id> --hybrid <run_id>

Ground truth is loaded HERE and nowhere else. Evaluation happens strictly after
prediction: this script reads finished prediction files and never invokes a
model or a pipeline, so it cannot influence what it measures.

Writes metrics per condition, a paired statistical comparison, and a case-level
table for the qualitative error analysis.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import load_paths_config, repo_root  # noqa: E402
from georef.evaluation.metrics import geodesic_error_km, summarise  # noqa: E402
from georef.evaluation.statistics import (  # noqa: E402
    mcnemar,
    pair_conditions,
    wilcoxon_signed_rank,
)


def load_predictions(directory: Path) -> dict:
    path = directory / "predictions.jsonl"
    if not path.is_file():
        raise SystemExit(f"no predictions at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return {json.loads(line)["benchmark_id"]: json.loads(line) for line in handle if line.strip()}


def score(rows: dict, truth: dict) -> dict:
    """Attach geodesic error to every prediction that produced a coordinate."""
    for identifier, row in rows.items():
        reference = truth.get(identifier)
        latitude, longitude = row.get("predicted_latitude"), row.get("predicted_longitude")
        if reference and latitude is not None and longitude is not None:
            row["error_km"] = round(
                geodesic_error_km(
                    float(latitude), float(longitude),
                    float(reference["decimal_latitude"]), float(reference["decimal_longitude"]),
                ), 4,
            )
            row["true_latitude"] = float(reference["decimal_latitude"])
            row["true_longitude"] = float(reference["decimal_longitude"])
        else:
            row["error_km"] = None
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate benchmark runs.")
    parser.add_argument("--baseline", required=True, help="Condition A run id")
    parser.add_argument("--hybrid", required=True, help="Condition B run id")
    parser.add_argument("--thresholds-km", default="1,5,10,50")
    parser.add_argument(
        "--dataset", choices=["benchmark", "dev", "validation"], default="benchmark"
    )
    args = parser.parse_args()

    root = repo_root()
    paths = load_paths_config()

    if args.dataset == "validation":
        truth_path = root / "data" / "validation" / "validation_ground_truth.csv"
        id_field, dir_a, dir_b = "validation_id", "validation_baseline", "validation_hybrid"
    elif args.dataset == "dev":
        truth_path = root / paths["dev_dir"] / "dev_ground_truth.csv"
        id_field, dir_a, dir_b = "dev_gt_id", "dev_baseline", "dev_hybrid"
    else:
        truth_path = root / paths["ground_truth_dir"] / "ground_truth.csv"
        id_field, dir_a, dir_b = "benchmark_id", "baseline", "hybrid"

    with open(truth_path, "r", encoding="utf-8", newline="") as handle:
        truth = {row[id_field]: row for row in csv.DictReader(handle)}

    rows_a = score(load_predictions(root / "experiments" / dir_a / args.baseline), truth)
    rows_b = score(load_predictions(root / "experiments" / dir_b / args.hybrid), truth)

    all_ids = sorted(set(truth))
    summary_a = summarise(list(rows_a.values()), total_records=len(all_ids)).to_dict()
    summary_b = summarise(list(rows_b.values()), total_records=len(all_ids)).to_dict()

    paired = pair_conditions(rows_a, rows_b, all_ids)
    wilcoxon = wilcoxon_signed_rank(paired.errors_a, paired.errors_b)

    thresholds = [float(t) for t in args.thresholds_km.split(",")]
    mcnemar_results = {}
    for threshold in thresholds:
        a_hits = [error <= threshold for error in paired.errors_a]
        b_hits = [error <= threshold for error in paired.errors_b]
        mcnemar_results[f"within_{threshold:g}km"] = mcnemar(a_hits, b_hits).to_dict()

    evaluation_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    directory = root / "experiments" / "evaluation" / evaluation_id
    directory.mkdir(parents=True, exist_ok=True)

    # Condition A restricted to the records Condition B also answered. Without
    # this, the headline columns compare A over 75 records against B over 60 --
    # different denominators, and the 15 B abstained on may be the hard ones, so
    # the difference could be entirely composition rather than method.
    paired_rows_a = [dict(rows_a[i]) for i in paired.ids]
    paired_rows_b = [dict(rows_b[i]) for i in paired.ids]
    summary_a_paired = summarise(paired_rows_a, total_records=paired.n_pairs).to_dict()
    summary_b_paired = summarise(paired_rows_b, total_records=paired.n_pairs).to_dict()

    # What Condition A scored on exactly the records Condition B declined.
    declined = [dict(rows_a[i]) for i in paired.only_a]
    summary_a_on_declined = summarise(declined, total_records=len(declined)).to_dict()

    metrics = {
        "evaluation_id": evaluation_id,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": args.baseline,
        "hybrid_run": args.hybrid,
        "denominators": paired.denominators(len(all_ids)),
        "condition_a_all_records": summary_a,
        "condition_b_all_records": summary_b,
        "condition_a_paired_subset": summary_a_paired,
        "condition_b_paired_subset": summary_b_paired,
        "condition_a_on_records_b_declined": summary_a_on_declined,
    }
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    statistics = {
        "note": (
            "Paired tests run over records where BOTH conditions produced a "
            "coordinate. That denominator is smaller than the benchmark and is "
            "stated above; comparing summary statistics computed over different "
            "denominators would not be a paired comparison."
        ),
        "denominators": paired.denominators(len(all_ids)),
        "wilcoxon_signed_rank": wilcoxon.to_dict(),
        "mcnemar_by_threshold": mcnemar_results,
    }
    (directory / "statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
    )

    # Case-level table for the qualitative analysis (master spec section 41).
    with open(directory / "paired_results.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "benchmark_id", "description", "status_a", "status_b",
            "a_latitude", "a_longitude", "b_latitude", "b_longitude",
            "true_latitude", "true_longitude", "error_a_km", "error_b_km",
            "difference_km", "selected_entity_b", "failure_reason_b",
        ])
        for identifier in all_ids:
            a, b = rows_a.get(identifier, {}), rows_b.get(identifier, {})
            reference = truth.get(identifier, {})
            error_a, error_b = a.get("error_km"), b.get("error_km")
            writer.writerow([
                identifier,
                (b.get("input_description") or a.get("input_description") or "")[:300],
                a.get("status"), b.get("status"),
                a.get("predicted_latitude"), a.get("predicted_longitude"),
                b.get("predicted_latitude"), b.get("predicted_longitude"),
                reference.get("decimal_latitude"), reference.get("decimal_longitude"),
                error_a, error_b,
                round(error_a - error_b, 4) if error_a is not None and error_b is not None else "",
                b.get("selected_candidate_name"), b.get("failure_reason"),
            ])

    def line(label, key, fmt="{:.3f}"):
        left = summary_a.get(key)
        right = summary_b.get(key)
        left = fmt.format(left) if isinstance(left, (int, float)) else "-"
        right = fmt.format(right) if isinstance(right, (int, float)) else "-"
        print(f"  {label:<22} {left:>12} {right:>12}")

    print(f"Evaluation {evaluation_id}\n")
    print("  All records each condition answered (DIFFERENT denominators):")
    print(f"{'':24}{'Condition A':>12} {'Condition B':>12}")
    line("coverage", "coverage")
    line("scored predictions", "denominator_evaluated_predictions", "{:.0f}")
    line("mean error km", "mean_error_km")
    line("median error km", "median_error_km")
    line("p90 error km", "p90_error_km")
    line("max error km", "max_error_km")
    line("within 1 km", "within_1km")
    line("within 5 km", "within_5km")
    line("within 10 km", "within_10km")
    line("over 50 km", "over_50km")

    def paired_line(label, key, fmt="{:.3f}"):
        left, right = summary_a_paired.get(key), summary_b_paired.get(key)
        left = fmt.format(left) if isinstance(left, (int, float)) else "-"
        right = fmt.format(right) if isinstance(right, (int, float)) else "-"
        print(f"  {label:<22} {left:>12} {right:>12}")

    print(f"\n  Like-for-like: the {paired.n_pairs} records BOTH conditions answered:")
    print(f"{'':24}{'Condition A':>12} {'Condition B':>12}")
    for label, key in (
        ("mean error km", "mean_error_km"),
        ("median error km", "median_error_km"),
        ("p90 error km", "p90_error_km"),
        ("within 1 km", "within_1km"),
        ("within 5 km", "within_5km"),
        ("over 50 km", "over_50km"),
    ):
        paired_line(label, key)

    if declined:
        print(f"\n  Condition A on the {len(declined)} records Condition B declined:")
        for key in ("mean_error_km", "median_error_km", "within_5km"):
            value = summary_a_on_declined.get(key)
            if isinstance(value, (int, float)):
                print(f"    {key:<22} {value:.3f}")
        print(
            "    If these are worse than A's paired subset, B abstained on the hard\n"
            "    cases and its coverage cost bought accuracy. If they are similar,\n"
            "    the abstentions were avoidable."
        )

    print(f"\n  Denominators: {json.dumps(paired.denominators(len(all_ids)))}")
    print("\n  Wilcoxon signed-rank on the paired errors (A minus B):")
    for key, value in wilcoxon.to_dict().items():
        print(f"    {key:<28} {value}")
    print("\n  McNemar on paired threshold outcomes:")
    for name, result in mcnemar_results.items():
        print(f"    {name:<14} p={result['p_value']:<10} discordant={result['discordant_pairs']}")

    print(f"\n  output   {directory}")
    print(
        "\n  A p-value is not a claim about practical significance, and the median\n"
        "  matters more than the mean on a distribution this skewed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
