#!/usr/bin/env python
"""Condition C: combine two finished runs under a deterministic rule.

    python scripts/combine_conditions.py --baseline <run> --hybrid <run> \
        --dataset dev                       # fix the thresholds here
    python scripts/combine_conditions.py --baseline <run> --hybrid <run> \
        --strategy agreement --agreement-km 10

POST-BENCHMARK EXPLORATORY ANALYSIS.

Condition C was conceived after the A-versus-B benchmark was evaluated. A rule
invented to fit an observed result and then reported against that same result
proves nothing, so this never replaces the frozen comparison and is never a
headline. It is held to the same standards as the rest of the work: rules
specified from the design rather than from the outcome, thresholds fixed on the
dev set, denominators stated, and the label attached to every artefact it
writes.

Runs no model and makes no request. It reads two prediction files and applies
arithmetic, so it costs nothing and cannot be confounded by a different model.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import load_paths_config, load_yaml, repo_root  # noqa: E402
from georef.evaluation.metrics import geodesic_error_km, summarise  # noqa: E402
from georef.evaluation.statistics import (  # noqa: E402
    mcnemar,
    pair_conditions,
    wilcoxon_signed_rank,
)
from georef.pipeline.condition_c import STRATEGIES, combine  # noqa: E402


def load_predictions(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"no predictions at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["benchmark_id"]: row for row in rows}


def score(rows: dict, truth: dict) -> dict:
    for identifier, row in rows.items():
        reference = truth.get(identifier)
        latitude, longitude = row.get("predicted_latitude"), row.get("predicted_longitude")
        row["error_km"] = (
            round(
                geodesic_error_km(
                    float(latitude), float(longitude),
                    float(reference["decimal_latitude"]), float(reference["decimal_longitude"]),
                ), 4,
            )
            if reference and latitude is not None and longitude is not None
            else None
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Condition C: combine two runs.")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--hybrid", default=None)
    parser.add_argument(
        "--pair", action="append", default=[],
        help="dataset:baseline_run:hybrid_run -- repeatable, so selection can pool "
             "the dev and benchmark sets into one selection set",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="search the pre-registered parameter grids and report every combination",
    )
    parser.add_argument(
        "--dataset", choices=["benchmark", "dev", "validation"], default="benchmark"
    )
    parser.add_argument("--strategy", choices=list(STRATEGIES) + ["all"], default="all")
    parser.add_argument("--agreement-km", type=float, default=10.0)
    parser.add_argument("--minimum-score", type=float, default=0.6)
    args = parser.parse_args()

    root = repo_root()
    paths = load_paths_config()

    def dataset_paths(name: str):
        if name == "validation":
            return (root / "data" / "validation" / "validation_ground_truth.csv",
                    "validation_id", "validation_baseline", "validation_hybrid")
        if name == "dev":
            return (root / paths["dev_dir"] / "dev_ground_truth.csv",
                    "dev_gt_id", "dev_baseline", "dev_hybrid")
        return (root / paths["ground_truth_dir"] / "ground_truth.csv",
                "benchmark_id", "baseline", "hybrid")

    # --- pooled selection over several sets -------------------------------
    if args.pair:
        pooled_a, pooled_b, pooled_truth = {}, {}, {}
        for spec in args.pair:
            try:
                name, run_a, run_b = spec.split(":")
            except ValueError:
                raise SystemExit(f"--pair must be dataset:baseline:hybrid, got {spec!r}")
            truth_file, id_field, dir_a, dir_b = dataset_paths(name)
            with open(truth_file, "r", encoding="utf-8", newline="") as handle:
                truth_rows = {row[id_field]: row for row in csv.DictReader(handle)}
            pooled_truth.update(truth_rows)
            pooled_a.update(
                score(load_predictions(root / "experiments" / dir_a / run_a / "predictions.jsonl"),
                      truth_rows)
            )
            pooled_b.update(
                score(load_predictions(root / "experiments" / dir_b / run_b / "predictions.jsonl"),
                      truth_rows)
            )
            print(f"pooled {name}: {len(truth_rows)} records")
        return report(root, args, pooled_a, pooled_b, sorted(pooled_truth), pooled_truth,
                      label="SELECTION (pooled)")

    if args.dataset == "validation":
        # The pre-registered run. It uses the frozen rule and nothing else, and
        # refuses to proceed if that rule has not been chosen on the dev set.
        c_config = load_yaml(root / "configs" / "condition_c.yaml")
        if not c_config.get("strategy"):
            print(
                "configs/condition_c.yaml has no strategy.\n\n"
                "The operating rule must be selected on the DEV set before the\n"
                "validation set is read -- see docs/CONDITION_C_PREREGISTRATION.md\n"
                "section 4. Refusing to run.",
                file=sys.stderr,
            )
            return 2
        args.strategy = c_config["strategy"]
        if c_config.get("agreement_km") is not None:
            args.agreement_km = float(c_config["agreement_km"])
        if c_config.get("minimum_score") is not None:
            args.minimum_score = float(c_config["minimum_score"])
        truth_path = root / "data" / "validation" / "validation_ground_truth.csv"
        id_field = "validation_id"
        baseline_dir, hybrid_dir = "validation_baseline", "validation_hybrid"
    elif args.dataset == "dev":
        truth_path = root / paths["dev_dir"] / "dev_ground_truth.csv"
        id_field, baseline_dir, hybrid_dir = "dev_gt_id", "dev_baseline", "dev_hybrid"
    else:
        truth_path = root / paths["ground_truth_dir"] / "ground_truth.csv"
        id_field, baseline_dir, hybrid_dir = "benchmark_id", "baseline", "hybrid"

    with open(truth_path, "r", encoding="utf-8", newline="") as handle:
        truth = {row[id_field]: row for row in csv.DictReader(handle)}

    if not args.baseline or not args.hybrid:
        raise SystemExit("--baseline and --hybrid are required unless --pair is used")
    rows_a = score(
        load_predictions(root / "experiments" / baseline_dir / args.baseline / "predictions.jsonl"),
        truth,
    )
    rows_b = score(
        load_predictions(root / "experiments" / hybrid_dir / args.hybrid / "predictions.jsonl"),
        truth,
    )
    return report(root, args, rows_a, rows_b, sorted(truth), truth)


def report(root, args, rows_a, rows_b, identifiers, truth, label=None):
    # A pooled selection run is not a benchmark evaluation, and must not print a
    # banner saying it is. The label is passed by the caller that knows.
    pooled = label is not None
    grids = load_yaml(repo_root() / "configs" / "condition_c.yaml").get("parameter_grids", {})
    if args.sweep:
        # Only the pre-registered grids. Fixed in configs/condition_c.yaml so the
        # search space cannot quietly grow once the numbers are in view.
        combinations = [("fallback", {}), ("prefer_baseline", {})]
        combinations += [("agreement", {"agreement_km": k}) for k in grids.get("agreement_km", [])]
        combinations += [
            ("confidence", {"minimum_score": v}) for v in grids.get("minimum_score", [])
        ]
    else:
        chosen = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
        combinations = [(name, {}) for name in chosen]

    strategies = [name for name, _ in combinations]
    evaluation_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    directory = root / "experiments" / "combined" / evaluation_id
    directory.mkdir(parents=True, exist_ok=True)

    summary_a = summarise(list(rows_a.values()), total_records=len(identifiers)).to_dict()
    summary_b = summarise(list(rows_b.values()), total_records=len(identifiers)).to_dict()

    if pooled:
        print("SELECTION RUN -- pooled selection set")
        print("Parameters chosen here are fixed before the held-out set is read.")
        print("These numbers are selection evidence, never a reported result.\n")
    elif args.dataset == "validation":
        print("PRE-REGISTERED VALIDATION RUN")
        print("Rule fixed in configs/condition_c.yaml, chosen on the dev set.")
        print(f"strategy {args.strategy}  agreement_km {args.agreement_km} "
              f"minimum_score {args.minimum_score}")
        print("Read once. The result stands whatever it says.\n")
    else:
        print("EXPLORATORY -- not the pre-registered validation run")
        print("Condition C was designed after A vs B was evaluated, so numbers on")
        print("the benchmark cannot validate it. Select the rule with --dataset dev,")
        print("then evaluate once with --dataset validation.\n")
    scope = "pooled selection set" if pooled else f"dataset {args.dataset}"
    print(f"{scope}   records {len(identifiers)}\n")
    header = f"{'':22}{'coverage':>10}{'mean km':>10}{'median km':>11}{'<1km':>8}{'>50km':>8}"
    print(header)

    def show(label, summary):
        print(
            f"  {label:<20}{summary.get('coverage', 0):>10.3f}"
            f"{summary.get('mean_error_km', float('nan')):>10.3f}"
            f"{summary.get('median_error_km', float('nan')):>11.3f}"
            f"{summary.get('within_1km', 0):>8.3f}{summary.get('over_50km', 0):>8.3f}"
        )

    show("A (baseline)", summary_a)
    show("B (hybrid)", summary_b)

    results = {}
    for strategy, overrides in combinations:
        agreement_km = overrides.get("agreement_km", args.agreement_km)
        minimum_score = overrides.get("minimum_score", args.minimum_score)
        variant = strategy + (
            f"@{agreement_km}km" if strategy == "agreement" and args.sweep else
            f"@{minimum_score}" if strategy == "confidence" and args.sweep else ""
        )
        combined = combine(
            rows_a, rows_b, identifiers,
            strategy=strategy,
            agreement_km=agreement_km,
            minimum_score=minimum_score,
        )
        rows_c = {}
        for identifier, result in combined.items():
            record = result.to_record()
            reference = truth.get(identifier)
            if reference and result.latitude is not None:
                record["error_km"] = round(
                    geodesic_error_km(
                        result.latitude, result.longitude,
                        float(reference["decimal_latitude"]),
                        float(reference["decimal_longitude"]),
                    ), 4,
                )
            else:
                record["error_km"] = None
            rows_c[identifier] = record

        summary_c = summarise(list(rows_c.values()), total_records=len(identifiers)).to_dict()
        show(f"C ({variant})", summary_c)

        paired = pair_conditions(rows_a, rows_c, identifiers)
        wilcoxon = wilcoxon_signed_rank(paired.errors_a, paired.errors_b)
        within_1km = mcnemar(
            [e <= 1 for e in paired.errors_a], [e <= 1 for e in paired.errors_b]
        )

        # H5 and H7 compare C against B, not against A. A rule that beats one
        # condition but not the other has not combined them; it has picked one.
        paired_b = pair_conditions(rows_b, rows_c, identifiers)
        wilcoxon_b = wilcoxon_signed_rank(paired_b.errors_a, paired_b.errors_b)
        over_50km_b = mcnemar(
            [e > 50 for e in paired_b.errors_a], [e > 50 for e in paired_b.errors_b]
        )
        sources = {}
        for record in rows_c.values():
            sources[record.get("source_condition")] = sources.get(record.get("source_condition"), 0) + 1

        results[variant] = {
            "strategy": strategy,
            "agreement_km": agreement_km,
            "minimum_score": minimum_score,
            "summary": summary_c,
            "sources_used": sources,
            "vs_baseline_wilcoxon": wilcoxon.to_dict(),
            "vs_baseline_mcnemar_1km": within_1km.to_dict(),
            "vs_hybrid_wilcoxon": wilcoxon_b.to_dict(),
            "vs_hybrid_mcnemar_over_50km": over_50km_b.to_dict(),
            "vs_hybrid_denominators": paired_b.denominators(len(identifiers)),
            "denominators": paired.denominators(len(identifiers)),
        }

        with open(directory / f"predictions_{variant.replace('@','_')}.jsonl", "w", encoding="utf-8",
                  newline="\n") as handle:
            for identifier in identifiers:
                handle.write(json.dumps(rows_c[identifier], ensure_ascii=False) + "\n")

    (directory / "combined_metrics.json").write_text(
        json.dumps(
            {
                "label": "POST-BENCHMARK EXPLORATORY -- not a frozen result",
                "evaluation_id": evaluation_id,
                "dataset": args.dataset,
                "baseline_run": args.baseline,
                "hybrid_run": args.hybrid,
                "parameters": {
                    "agreement_km": args.agreement_km,
                    "minimum_score": args.minimum_score,
                },
                "condition_a": summary_a,
                "condition_b": summary_b,
                "strategies": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.sweep:
        control = results.get("prefer_baseline", {}).get("summary", {})
        print("\n  Ranked by the pre-registered criterion (median error, then >50 km):")
        ranked = sorted(
            ((name, payload) for name, payload in results.items() if name != "prefer_baseline"),
            key=lambda item: (
                item[1]["summary"].get("median_error_km", float("inf")),
                item[1]["summary"].get("over_50km", 1.0),
            ),
        )
        for position, (name, payload) in enumerate(ranked, start=1):
            summary = payload["summary"]
            beats = (
                "beats control"
                if summary.get("median_error_km", 9e9) < control.get("median_error_km", 0)
                else "does NOT beat control"
            )
            print(
                f"    {position:>2}. {name:<22} median {summary.get('median_error_km'):>7.3f}"
                f"  >50km {summary.get('over_50km'):>6.3f}  {beats}"
            )

    print("\n  Which condition each rule drew from:")
    for strategy, payload in results.items():
        print(f"    {strategy:<16} {payload['sources_used']}")

    print("\n  Against Condition A, paired (H4, H8):")
    for strategy, payload in results.items():
        w = payload["vs_baseline_wilcoxon"]
        m = payload["vs_baseline_mcnemar_1km"]
        print(
            f"    {strategy:<16} Wilcoxon p={w['p_value']:<10} "
            f"median diff {w['median_difference_km']:>8} km   "
            f"McNemar<1km p={m['p_value']}   n={w['n_used']}"
        )

    print("\n  Against Condition B, paired (H5, H7):")
    for strategy, payload in results.items():
        w = payload["vs_hybrid_wilcoxon"]
        m = payload["vs_hybrid_mcnemar_over_50km"]
        print(
            f"    {strategy:<16} Wilcoxon p={w['p_value']:<10} "
            f"median diff {w['median_difference_km']:>8} km   "
            f"McNemar>50km p={m['p_value']}   n={w['n_used']}"
        )
        print(f"      denominators: {json.dumps(payload['vs_hybrid_denominators'])}")

    print(f"\n  output   {directory}")
    if pooled:
        print(
            "\n  Selection evidence. Write the winning rule into configs/condition_c.yaml,\n"
            "  freeze, then read the held-out set once."
        )
    elif args.dataset == "benchmark":
        print(
            "\n  Exploratory. Any threshold these numbers suggest was chosen with the\n"
            "  benchmark in view, so a rule tuned here cannot be reported as validated.\n"
            "  Select on --dataset dev, then evaluate once on --dataset validation."
        )
    elif args.dataset == "validation":
        print(
            "\n  The validation set has now been read. There is no further unused data\n"
            "  in this corpus: any subsequent hypothesis is exploratory in the strict\n"
            "  sense and must be reported as such."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
