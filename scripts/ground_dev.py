#!/usr/bin/env python
"""Ground the interpretations you already have, and score them on the dev set.

    python scripts/ground_dev.py
    python scripts/ground_dev.py --policy shortest --no-dense

Runs the offline half of Condition B over data/dev/interpretations.jsonl:
retrieval, disambiguation, deterministic geodesy, coordinate sanity. No network
and no model -- only the frozen LINZ snapshot and the index built from it.

Then joins to the dev ground truth and reports geodesic error. This is the DEV
set, whose declared purpose is exactly this: choosing the configuration once,
before the freeze. It is not the benchmark, and no headline result comes from it.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import load_paths_config, load_yaml, repo_root  # noqa: E402
from georef.disambiguation.resolver import Disambiguator  # noqa: E402
from georef.evaluation.metrics import geodesic_error_km, summarise  # noqa: E402
from georef.pipeline.grounding import Grounder  # noqa: E402
from georef.retrieval.hybrid import HybridRetriever  # noqa: E402
from georef.retrieval.lexical import LexicalIndex  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ground and score the dev interpretations.")
    parser.add_argument("--policy", default=None, choices=["first", "shortest", "fuse"])
    parser.add_argument("--no-dense", action="store_true", help="lexical channels only")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="load the cross-encoder reranker (downloads ~2.3 GB on first use, and "
             "adds seconds per record on CPU)",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-coherence", action="store_true",
                        help="disable the spatial-coherence check against other named places")
    parser.add_argument("--interpretations", default=None,
                        help="path to an interpretations.jsonl, for comparing variants")
    args = parser.parse_args()

    root = repo_root()
    paths = load_paths_config()
    retrieval_config = load_yaml(root / "configs" / "retrieval.yaml")
    pipeline_config = load_yaml(root / "configs" / "pipeline.yaml")
    disambiguation_config = load_yaml(root / "configs" / "disambiguation.yaml")
    linz_config = load_yaml(root / "configs" / "linz.yaml")
    models_config = load_yaml(root / "configs" / "models.yaml")

    if args.policy:
        pipeline_config["multi_anchor"]["policy"] = args.policy
    # Reranking is opt-in: the model is a large download and costs seconds per
    # record on CPU. Whether it earns that is one of the questions the dev set
    # exists to answer.
    retrieval_config["rerank"]["enabled"] = bool(args.rerank) and not args.no_rerank
    if args.no_dense:
        retrieval_config["channels"]["dense"]["enabled"] = False

    interpretations_path = (
        Path(args.interpretations)
        if args.interpretations
        else root / paths["dev_dir"] / "interpretations.jsonl"
    )
    if not interpretations_path.is_file():
        print("No interpretations found. Run scripts/extract_dev.py first.", file=sys.stderr)
        return 2
    interpretations = load_jsonl(interpretations_path)

    snapshots = sorted((root / linz_config["paths"]["normalised_dir"]).glob("*.jsonl"))
    if not snapshots:
        print("No LINZ snapshot found. Run scripts/fetch_linz.py first.", file=sys.stderr)
        return 2
    records = load_jsonl(snapshots[-1])
    print(f"Snapshot     {snapshots[-1].name}  ({len(records)} records)")

    lexical = LexicalIndex(records, retrieval_config["document_template"])

    embedder = vector_store = None
    if retrieval_config["channels"]["dense"]["enabled"]:
        try:
            from georef.retrieval.dense import ChromaVectorStore, SentenceTransformerEmbedder

            embedding_config = models_config["embedding"]
            embedder = SentenceTransformerEmbedder(
                embedding_config["model"], embedding_config.get("normalise_embeddings", True)
            )
            vector_store = ChromaVectorStore(
                root / "data" / "chroma",
                retrieval_config["collection_name"],
                embedding_config["similarity"],
            )
            print(f"Dense        {embedding_config['model']}  ({vector_store.count()} vectors)")
        except Exception as error:  # noqa: BLE001
            print(f"Dense channel unavailable ({error}); continuing with lexical only.")
            retrieval_config["channels"]["dense"]["enabled"] = False

    reranker = None
    if retrieval_config["rerank"]["enabled"]:
        try:
            from georef.retrieval.rerank import CrossEncoderReranker

            reranker_config = models_config["reranker"]
            reranker = CrossEncoderReranker(
                reranker_config["model"], reranker_config.get("device", "cpu")
            )
            print(f"Reranker     {reranker_config['model']} on {reranker_config.get('device','cpu')}")
        except Exception as error:  # noqa: BLE001
            print(f"Reranker unavailable ({error}); continuing without it.")
            retrieval_config["rerank"]["enabled"] = False

    channels = [name for name, cfg in retrieval_config["channels"].items() if cfg["enabled"]]
    print(f"Channels     {', '.join(channels)}")
    spotter = None
    if not args.no_coherence:
        from georef.retrieval.name_spotter import NameSpotter

        spotter = NameSpotter(
            records,
            ambiguity_radius_km=float(
                disambiguation_config["feature_scores"].get(
                    "coherence_context_ambiguity_km", 25
                )
            ),
        )

    # Report what is actually wired, not what the config asks for: a run that
    # says it reranked and did not would quietly invalidate the comparison.
    print(f"Rerank       {reranker is not None}")
    print(f"Coherence    {spotter is not None}")
    print(f"Policy       {pipeline_config['multi_anchor']['policy']}\n")

    retriever = HybridRetriever(lexical, retrieval_config, embedder, vector_store, reranker)
    grounder = Grounder(
        retriever, Disambiguator(disambiguation_config), pipeline_config, spotter
    )

    truth_path = root / paths["dev_dir"] / "dev_ground_truth.csv"
    with open(truth_path, "r", encoding="utf-8", newline="") as handle:
        truth = {row["dev_gt_id"]: row for row in csv.DictReader(handle)}

    rows = []
    for interpretation in interpretations:
        prediction = grounder.predict(interpretation)
        row = prediction.to_record()
        row["input_description"] = interpretation.get("input_description")
        row["interpretation_status"] = interpretation.get("status")

        reference = truth.get(prediction.benchmark_id)
        if prediction.is_success and reference:
            row["error_km"] = round(
                geodesic_error_km(
                    prediction.latitude,
                    prediction.longitude,
                    float(reference["decimal_latitude"]),
                    float(reference["decimal_longitude"]),
                ),
                4,
            )
            row["true_latitude"] = float(reference["decimal_latitude"])
            row["true_longitude"] = float(reference["decimal_longitude"])
        rows.append(row)

    output_path = Path(args.output) if args.output else root / paths["dev_dir"] / "dev_predictions.jsonl"
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda r: r["benchmark_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # A feature that silently evaluates to its neutral value on every record is
    # indistinguishable from a feature that is switched off, and would make an
    # ablation report a comparison of nothing against nothing. Checked rather
    # than assumed, because it already happened once.
    if spotter is not None:
        neutral = disambiguation_config["feature_scores"]["coherence_absent"]
        values = {
            candidate["features"].get("spatial_coherence")
            for row in rows
            for selection in row.get("clause_selections", [])
            for candidate in selection.get("candidates_considered", [])
        }
        if values and values <= {neutral}:
            print(
                "\n  WARNING: coherence is enabled but scored neutral on every candidate.\n"
                "  Either no description named another gazetteer place, or the spotter is\n"
                "  not wired. These results are NOT a test of coherence."
            )

    summary = summarise(rows, total_records=len(truth))
    print("Outcome by status:")
    for status, count in sorted(summary.status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<26} {count}")

    stats = summary.to_dict()
    print("\nGeodesic error, evaluated predictions only:")
    print(f"  denominator (dev records)  {stats['denominator_all_records']}")
    print(f"  scored predictions         {stats['denominator_evaluated_predictions']}")
    print(f"  coverage                   {stats['coverage']:.1%}")
    if stats.get("mean_error_km") is not None:
        for key in ("mean_error_km", "median_error_km", "p90_error_km", "max_error_km"):
            print(f"  {key:<26} {stats[key]}")
        for key in ("within_1km", "within_5km", "within_10km", "over_50km"):
            print(f"  {key:<26} {stats[key]:.1%}")

    worst = sorted(
        (r for r in rows if r.get("error_km") is not None), key=lambda r: -r["error_km"]
    )[:5]
    if worst:
        print("\nLargest errors:")
        for row in worst:
            print(
                f"  {row['benchmark_id']}  {row['error_km']:>8.1f} km  "
                f"{(row.get('selected_candidate_name') or '?')[:28]:<28}"
                f"{(row.get('input_description') or '')[:60]}"
            )

    print(f"\n  output   {output_path}")
    print(
        "\n  Reminder: this is the DEV set, used once to choose a configuration\n"
        "  before the freeze. No headline result comes from it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
