#!/usr/bin/env python
"""Run the benchmark for one or both conditions (master spec sections 12 and 52).

    python scripts/run_benchmark.py --condition A
    python scripts/run_benchmark.py --condition B
    python scripts/run_benchmark.py --condition both

Both conditions see the SAME descriptions and the SAME model. Ground truth is
never loaded here: this script cannot see it, which is the point.

Refuses to run against a changed configuration unless --allow-unfrozen is given,
because a benchmark run that does not match the freeze is not the frozen
experiment. Resumable by record id, so an interruption costs nothing and a rerun
cannot silently overwrite a completed record.
"""

import argparse
import csv
import json
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import (  # noqa: E402
    load_paths_config,
    load_yaml,
    repo_root,
    resolve_llm_profile,
)
from georef.disambiguation.resolver import Disambiguator  # noqa: E402
from georef.linz.ingest import _load_dotenv  # noqa: E402
from georef.llm.baseline import BaselineGeoreferencer  # noqa: E402
from georef.llm.client import LlmClient  # noqa: E402
from georef.llm.extraction import SpatialExtractor  # noqa: E402
from georef.pipeline.condition_b import ConditionBPipeline  # noqa: E402
from georef.pipeline.grounding import Grounder  # noqa: E402
from georef.pipeline.runner import JsonlCheckpoint, run_parallel  # noqa: E402
from georef.provenance.freeze import verify_against_freeze  # noqa: E402
from georef.retrieval.hybrid import HybridRetriever  # noqa: E402
from georef.retrieval.lexical import LexicalIndex  # noqa: E402


def build_condition_b(root, models, apis):
    """Assemble the hybrid pipeline from the frozen snapshot and index."""
    from georef.linz.verify import load_snapshot
    from georef.retrieval.name_spotter import NameSpotter

    retrieval_config = load_yaml(root / "configs" / "retrieval.yaml")
    pipeline_config = load_yaml(root / "configs" / "pipeline.yaml")
    disambiguation_config = load_yaml(root / "configs" / "disambiguation.yaml")
    linz_config = load_yaml(root / "configs" / "linz.yaml")

    snapshots = sorted((root / linz_config["paths"]["normalised_dir"]).glob("*.jsonl"))
    if not snapshots:
        raise SystemExit("No LINZ snapshot. Run scripts/fetch_linz.py first.")
    records = load_snapshot(snapshots[-1])

    lexical = LexicalIndex(records, retrieval_config["document_template"])

    embedder = store = None
    if retrieval_config["channels"]["dense"]["enabled"]:
        from georef.retrieval.dense import ChromaVectorStore, SentenceTransformerEmbedder

        embedding_config = models["embedding"]
        embedder = SentenceTransformerEmbedder(
            embedding_config["model"], embedding_config.get("normalise_embeddings", True)
        )
        store = ChromaVectorStore(
            root / "data" / "chroma",
            retrieval_config["collection_name"],
            embedding_config["similarity"],
        )

    spotter = NameSpotter(
        records,
        ambiguity_radius_km=float(
            disambiguation_config["feature_scores"].get("coherence_context_ambiguity_km", 25)
        ),
    )
    grounder = Grounder(
        HybridRetriever(lexical, retrieval_config, embedder, store),
        Disambiguator(disambiguation_config),
        pipeline_config,
        spotter,
    )
    return ConditionBPipeline(SpatialExtractor(LlmClient(models, apis)), grounder), {
        "snapshot": snapshots[-1].name,
        "gazetteer_records": len(records),
        "channels": [n for n, c in retrieval_config["channels"].items() if c["enabled"]],
        "multi_anchor_policy": pipeline_config["multi_anchor"]["policy"],
        "minimum_score": disambiguation_config["abstention"]["minimum_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument("--condition", choices=["A", "B", "both"], required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="pilot subset, not a benchmark run")
    parser.add_argument(
        "--dataset", choices=["benchmark", "dev", "validation"], default="benchmark",
        help="dev runs a condition over the development descriptions, for fixing "
             "parameters before they are applied to the benchmark",
    )
    parser.add_argument("--run-id", default=None, help="resume a specific run")
    parser.add_argument("--allow-unfrozen", action="store_true",
                        help="run despite configuration drift; results are NOT the frozen experiment")
    args = parser.parse_args()

    root = repo_root()
    _load_dotenv(root / ".env")
    paths = load_paths_config()
    apis = load_yaml(root / "configs" / "apis.yaml")
    models = resolve_llm_profile(load_yaml(root / "configs" / "models.yaml"), args.profile)

    # --- freeze gate -------------------------------------------------------
    freeze_path = root / "experiments" / "freeze_manifest.json"
    freeze = {}
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        # The validation run is a frozen experiment too: it tests a
        # pre-registered rule and must be held to the same standard.
        differences = (
            verify_against_freeze(root, freeze)
            if args.dataset in ("benchmark", "validation")
            else []
        )
        if differences:
            print(f"CONFIGURATION HAS CHANGED SINCE FREEZE {freeze['freeze_sha256'][:16]}\n")
            for difference in differences[:12]:
                print(f"  {difference}")
            if not args.allow_unfrozen:
                print(
                    "\nRefusing to run. A benchmark against this state is not the frozen\n"
                    "experiment. Revert, or freeze a new version, or pass --allow-unfrozen\n"
                    "and accept that the output is a pilot rather than a result."
                )
                return 2
            print("\n  --allow-unfrozen given: continuing. This is NOT the frozen experiment.\n")
    elif args.dataset == "dev":
        pass  # dev runs are for fixing parameters, and precede the freeze
    elif not args.allow_unfrozen:
        print("No freeze manifest. Run scripts/freeze.py first, or pass --allow-unfrozen.",
              file=sys.stderr)
        return 2

    if args.dataset in ("dev", "validation"):
        # Descriptions only, and the id column is named differently. Normalised
        # here so downstream code never needs to know which set it is reading.
        if args.dataset == "dev":
            source_path = root / paths["dev_dir"] / "dev_descriptions.csv"
            id_column = "dev_gt_id"
        else:
            source_path = root / "data" / "validation" / "validation.csv"
            id_column = "validation_id"
        with open(source_path, "r", encoding="utf-8", newline="") as handle:
            rows = [
                {"benchmark_id": row[id_column], "description": row["description"]}
                for row in csv.DictReader(handle)
            ]
    else:
        source_path = root / paths["benchmark_dir"] / "benchmark.csv"
        with open(source_path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    conditions = ["A", "B"] if args.condition == "both" else [args.condition]
    workers = args.workers or int(apis[models["llm"]["provider"]].get("max_workers", 1))

    for condition in conditions:
        folder = "baseline" if condition == "A" else "hybrid"
        if args.dataset != "benchmark":
            folder = f"{args.dataset}_{folder}"
        run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        directory = root / "experiments" / folder / run_id
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = JsonlCheckpoint(directory / "predictions.jsonl", "benchmark_id")

        extra = {}
        if condition == "A":
            engine = BaselineGeoreferencer(LlmClient(models, apis))
            work = lambda row: engine.predict(row["benchmark_id"], row["description"]).to_record()
        else:
            pipeline, extra = build_condition_b(root, models, apis)
            work = lambda row: pipeline.predict(row["benchmark_id"], row["description"]).to_record()

        done = checkpoint.completed()
        pending = [r for r in rows if r["benchmark_id"] not in done]
        print(f"\n=== CONDITION {condition} ({folder}) ===")
        print(f"run id   {run_id}")
        print(f"model    {models['llm']['model']}  ({models['llm'].get('profile')})")
        print(f"records  {len(rows)}, {len(done)} done, {len(pending)} to run, {workers} workers")
        for key, value in extra.items():
            print(f"  {key:<22} {value}")

        started = datetime.now(timezone.utc)
        if pending:
            run_parallel(
                pending, work, checkpoint, workers=workers,
                on_result=lambda record, position, total: print(
                    f"  [{position}/{total}] {record['benchmark_id']}  {record['status']}",
                    flush=True,
                ),
            )
        total_records = checkpoint.finalise()
        everything = checkpoint.completed()
        statuses = Counter(r["status"] for r in everything.values())

        manifest = {
            "run_id": run_id,
            "condition": condition,
            "dataset": args.dataset,
            "benchmark_size": len(rows),
            "records_written": total_records,
            "status_counts": dict(statuses),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "llm": {k: v for k, v in models["llm"].items()},
            "served_models": sorted(
                {r["served_model"] for r in everything.values() if r.get("served_model")}
            ),
            "freeze_sha256": freeze.get("freeze_sha256"),
            "allow_unfrozen": bool(args.allow_unfrozen),
            "is_pilot": bool(args.limit),
            "configuration": extra,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        print("\n  status counts:")
        for status, count in statuses.most_common():
            print(f"    {status:<28} {count}")
        if len(manifest["served_models"]) > 1:
            print(f"\n  WARNING: more than one served model: {manifest['served_models']}")
        print(f"\n  output   {directory}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
