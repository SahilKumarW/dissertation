#!/usr/bin/env python
"""Step 4: run semantic interpretation over the dev ground-truth set.

    python scripts/extract_dev.py --limit 10        # pilot first
    python scripts/extract_dev.py                   # full set, resumes
    python scripts/extract_dev.py --workers 8       # override configured workers

Writes one JSON object per record to data/dev/interpretations.jsonl, sorted by
record id when the run completes. Resume is by record id: a completed record is
never re-requested and never overwritten, so an interrupted run costs nothing.
"""

import argparse
import csv
import sys
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
from georef.linz.ingest import _load_dotenv  # noqa: E402
from georef.llm.client import LlmClient  # noqa: E402
from georef.llm.extraction import PROMPT_VERSION, SpatialExtractor  # noqa: E402
from georef.pipeline.runner import JsonlCheckpoint, run_parallel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Interpret the dev descriptions.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="defaults to nvidia.max_workers in configs/apis.yaml")
    parser.add_argument("--hints", action="store_true",
                        help="two-pass: query the gazetteer first and tell the model which "
                             "names in the description are real NZ places (names only)")
    parser.add_argument("--profile", default=None,
                        help="LLM profile from configs/models.yaml, e.g. gemini_flash")
    parser.add_argument("--retry-failures", action="store_true",
                        help="re-request records whose previous attempt failed")
    args = parser.parse_args()

    root = repo_root()
    _load_dotenv(root / ".env")
    paths = load_paths_config()
    apis = load_yaml(root / "configs" / "apis.yaml")
    models = resolve_llm_profile(load_yaml(root / "configs" / "models.yaml"), args.profile)

    # Descriptions only. This stage must not open a file containing answers,
    # for the same reason the benchmark run cannot: what it cannot see, it
    # cannot be influenced by.
    dev_path = root / paths["dev_dir"] / "dev_descriptions.csv"
    if not dev_path.is_file():
        print("dev_descriptions.csv not found. Run scripts/build_benchmark.py.", file=sys.stderr)
        return 2

    with open(dev_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    checkpoint = JsonlCheckpoint(root / paths["dev_dir"] / "interpretations.jsonl", "benchmark_id")
    done = checkpoint.completed()

    pending = [
        row
        for row in rows
        if row["dev_gt_id"] not in done
        or (args.retry_failures and done[row["dev_gt_id"]]["status"] != "SUCCESS")
    ]

    provider = models["llm"]["provider"]
    workers = args.workers or int(apis[provider].get("max_workers", 1))
    print(
        f"{len(rows)} records, {len(done)} already done, {len(pending)} to request, "
        f"{workers} worker(s)."
    )
    print(
        f"profile {models['llm'].get('profile')}  provider {provider}  "
        f"model {models['llm']['model']}\n"
    )

    # Two-pass retrieval. Off unless asked for: it is an ablation arm, not a
    # change to the pipeline, and enabling it silently would bundle another
    # component into Condition B with no way to attribute the effect.
    retrieval_config = load_yaml(root / "configs" / "retrieval.yaml")
    hint_config = retrieval_config.get("pre_retrieval_hints", {})
    use_hints = args.hints or bool(hint_config.get("enabled"))

    spotter = None
    if use_hints:
        from georef.linz.verify import load_snapshot
        from georef.retrieval.name_spotter import NameSpotter

        linz_config = load_yaml(root / "configs" / "linz.yaml")
        snapshots = sorted((root / linz_config["paths"]["normalised_dir"]).glob("*.jsonl"))
        if not snapshots:
            print("--hints needs a LINZ snapshot. Run scripts/fetch_linz.py.", file=sys.stderr)
            return 2
        spotter = NameSpotter(
            load_snapshot(snapshots[-1]),
            max_name_tokens=int(hint_config.get("max_name_tokens", 4)),
        )
        print(f"hints    on, from {snapshots[-1].name}")

    client = LlmClient(models, apis)
    extractor = SpatialExtractor(
        client,
        name_spotter=spotter,
        max_hints=int(hint_config.get("max_hints", 12)),
    )

    def work(row: dict) -> dict:
        result = extractor.extract(row["dev_gt_id"], row["description"])
        record = result.to_record()
        record["requested_at_utc"] = datetime.now(timezone.utc).isoformat()
        return record

    def report(record: dict, position: int, total: int) -> None:
        print(
            f"  [{position}/{total}] {record['benchmark_id']}  {record['status']:<28}"
            f"{(record.get('anchor_entity') or '-')[:34]}",
            flush=True,
        )

    started = datetime.now(timezone.utc)
    if pending:
        run_parallel(pending, work, checkpoint, workers=workers, on_result=report)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    total_records = checkpoint.finalise()
    everything = checkpoint.completed()
    statuses = Counter(record["status"] for record in everything.values())

    print("\nStatus counts:")
    for status, count in statuses.most_common():
        print(f"  {status:<30} {count}")

    usable = statuses.get("SUCCESS", 0)
    print(f"\n  usable interpretations   {usable}/{total_records}")
    print(f"  prompt version           {PROMPT_VERSION}")
    print(f"  gazetteer hints          {'on' if use_hints else 'off'}")
    hinted = [r for r in everything.values() if r.get("gazetteer_hints")]
    if hinted:
        total_hints = sum(len(r["gazetteer_hints"]) for r in hinted)
        print(
            f"  records given hints      {len(hinted)}"
            f" (mean {total_hints / len(hinted):.1f} names each)"
        )
    print(f"  max_tokens               {models['llm']['max_tokens']}")
    if pending:
        print(f"  wall clock               {elapsed:.1f}s for {len(pending)} records")
        # processing_time spans every attempt including backoff waits, so it is
        # not a per-request latency and is not labelled as one.
        per_record = [r.get("processing_time") or 0 for r in everything.values()]
        if per_record:
            print(f"  mean time per record     {sum(per_record) / len(per_record):.1f}s (incl. retries)")

    attempts = Counter(r.get("attempt_count", 0) for r in everything.values())
    retried = sum(count for attempt, count in attempts.items() if attempt > 1)
    print(f"  records needing a retry  {retried}/{total_records}")
    if retried > total_records * 0.2:
        print(
            "  The endpoint is throttling or dropping connections. Lower --workers"
            " before raising it: more concurrency here buys queueing, not throughput."
        )

    served = {r.get("served_model") for r in everything.values() if r.get("served_model")}
    profiles = {r.get("llm_profile") for r in everything.values() if r.get("llm_profile")}
    if len(served) > 1 or len(profiles) > 1:
        print(
            f"\n  WARNING: this file mixes models or profiles -- served {sorted(served)}, "
            f"profiles {sorted(profiles)}.\n"
            "  Fine for development. NOT acceptable for the frozen benchmark, where both\n"
            "  conditions must run on one profile or the comparison measures model\n"
            "  capability rather than the intervention."
        )

    disagreements = [
        r for r in everything.values() if (r.get("bearing_disagreement_degrees") or 0) > 0
    ]
    if disagreements:
        print(
            f"\n  NOTE: on {len(disagreements)} record(s) the model's own bearing "
            "disagreed with the deterministic table. The table value was used."
        )
    print(f"\n  output   {checkpoint.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
