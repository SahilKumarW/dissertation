#!/usr/bin/env python
"""Step 3: embed the frozen LINZ snapshot into ChromaDB.

    python scripts/build_index.py

Reads the latest snapshot under data/linz/normalised/, embeds every record with
the configured model, loads them into the collection, and writes a manifest
tying the index to the snapshot hash.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import load_yaml, repo_root  # noqa: E402
from georef.provenance.hashing import environment_fingerprint, sha256_file  # noqa: E402
from georef.retrieval.dense import (  # noqa: E402
    ChromaVectorStore,
    SentenceTransformerEmbedder,
    build_index,
    load_snapshot,
)
from georef.retrieval.lexical import LexicalIndex  # noqa: E402


def main() -> int:
    root = repo_root()
    models = load_yaml(root / "configs" / "models.yaml")
    retrieval = load_yaml(root / "configs" / "retrieval.yaml")
    linz = load_yaml(root / "configs" / "linz.yaml")

    snapshots = sorted((root / linz["paths"]["normalised_dir"]).glob("*.jsonl"))
    if not snapshots:
        print("No LINZ snapshot found. Run scripts/fetch_linz.py first.", file=sys.stderr)
        return 2
    snapshot_path = snapshots[-1]
    snapshot_hash = sha256_file(snapshot_path)

    records = load_snapshot(snapshot_path)
    print(f"Snapshot   {snapshot_path.name}  ({len(records)} records)")
    print(f"sha256     {snapshot_hash}")

    embedding_config = models["embedding"]
    embedder = SentenceTransformerEmbedder(
        embedding_config["model"], embedding_config.get("normalise_embeddings", True)
    )
    print(f"Embedder   {embedding_config['model']}  (dim {embedder.dimension})")
    if embedder.dimension != embedding_config["dimension"]:
        print(
            f"FAILED: model reports {embedder.dimension} dimensions but config "
            f"declares {embedding_config['dimension']}.",
            file=sys.stderr,
        )
        return 2

    chroma_dir = root / "data" / "chroma"
    store = ChromaVectorStore(chroma_dir, retrieval["collection_name"], embedding_config["similarity"])

    # Document text comes from the lexical index so BM25 and the embedder see
    # identical evidence for every record.
    lexical = LexicalIndex(records, retrieval["document_template"])

    print("Embedding and indexing...")
    result = build_index(
        records,
        embedder,
        store,
        lexical.document_text,
        snapshot_path.name,
        snapshot_hash,
    )

    manifest = {
        "collection_name": result.collection_name,
        "record_count": result.record_count,
        "embedding_model": result.embedding_model,
        "embedding_dimension": result.dimension,
        "similarity": embedding_config["similarity"],
        "document_template": retrieval["document_template"],
        "snapshot_file": result.snapshot_file,
        "snapshot_sha256": result.snapshot_sha256,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_fingerprint(),
    }
    manifest_path = root / "data" / "chroma" / "index_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\n  collection   {result.collection_name}")
    print(f"  records      {result.record_count}")
    print(f"  dimension    {result.dimension}")
    print(f"  manifest     {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
