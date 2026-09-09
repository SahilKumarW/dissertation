"""Dense retrieval channel: embedding the frozen LINZ snapshot into ChromaDB.

    snapshot -> document text -> embedding -> ChromaDB collection

ChromaDB is a retrieval mechanism, not the authoritative geographic source
(master spec section 18). Every record it holds carries `source: LINZ` and the
snapshot version it came from, so a candidate can always be traced back to the
hashed file it originated in.

The embedder is injected rather than constructed here, so index construction and
querying can be tested without model weights, and so a change of embedding model
is an explicit act rather than an import side effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["SentenceTransformerEmbedder", "ChromaVectorStore", "IndexBuildResult", "build_index"]


@dataclass
class IndexBuildResult:
    collection_name: str
    record_count: int
    dimension: int
    embedding_model: str
    snapshot_file: str
    snapshot_sha256: str
    manifest_path: Path | None = None


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. Constructed only by the build script."""

    def __init__(self, model_name: str, normalise_embeddings: bool = True, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalise_embeddings = normalise_embeddings
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalise_embeddings,
            show_progress_bar=len(texts) > 5000,
        )
        return vectors.tolist()


class ChromaVectorStore:
    """Persistent ChromaDB collection keyed by LINZ identifier."""

    def __init__(self, path: str | Path, collection_name: str, similarity: str = "cosine"):
        import chromadb

        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": similarity}
        )

    def reset(self) -> None:
        """Drop and recreate, so a rebuild cannot leave stale records behind."""
        metadata = self._collection.metadata
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata=metadata
        )

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str],
            metadatas: list[dict]) -> None:
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def count(self) -> int:
        return int(self._collection.count())

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        identifiers = result["ids"][0]
        distances = result.get("distances", [[None] * len(identifiers)])[0]
        # Chroma returns cosine DISTANCE; similarity is 1 - distance. Reporting
        # the distance as a score would invert the ordering downstream.
        return [
            (identifier, 1.0 - float(distance) if distance is not None else 0.0)
            for identifier, distance in zip(identifiers, distances)
        ]


# Metadata carried per record. Kept flat because Chroma metadata values must be
# scalars, and kept small because everything here is also available from the
# snapshot; the store holds what the disambiguator needs to filter on.
_METADATA_FIELDS = (
    "name",
    "feature_type",
    "land_district",
    "region",
    "territorial_authority",
    "geometry_type",
    "name_status",
    "latitude",
    "longitude",
    "country",
    "source",
    "dataset_version",
    "bounding_box",
)


def _metadata_for(record: dict) -> dict:
    metadata = {}
    for field in _METADATA_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        metadata[field] = value if isinstance(value, (int, float, bool)) else str(value)
    return metadata


def build_index(
    records: list[dict],
    embedder,
    store,
    document_for,
    snapshot_file: str,
    snapshot_sha256: str,
    batch_size: int = 1000,
) -> IndexBuildResult:
    """Embed every snapshot record and load it into the vector store.

    The collection is reset first: an incremental add over an existing
    collection would leave records from a previous snapshot behind, and a
    candidate traced to a superseded snapshot is a provenance failure that
    nothing downstream could detect.
    """
    store.reset()

    dimension = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        documents = [document_for(record) for record in batch]
        embeddings = embedder.encode(documents)
        if embeddings:
            dimension = len(embeddings[0])
        store.add(
            ids=[str(record["linz_id"]) for record in batch],
            embeddings=embeddings,
            documents=documents,
            metadatas=[_metadata_for(record) for record in batch],
        )

    loaded = store.count()
    if loaded != len(records):
        raise RuntimeError(
            f"index holds {loaded} records but the snapshot has {len(records)}; "
            "a partial index would silently narrow retrieval"
        )

    return IndexBuildResult(
        collection_name=getattr(store, "collection_name", "unknown"),
        record_count=loaded,
        dimension=dimension,
        embedding_model=getattr(embedder, "model_name", "unknown"),
        snapshot_file=snapshot_file,
        snapshot_sha256=snapshot_sha256,
    )


def load_snapshot(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
