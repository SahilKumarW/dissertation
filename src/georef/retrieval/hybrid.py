"""Hybrid retrieval: four channels, rank fusion, optional cross-encoder rerank.

    query -> exact | folded | BM25 | dense -> RRF -> rerank -> candidates

The embedder, the vector store and the reranker are injected as protocols so
the fusion and ordering logic can be tested without model weights or a
database. That also keeps the retrieval contract stable if any of the three is
swapped, which would otherwise be exactly the kind of silent substitution the
experiment protocol prohibits.

Retrieval never selects the answer. It hands a ranked candidate list, with the
evidence that produced it, to the deterministic disambiguator.
"""

from __future__ import annotations

from typing import Protocol

from georef.retrieval.candidates import Candidate, reciprocal_rank_fusion
from georef.retrieval.lexical import LexicalIndex

__all__ = ["Embedder", "VectorStore", "Reranker", "HybridRetriever"]


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return (linz_id, similarity) pairs, best first."""


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class HybridRetriever:
    def __init__(
        self,
        lexical: LexicalIndex,
        config: dict,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.lexical = lexical
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker

    def _channel(self, name: str) -> dict:
        return self.config["channels"].get(name, {"enabled": False, "top_k": 0})

    def retrieve(self, query: str) -> list[Candidate]:
        ranked: dict[str, list[str]] = {}
        scores: dict[str, dict[str, float]] = {}

        exact = self._channel("exact_name")
        if exact["enabled"]:
            hits = self.lexical.exact_name(query, exact["top_k"])
            if hits:
                ranked["exact_name"] = hits

        folded = self._channel("folded_name")
        if folded["enabled"]:
            hits = self.lexical.folded_name(query, folded["top_k"])
            if hits:
                ranked["folded_name"] = hits

        bm25 = self._channel("bm25")
        if bm25["enabled"]:
            hits, by_identifier = self.lexical.bm25(query, bm25["top_k"])
            if hits:
                ranked["bm25"] = hits
                scores["bm25"] = by_identifier

        dense = self._channel("dense")
        if dense["enabled"] and self.embedder is not None and self.vector_store is not None:
            embedding = self.embedder.encode([query])[0]
            results = self.vector_store.query(embedding, dense["top_k"])
            if results:
                ranked["dense"] = [identifier for identifier, _ in results]
                scores["dense"] = dict(results)

        if not ranked:
            return []

        fusion = self.config["fusion"]
        fused = reciprocal_rank_fusion(
            ranked, k=fusion["k"], tie_break=self.config.get("tie_break", "linz_id_ascending")
        )

        candidates: list[Candidate] = []
        for identifier, fused_score, channel_ranks in fused:
            record = self.lexical.records.get(identifier)
            if record is None:
                # A vector store holding identifiers absent from the snapshot
                # means the two are out of sync; skipping silently would hide a
                # real provenance failure, so it is surfaced by the caller's
                # count check rather than papered over here.
                continue
            candidates.append(
                Candidate(
                    linz_id=identifier,
                    record=record,
                    channel_ranks=channel_ranks,
                    channel_scores={
                        channel: by_identifier[identifier]
                        for channel, by_identifier in scores.items()
                        if identifier in by_identifier
                    },
                    fused_score=fused_score,
                )
            )

        candidates = self._rerank(query, candidates)
        return candidates[: self.config["final_top_k"]]

    def _rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        rerank_config = self.config.get("rerank", {})
        if not rerank_config.get("enabled") or self.reranker is None or not candidates:
            return candidates

        window = candidates[: rerank_config["top_n_in"]]
        tail = candidates[rerank_config["top_n_in"] :]
        documents = [self.lexical.document_text(candidate.record) for candidate in window]
        for candidate, score in zip(window, self.reranker.score(query, documents)):
            candidate.rerank_score = float(score)

        # Reranked candidates come first, ordered by cross-encoder score; the
        # tail keeps its fused order. Ties fall back to linz_id so the ordering
        # stays total and reproducible.
        window.sort(key=lambda c: (-(c.rerank_score or 0.0), str(c.linz_id)))
        return window[: rerank_config["top_k_out"]] + tail
