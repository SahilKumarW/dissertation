"""Cross-encoder reranking.

Optional, and off by default. The model is a large download and costs seconds
per query on CPU; whether that buys enough accuracy to be worth including is one
of the questions the dev set exists to answer, so it is a flag rather than an
assumption.

Kept behind the `Reranker` protocol so the retrieval tests never need the
weights.
"""

from __future__ import annotations

__all__ = ["CrossEncoderReranker"]


class CrossEncoderReranker:
    """Scores (query, document) pairs with a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        return [float(score) for score in self._model.predict(pairs, show_progress_bar=False)]
