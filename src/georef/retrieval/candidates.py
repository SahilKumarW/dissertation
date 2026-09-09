"""Retrieval candidates and rank fusion.

A candidate is a gazetteer record that one or more retrieval channels proposed,
carrying the per-channel evidence that produced it. Nothing here selects an
answer: retrieval proposes, the disambiguator disposes (master spec section 19).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Candidate", "reciprocal_rank_fusion"]


@dataclass
class Candidate:
    """One retrieved gazetteer record plus its retrieval provenance."""

    linz_id: str
    record: dict
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def name(self) -> str:
        return self.record.get("name", "")

    @property
    def feature_type(self) -> str | None:
        return self.record.get("feature_type")

    @property
    def land_district(self) -> str | None:
        return self.record.get("land_district")

    @property
    def latitude(self) -> float:
        return self.record["latitude"]

    @property
    def longitude(self) -> float:
        return self.record["longitude"]

    def to_provenance(self) -> dict:
        """The retrieval evidence recorded for this candidate in the output."""
        return {
            "linz_id": self.linz_id,
            "name": self.name,
            "feature_type": self.feature_type,
            "land_district": self.land_district,
            "source": self.record.get("source"),
            "channel_ranks": dict(self.channel_ranks),
            "channel_scores": {k: round(v, 6) for k, v in self.channel_scores.items()},
            "fused_score": round(self.fused_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    k: int = 60,
    tie_break: str = "linz_id_ascending",
) -> list[tuple[str, float, dict[str, int]]]:
    """Fuse ranked identifier lists into one ordering.

    Each channel contributes ``1 / (k + rank)`` for every identifier it ranked,
    with ``rank`` starting at 1. RRF is used rather than weighted score fusion
    because it needs no per-channel weights: channel scores here are not
    comparable (a BM25 score and a cosine similarity are different quantities),
    and inventing weights to combine them would be tuning by another name.

    ``k`` damps the influence of top ranks so a single confident channel cannot
    dominate the others outright. 60 is the value from the original paper,
    adopted unchanged.

    Ties are broken deterministically so a rerun cannot silently reorder
    equally-scored candidates.
    """
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for channel, identifiers in ranked_lists.items():
        for position, identifier in enumerate(identifiers, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + position)
            ranks.setdefault(identifier, {})[channel] = position

    if tie_break == "linz_id_ascending":
        ordered = sorted(scores, key=lambda i: (-scores[i], str(i)))
    else:
        raise ValueError(f"unknown tie_break policy: {tie_break}")

    return [(identifier, scores[identifier], ranks[identifier]) for identifier in ordered]
