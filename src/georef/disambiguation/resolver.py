"""Deterministic candidate selection and abstention (master spec sections 20–24).

Retrieval proposes; this decides. The decision is a weighted sum of features
that are each individually inspectable, so a wrong selection can be traced to
the evidence that caused it rather than to an opaque score.

Nothing here is learned or tuned. The weights in `configs/disambiguation.yaml`
express judgements about what counts as evidence — an exact name match beats
regional corroboration, a true point beats the representative point of a river —
and they are never fitted to the benchmark.

Two abstention routes, both deliberate:

* **Not good enough.** The best candidate scores below the minimum.
* **Not distinguishable.** The best two are within the margin. Forcing a choice
  there is a coin flip recorded as a prediction, and a positional-error metric
  punishes a confident wrong answer far harder than an honest abstention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from georef.disambiguation.anchors import feature_terms, is_generic_only, name_similarity
from georef.normalise import match_key, normalise
from georef.spatial.geodesic import distance_between

__all__ = ["Selection", "SelectionStatus", "CandidateScore", "Disambiguator"]


class SelectionStatus:
    SELECTED = "SELECTED"
    NO_CANDIDATES = "NO_CANDIDATES"
    GENERIC_ANCHOR = "GENERIC_ANCHOR"
    BELOW_MINIMUM_SCORE = "BELOW_MINIMUM_SCORE"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"


@dataclass
class CandidateScore:
    linz_id: str
    name: str
    record: dict
    score: float
    features: dict = field(default_factory=dict)

    def to_provenance(self) -> dict:
        return {
            "linz_id": self.linz_id,
            "name": self.name,
            "feature_type": self.record.get("feature_type"),
            "land_district": self.record.get("land_district"),
            "geometry_type": self.record.get("geometry_type"),
            "name_status": self.record.get("name_status"),
            "source": self.record.get("source"),
            # Kept so a spatially resolved tie can be audited after the fact.
            "latitude": self.record.get("latitude"),
            "longitude": self.record.get("longitude"),
            "score": round(self.score, 6),
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


@dataclass
class Selection:
    status: str
    selected: CandidateScore | None = None
    runner_up: CandidateScore | None = None
    margin: float | None = None
    considered: list[CandidateScore] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def is_selected(self) -> bool:
        return self.status == SelectionStatus.SELECTED

    def to_provenance(self) -> dict:
        return {
            "disambiguation_status": self.status,
            "selected_candidate_name": self.selected.name if self.selected else None,
            "selected_linz_id": self.selected.linz_id if self.selected else None,
            "selected_feature_type": (
                self.selected.record.get("feature_type") if self.selected else None
            ),
            "selected_region": (
                self.selected.record.get("land_district") if self.selected else None
            ),
            "selected_source": self.selected.record.get("source") if self.selected else None,
            "disambiguation_score": round(self.selected.score, 6) if self.selected else None,
            "score_margin": round(self.margin, 6) if self.margin is not None else None,
            "runner_up_name": self.runner_up.name if self.runner_up else None,
            "candidates_considered": [c.to_provenance() for c in self.considered],
            "disambiguation_failure_reason": self.failure_reason,
        }


def _split_districts(value) -> list[str]:
    """'Canterbury, Westland' is two districts, not one place called both."""
    if not value:
        return []
    return [part.strip() for part in re.split(r"[,/;]", str(value)) if part.strip()]


class Disambiguator:
    def __init__(self, config: dict):
        self.weights = config["weights"]
        self.scores = config["feature_scores"]
        self.abstention = config["abstention"]
        self.limit = int(config.get("candidates_considered", 10))
        self.settlement_types = {
            t.strip().lower() for t in config.get("settlement_feature_types", [])
        }
        self.rank_decay_k = float(config.get("retrieval_rank_decay_k", 10))
        self.spatial_agreement_metres = float(
            self.abstention.get("spatial_agreement_metres", 0) or 0
        )

    # -- individual features -------------------------------------------------

    def _name_features(self, anchor: str, variants: list[str], record: dict) -> dict:
        name = record.get("name") or ""
        exact = normalise(name) == normalise(anchor)
        folded = match_key(name) == match_key(anchor)
        variant = any(match_key(name) == match_key(v) for v in variants)
        best_variant_similarity = max(
            (name_similarity(v, name) for v in variants or [anchor]), default=0.0
        )
        return {
            "exact_name": 1.0 if exact else 0.0,
            "folded_name": 1.0 if folded and not exact else 0.0,
            "variant_name": 1.0 if variant and not folded else 0.0,
            # Graded, so a partial match earns partial credit and an unrelated
            # name earns none. Without this, a candidate matching nothing but
            # the generic word ("Mount Guinevere" for "Mount Benoit") could
            # still clear the minimum on feature type and status alone.
            "name_similarity": max(
                name_similarity(anchor, name), best_variant_similarity
            ),
        }

    def _feature_type(self, anchor: str, record: dict) -> float:
        expected = feature_terms(anchor)
        actual = (record.get("feature_type") or "").strip().lower()

        if not expected:
            # The anchor names no feature type. The candidate's own type is
            # still evidence: an offset measured from a bare place name usually
            # references a settlement rather than a terrain feature sharing the
            # name. A soft preference, so name and region evidence can override
            # it.
            if actual in self.settlement_types:
                return self.scores.get(
                    "feature_type_settlement_prior", self.scores["feature_type_absent"]
                )
            return self.scores["feature_type_absent"]

        if not actual:
            return self.scores["feature_type_absent"]
        for candidate_type in expected:
            if candidate_type.lower() == actual or candidate_type.lower() in actual:
                return self.scores["feature_type_match"]
        return self.scores["feature_type_mismatch"]

    def _region(self, context: str | None, record: dict) -> float:
        if not context:
            return self.scores["region_absent"]
        stated = match_key(context)
        if not stated:
            return self.scores["region_absent"]
        fields = _split_districts(record.get("land_district")) + _split_districts(
            record.get("territorial_authority")
        ) + _split_districts(record.get("region"))
        if not fields:
            return self.scores["region_absent"]
        for value in fields:
            key = match_key(value)
            # Substring either way: "Buller" should match "Buller County", and
            # "Westland District" should match a stated "Westland".
            if key and (key in stated or stated in key):
                return self.scores["region_match"]
        return self.scores["region_mismatch"]

    def _geometry(self, record: dict) -> float:
        geometry = (record.get("geometry_type") or "POINT").strip().upper()
        if geometry == "POINT":
            return self.scores["geometry_point"]
        if geometry == "POLYGON":
            return self.scores["geometry_polygon"]
        return self.scores["geometry_line"]

    def _spatial_coherence(self, record: dict, context_points: list[tuple]) -> float:
        """How well the candidate agrees with other places the description names.

        Deterministic and gazetteer-only: the reference points come from the
        frozen snapshot, never from ground truth.
        """
        if not context_points:
            return self.scores.get("coherence_absent", 0.5)
        try:
            latitude = float(record["latitude"])
            longitude = float(record["longitude"])
        except (KeyError, TypeError, ValueError):
            return self.scores.get("coherence_absent", 0.5)

        near = float(self.scores.get("coherence_near_km", 50)) * 1000.0
        far = float(self.scores.get("coherence_far_km", 200)) * 1000.0
        closest = min(
            distance_between(latitude, longitude, point_lat, point_lon)
            for point_lat, point_lon in context_points
        )
        if closest <= near:
            return 1.0
        if closest >= far:
            return 0.0
        return 1.0 - (closest - near) / (far - near)

    def _status(self, record: dict) -> float:
        status = (record.get("name_status") or "").strip().lower()
        return (
            self.scores["status_official"]
            if status.startswith("official")
            else self.scores["status_unofficial"]
        )

    # -- scoring -------------------------------------------------------------

    def score_candidate(
        self,
        anchor: str,
        variants: list[str],
        candidate,
        rank: int,
        total: int,
        context: str | None,
        context_points: list[tuple] | None = None,
    ) -> CandidateScore:
        record = candidate.record if hasattr(candidate, "record") else candidate
        features = self._name_features(anchor, variants, record)
        # k / (k + rank): gentle decay, and independent of list length. The
        # earlier form, 1 - rank/total, made the same candidate score higher in
        # a two-item list than a ten-item one, which is a property of the query
        # rather than of the candidate.
        features["retrieval_rank"] = self.rank_decay_k / (self.rank_decay_k + rank)
        rerank_score = getattr(candidate, "rerank_score", None)
        features["rerank"] = 0.5 if rerank_score is None else _squash(rerank_score)
        features["feature_type"] = self._feature_type(anchor, record)
        features["region"] = self._region(context, record)
        features["spatial_coherence"] = self._spatial_coherence(record, context_points or [])
        features["geometry_type"] = self._geometry(record)
        features["name_status"] = self._status(record)

        total_weight = sum(self.weights.values())
        score = sum(self.weights[name] * value for name, value in features.items()) / total_weight

        return CandidateScore(
            linz_id=str(record.get("linz_id")),
            name=record.get("name") or "",
            record=record,
            score=score,
            features=features,
        )

    def select(
        self,
        anchor: str | None,
        candidates: list,
        variants: list[str] | None = None,
        context: str | None = None,
        context_points: list[tuple] | None = None,
    ) -> Selection:
        if self.abstention.get("abstain_on_generic_anchor", True) and is_generic_only(anchor):
            return Selection(
                status=SelectionStatus.GENERIC_ANCHOR,
                failure_reason=(
                    f"anchor {anchor!r} names no specific place; a gazetteer of "
                    "proper names cannot resolve it"
                ),
            )

        if not candidates:
            return Selection(
                status=SelectionStatus.NO_CANDIDATES,
                failure_reason="retrieval returned no candidates",
            )

        variants = variants or [anchor or ""]
        window = candidates[: self.limit]
        scored = [
            self.score_candidate(
                anchor or "", variants, candidate, rank, len(window), context, context_points
            )
            for rank, candidate in enumerate(window)
        ]
        # Ties break on linz_id so a rerun cannot reorder equal candidates.
        scored.sort(key=lambda c: (-c.score, str(c.linz_id)))

        best = scored[0]
        second = scored[1] if len(scored) > 1 else None
        margin = (best.score - second.score) if second else None

        if best.score < self.abstention["minimum_score"]:
            return Selection(
                status=SelectionStatus.BELOW_MINIMUM_SCORE,
                considered=scored,
                runner_up=second,
                margin=margin,
                failure_reason=(
                    f"best candidate scored {best.score:.3f}, below the "
                    f"{self.abstention['minimum_score']} minimum"
                ),
            )

        if margin is not None and margin < self.abstention["minimum_margin"]:
            # A gazetteer holds the same place more than once: Akaroa is both a
            # Locality and a Town, a few hundred metres apart. Abstaining
            # between those records answers a question that does not matter,
            # because either choice yields the same coordinate. The margin
            # exists to stop the wrong PLACE being chosen, so when the tied
            # candidates agree spatially the tie is resolved rather than
            # abstained on -- and the resolution is recorded.
            if self._agree_spatially(best, second):
                return Selection(
                    status=SelectionStatus.SELECTED,
                    selected=best,
                    runner_up=second,
                    margin=margin,
                    considered=scored,
                    failure_reason=(
                        f"tie with {second.name!r} resolved spatially: the "
                        "candidates are effectively the same location"
                    ),
                )
            return Selection(
                status=SelectionStatus.INSUFFICIENT_MARGIN,
                considered=scored,
                runner_up=second,
                margin=margin,
                failure_reason=(
                    f"{best.name!r} and {second.name!r} differ by {margin:.3f}, "
                    f"below the {self.abstention['minimum_margin']} margin"
                ),
            )

        return Selection(
            status=SelectionStatus.SELECTED,
            selected=best,
            runner_up=second,
            margin=margin,
            considered=scored,
        )


    def _agree_spatially(self, first: CandidateScore, second: CandidateScore) -> bool:
        """True when two candidates are close enough that the choice is immaterial."""
        if not self.spatial_agreement_metres:
            return False
        try:
            gap = distance_between(
                float(first.record["latitude"]),
                float(first.record["longitude"]),
                float(second.record["latitude"]),
                float(second.record["longitude"]),
            )
        except (KeyError, TypeError, ValueError):
            return False
        return gap <= self.spatial_agreement_metres


def _squash(value: float) -> float:
    """Map a cross-encoder logit onto [0, 1] without assuming its range."""
    import math

    return 1.0 / (1.0 + math.exp(-float(value)))
