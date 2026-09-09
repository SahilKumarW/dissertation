"""Condition C: combining the baseline and the hybrid.

The benchmark showed the two conditions failing differently rather than one
dominating. The hybrid is nearly twice as likely to land within a kilometre and
three times as likely to miss by fifty; the baseline never abstains and its
median is the same. Two methods with different failure modes are exactly the
case where a combination rule can beat both.

**This is a derived condition.** It runs no model and makes no request: it reads
two finished prediction files and applies a deterministic rule. That means it
can be evaluated for free, and it means it cannot be confounded by a different
model or a different prompt.

## The integrity problem, and how it is handled

Condition C was conceived AFTER the benchmark was evaluated. A rule invented to
fit an observed result and then reported against that same result is worthless,
and no amount of care in the implementation fixes that.

Two things keep it honest:

1. **The rules take no parameters from the benchmark.** Each is specified from
   the design alone -- "the hybrid abstains and the baseline does not", "two
   independent estimates agreeing is evidence" -- and each could have been
   written before any run.
2. **Any threshold is fixed on the dev set**, exactly like the abstention
   threshold and the coherence switch, using a Condition A dev run.

Even so, a combination reported against the benchmark that motivated it is
**exploratory analysis and must be labelled as such**. It does not replace the
frozen A-versus-B result and it is not a headline. `AGREEMENT` is the only rule
here that uses no threshold at all and so carries the least of this baggage.
"""

from __future__ import annotations

from dataclasses import dataclass

from georef.spatial.geodesic import distance_between

__all__ = ["STRATEGIES", "CombinationResult", "combine_record", "combine"]

STRATEGIES = ("fallback", "confidence", "agreement", "prefer_baseline")


@dataclass
class CombinationResult:
    benchmark_id: str
    status: str
    latitude: float | None
    longitude: float | None
    source_condition: str | None
    reason: str
    agreement_km: float | None = None

    def to_record(self, description: str | None = None) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "condition": "C",
            "status": self.status,
            "input_description": description,
            "predicted_latitude": self.latitude,
            "predicted_longitude": self.longitude,
            "source_condition": self.source_condition,
            "combination_reason": self.reason,
            "agreement_km": self.agreement_km,
        }


def _coordinates(row: dict) -> tuple[float, float] | None:
    latitude, longitude = row.get("predicted_latitude"), row.get("predicted_longitude")
    if latitude is None or longitude is None:
        return None
    return float(latitude), float(longitude)


def combine_record(
    benchmark_id: str,
    row_a: dict,
    row_b: dict,
    strategy: str = "agreement",
    agreement_km: float = 10.0,
    minimum_score: float = 0.6,
) -> CombinationResult:
    """Apply one combination rule to a single paired record.

    * ``fallback``        -- the hybrid where it answered, the baseline where it
      abstained. The obvious rule, and what a deployed system would do: it
      recovers full coverage without discarding any hybrid prediction.
    * ``confidence``      -- the hybrid only when its disambiguation score
      clears ``minimum_score``, otherwise the baseline. Uses the hybrid's own
      self-assessment, which is available at prediction time.
    * ``agreement``       -- the hybrid when the two independent estimates agree
      within ``agreement_km``, otherwise the baseline. Two methods that share no
      machinery arriving at the same place is evidence neither can produce
      alone; disagreement means at least one is wrong and the safer of the two
      is preferred.
    * ``prefer_baseline`` -- the baseline unless it abstained. The control:
      whatever a combination rule achieves must beat this, or it is achieving
      nothing.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown combination strategy: {strategy!r}")

    point_a = _coordinates(row_a) if row_a else None
    point_b = _coordinates(row_b) if row_b else None

    separation = None
    if point_a and point_b:
        separation = distance_between(*point_a, *point_b) / 1000.0

    def take(point, condition, reason):
        if point is None:
            return None
        return CombinationResult(
            benchmark_id=benchmark_id,
            status="SUCCESS",
            latitude=point[0],
            longitude=point[1],
            source_condition=condition,
            reason=reason,
            agreement_km=None if separation is None else round(separation, 4),
        )

    chosen = None
    if strategy == "fallback":
        chosen = take(point_b, "B", "hybrid answered") or take(
            point_a, "A", "hybrid abstained, baseline used"
        )

    elif strategy == "prefer_baseline":
        chosen = take(point_a, "A", "baseline answered") or take(
            point_b, "B", "baseline abstained, hybrid used"
        )

    elif strategy == "confidence":
        score = (row_b or {}).get("disambiguation_score")
        if point_b is not None and score is not None and float(score) >= minimum_score:
            chosen = take(point_b, "B", f"hybrid score {float(score):.3f} >= {minimum_score}")
        else:
            detail = "hybrid abstained" if point_b is None else f"hybrid score below {minimum_score}"
            chosen = take(point_a, "A", detail) or take(point_b, "B", "baseline unavailable")

    elif strategy == "agreement":
        if separation is not None and separation <= agreement_km:
            chosen = take(point_b, "B", f"conditions agree within {separation:.1f} km")
        elif separation is not None:
            chosen = take(point_a, "A", f"conditions disagree by {separation:.1f} km")
        else:
            chosen = take(point_b, "B", "only the hybrid answered") or take(
                point_a, "A", "only the baseline answered"
            )

    if chosen is None:
        return CombinationResult(
            benchmark_id=benchmark_id,
            status="NO_PREDICTION",
            latitude=None,
            longitude=None,
            source_condition=None,
            reason="neither condition produced a coordinate",
            agreement_km=None,
        )
    return chosen


def combine(
    rows_a: dict,
    rows_b: dict,
    identifiers: list[str],
    strategy: str = "agreement",
    agreement_km: float = 10.0,
    minimum_score: float = 0.6,
) -> dict:
    """Apply a rule across every record, keyed by benchmark id."""
    combined = {}
    for identifier in identifiers:
        combined[identifier] = combine_record(
            identifier,
            rows_a.get(identifier, {}),
            rows_b.get(identifier, {}),
            strategy=strategy,
            agreement_km=agreement_km,
            minimum_score=minimum_score,
        )
    return combined
