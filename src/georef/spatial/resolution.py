"""Turning one or more grounded clauses into a single predicted coordinate.

A description often states two relations for the same point:

    "54 km ESE of New Plymouth, 4 km W of Whangamommona"

Once both anchors are grounded, each clause yields a destination. Those are two
independent estimates of the same location, which gives two things a single
clause cannot:

* **A better estimate.** A 4 km offset compounds far less anchor error than a
  54 km one, so the shorter clause deserves more weight.
* **A consistency check that needs no ground truth.** If the two estimates are
  40 km apart, something is wrong — wrong anchor selected, misread direction,
  misread distance — and that is knowable at prediction time. Abstaining there
  prevents a confident wrong coordinate, which is what a positional-error metric
  punishes hardest.

Three policies are implemented so the dev set can choose between them rather
than the choice being asserted. Whichever wins is frozen before the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from georef.spatial.geodesic import (
    GEODESIC_METHOD,
    destination,
    distance_between,
    interpolate,
)

__all__ = [
    "GroundedClause",
    "Resolution",
    "ResolutionStatus",
    "resolve_clauses",
    "POLICIES",
]

POLICIES = ("first", "shortest", "fuse")


class ResolutionStatus:
    SUCCESS = "SUCCESS"
    NO_GROUNDED_CLAUSE = "NO_GROUNDED_CLAUSE"
    CLAUSE_DISAGREEMENT = "CLAUSE_DISAGREEMENT"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


@dataclass
class GroundedClause:
    """A clause whose anchor has been resolved to authoritative coordinates."""

    anchor_name: str
    anchor_latitude: float
    anchor_longitude: float
    bearing_degrees: float
    distance_metres: float
    anchor_source: str = "LINZ"
    anchor_linz_id: str | None = None

    def destination(self):
        return destination(
            self.anchor_latitude,
            self.anchor_longitude,
            self.bearing_degrees,
            self.distance_metres,
        )


@dataclass
class Resolution:
    status: str
    latitude: float | None = None
    longitude: float | None = None
    policy: str | None = None
    clause_estimates: list[dict] = field(default_factory=list)
    disagreement_metres: float | None = None
    failure_reason: str | None = None
    geodesic_method: str = GEODESIC_METHOD

    @property
    def is_success(self) -> bool:
        return self.status == ResolutionStatus.SUCCESS

    def to_provenance(self) -> dict:
        return {
            "resolution_status": self.status,
            "resolution_policy": self.policy,
            "predicted_latitude": self.latitude,
            "predicted_longitude": self.longitude,
            "clause_estimates": self.clause_estimates,
            "clause_disagreement_metres": self.disagreement_metres,
            "geodesic_method": self.geodesic_method,
            "resolution_failure_reason": self.failure_reason,
        }


def _estimates(clauses: list[GroundedClause]) -> list[dict]:
    rows = []
    for clause in clauses:
        point = clause.destination()
        rows.append(
            {
                "anchor_name": clause.anchor_name,
                "anchor_linz_id": clause.anchor_linz_id,
                "anchor_latitude": clause.anchor_latitude,
                "anchor_longitude": clause.anchor_longitude,
                "anchor_source": clause.anchor_source,
                "bearing_degrees": clause.bearing_degrees,
                "distance_metres": clause.distance_metres,
                "predicted_latitude": point.latitude,
                "predicted_longitude": point.longitude,
            }
        )
    return rows


def _max_disagreement(estimates: list[dict]) -> float:
    """Largest geodesic gap between any two clause estimates."""
    worst = 0.0
    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            gap = distance_between(
                estimates[i]["predicted_latitude"],
                estimates[i]["predicted_longitude"],
                estimates[j]["predicted_latitude"],
                estimates[j]["predicted_longitude"],
            )
            worst = max(worst, gap)
    return worst


def _fuse_two(first: dict, second: dict, minimum_metres: float) -> tuple[float, float]:
    """Combine two estimates, weighting the shorter offset more heavily.

    A clause's positional error scales with its offset: an error in the anchor
    or the bearing is amplified by distance. Weighting by inverse offset is a
    simple expression of that, floored so a stated offset of zero cannot take
    infinite weight.
    """
    weight_first = 1.0 / max(first["distance_metres"], minimum_metres)
    weight_second = 1.0 / max(second["distance_metres"], minimum_metres)
    fraction = weight_second / (weight_first + weight_second)
    point = interpolate(
        first["predicted_latitude"],
        first["predicted_longitude"],
        second["predicted_latitude"],
        second["predicted_longitude"],
        fraction,
    )
    return point.latitude, point.longitude


def resolve_clauses(
    clauses: list[GroundedClause],
    policy: str = "fuse",
    disagreement_threshold_metres: float = 25_000.0,
    minimum_weight_metres: float = 100.0,
) -> Resolution:
    """Produce one predicted coordinate from the grounded clauses.

    ``policy``:

    * ``first``    -- use the first clause as written. No heuristic to defend,
      and the natural baseline for the ablation.
    * ``shortest`` -- use the clause with the smallest offset, on the grounds
      that it compounds the least anchor and bearing error.
    * ``fuse``     -- combine the estimates, weighted by inverse offset, and
      abstain when they disagree by more than the threshold.

    The threshold is a fixed pre-benchmark parameter. It is not tuned against
    the benchmark, and its value is recorded in the run manifest.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown resolution policy: {policy!r}")

    if not clauses:
        return Resolution(
            status=ResolutionStatus.NO_GROUNDED_CLAUSE,
            policy=policy,
            failure_reason="no clause had a grounded anchor",
        )

    estimates = _estimates(clauses)
    disagreement = _max_disagreement(estimates) if len(estimates) > 1 else None

    if policy == "first" or len(estimates) == 1:
        chosen = estimates[0]
        latitude, longitude = chosen["predicted_latitude"], chosen["predicted_longitude"]

    elif policy == "shortest":
        chosen = min(estimates, key=lambda e: e["distance_metres"])
        latitude, longitude = chosen["predicted_latitude"], chosen["predicted_longitude"]

    else:  # fuse
        if disagreement is not None and disagreement > disagreement_threshold_metres:
            return Resolution(
                status=ResolutionStatus.CLAUSE_DISAGREEMENT,
                policy=policy,
                clause_estimates=estimates,
                disagreement_metres=disagreement,
                failure_reason=(
                    f"clause estimates disagree by {disagreement / 1000:.1f} km, "
                    f"above the {disagreement_threshold_metres / 1000:.1f} km threshold"
                ),
            )
        # Fuse pairwise, shortest offset first, so the most reliable estimate
        # anchors the result when there are more than two clauses.
        ordered = sorted(estimates, key=lambda e: e["distance_metres"])
        latitude = ordered[0]["predicted_latitude"]
        longitude = ordered[0]["predicted_longitude"]
        accumulated = ordered[0]["distance_metres"]
        for nxt in ordered[1:]:
            latitude, longitude = _fuse_two(
                {
                    "predicted_latitude": latitude,
                    "predicted_longitude": longitude,
                    "distance_metres": accumulated,
                },
                nxt,
                minimum_weight_metres,
            )
            accumulated = min(accumulated, nxt["distance_metres"])

    return Resolution(
        status=ResolutionStatus.SUCCESS,
        latitude=latitude,
        longitude=longitude,
        policy=policy,
        clause_estimates=estimates,
        disagreement_metres=disagreement,
    )
