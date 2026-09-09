"""Positional error metrics (master spec sections 37 and 38).

Evaluation happens ONLY after prediction, and only here. Nothing in this module
is importable from the prediction path -- the isolation test enforces that -- so
ground truth cannot reach a prediction by accident.

The primary metric is WGS84 ellipsoidal geodesic error in kilometres. Lower is
better positional accuracy. It is not "higher precision": precision and accuracy
are different properties and the benchmark measures only one of them.

Every denominator is stated explicitly. A mean error computed over successes
alone, next to a coverage figure computed over everything, is the standard way
to make an abstaining system look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from georef.spatial.geodesic import distance_between

__all__ = ["ErrorSummary", "geodesic_error_km", "summarise"]


def geodesic_error_km(
    predicted_latitude: float,
    predicted_longitude: float,
    true_latitude: float,
    true_longitude: float,
) -> float:
    """WGS84 geodesic distance between prediction and ground truth, in km."""
    return (
        distance_between(
            predicted_latitude, predicted_longitude, true_latitude, true_longitude
        )
        / 1000.0
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile; no numpy dependency in this path."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass
class ErrorSummary:
    total_records: int
    predicted: int
    abstained: int
    errors_km: list[float] = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Share of records that produced a coordinate at all."""
        return self.predicted / self.total_records if self.total_records else 0.0

    def to_dict(self) -> dict:
        values = sorted(self.errors_km)
        count = len(values)
        summary = {
            "denominator_all_records": self.total_records,
            "denominator_evaluated_predictions": count,
            "predicted": self.predicted,
            "abstained": self.abstained,
            "coverage": round(self.coverage, 4),
            "status_counts": dict(self.status_counts),
        }
        if not count:
            summary["note"] = "no predictions to evaluate"
            return summary

        mean = sum(values) / count
        variance = sum((v - mean) ** 2 for v in values) / count
        summary.update(
            {
                "mean_error_km": round(mean, 3),
                "median_error_km": round(_percentile(values, 0.5), 3),
                "std_error_km": round(variance**0.5, 3),
                "min_error_km": round(values[0], 3),
                "max_error_km": round(values[-1], 3),
                "p25_error_km": round(_percentile(values, 0.25), 3),
                "p75_error_km": round(_percentile(values, 0.75), 3),
                "p90_error_km": round(_percentile(values, 0.90), 3),
                "p95_error_km": round(_percentile(values, 0.95), 3),
                "within_1km": round(sum(1 for v in values if v <= 1) / count, 4),
                "within_5km": round(sum(1 for v in values if v <= 5) / count, 4),
                "within_10km": round(sum(1 for v in values if v <= 10) / count, 4),
                "over_50km": round(sum(1 for v in values if v > 50) / count, 4),
            }
        )
        return summary


def summarise(rows: list[dict], total_records: int | None = None) -> ErrorSummary:
    """Summarise per-record results.

    Each row needs ``status`` and, when successful, ``error_km``. Records that
    abstained are counted, never dropped: an abstention is an outcome, and
    silently excluding it would inflate every accuracy figure.
    """
    from collections import Counter

    statuses = Counter(row.get("status") for row in rows)
    errors = [row["error_km"] for row in rows if row.get("error_km") is not None]
    predicted = sum(1 for row in rows if row.get("error_km") is not None)
    total = total_records if total_records is not None else len(rows)
    return ErrorSummary(
        total_records=total,
        predicted=predicted,
        abstained=total - predicted,
        errors_km=errors,
        status_counts=dict(statuses),
    )
