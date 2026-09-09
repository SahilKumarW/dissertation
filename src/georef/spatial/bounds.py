"""Bounding-box containment, shared by the benchmark builder and LINZ ingest.

Kept in one place because the Chatham Islands are easy to lose: they lie east
of the antimeridian and carry negative longitudes, so a single naive New
Zealand box silently discards them. Any code deciding whether a coordinate is
"in New Zealand" uses this function against a configured set of boxes.
"""

from __future__ import annotations

__all__ = ["containing_box", "is_inside_any"]


def containing_box(latitude: float, longitude: float, boxes: dict) -> str | None:
    """Name of the first box containing the point, or ``None``.

    ``boxes`` maps a box name to a mapping with ``lat_min``, ``lat_max``,
    ``lon_min`` and ``lon_max``. Iteration order is the caller's, so the
    configuration file decides precedence when boxes overlap.
    """
    for name, box in boxes.items():
        if (
            box["lat_min"] <= latitude <= box["lat_max"]
            and box["lon_min"] <= longitude <= box["lon_max"]
        ):
            return name
    return None


def is_inside_any(latitude: float, longitude: float, boxes: dict) -> bool:
    return containing_box(latitude, longitude, boxes) is not None
