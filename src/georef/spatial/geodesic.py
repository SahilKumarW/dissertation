"""Deterministic WGS84 geodesic computation (master spec section 26).

The final positional calculation is done here, in Python, by GeographicLib. The
LLM never performs it. Given an anchor coordinate, a bearing and a distance, the
destination is a solved geometry problem with one right answer, and routing it
through a language model would replace an exact result with an approximation
that cannot be audited.

No flat-earth shortcuts. `latitude += distance / 111000` is wrong by kilometres
over the distances in this benchmark and wrong in a direction that varies with
latitude, which would show up as a systematic bias in the error metric rather
than as an obvious bug.
"""

from __future__ import annotations

from dataclasses import dataclass

from geographiclib.geodesic import Geodesic

__all__ = ["GeodesicResult", "destination", "distance_between", "GEODESIC_METHOD"]

GEODESIC_METHOD = "GeographicLib WGS84 (Karney) direct/inverse"

_WGS84 = Geodesic.WGS84


@dataclass(frozen=True)
class GeodesicResult:
    latitude: float
    longitude: float
    method: str = GEODESIC_METHOD

    def as_tuple(self) -> tuple[float, float]:
        return self.latitude, self.longitude


def _normalise_longitude(longitude: float) -> float:
    """Wrap to [-180, 180).

    GeographicLib already returns values in range, but any arithmetic done on a
    longitude afterwards can push it out, and a point at 181 degrees is silently
    in the wrong hemisphere rather than obviously invalid.
    """
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    return wrapped


def destination(
    latitude: float, longitude: float, bearing_degrees: float, distance_metres: float
) -> GeodesicResult:
    """Solve the direct geodesic problem on the WGS84 ellipsoid.

    ``bearing_degrees`` is measured clockwise from true north.
    """
    if distance_metres < 0:
        raise ValueError(f"distance must not be negative: {distance_metres}")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude out of range: {latitude}")

    solution = _WGS84.Direct(latitude, longitude, bearing_degrees % 360.0, distance_metres)
    return GeodesicResult(
        latitude=solution["lat2"], longitude=_normalise_longitude(solution["lon2"])
    )


def distance_between(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
) -> float:
    """Solve the inverse problem: geodesic distance in metres.

    This is the primary evaluation metric (master spec section 37) and the
    measure of disagreement between two anchors, so it is the same function in
    both places rather than two implementations that could diverge.
    """
    return float(
        _WGS84.Inverse(latitude_1, longitude_1, latitude_2, longitude_2)["s12"]
    )


def bearing_between(
    latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float
) -> float:
    """Initial bearing from point 1 to point 2, in degrees clockwise from north."""
    return float(
        _WGS84.Inverse(latitude_1, longitude_1, latitude_2, longitude_2)["azi1"]
    ) % 360.0


def interpolate(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
    fraction: float,
) -> GeodesicResult:
    """Point at ``fraction`` of the way along the geodesic from 1 to 2.

    Used to combine two independent estimates of the same location. Averaging
    latitudes and longitudes arithmetically would be wrong near the
    antimeridian and subtly wrong everywhere else; walking the geodesic is
    correct in both cases.
    """
    fraction = min(1.0, max(0.0, float(fraction)))
    line = _WGS84.InverseLine(latitude_1, longitude_1, latitude_2, longitude_2)
    solution = line.Position(line.s13 * fraction)
    return GeodesicResult(
        latitude=solution["lat2"], longitude=_normalise_longitude(solution["lon2"])
    )
