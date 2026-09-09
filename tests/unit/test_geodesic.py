"""UNIT TEST -- WGS84 geodesic computation (master spec section 27).

Correctness is checked three ways: against independently published WGS84
constants, by round-tripping the direct and inverse problems against each other,
and by structural properties that must hold in any correct implementation.
"""

from __future__ import annotations

import pytest

from georef.spatial.geodesic import (
    bearing_between,
    destination,
    distance_between,
    interpolate,
)

WAIOURU = (-39.4783, 175.6689)
AKAROA = (-43.8103, 172.9679)
CHATHAM = (-43.95, -176.55)


class TestAgainstPublishedConstants:
    """Values from the WGS84 definition, not from this library's own output."""

    def test_one_degree_of_longitude_at_the_equator(self):
        assert distance_between(0, 0, 0, 1) == pytest.approx(111319.49, abs=0.1)

    def test_one_degree_of_latitude_at_the_equator(self):
        assert distance_between(0, 0, 1, 0) == pytest.approx(110574.39, abs=0.1)

    def test_a_degree_of_latitude_lengthens_toward_the_pole(self):
        """Consequence of the ellipsoid; a spherical model would give equal values."""
        equatorial = distance_between(0, 0, 1, 0)
        polar = distance_between(89, 0, 90, 0)
        assert polar > equatorial
        assert polar == pytest.approx(111693.9, abs=1.0)


class TestDirectProblem:
    def test_north_and_south_hold_longitude(self):
        start_lat, start_lon = WAIOURU
        north = destination(start_lat, start_lon, 0, 50_000)
        south = destination(start_lat, start_lon, 180, 50_000)
        assert north.latitude > start_lat
        assert south.latitude < start_lat
        assert north.longitude == pytest.approx(start_lon, abs=1e-9)
        assert south.longitude == pytest.approx(start_lon, abs=1e-9)

    @pytest.mark.parametrize("bearing, direction", [(90, 1), (270, -1)])
    def test_east_and_west_move_longitude_the_right_way(self, bearing, direction):
        start_lat, start_lon = WAIOURU
        point = destination(start_lat, start_lon, bearing, 50_000)
        assert (point.longitude - start_lon) * direction > 0

    @pytest.mark.parametrize("bearing", [90, 270])
    def test_a_geodesic_due_east_is_not_a_parallel_of_latitude(self, bearing):
        """Heading due east from a southern latitude, the geodesic curves toward
        the equator: a great circle is not a line of constant latitude. The
        drift is small at benchmark distances but it is real, and a naive
        implementation that held latitude fixed would be wrong here."""
        start_lat, start_lon = WAIOURU
        point = destination(start_lat, start_lon, bearing, 50_000)
        assert point.latitude > start_lat          # toward the equator, southern hemisphere
        assert point.latitude - start_lat < 0.01   # but only slightly, at this distance

    def test_the_same_curvature_appears_in_the_northern_hemisphere(self):
        point = destination(51.5, -0.12, 90, 50_000)
        assert point.latitude < 51.5

    @pytest.mark.parametrize("bearing", [0, 45, 90, 135, 180, 225, 270, 315, 22.5, 337.5])
    @pytest.mark.parametrize("metres", [100, 5_000, 19_000, 120_000])
    def test_the_destination_is_the_requested_distance_away(self, bearing, metres):
        """The single most important property: ask for 19 km, get 19 km."""
        point = destination(*WAIOURU, bearing, metres)
        assert distance_between(*WAIOURU, point.latitude, point.longitude) == pytest.approx(
            metres, abs=0.01
        )

    @pytest.mark.parametrize("bearing", [0, 45, 90, 200, 315])
    def test_direct_and_inverse_agree_on_bearing(self, bearing):
        point = destination(*WAIOURU, bearing, 30_000)
        assert bearing_between(*WAIOURU, point.latitude, point.longitude) == pytest.approx(
            bearing, abs=1e-6
        )

    def test_zero_distance_returns_the_anchor(self):
        point = destination(*WAIOURU, 45, 0)
        assert (point.latitude, point.longitude) == pytest.approx(WAIOURU)

    def test_negative_distance_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            destination(*WAIOURU, 45, -1000)

    def test_bearings_beyond_360_wrap(self):
        assert destination(*WAIOURU, 405, 10_000).as_tuple() == pytest.approx(
            destination(*WAIOURU, 45, 10_000).as_tuple()
        )


class TestAntimeridianAndHemispheres:
    def test_crossing_the_antimeridian_eastward_wraps_longitude(self):
        """The Chatham Islands sit east of 180 with negative longitudes; naive
        arithmetic puts a point there in the wrong hemisphere."""
        point = destination(-43.95, 179.5, 90, 100_000)
        assert point.longitude < 0
        assert -180 <= point.longitude < 180

    def test_crossing_westward_from_the_chathams(self):
        point = destination(*CHATHAM, 270, 400_000)
        assert -180 <= point.longitude <= 180
        assert distance_between(*CHATHAM, point.latitude, point.longitude) == pytest.approx(
            400_000, abs=0.01
        )

    def test_southern_hemisphere_north_increases_latitude(self):
        assert destination(*AKAROA, 0, 20_000).latitude > AKAROA[0]

    def test_northern_hemisphere_behaves_the_same_way(self):
        assert destination(51.5, -0.12, 0, 20_000).latitude > 51.5

    def test_high_latitude_case(self):
        point = destination(-77.5, 166.5, 90, 50_000)
        assert distance_between(-77.5, 166.5, point.latitude, point.longitude) == pytest.approx(
            50_000, abs=0.01
        )


class TestWhyTheEllipsoidMatters:
    def test_a_flat_earth_approximation_would_be_kilometres_out(self):
        """Justifies the prohibition on `latitude += distance / 111000`."""
        metres = 100_000
        exact = destination(*AKAROA, 45, metres)
        naive_lat = AKAROA[0] + (metres * 0.70710678) / 111_000
        naive_lon = AKAROA[1] + (metres * 0.70710678) / 111_000
        error = distance_between(exact.latitude, exact.longitude, naive_lat, naive_lon)
        assert error > 15_000


class TestInterpolate:
    def test_endpoints(self):
        assert interpolate(*WAIOURU, *AKAROA, 0.0).as_tuple() == pytest.approx(WAIOURU)
        assert interpolate(*WAIOURU, *AKAROA, 1.0).as_tuple() == pytest.approx(AKAROA, abs=1e-9)

    def test_midpoint_is_equidistant(self):
        middle = interpolate(*WAIOURU, *AKAROA, 0.5)
        to_start = distance_between(*WAIOURU, middle.latitude, middle.longitude)
        to_end = distance_between(middle.latitude, middle.longitude, *AKAROA)
        assert to_start == pytest.approx(to_end, rel=1e-9)

    def test_fraction_is_clamped(self):
        assert interpolate(*WAIOURU, *AKAROA, 5.0).as_tuple() == pytest.approx(
            interpolate(*WAIOURU, *AKAROA, 1.0).as_tuple()
        )

    def test_interpolating_across_the_antimeridian_stays_in_range(self):
        """Arithmetic averaging of 179.9 and -179.9 gives 0, the wrong side of the planet."""
        middle = interpolate(-43.9, 179.9, -43.9, -179.9, 0.5)
        assert abs(middle.longitude) > 179.0
