"""UNIT TEST -- deterministic bearing and distance conversion. No model involved."""

from __future__ import annotations

import pytest

from georef.spatial.units import (
    UnknownDirection,
    UnknownUnit,
    bearing_for,
    distance_in_metres,
)


class TestBearings:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("N", 0.0), ("NE", 45.0), ("E", 90.0), ("SE", 135.0),
            ("S", 180.0), ("SW", 225.0), ("W", 270.0), ("NW", 315.0),
        ],
    )
    def test_eight_point_compass(self, text, expected):
        assert bearing_for(text) == expected

    @pytest.mark.parametrize(
        "text, expected",
        [("NNE", 22.5), ("ENE", 67.5), ("ESE", 112.5), ("SSE", 157.5),
         ("SSW", 202.5), ("WSW", 247.5), ("WNW", 292.5), ("NNW", 337.5)],
    )
    def test_sixteen_point_compass(self, text, expected):
        assert bearing_for(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["SW", "sw", "s.w.", "S.W.", "south-west", "southwest", "south west",
         "South West", " southwest "],
    )
    def test_every_spelling_of_southwest_agrees(self, text):
        assert bearing_for(text) == 225.0

    @pytest.mark.parametrize(
        "text", ["NNW", "n.n.w.", "north-north-west", "northnorthwest", "north north west"]
    )
    def test_every_spelling_of_nnw_agrees(self, text):
        assert bearing_for(text) == 337.5

    def test_archaic_abbreviations(self):
        assert bearing_for("Sth") == 180.0
        assert bearing_for("Nth") == 0.0

    @pytest.mark.parametrize("text", [None, "", "upstream", "inland", "seaward", "above", "xyz"])
    def test_unmappable_directions_raise_rather_than_guess(self, text):
        """A guessed bearing turns a failure into a confident wrong coordinate."""
        with pytest.raises(UnknownDirection):
            bearing_for(text)

    def test_bearings_are_clockwise_from_north(self):
        assert bearing_for("E") == 90.0
        assert bearing_for("W") == 270.0


class TestDistances:
    @pytest.mark.parametrize(
        "value, unit, expected",
        [
            (19, "km", 19000.0), (19, "kilometres", 19000.0), (19, "KM", 19000.0),
            (500, "m", 500.0), (500, "metres", 500.0),
            (1, "mile", 1609.344), (3.5, "miles", 5632.704), (2, "mls", 3218.688),
        ],
    )
    def test_conversions(self, value, unit, expected):
        assert distance_in_metres(value, unit) == pytest.approx(expected)

    def test_trailing_period_is_tolerated(self):
        assert distance_in_metres(2, "km.") == 2000.0

    @pytest.mark.parametrize(
        "value, unit",
        [(None, "km"), (5, None), (5, "leagues"), (5, "nautical miles"), (-3, "km")],
    )
    def test_unusable_distances_raise(self, value, unit):
        with pytest.raises(UnknownUnit):
            distance_in_metres(value, unit)

    def test_nautical_miles_are_rejected_not_assumed(self):
        """Terrestrial collection localities; assuming a marine unit would be wrong."""
        with pytest.raises(UnknownUnit):
            distance_in_metres(5, "nmi")
