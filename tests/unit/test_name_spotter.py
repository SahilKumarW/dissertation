"""UNIT TEST -- gazetteer name spotting for the optional two-pass variant.

Not enabled by default. This covers the mechanism so the experiment can be run
later without first debugging it.
"""

from __future__ import annotations

import pytest

from georef.retrieval.name_spotter import NameSpotter

NAMES = [
    "Lindis Pass", "Perpendicular Point", "Mangahowhi Stream", "Mt Maungatua",
    "Maungatua", "Dunedin", "Nelson", "Whangamomona", "New Plymouth",
    "Lake Rotoiti", "Bush", "The Pass",
]


@pytest.fixture(scope="module")
def spotter():
    return NameSpotter([{"name": name} for name in NAMES])


class TestSpotting:
    @pytest.mark.parametrize(
        "description, expected",
        [
            ("Lindis Pass, gully crossing Highway 8 about 200m west of Pass", "Lindis Pass"),
            ("S.W. Nelson, Perpendicular Point by trig", "Perpendicular Point"),
            ("c. 54 km ESE of New Plymouth, 4 km W of Whangamomona", "New Plymouth"),
        ],
    )
    def test_finds_names_present_in_the_description(self, spotter, description, expected):
        assert expected in spotter.spot(description)

    def test_abbreviated_forms_match_the_gazetteer_spelling(self, spotter):
        """'Mt. Maungatua' carries a period the gazetteer form lacks."""
        assert "Mt Maungatua" in spotter.spot("Mt. Maungatua south west of Dunedin")

    def test_the_longest_match_wins(self, spotter):
        """'Lindis Pass' is what the description says; 'Pass' alone is not a hint."""
        hints = spotter.spot("Lindis Pass, 200m west of Pass")
        assert "Lindis Pass" in hints
        assert "The Pass" not in hints

    def test_names_made_only_of_generic_words_are_never_hints(self, spotter):
        """'The Pass' identifies no place and would otherwise fire everywhere."""
        assert "The Pass" not in spotter.spot("west of the pass on the track")

    def test_lowercase_occurrences_are_ignored(self, spotter):
        """Proper-noun shape is required, or common words match constantly."""
        assert spotter.spot("scrubby bush beside the stream") == []

    def test_hints_are_capped(self, spotter):
        description = "Nelson, Dunedin, New Plymouth, Whangamomona, Lake Rotoiti"
        assert len(spotter.spot(description, limit=2)) == 2

    def test_an_empty_description_yields_nothing(self, spotter):
        assert spotter.spot(None) == []
        assert spotter.spot("") == []

    def test_spotting_is_deterministic(self, spotter):
        description = "Lindis Pass, 200m west of Pass, near Dunedin"
        assert len({tuple(spotter.spot(description)) for _ in range(5)}) == 1


class TestNoCoordinatesLeaveTheModule:
    def test_only_names_are_returned(self):
        """The architecture's responsibility split holds because the model does
        not know where anything is. Hints must never carry coordinates."""
        records = [{"name": "Dunedin", "latitude": -45.87, "longitude": 170.50}]
        hints = NameSpotter(records).spot("Collected near Dunedin")
        assert hints == ["Dunedin"]
        assert all(isinstance(hint, str) for hint in hints)


class TestAmbiguousContextNames:
    """A name held by features hundreds of kilometres apart locates nothing.
    Regression: on the dev set, keeping such names made coherence worse than no
    coherence at all -- 'North Island' in a description pulled a Puketi in Otago
    ahead of the Puketi in the Bay of Islands, 1,249 km wrong."""

    RECORDS = [
        {"name": "North Island", "latitude": -38.0, "longitude": 176.0},
        {"name": "North Island", "latitude": -46.9, "longitude": 168.1},
        {"name": "Lake Forsyth", "latitude": -43.80, "longitude": 172.72},
        {"name": "Puketi", "latitude": -35.25, "longitude": 173.75},
        {"name": "Twin Peak", "latitude": -44.10, "longitude": 169.30},
        {"name": "Twin Peak", "latitude": -44.15, "longitude": 169.35},
    ]

    def spotter(self):
        return NameSpotter(self.RECORDS, ambiguity_radius_km=25.0)

    def test_a_scattered_name_is_dropped_from_context(self):
        names = [r["name"] for r in self.spotter().spot_records("North Island, near Puketi")]
        assert "North Island" not in names

    def test_it_is_still_spotted_as_a_name(self):
        """Dropping it as a REFERENCE POINT is different from not seeing it."""
        assert "North Island" in self.spotter().spot("North Island, near Puketi")

    def test_an_unambiguous_name_survives(self):
        names = [r["name"] for r in self.spotter().spot_records("near Lake Forsyth")]
        assert names == ["Lake Forsyth"]

    def test_duplicates_in_one_locality_are_not_ambiguous(self):
        """Two records of the same place a few km apart still locate it."""
        names = [r["name"] for r in self.spotter().spot_records("below Twin Peak")]
        assert len(names) == 2

    def test_the_radius_is_configurable(self):
        wide = NameSpotter(self.RECORDS, ambiguity_radius_km=2000.0)
        names = [r["name"] for r in wide.spot_records("North Island, near Puketi")]
        assert "North Island" in names


class TestGazetteerNameForms:
    """LINZ gazettes dual names as one string, and descriptions space words
    differently. Both broke coherence on the dev set: 'Lake Forsyth' failed to
    match 'Wairewa/Lake Forsyth' and a southern feature was used as the context
    point instead, putting Little River 502 km wrong."""

    RECORDS = [
        {"name": "Wairewa/Lake Forsyth", "latitude": -43.80, "longitude": 172.72},
        {"name": "Lake Clearwater", "latitude": -43.60, "longitude": 171.04},
        {"name": "Cook River/Weheka", "latitude": -43.47, "longitude": 170.00},
        {"name": "Aoraki/Mount Cook", "latitude": -43.59, "longitude": 170.14},
    ]

    def spotter(self):
        return NameSpotter(self.RECORDS)

    def test_either_side_of_a_dual_name_matches(self):
        for description, expected in [
            ("near Lake Forsyth", "Wairewa/Lake Forsyth"),
            ("near Wairewa", "Wairewa/Lake Forsyth"),
            ("1 mile N of Weheka", "Cook River/Weheka"),
            ("summit of Mount Cook", "Aoraki/Mount Cook"),
            ("slopes of Aoraki", "Aoraki/Mount Cook"),
        ]:
            names = [r["name"] for r in self.spotter().spot_records(description)]
            assert expected in names, description

    def test_spacing_differences_still_match(self):
        """The description writes 'Lake Clear Water'; the gazetteer has 'Lake Clearwater'."""
        names = [r["name"] for r in self.spotter().spot_records("Lake Clear Water [Clearwater]")]
        assert "Lake Clearwater" in names

    def test_a_dual_name_is_not_double_counted(self):
        records = self.spotter().spot_records("Wairewa/Lake Forsyth")
        assert len(records) == 1
