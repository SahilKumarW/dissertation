"""UNIT TEST -- snapshot summary and the diagnostic anchor extractor."""

from __future__ import annotations

import pytest

from georef.linz.verify import anchor_coverage, extract_anchor, summarise_snapshot


def record(**overrides) -> dict:
    base = {
        "linz_id": "1",
        "name": "Waiouru",
        "feature_type": "Locality",
        "land_district": "Wellington",
        "region": None,
        "territorial_authority": "Ruapehu District",
        "latitude": -39.4783,
        "longitude": 175.6689,
        "geometry_type": "POINT",
        "name_status": "Official Assigned",
        "bounding_box": "nz_mainland",
        "source": "LINZ",
    }
    base.update(overrides)
    return base


class TestAnchorExtraction:
    @pytest.mark.parametrize(
        "description, expected",
        [
            ("About 8 km north of Waiouru, east side of main highway", "Waiouru"),
            ("ca 40 km south-east of Auckland", "Auckland"),
            ("3km east-northeast of Akaroa, Cabstand Road", "Akaroa"),
            ("2 km S.E. of Stockton, Nelson", "Stockton"),
            ("25 km S from Wanaka to Queenstown", "Wanaka"),
            ("5 km north of Lake Rotoiti", "Lake Rotoiti"),
        ],
    )
    def test_extracts_the_anchor(self, description, expected):
        assert extract_anchor(description) == expected

    def test_strips_a_trailing_relative_clause(self):
        assert extract_anchor("8 km N of Waiouru on the main highway") == "Waiouru"

    @pytest.mark.parametrize(
        "description, expected",
        [
            ("2 km S of Mt. Cook", "Mt. Cook"),
            ("3 km W of St. Bathans", "St. Bathans"),
            ("1 km E of Rte. 6", "Rte. 6"),
        ],
    )
    def test_keeps_abbreviation_periods(self, description, expected):
        """A period inside 'Mt.' is part of the token, not the end of the anchor."""
        assert extract_anchor(description) == expected

    def test_cuts_at_a_sentence_ending_period(self):
        assert extract_anchor("5 km N of Waiouru. Collected by J Smith") == "Waiouru"

    @pytest.mark.parametrize(
        "description, expected",
        [
            ("2 km W of the town of Otautau", "Otautau"),
            ("0.5 km E of Erua township", "Erua"),
            ("4 km S of the village of Kaeo", "Kaeo"),
        ],
    )
    def test_strips_settlement_wrappers(self, description, expected):
        assert extract_anchor(description) == expected

    @pytest.mark.parametrize(
        "description",
        [
            "Lake Rotoiti, by the outlet of the Buller River",
            "44.3 km upstream of Pipiriki",
            "Northern slopes Mt Anglem",
        ],
    )
    def test_returns_none_when_there_is_no_distance_direction(self, description):
        assert extract_anchor(description) is None


class TestSummary:
    def test_counts_and_ranges(self):
        records = [
            record(),
            record(linz_id="2", name="Akaroa", latitude=-43.81, longitude=172.97),
            record(linz_id="3", name="Waiouru", feature_type="Railway Station"),
        ]
        summary = summarise_snapshot(records)
        assert summary["record_count"] == 3
        assert summary["distinct_names"] == 2  # Waiouru appears twice
        assert summary["latitude_range"] == [-43.81, -39.4783]
        assert summary["top_feature_types"]["Locality"] == 2

    def test_reports_how_populated_each_administrative_field_is(self):
        """region is sparse in this layer; land_district is the usable one."""
        records = [record(), record(linz_id="2", region="Manawatu-Whanganui")]
        summary = summarise_snapshot(records)
        assert summary["land_district_populated"] == 2
        assert summary["region_populated"] == 1
        assert summary["territorial_authority_populated"] == 2


class TestAnchorCoverage:
    def test_matches_anchors_against_gazetteer_names(self):
        records = [record(), record(linz_id="2", name="Akaroa")]
        result = anchor_coverage(
            ["8 km N of Waiouru", "3 km NE of Akaroa", "5 km W of Nowhereville"], records
        )
        assert result["anchors_extracted"] == 3
        assert result["exact_match_rate"] == pytest.approx(round(2 / 3, 4))
        assert result["unmatched_examples"] == ["Nowhereville"]

    def test_macronised_gazetteer_names_match_unmacronised_descriptions(self):
        """The gazetteer holds 'Opotiki' with macrons; collection labels do not."""
        records = [record(linz_id="9", name="\u014cp\u014dtiki")]
        result = anchor_coverage(["12 km SE of Opotiki"], records)
        assert result["exact_match_rate"] == 0.0
        assert result["matched_only_after_diacritic_folding"] == 1
        assert result["match_rate_with_folding"] == 1.0
        assert result["folding_examples"] == ["Opotiki"]

    def test_matching_is_case_and_punctuation_insensitive(self):
        result = anchor_coverage(["8 km N of WAIOURU."], [record()])
        assert result["exact_match_rate"] == 1.0

    def test_descriptions_without_anchors_are_not_counted_as_failures(self):
        result = anchor_coverage(["Mt Anglem northern slopes"], [record()])
        assert result["anchors_extracted"] == 0
        assert result["exact_match_rate"] == 0.0
