"""UNIT TEST -- audit categorisation, exercised against synthetic indexes."""

from georef.data.benchmark_builder import (
    DUPLICATE_ROWS_SAME_COORDINATE,
    NO_COORDINATE,
    NO_MATCH,
    SINGLE_ROW_EXACT,
    SINGLE_ROW_NORMALIZED,
    UNRESOLVED_MULTI_COORD,
    audit_description,
    compile_pattern,
)
from georef.config.loader import load_benchmark_config
from georef.normalise import normalise

DECIMALS = 6


def row(event_id, lat, lon):
    return {"event_id": event_id, "lat": lat, "lon": lon, "metadata": {}}


def test_single_row_exact():
    raw = "2 km NE of Rakaia"
    result = audit_description(raw, normalise(raw), {raw: [row("a", -43.7, 172.0)]}, {}, DECIMALS)
    assert result["audit_category"] == SINGLE_ROW_EXACT
    assert result["latitude"] == -43.7


def test_duplicate_rows_same_coordinate():
    raw = "2 km NE of Rakaia"
    rows = [row("a", -43.7, 172.0), row("b", -43.7000001, 172.0)]
    result = audit_description(raw, normalise(raw), {raw: rows}, {}, DECIMALS)
    assert result["audit_category"] == DUPLICATE_ROWS_SAME_COORDINATE
    assert result["source_row_count"] == 2


def test_conflicting_coordinates_are_unresolved():
    raw = "2 km NE of Rakaia"
    rows = [row("a", -43.7, 172.0), row("b", -44.9, 171.0)]
    result = audit_description(raw, normalise(raw), {raw: rows}, {}, DECIMALS)
    assert result["audit_category"] == UNRESOLVED_MULTI_COORD
    assert result["latitude"] is None


def test_normalised_fallback_is_labelled_distinctly():
    raw = "2 km NE of Rakaia."
    key = normalise(raw)
    result = audit_description(raw, key, {}, {key: [row("a", -43.7, 172.0)]}, DECIMALS)
    assert result["audit_category"] == SINGLE_ROW_NORMALIZED
    assert result["matched_by"] == "normalised"


def test_no_match_and_no_coordinate():
    raw = "nowhere at all"
    assert audit_description(raw, normalise(raw), {}, {}, DECIMALS)["audit_category"] == NO_MATCH
    result = audit_description(raw, normalise(raw), {raw: [row("a", None, None)]}, {}, DECIMALS)
    assert result["audit_category"] == NO_COORDINATE


class TestEligibilityPattern:
    pattern = compile_pattern(load_benchmark_config()["distance_direction_pattern"])

    def test_matches_expected_forms(self):
        for text in [
            "ca 40 km south-east of Auckland",
            "3km east-northeast of Akaroa",
            "About 8 km north of Waiouru",
            "54 km ESE of New Plymouth",
            "3.5 miles east of Route 6",
            "Approx 10 m NW from the bridge",
            "25 km S from Wanaka",
            "2 km S.E. of Stockton",
            "1.5 km N.N.W. of Erua",
            "7 mile N.N.E. of Waiau",
        ]:
            assert self.pattern.search(text), text

    def test_rejects_non_distance_direction(self):
        for text in [
            "Lake Rotoiti, by the outlet of the Buller River",
            "Northern slopes Mt Anglem above Christmas Village",
            "South Head, upper Kawau Creek",
            "gully W of road, below entrance gate",
            "Along the 9.2 km section of road from the junction",
            "44.3 km upstream of Pipiriki",
            "extending along at least 1 km of coastline",
        ]:
            assert not self.pattern.search(text), text
