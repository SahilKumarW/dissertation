"""UNIT TEST -- normalisation behaviour. No data files involved."""

import pytest

from georef.normalise import normalise


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Coulthard's  Scenic Reserve ", "coulthard's scenic reserve"),
        ("Coulthard\u2019s Scenic Reserve", "coulthard's scenic reserve"),
        ("Carter\u2018s Beach", "carter's beach"),
        ("MT CARGILL, DUNEDIN.", "mt cargill, dunedin"),
        ("Lake\u00a0Rotoiti", "lake rotoiti"),
        ("Buller \u2013 Punakaiki", "buller - punakaiki"),
        ('He said "Bluff View Pt."', 'he said "bluff view pt."'),
        ("trailing;;  ", "trailing"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_normalise_cases(raw, expected):
    assert normalise(raw) == expected


def test_normalise_is_idempotent():
    raw = "  Stewart\u00a0Island:  Mt\u2019 Anglem.  "
    once = normalise(raw)
    assert normalise(once) == once


def test_normalise_preserves_geographic_tokens():
    raw = "8 km SE of New Plymouth, Waiwakaiho River"
    out = normalise(raw)
    for token in ("8 km", "se", "new plymouth", "waiwakaiho river"):
        assert token in out


def test_distinct_localities_do_not_collide():
    assert normalise("North Head") != normalise("South Head")


class TestMatchKey:
    """The gazetteer carries macrons; historical collection labels rarely do."""

    @pytest.mark.parametrize(
        "gazetteer, label",
        [
            ("\u014cp\u014dtiki", "Opotiki"),
            ("Whangamat\u0101", "Whangamata"),
            ("Manawat\u016b-Whanganui", "Manawatu-Whanganui"),
            ("T\u0101maki Makaurau", "Tamaki Makaurau"),
        ],
    )
    def test_folded_forms_agree(self, gazetteer, label):
        from georef.normalise import match_key

        assert match_key(gazetteer) == match_key(label)

    def test_normalise_itself_is_unchanged_by_folding(self):
        """normalise() defines the benchmark join key and must keep macrons."""
        assert normalise("\u014cp\u014dtiki") != normalise("Opotiki")

    def test_folding_is_only_additive(self):
        from georef.normalise import fold_diacritics, match_key

        assert fold_diacritics("Waiouru") == "Waiouru"
        assert match_key("  WAIOURU. ") == "waiouru"


class TestSourceFilenameResolution:
    """The same supplier file circulates under two spellings."""

    def test_exact_match_wins(self, tmp_path):
        from georef.data.benchmark_builder import resolve_source_file

        (tmp_path / "SampleData-NoCoords.xlsx").write_text("x")
        (tmp_path / "SampleDataNoCoords.xlsx").write_text("y")
        found = resolve_source_file(tmp_path, "SampleDataNoCoords.xlsx")
        assert found.name == "SampleDataNoCoords.xlsx"

    def test_separator_and_case_differences_resolve(self, tmp_path):
        from georef.data.benchmark_builder import resolve_source_file

        (tmp_path / "MWLR-Data-CHR-20230307 nz only.csv").write_text("x")
        found = resolve_source_file(tmp_path, "MWLRDataCHR20230307_nz_only.csv")
        assert found is not None

    def test_an_unrelated_filename_does_not_resolve(self, tmp_path):
        from georef.data.benchmark_builder import resolve_source_file

        (tmp_path / "some_other_file.csv").write_text("x")
        assert resolve_source_file(tmp_path, "MWLRDataCHR20230307.csv") is None

    def test_ambiguous_matches_are_refused(self, tmp_path):
        """Two files differing only by separators is not a resolvable case."""
        from georef.data.benchmark_builder import resolve_source_file

        (tmp_path / "a-b.csv").write_text("x")
        (tmp_path / "a_b.csv").write_text("y")
        assert resolve_source_file(tmp_path, "ab.csv") is None
