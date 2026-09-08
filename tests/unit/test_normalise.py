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
