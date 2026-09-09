"""UNIT TEST -- anchor variants and generic-term handling.

Every case below is a real anchor string from the 100-record dev run.
"""

from __future__ import annotations

import pytest

from georef.disambiguation.anchors import (
    anchor_variants,
    feature_terms,
    is_generic_only,
)


class TestVariants:
    def test_the_original_is_always_tried_first(self):
        assert anchor_variants("Waiouru")[0] == "Waiouru"

    def test_bracketed_alternative_names_become_separate_queries(self):
        variants = anchor_variants("Minaret Creek [Minaret Burn]")
        assert "Minaret Creek [Minaret Burn]" in variants
        assert "Minaret Creek" in variants
        assert "Minaret Burn" in variants

    @pytest.mark.parametrize(
        "anchor, expected",
        [
            ("upper end of Lake Orbell", "Lake Orbell"),
            ("west end of Lake Orbell", "Lake Orbell"),
            ("the town of Otautau", "Otautau"),
            ("Erua township", "Erua"),
            ("Tauherenikau River mouth", "Tauherenikau River"),
        ],
    )
    def test_qualifiers_and_wrappers_are_stripped(self, anchor, expected):
        assert expected in anchor_variants(anchor)

    def test_variants_never_replace_the_original(self):
        """A variant can only widen the pool, never lose a match the original found."""
        for anchor in ["upper end of Lake Orbell", "the town of Otautau", "Erua township"]:
            assert anchor_variants(anchor)[0] == anchor

    def test_no_duplicate_query_forms(self):
        from georef.normalise import match_key

        variants = anchor_variants("Lake Rotoiti")
        assert len({match_key(v) for v in variants}) == len(variants)

    def test_empty_anchor_has_no_variants(self):
        assert anchor_variants(None) == []
        assert anchor_variants("   ") == []


class TestFeatureTerms:
    @pytest.mark.parametrize(
        "anchor, expected_type",
        [
            ("Porarari River", "Stream"),
            ("Lake Hankinson", "Lake"),
            ("Whatakao Stream", "Stream"),
            ("Sinclair Head", "Point"),
            ("Rock and Pillar railway station", "Railway Station"),
            ("Te Pukaha Trig", "Trig Station"),
            ("Jack's Hut", "Hut"),
        ],
    )
    def test_generic_words_imply_a_feature_type(self, anchor, expected_type):
        assert expected_type in feature_terms(anchor)

    def test_a_plain_place_name_implies_nothing(self):
        assert feature_terms("Waiouru") == ()

    def test_macronised_anchors_still_resolve_terms(self):
        assert "Lake" in feature_terms("Lake \u014ctamangakau")


class TestGenericOnly:
    @pytest.mark.parametrize(
        "anchor",
        ["the bridge across the stream", "the summit", "the point", "hut track", "Pass"],
    )
    def test_anchors_naming_no_place_are_flagged(self, anchor):
        """A gazetteer of proper names cannot resolve these, however good
        retrieval is. Detecting it turns a guaranteed wrong answer into an
        honest abstention."""
        assert is_generic_only(anchor)

    @pytest.mark.parametrize(
        "anchor",
        ["Waiouru", "Lake Rotoiti", "Porarari River", "Akaroa Harbour", "Jack's Hut"],
    )
    def test_named_places_are_not_flagged(self, anchor):
        """Generic words are not discarded: 'Lake Rotoiti' carries real
        feature-type information in exactly that word."""
        assert not is_generic_only(anchor)

    def test_empty_anchor_counts_as_generic(self):
        assert is_generic_only(None)
        assert is_generic_only("the")


class TestBackReferenceResolution:
    """Descriptions name a feature once then refer back to it. The referent is
    in the text, so resolving it is text processing, not geography. Every case
    below is a real dev-run description that abstained without this."""

    @pytest.mark.parametrize(
        "anchor, description, expected",
        [
            (
                "Pass",
                "Lindis Pass, Stream gully crossing Highway 8 about 200m west of Pass",
                "Lindis Pass",
            ),
            (
                "the point",
                "S.W. Nelson, Perpendicular Point by trig. 1/2 mile N of the point",
                "Perpendicular Point",
            ),
            (
                "the summit",
                "Mt. Maungatua south west of Dunedin, rock c. 500 m west of the summit",
                "Mt. Maungatua",
            ),
            (
                "the bridge across the stream",
                "bank of Mangahowhi Stream, between Mangahoe Road and stream. Approx 10 m "
                "NW from the bridge across the stream to Sutherlands Bush No. 2.",
                "Mangahowhi Stream",
            ),
        ],
    )
    def test_real_dev_cases(self, anchor, description, expected):
        from georef.disambiguation.anchors import resolve_back_reference

        assert resolve_back_reference(anchor, description) == expected

    def test_it_declines_when_the_description_names_nothing_matching(self):
        """'100m west of hut track' names no track and no hut, so this stays an
        abstention rather than becoming an arbitrary substitution."""
        from georef.disambiguation.anchors import resolve_back_reference

        assert (
            resolve_back_reference(
                "hut track",
                "Chatham Islands; South East Island (Rangatira); Northwest headland, "
                "100m west of hut track.",
            )
            is None
        )

    def test_a_named_anchor_is_never_rewritten(self):
        from georef.disambiguation.anchors import resolve_back_reference

        assert resolve_back_reference("Waiouru", "About 8 km north of Waiouru") is None

    def test_it_never_invents_a_name_absent_from_the_description(self):
        from georef.disambiguation.anchors import resolve_back_reference

        assert resolve_back_reference("the summit", "rock 500 m west of the summit") is None

    def test_abbreviated_and_spelled_forms_agree(self):
        from georef.disambiguation.anchors import name_similarity

        assert name_similarity("Mt. Cook", "Mount Cook") == 1.0
