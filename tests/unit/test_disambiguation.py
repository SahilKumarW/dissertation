"""UNIT TEST -- deterministic candidate selection and abstention."""

from __future__ import annotations

import pytest

from georef.config.loader import load_yaml, repo_root
from georef.disambiguation.anchors import anchor_variants
from georef.disambiguation.resolver import Disambiguator, SelectionStatus

CONFIG = load_yaml(repo_root() / "configs" / "disambiguation.yaml")


def record(name, **overrides):
    base = {
        "linz_id": overrides.pop("linz_id", "1"),
        "name": name,
        "feature_type": "Locality",
        "land_district": "Wellington",
        "territorial_authority": "Ruapehu District",
        "region": None,
        "geometry_type": "POINT",
        "name_status": "Official Assigned",
        "latitude": -39.4783,
        "longitude": 175.6689,
        "source": "LINZ",
    }
    base.update(overrides)
    return base


class Candidate:
    """Stands in for a retrieval Candidate."""

    def __init__(self, rec, rerank_score=None):
        self.record = rec
        self.rerank_score = rerank_score


@pytest.fixture
def resolver():
    return Disambiguator(CONFIG)


class TestSelection:
    def test_exact_name_match_wins(self, resolver):
        candidates = [
            Candidate(record("Waiouru Junction", linz_id="2")),
            Candidate(record("Waiouru", linz_id="1")),
        ]
        result = resolver.select("Waiouru", candidates, anchor_variants("Waiouru"))
        assert result.is_selected
        assert result.selected.name == "Waiouru"

    def test_feature_type_breaks_a_name_tie(self, resolver):
        """'Porarari River' should prefer the Stream over a Locality of the same name."""
        candidates = [
            Candidate(record("Porarari River", linz_id="1", feature_type="Locality")),
            Candidate(record("Porarari River", linz_id="2", feature_type="Stream")),
        ]
        result = resolver.select("Porarari River", candidates, anchor_variants("Porarari River"))
        assert result.is_selected
        assert result.selected.linz_id == "2"

    def test_stated_region_breaks_a_tie(self, resolver):
        candidates = [
            Candidate(record("Stockton", linz_id="1", land_district="Otago")),
            Candidate(record("Stockton", linz_id="2", land_district="Nelson")),
        ]
        result = resolver.select("Stockton", candidates, ["Stockton"], context="Nelson")
        assert result.is_selected
        assert result.selected.linz_id == "2"

    def test_multi_valued_districts_are_split(self, resolver):
        """'Canterbury, Westland' is two districts, not one place called both."""
        candidates = [
            Candidate(record("Harihari", linz_id="1", land_district="Otago")),
            Candidate(record("Harihari", linz_id="2", land_district="Canterbury, Westland")),
        ]
        result = resolver.select("Harihari", candidates, ["Harihari"], context="Westland")
        assert result.is_selected
        assert result.selected.linz_id == "2"

    def test_a_point_beats_the_representative_point_of_a_river(self, resolver):
        candidates = [
            Candidate(record("Karamea", linz_id="1", geometry_type="LINE")),
            Candidate(record("Karamea", linz_id="2", geometry_type="POINT")),
        ]
        result = resolver.select("Karamea", candidates, ["Karamea"])
        assert result.selected.linz_id == "2"

    def test_name_status_alone_is_too_weak_to_decide(self, resolver):
        """Official status is real but weak evidence. On its own, against the
        retrieval order pointing the other way, it does not clear the margin --
        so two candidates in DIFFERENT places abstain. One weak signal is not
        enough to choose between two distinct places."""
        candidates = [
            Candidate(record("Peria", linz_id="1", name_status="Unofficial Recorded")),
            Candidate(
                record("Peria", linz_id="2", name_status="Official Assigned",
                       latitude=-41.0, longitude=174.0)
            ),
        ]
        result = resolver.select("Peria", candidates, ["Peria"])
        assert result.status == SelectionStatus.INSUFFICIENT_MARGIN

    def test_status_does_decide_once_another_feature_agrees(self, resolver):
        candidates = [
            Candidate(
                record("Peria", linz_id="1", name_status="Unofficial Recorded", geometry_type="LINE")
            ),
            Candidate(record("Peria", linz_id="2", name_status="Official Assigned")),
        ]
        result = resolver.select("Peria", candidates, ["Peria"])
        assert result.is_selected
        assert result.selected.linz_id == "2"

    def test_macronised_gazetteer_name_matches_an_unmacronised_anchor(self, resolver):
        candidates = [Candidate(record("\u014cp\u014dtiki", linz_id="9"))]
        result = resolver.select("Opotiki", candidates, anchor_variants("Opotiki"))
        assert result.is_selected
        assert result.selected.features["folded_name"] == 1.0

    def test_a_bracketed_anchor_still_matches_through_a_variant(self, resolver):
        candidates = [Candidate(record("Minaret Creek", linz_id="3", feature_type="Stream"))]
        anchor = "Minaret Creek [Minaret Burn]"
        result = resolver.select(anchor, candidates, anchor_variants(anchor))
        assert result.is_selected


class TestAbstention:
    def test_a_generic_anchor_abstains_before_scoring(self, resolver):
        candidates = [Candidate(record("Some Bridge", linz_id="1"))]
        result = resolver.select("the bridge across the stream", candidates, ["the bridge"])
        assert result.status == SelectionStatus.GENERIC_ANCHOR
        assert result.selected is None

    def test_no_candidates_is_a_named_failure(self, resolver):
        result = resolver.select("Waiouru", [], ["Waiouru"])
        assert result.status == SelectionStatus.NO_CANDIDATES

    def test_two_different_places_with_the_same_name_abstain(self, resolver):
        """The Stockton in Nelson and a Stockton elsewhere are a coin flip, and
        forcing a choice would record one as a prediction."""
        candidates = [
            Candidate(record("Stockton", linz_id="1")),
            Candidate(record("Stockton", linz_id="2", latitude=-45.0, longitude=170.0)),
        ]
        result = resolver.select("Stockton", candidates, ["Stockton"])
        assert result.status == SelectionStatus.INSUFFICIENT_MARGIN
        assert result.margin is not None and result.margin < CONFIG["abstention"]["minimum_margin"]

    def test_duplicate_records_of_one_place_resolve_rather_than_abstain(self, resolver):
        """Akaroa is both a Locality and a Town a few hundred metres apart.
        Abstaining between them answers a question that does not matter: either
        choice yields the same coordinate."""
        candidates = [
            Candidate(
                record("Akaroa", linz_id="1", feature_type="Locality",
                       latitude=-43.8103, longitude=172.9679)
            ),
            Candidate(
                record("Akaroa", linz_id="2", feature_type="Town",
                       latitude=-43.8110, longitude=172.9685)
            ),
        ]
        result = resolver.select("Akaroa", candidates, ["Akaroa"])
        assert result.is_selected
        assert "resolved spatially" in result.failure_reason

    def test_the_spatial_tie_break_respects_its_threshold(self, resolver):
        """Just outside the threshold, the candidates are different places again."""
        candidates = [
            Candidate(record("Akaroa", linz_id="1", latitude=-43.8103, longitude=172.9679)),
            Candidate(record("Akaroa", linz_id="2", latitude=-43.8600, longitude=172.9679)),
        ]
        result = resolver.select("Akaroa", candidates, ["Akaroa"])
        assert result.status == SelectionStatus.INSUFFICIENT_MARGIN

    def test_weak_evidence_abstains(self, resolver):
        """Nothing matches the anchor; the best candidate is not good enough."""
        candidates = [
            Candidate(
                record(
                    "Somewhere Else",
                    linz_id="1",
                    feature_type="Stream",
                    geometry_type="LINE",
                    name_status="Unofficial Recorded",
                    land_district="Otago",
                )
            )
        ]
        result = resolver.select("Waiouru", candidates, ["Waiouru"], context="Nelson")
        assert result.status == SelectionStatus.BELOW_MINIMUM_SCORE

    def test_abstentions_still_record_what_was_considered(self, resolver):
        candidates = [
            Candidate(record("Stockton", linz_id="1")),
            Candidate(record("Stockton", linz_id="2", latitude=-45.0, longitude=170.0)),
        ]
        provenance = resolver.select("Stockton", candidates, ["Stockton"]).to_provenance()
        assert len(provenance["candidates_considered"]) == 2
        assert provenance["disambiguation_failure_reason"]


class TestDeterminism:
    def test_the_same_input_always_gives_the_same_order(self, resolver):
        candidates = [
            Candidate(record("Stockton", linz_id="9")),
            Candidate(record("Stockton", linz_id="2")),
        ]
        orders = {
            tuple(c.linz_id for c in resolver.select("Stockton", candidates, ["Stockton"]).considered)
            for _ in range(5)
        }
        assert len(orders) == 1

    def test_equal_scores_break_on_identifier(self, resolver):
        """Retrieval order is real evidence, so reversing the input legitimately
        changes the result. Only genuinely equal scores fall through to the
        identifier tie-break."""
        scored = [
            resolver.score_candidate("Stockton", ["Stockton"], Candidate(record("Stockton", linz_id=i)), 0, 2, None)
            for i in ("9", "2")
        ]
        assert scored[0].score == scored[1].score
        scored.sort(key=lambda c: (-c.score, str(c.linz_id)))
        assert [c.linz_id for c in scored] == ["2", "9"]

    def test_scores_stay_within_range(self, resolver):
        candidates = [Candidate(record("Waiouru", linz_id="1"), rerank_score=12.0)]
        result = resolver.select("Waiouru", candidates, ["Waiouru"], context="Wellington")
        assert 0.0 <= result.selected.score <= 1.0

    def test_only_the_configured_number_of_candidates_is_considered(self, resolver):
        candidates = [Candidate(record(f"Place {i}", linz_id=str(i))) for i in range(50)]
        result = resolver.select("Place 0", candidates, ["Place 0"])
        assert len(result.considered) == CONFIG["candidates_considered"]


class TestNameSimilarity:
    """Graded name evidence. Without it a candidate sharing only a generic word
    could clear the minimum score on feature type and status alone."""

    def test_an_unrelated_name_is_pushed_below_the_minimum(self, resolver):
        """'Mount Guinevere' is not 'Mount Benoit'; they share only 'mount'."""
        candidates = [
            Candidate(record("Mount Guinevere", linz_id="1", feature_type="Hill")),
            Candidate(record("Mount Findlay", linz_id="2", feature_type="Hill")),
        ]
        result = resolver.select("Mount Benoit", candidates, ["Mount Benoit"])
        assert result.status == SelectionStatus.BELOW_MINIMUM_SCORE

    def test_a_partial_match_differing_only_by_a_generic_word_is_selected(self, resolver):
        """'Te Pukaha' is the gazetteer form of the anchor 'Te Pukaha Trig'."""
        candidates = [
            Candidate(record("Te Pukaha", linz_id="1", feature_type="Trig Station")),
            Candidate(record("Te Tahi", linz_id="2", feature_type="Trig Station",
                             latitude=-40.0, longitude=176.0)),
        ]
        result = resolver.select("Te Pukaha Trig", candidates, ["Te Pukaha Trig"])
        assert result.is_selected
        assert result.selected.name == "Te Pukaha"

    def test_a_shared_generic_word_alone_scores_nothing(self, resolver):
        candidates = [Candidate(record("Makarora Township", linz_id="1"))]
        result = resolver.select("Township Site", candidates, ["Township Site"])
        assert result.status == SelectionStatus.BELOW_MINIMUM_SCORE


class TestProvenance:
    def test_every_feature_contribution_is_recorded(self, resolver):
        candidates = [Candidate(record("Waiouru", linz_id="1"))]
        result = resolver.select("Waiouru", candidates, ["Waiouru"], context="Wellington")
        features = result.selected.features
        for name in CONFIG["weights"]:
            assert name in features, name

    def test_selection_provenance_names_the_authoritative_source(self, resolver):
        candidates = [Candidate(record("Waiouru", linz_id="1"))]
        provenance = resolver.select("Waiouru", candidates, ["Waiouru"]).to_provenance()
        assert provenance["selected_source"] == "LINZ"
        assert provenance["selected_linz_id"] == "1"
        assert provenance["disambiguation_score"] is not None

    def test_candidate_provenance_carries_coordinates(self):
        """Needed to audit a spatially resolved tie after the fact."""
        resolver = Disambiguator(CONFIG)
        candidates = [Candidate(record("Waiouru", linz_id="1"))]
        considered = resolver.select("Waiouru", candidates, ["Waiouru"]).considered
        assert considered[0].to_provenance()["latitude"] == -39.4783


class TestSettlementPrior:
    """When the anchor carries no generic term, the candidate's own type is
    still evidence. Regression from the dev run: a Hill named Pukenui beat the
    Locality named Pukenui and landed 203 km from the truth."""

    def northland(self, name, feature_type, linz_id, lat=-34.80, lon=173.00):
        return Candidate(
            record(name, linz_id=linz_id, feature_type=feature_type,
                   land_district="North Auckland", latitude=lat, longitude=lon)
        )

    def test_a_settlement_beats_a_hill_of_the_same_name(self, resolver):
        candidates = [
            self.northland("Pukenui", "Hill", "1"),
            self.northland("Pukenui", "Locality", "2", lat=-34.85, lon=173.05),
        ]
        result = resolver.select("Pukenui", candidates, ["Pukenui"])
        assert result.is_selected
        assert result.selected.record["feature_type"] == "Locality"

    def test_the_prior_does_not_apply_when_the_anchor_names_a_type(self, resolver):
        """'Pukenui Hill' asks for a hill; the prior must not override that."""
        candidates = [
            self.northland("Pukenui", "Locality", "1"),
            self.northland("Pukenui", "Hill", "2", lat=-34.85, lon=173.05),
        ]
        result = resolver.select("Pukenui Hill", candidates, ["Pukenui Hill"])
        assert result.is_selected
        assert result.selected.record["feature_type"] == "Hill"

    def test_it_is_a_preference_not_a_filter(self, resolver):
        """A terrain feature can still win on region evidence."""
        candidates = [
            Candidate(
                record("Karamea", linz_id="1", feature_type="Locality",
                       land_district="Otago", latitude=-45.0, longitude=170.0)
            ),
            Candidate(
                record("Karamea", linz_id="2", feature_type="Hill",
                       land_district="Nelson", latitude=-41.2, longitude=172.1)
            ),
        ]
        result = resolver.select("Karamea", candidates, ["Karamea"], context="Nelson")
        assert result.is_selected
        assert result.selected.linz_id == "2"


class TestSpatialCoherence:
    """A description usually names several places, and they pin each other down.
    Every catastrophic dev-set error (475-559 km) had a description naming other
    features that ruled the chosen candidate out."""

    def southland(self, name, linz_id):
        return Candidate(
            record(name, linz_id=linz_id, feature_type="Place",
                   land_district="Southland", latitude=-45.9, longitude=168.0)
        )

    def westland(self, name, linz_id):
        return Candidate(
            record(name, linz_id=linz_id, feature_type="Building",
                   land_district="Westland", latitude=-43.46, longitude=170.02)
        )

    FOX_GLACIER = [(-43.4645, 170.0176)]

    def test_a_distant_candidate_loses_to_a_nearby_one(self, resolver):
        """DEVGT-044: 'Weheka' near Fox Glacier Hotel picked a Te Weheka in
        Southland, 475 km away.

        Asserted as an ORDERING effect rather than a selection, because whether
        either candidate clears the minimum score is a separate question
        governed by a separately chosen threshold."""
        candidates = [self.southland("Te Weheka", "1"), self.westland("Weheka Post Office", "2")]

        without = resolver.select("Weheka", candidates, ["Weheka"], None, [])
        with_context = resolver.select("Weheka", candidates, ["Weheka"], None, self.FOX_GLACIER)

        assert without.considered[0].name == "Te Weheka"
        assert with_context.considered[0].name == "Weheka Post Office"

    def test_coherence_can_carry_a_candidate_over_the_minimum(self, resolver):
        """With an exact name match as well, the nearby candidate is selected."""
        candidates = [
            self.southland("Weheka", "1"),
            Candidate(
                record("Weheka", linz_id="2", feature_type="Locality",
                       land_district="Westland", latitude=-43.46, longitude=170.02)
            ),
        ]
        result = resolver.select("Weheka", candidates, ["Weheka"], None, self.FOX_GLACIER)
        assert result.is_selected
        assert result.selected.linz_id == "2"

    def test_without_context_points_it_stays_neutral(self, resolver):
        """A description naming nothing else must not be penalised."""
        candidates = [self.southland("Te Weheka", "1"), self.westland("Weheka Post Office", "2")]
        scored = resolver.select("Weheka", candidates, ["Weheka"], None, []).considered
        assert {c.features["spatial_coherence"] for c in scored} == {
            CONFIG["feature_scores"]["coherence_absent"]
        }

    def test_coherence_decays_with_distance(self, resolver):
        near = resolver.score_candidate(
            "X", ["X"], self.westland("X", "1"), 0, 1, None, self.FOX_GLACIER
        )
        far = resolver.score_candidate(
            "X", ["X"], self.southland("X", "2"), 0, 1, None, self.FOX_GLACIER
        )
        assert near.features["spatial_coherence"] == 1.0
        assert far.features["spatial_coherence"] == 0.0

    def test_a_candidate_between_the_thresholds_scores_partially(self, resolver):
        """100 km from the nearest named place: some support, not full."""
        middle = Candidate(
            record("X", linz_id="1", latitude=-42.56, longitude=170.02)
        )
        scored = resolver.score_candidate("X", ["X"], middle, 0, 1, None, self.FOX_GLACIER)
        assert 0.0 < scored.features["spatial_coherence"] < 1.0

    def test_it_cannot_override_an_exact_name_match(self, resolver):
        """Coherence is corroboration. An exact name still outranks it, or a
        cluster of nearby wrong names could beat the right place."""
        candidates = [
            Candidate(record("Weheka", linz_id="1", land_district="Southland",
                             latitude=-45.9, longitude=168.0)),
            self.westland("Weheka Post Office", "2"),
        ]
        result = resolver.select("Weheka", candidates, ["Weheka"], None, self.FOX_GLACIER)
        assert result.selected.name == "Weheka"
