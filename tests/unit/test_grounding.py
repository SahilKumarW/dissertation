"""UNIT TEST -- the offline half of Condition B.

Real lexical retrieval, real disambiguation, real geodesy, over a small
synthetic gazetteer. No network and no model.
"""

from __future__ import annotations

import pytest

from georef.config.loader import load_yaml, repo_root
from georef.disambiguation.resolver import Disambiguator
from georef.pipeline.grounding import Grounder, PredictionStatus
from georef.retrieval.hybrid import HybridRetriever
from georef.retrieval.lexical import LexicalIndex
from georef.spatial.geodesic import distance_between

RETRIEVAL = load_yaml(repo_root() / "configs" / "retrieval.yaml")
PIPELINE = load_yaml(repo_root() / "configs" / "pipeline.yaml")
DISAMBIGUATION = load_yaml(repo_root() / "configs" / "disambiguation.yaml")

GAZETTEER = [
    {"linz_id": "1", "name": "Waiouru", "feature_type": "Locality", "land_district": "Wellington",
     "territorial_authority": "Ruapehu District", "region": None, "geometry_type": "POINT",
     "name_status": "Official Assigned", "latitude": -39.4783, "longitude": 175.6689,
     "country": "New Zealand", "source": "LINZ"},
    {"linz_id": "2", "name": "Akaroa", "feature_type": "Locality", "land_district": "Canterbury",
     "territorial_authority": "Christchurch City", "region": None, "geometry_type": "POINT",
     "name_status": "Official Assigned", "latitude": -43.8103, "longitude": 172.9679,
     "country": "New Zealand", "source": "LINZ"},
    {"linz_id": "3", "name": "\u014cp\u014dtiki", "feature_type": "Locality",
     "land_district": "Gisborne", "territorial_authority": "\u014cp\u014dtiki District",
     "region": None, "geometry_type": "POINT", "name_status": "Official Assigned",
     "latitude": -38.0055, "longitude": 177.2870, "country": "New Zealand", "source": "LINZ"},
    {"linz_id": "4", "name": "Lake Orbell", "feature_type": "Lake", "land_district": "Southland",
     "territorial_authority": "Southland District", "region": None, "geometry_type": "POLYGON",
     "name_status": "Official Assigned", "latitude": -45.2500, "longitude": 167.4500,
     "country": "New Zealand", "source": "LINZ"},
]


@pytest.fixture(scope="module")
def grounder():
    lexical = LexicalIndex(GAZETTEER, RETRIEVAL["document_template"])
    config = {**RETRIEVAL, "channels": {**RETRIEVAL["channels"],
                                        "dense": {"enabled": False, "top_k": 0}}}
    retriever = HybridRetriever(lexical, config)
    return Grounder(retriever, Disambiguator(DISAMBIGUATION), PIPELINE)


def interpretation(clauses, status="SUCCESS", context=None, benchmark_id="DEV-001"):
    return {
        "benchmark_id": benchmark_id,
        "status": status,
        "geographic_context": context,
        "clauses": clauses,
    }


def clause(anchor, bearing, metres):
    return {
        "anchor_entity": anchor,
        "bearing_degrees": bearing,
        "distance_metres": metres,
        "anchor_is_generic": False,
    }


class TestEndToEnd:
    def test_a_single_clause_produces_a_coordinate(self, grounder):
        result = grounder.predict(interpretation([clause("Waiouru", 225, 19_000)]))
        assert result.status == PredictionStatus.SUCCESS
        # 19 km south-west of the anchor, computed on the ellipsoid.
        assert distance_between(
            -39.4783, 175.6689, result.latitude, result.longitude
        ) == pytest.approx(19_000, abs=1.0)

    def test_the_grounded_anchor_is_recorded_as_authoritative(self, grounder):
        record = grounder.predict(interpretation([clause("Waiouru", 225, 19_000)])).to_record()
        assert record["selected_source"] == "LINZ"
        assert record["selected_candidate_name"] == "Waiouru"
        assert record["geodesic_method"].startswith("GeographicLib")

    def test_an_unmacronised_anchor_grounds_against_the_macronised_gazetteer(self, grounder):
        result = grounder.predict(interpretation([clause("Opotiki", 135, 12_000)]))
        assert result.status == PredictionStatus.SUCCESS

    def test_a_qualified_anchor_grounds_through_a_variant(self, grounder):
        result = grounder.predict(interpretation([clause("upper end of Lake Orbell", 270, 1_000)]))
        assert result.status == PredictionStatus.SUCCESS


class TestAbstention:
    def test_a_failed_interpretation_never_reaches_retrieval(self, grounder):
        result = grounder.predict(interpretation([], status="API_FAILURE"))
        assert result.status == PredictionStatus.NO_INTERPRETATION

    def test_a_generic_anchor_abstains(self, grounder):
        """Real dev-run anchor. No gazetteer of proper names can resolve it."""
        result = grounder.predict(
            interpretation([clause("the bridge across the stream", 315, 10)])
        )
        assert result.status == PredictionStatus.GROUNDING_FAILURE
        assert result.latitude is None

    def test_an_absent_anchor_abstains_rather_than_guessing(self, grounder):
        result = grounder.predict(interpretation([clause("Nowhereville", 0, 5_000)]))
        assert result.status in {
            PredictionStatus.GROUNDING_FAILURE,
            PredictionStatus.RETRIEVAL_FAILURE,
        }
        assert result.latitude is None

    def test_abstentions_still_record_what_was_tried(self, grounder):
        record = grounder.predict(interpretation([clause("Nowhereville", 0, 5_000)])).to_record()
        assert record["clause_selections"]
        assert record["failure_reason"]


class TestMultipleClauses:
    def test_disagreeing_clauses_abstain_under_the_fuse_policy(self, grounder):
        """Two anchors 400 km apart cannot both describe one point, and that is
        knowable without ground truth."""
        result = grounder.predict(
            interpretation([clause("Waiouru", 0, 5_000), clause("Akaroa", 0, 5_000)])
        )
        assert result.status == PredictionStatus.CLAUSE_DISAGREEMENT
        assert result.resolution.disagreement_metres > 100_000

    def test_one_ungroundable_clause_does_not_sink_a_groundable_one(self, grounder):
        result = grounder.predict(
            interpretation([clause("Nowhereville", 0, 5_000), clause("Waiouru", 225, 19_000)])
        )
        assert result.status == PredictionStatus.SUCCESS
        assert len(result.clause_selections) == 2


class TestCoordinateSanity:
    def test_a_prediction_outside_new_zealand_is_rejected(self, grounder):
        """Master spec section 47: never plot an invalid coordinate."""
        result = grounder.predict(interpretation([clause("Waiouru", 0, 3_000_000)]))
        assert result.status == PredictionStatus.VALIDATION_FAILURE
        assert result.latitude is None
        assert "outside every configured" in result.failure_reason

    def test_sanity_can_be_disabled_by_configuration(self):
        lexical = LexicalIndex(GAZETTEER, RETRIEVAL["document_template"])
        config = {**RETRIEVAL, "channels": {**RETRIEVAL["channels"],
                                            "dense": {"enabled": False, "top_k": 0}}}
        relaxed = {**PIPELINE, "coordinate_sanity": {"enforce_bounding_boxes": False}}
        grounder = Grounder(HybridRetriever(lexical, config), Disambiguator(DISAMBIGUATION), relaxed)
        result = grounder.predict(interpretation([clause("Waiouru", 0, 3_000_000)]))
        assert result.status == PredictionStatus.SUCCESS


class TestBackReferenceGrounding:
    """The generic-anchor abstention should not fire when the description names
    the referent outright."""

    @pytest.fixture(scope="class")
    def lindis_grounder(self):
        gazetteer = GAZETTEER + [
            {"linz_id": "5", "name": "Lindis Pass", "feature_type": "Pass",
             "land_district": "Otago", "territorial_authority": None, "region": None,
             "geometry_type": "POINT", "name_status": "Official Assigned",
             "latitude": -44.5867, "longitude": 169.6667, "country": "New Zealand",
             "source": "LINZ"},
        ]
        lexical = LexicalIndex(gazetteer, RETRIEVAL["document_template"])
        config = {**RETRIEVAL, "channels": {**RETRIEVAL["channels"],
                                            "dense": {"enabled": False, "top_k": 0}}}
        return Grounder(HybridRetriever(lexical, config), Disambiguator(DISAMBIGUATION), PIPELINE)

    def test_a_back_reference_grounds_to_the_named_feature(self, lindis_grounder):
        record = {
            **interpretation([clause("Pass", 270, 200)]),
            "input_description": "Lindis Pass, Stream gully crossing Highway 8 about "
                                 "200m west of Pass",
        }
        result = lindis_grounder.predict(record)
        assert result.status == PredictionStatus.SUCCESS
        assert result.to_record()["selected_candidate_name"] == "Lindis Pass"

    def test_the_original_anchor_is_still_recorded(self, lindis_grounder):
        """Provenance keeps what the model said as well as what it resolved to."""
        record = {
            **interpretation([clause("Pass", 270, 200)]),
            "input_description": "Lindis Pass, Stream gully crossing Highway 8 about "
                                 "200m west of Pass",
        }
        selection = lindis_grounder.predict(record).clause_selections[0]
        assert selection["anchor_entity"] == "Pass"
        assert selection["resolved_anchor_entity"] == "Lindis Pass"

    def test_without_a_description_it_still_abstains(self, lindis_grounder):
        result = lindis_grounder.predict(interpretation([clause("Pass", 270, 200)]))
        assert result.status == PredictionStatus.GROUNDING_FAILURE


class TestContextPoints:
    def test_the_anchor_does_not_corroborate_itself(self):
        """Otherwise every candidate for an ambiguous name looks coherent."""
        from georef.retrieval.name_spotter import NameSpotter

        lexical = LexicalIndex(GAZETTEER, RETRIEVAL["document_template"])
        config = {**RETRIEVAL, "channels": {**RETRIEVAL["channels"],
                                            "dense": {"enabled": False, "top_k": 0}}}
        grounder = Grounder(
            HybridRetriever(lexical, config),
            Disambiguator(DISAMBIGUATION),
            PIPELINE,
            NameSpotter(GAZETTEER),
        )
        points = grounder._context_points("19 km SW of Waiouru", "Waiouru")
        assert points == []

    def test_other_named_places_become_context_points(self):
        from georef.retrieval.name_spotter import NameSpotter

        lexical = LexicalIndex(GAZETTEER, RETRIEVAL["document_template"])
        config = {**RETRIEVAL, "channels": {**RETRIEVAL["channels"],
                                            "dense": {"enabled": False, "top_k": 0}}}
        grounder = Grounder(
            HybridRetriever(lexical, config),
            Disambiguator(DISAMBIGUATION),
            PIPELINE,
            NameSpotter(GAZETTEER),
        )
        points = grounder._context_points("Akaroa Harbour, 19 km SW of Waiouru", "Waiouru")
        assert points and points[0][0] == pytest.approx(-43.8103)
