"""UNIT TEST -- multi-anchor resolution policies and disagreement abstention."""

from __future__ import annotations

import pytest

from georef.spatial.geodesic import destination, distance_between
from georef.spatial.resolution import (
    GroundedClause,
    ResolutionStatus,
    resolve_clauses,
)

NEW_PLYMOUTH = (-39.0556, 174.0752)
WHANGAMOMONA = (-39.1447, 174.8722)
WAIOURU = (-39.4783, 175.6689)


def clause(anchor, coords, bearing, metres, name="anchor"):
    return GroundedClause(
        anchor_name=anchor,
        anchor_latitude=coords[0],
        anchor_longitude=coords[1],
        bearing_degrees=bearing,
        distance_metres=metres,
        anchor_linz_id="1",
    )


class TestSingleClause:
    def test_predicts_the_geodesic_destination(self):
        result = resolve_clauses([clause("Waiouru", WAIOURU, 225, 19_000)])
        assert result.status == ResolutionStatus.SUCCESS
        expected = destination(*WAIOURU, 225, 19_000)
        assert result.latitude == pytest.approx(expected.latitude)
        assert result.longitude == pytest.approx(expected.longitude)

    def test_disagreement_is_undefined_with_one_clause(self):
        assert resolve_clauses([clause("Waiouru", WAIOURU, 225, 19_000)]).disagreement_metres is None

    def test_policy_does_not_matter_for_one_clause(self):
        points = {
            policy: resolve_clauses([clause("W", WAIOURU, 225, 19_000)], policy=policy)
            for policy in ("first", "shortest", "fuse")
        }
        coordinates = {(round(r.latitude, 9), round(r.longitude, 9)) for r in points.values()}
        assert len(coordinates) == 1


class TestPolicies:
    @pytest.fixture
    def agreeing(self):
        """Two clauses describing very nearly the same point, as a real
        two-clause description does."""
        target = destination(*NEW_PLYMOUTH, 112.5, 58_000)
        long_clause = clause("New Plymouth", NEW_PLYMOUTH, 112.5, 58_000)
        bearing = 270.0
        short_clause = clause(
            "Whangamommona",
            (target.latitude, WHANGAMOMONA[1] + 0.05),
            bearing,
            distance_between(
                target.latitude, WHANGAMOMONA[1] + 0.05, target.latitude, target.longitude
            ),
        )
        return [long_clause, short_clause]

    def test_first_uses_the_leading_clause(self, agreeing):
        result = resolve_clauses(agreeing, policy="first")
        expected = destination(*NEW_PLYMOUTH, 112.5, 58_000)
        assert result.latitude == pytest.approx(expected.latitude)

    def test_shortest_uses_the_smallest_offset(self, agreeing):
        result = resolve_clauses(agreeing, policy="shortest")
        shortest = min(agreeing, key=lambda c: c.distance_metres)
        expected = shortest.destination()
        assert result.latitude == pytest.approx(expected.latitude)

    def test_fuse_lands_between_the_two_estimates(self, agreeing):
        result = resolve_clauses(agreeing, policy="fuse")
        estimates = result.clause_estimates
        gaps = [
            distance_between(
                result.latitude, result.longitude, e["predicted_latitude"], e["predicted_longitude"]
            )
            for e in estimates
        ]
        separation = distance_between(
            estimates[0]["predicted_latitude"], estimates[0]["predicted_longitude"],
            estimates[1]["predicted_latitude"], estimates[1]["predicted_longitude"],
        )
        assert max(gaps) <= separation + 1.0

    def test_fuse_leans_toward_the_shorter_offset(self):
        """A 4 km offset compounds far less anchor error than a 54 km one, so
        the fused point should sit much closer to the short clause's estimate."""
        far = clause("far", WAIOURU, 0, 54_000)
        far_point = far.destination()

        # Put the short clause's anchor 3 km east of the long clause's estimate
        # and point it west, so the two estimates differ by about 1 km rather
        # than coinciding.
        near_anchor = destination(far_point.latitude, far_point.longitude, 90, 3_000)
        near = clause("near", (near_anchor.latitude, near_anchor.longitude), 270, 4_000)
        near_point = near.destination()

        separation = distance_between(*far_point.as_tuple(), *near_point.as_tuple())
        assert separation > 500, "fixture must produce two distinct estimates"

        result = resolve_clauses([far, near], policy="fuse")
        to_far = distance_between(result.latitude, result.longitude, *far_point.as_tuple())
        to_near = distance_between(result.latitude, result.longitude, *near_point.as_tuple())

        assert to_near < to_far
        # Inverse-offset weighting puts it around 93% of the way toward the
        # short clause: (1/4000) / (1/4000 + 1/54000).
        assert to_near < separation * 0.15

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown resolution policy"):
            resolve_clauses([clause("W", WAIOURU, 0, 1000)], policy="whatever")


class TestDisagreementAbstention:
    def test_wildly_disagreeing_clauses_abstain(self):
        """Two estimates 200 km apart mean something is wrong, and that is
        knowable without ground truth."""
        first = clause("a", WAIOURU, 0, 10_000)
        second = clause("b", NEW_PLYMOUTH, 180, 10_000)
        result = resolve_clauses([first, second], policy="fuse")
        assert result.status == ResolutionStatus.CLAUSE_DISAGREEMENT
        assert result.latitude is None
        assert result.disagreement_metres > 100_000

    def test_the_abstention_records_how_far_apart_they_were(self):
        result = resolve_clauses(
            [clause("a", WAIOURU, 0, 10_000), clause("b", NEW_PLYMOUTH, 180, 10_000)],
            policy="fuse",
        )
        assert "disagree by" in result.failure_reason
        assert result.clause_estimates  # both estimates preserved for analysis

    def test_agreement_within_the_threshold_proceeds(self):
        near_anchor = destination(*WAIOURU, 0, 5_000)
        result = resolve_clauses(
            [
                clause("a", WAIOURU, 0, 10_000),
                clause("b", (near_anchor.latitude, near_anchor.longitude), 0, 5_000),
            ],
            policy="fuse",
            disagreement_threshold_metres=25_000,
        )
        assert result.status == ResolutionStatus.SUCCESS

    def test_only_fuse_abstains(self):
        """first and shortest have no basis to abstain; that is the point of
        comparing them against fuse on the dev set."""
        clauses = [clause("a", WAIOURU, 0, 10_000), clause("b", NEW_PLYMOUTH, 180, 10_000)]
        for policy in ("first", "shortest"):
            assert resolve_clauses(clauses, policy=policy).status == ResolutionStatus.SUCCESS

    def test_disagreement_is_still_recorded_when_not_abstaining(self):
        clauses = [clause("a", WAIOURU, 0, 10_000), clause("b", NEW_PLYMOUTH, 180, 10_000)]
        result = resolve_clauses(clauses, policy="first")
        assert result.disagreement_metres > 100_000


class TestFailureAndProvenance:
    def test_no_clauses_is_a_named_failure(self):
        result = resolve_clauses([], policy="fuse")
        assert result.status == ResolutionStatus.NO_GROUNDED_CLAUSE
        assert result.latitude is None

    def test_provenance_records_every_clause_estimate(self):
        provenance = resolve_clauses(
            [clause("Waiouru", WAIOURU, 225, 19_000)]
        ).to_provenance()
        assert provenance["resolution_status"] == "SUCCESS"
        assert provenance["resolution_policy"] == "fuse"
        assert provenance["geodesic_method"].startswith("GeographicLib")
        estimate = provenance["clause_estimates"][0]
        assert estimate["anchor_source"] == "LINZ"
        assert estimate["anchor_linz_id"] == "1"
        assert estimate["bearing_degrees"] == 225
