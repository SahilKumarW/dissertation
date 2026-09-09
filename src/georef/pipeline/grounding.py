"""Grounding: turn an interpretation into a predicted coordinate.

    clauses -> anchor variants -> retrieval -> disambiguation -> LINZ anchor
            -> deterministic geodesy -> coordinate sanity -> prediction

This is the part of Condition B that runs entirely on frozen local assets: the
hashed LINZ snapshot and the index built from it. No network, no model.

Every stage can abstain, and each abstention has its own reason. A record that
fails here is a record where the pipeline knew it did not know, which is a
different outcome from a wrong coordinate and is counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from georef.disambiguation.anchors import anchor_variants, resolve_back_reference
from georef.disambiguation.resolver import Disambiguator, Selection
from georef.normalise import match_key
from georef.spatial.bounds import containing_box
from georef.spatial.resolution import (
    GroundedClause,
    Resolution,
    ResolutionStatus,
    resolve_clauses,
)

__all__ = ["PredictionStatus", "Prediction", "Grounder"]


class PredictionStatus:
    SUCCESS = "SUCCESS"
    NO_INTERPRETATION = "NO_INTERPRETATION"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    CLAUSE_DISAGREEMENT = "CLAUSE_DISAGREEMENT"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"


@dataclass
class Prediction:
    benchmark_id: str
    status: str
    latitude: float | None = None
    longitude: float | None = None
    failure_reason: str | None = None
    clause_selections: list[dict] = field(default_factory=list)
    resolution: Resolution | None = None

    @property
    def is_success(self) -> bool:
        return self.status == PredictionStatus.SUCCESS

    def to_record(self) -> dict:
        record = {
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "predicted_latitude": self.latitude,
            "predicted_longitude": self.longitude,
            "failure_reason": self.failure_reason,
            "clause_selections": self.clause_selections,
        }
        if self.resolution is not None:
            record.update(self.resolution.to_provenance())

        # The output contract (master spec section 29) asks for the selected
        # candidate at the top level of each record. With one clause that is
        # unambiguous; with several, the first successfully grounded selection
        # is lifted and the full per-clause detail stays in clause_selections.
        chosen = next(
            (s for s in self.clause_selections if s.get("disambiguation_status") == "SELECTED"),
            None,
        )
        for key in (
            "selected_candidate_name",
            "selected_linz_id",
            "selected_feature_type",
            "selected_region",
            "selected_source",
            "disambiguation_score",
            "score_margin",
            "disambiguation_status",
        ):
            record[key] = chosen.get(key) if chosen else None

        # The prediction's own status is authoritative; the resolution's stays
        # under its own key rather than overwriting it.
        record["status"] = self.status
        record["predicted_latitude"] = self.latitude
        record["predicted_longitude"] = self.longitude
        return record


class Grounder:
    def __init__(
        self,
        retriever,
        disambiguator: Disambiguator,
        pipeline_config: dict,
        name_spotter=None,
    ) -> None:
        self.retriever = retriever
        self.disambiguator = disambiguator
        # Used to find the OTHER places a description names, so a candidate can
        # be checked against them. Gazetteer only; never ground truth.
        self.name_spotter = name_spotter
        multi_anchor = pipeline_config["multi_anchor"]
        self.policy = multi_anchor["policy"]
        self.disagreement_threshold = float(multi_anchor["disagreement_threshold_metres"])
        self.minimum_weight = float(multi_anchor["minimum_weight_metres"])
        sanity = pipeline_config.get("coordinate_sanity", {})
        self.enforce_boxes = bool(sanity.get("enforce_bounding_boxes", False))
        self.boxes = sanity.get("boxes", {})

    def _context_points(self, description: str | None, anchor: str | None) -> list[tuple]:
        """Coordinates of other places the description names.

        The anchor's own matches are excluded: a candidate must not corroborate
        itself, or every candidate for an ambiguous name would look coherent.
        """
        if not self.name_spotter or not description:
            return []
        anchor_key = match_key(anchor or "")
        points = []
        self._last_context_names = []
        for record in self.name_spotter.spot_records(description):
            name = record.get("name") or ""
            if match_key(name) == anchor_key:
                continue
            try:
                points.append((float(record["latitude"]), float(record["longitude"])))
            except (KeyError, TypeError, ValueError):
                continue
            self._last_context_names.append(name)
        return points

    def _ground_clause(
        self, clause: dict, context: str | None, description: str | None = None
    ) -> tuple[GroundedClause | None, Selection]:
        anchor = clause.get("anchor_entity")

        # A generic anchor often refers back to a feature the description names
        # outright: "Lindis Pass ... 200m west of Pass". Resolving that uses the
        # description text only -- no gazetteer, no geography -- and the
        # resolved anchor still has to survive retrieval and disambiguation.
        resolved = resolve_back_reference(anchor, description)
        if resolved:
            anchor = resolved

        variants = anchor_variants(anchor)

        # Each variant is a separate query; results are concatenated in variant
        # order and de-duplicated, so the literal form's candidates rank ahead
        # of a stripped form's.
        candidates: list = []
        seen: set[str] = set()
        for variant in variants:
            for candidate in self.retriever.retrieve(variant):
                identifier = str(candidate.linz_id)
                if identifier not in seen:
                    seen.add(identifier)
                    candidates.append(candidate)

        selection = self.disambiguator.select(
            anchor, candidates, variants, context, self._context_points(description, anchor)
        )
        if not selection.is_selected:
            return None, selection

        record = selection.selected.record
        return (
            GroundedClause(
                anchor_name=selection.selected.name,
                anchor_latitude=float(record["latitude"]),
                anchor_longitude=float(record["longitude"]),
                bearing_degrees=float(clause["bearing_degrees"]),
                distance_metres=float(clause["distance_metres"]),
                anchor_source=record.get("source", "LINZ"),
                anchor_linz_id=str(record.get("linz_id")),
            ),
            selection,
        )

    def predict(self, interpretation: dict) -> Prediction:
        benchmark_id = interpretation.get("benchmark_id", "")

        if interpretation.get("status") != "SUCCESS":
            return Prediction(
                benchmark_id=benchmark_id,
                status=PredictionStatus.NO_INTERPRETATION,
                failure_reason=f"interpretation status {interpretation.get('status')}",
            )

        usable = [
            clause
            for clause in interpretation.get("clauses", [])
            if clause.get("bearing_degrees") is not None
            and clause.get("distance_metres") is not None
            and clause.get("anchor_entity")
        ]
        if not usable:
            return Prediction(
                benchmark_id=benchmark_id,
                status=PredictionStatus.NO_INTERPRETATION,
                failure_reason="no clause carried a usable bearing and distance",
            )

        context = interpretation.get("geographic_context")
        grounded: list[GroundedClause] = []
        selections: list[dict] = []
        description = interpretation.get("input_description")
        for clause in usable:
            clause_grounded, selection = self._ground_clause(clause, context, description)
            record = selection.to_provenance()
            record["anchor_entity"] = clause.get("anchor_entity")
            resolved = resolve_back_reference(clause.get("anchor_entity"), description)
            record["resolved_anchor_entity"] = resolved
            # Which places corroborated the choice. Without this, a coherence
            # failure can only be guessed at from the outcome.
            record["context_names"] = list(getattr(self, "_last_context_names", []))
            selections.append(record)
            if clause_grounded is not None:
                grounded.append(clause_grounded)

        if not grounded:
            reasons = "; ".join(
                sorted({s["disambiguation_failure_reason"] for s in selections if s.get("disambiguation_failure_reason")})
            )
            status = (
                PredictionStatus.RETRIEVAL_FAILURE
                if all(s["disambiguation_status"] == "NO_CANDIDATES" for s in selections)
                else PredictionStatus.GROUNDING_FAILURE
            )
            return Prediction(
                benchmark_id=benchmark_id,
                status=status,
                failure_reason=reasons or "no anchor could be grounded",
                clause_selections=selections,
            )

        resolution = resolve_clauses(
            grounded,
            policy=self.policy,
            disagreement_threshold_metres=self.disagreement_threshold,
            minimum_weight_metres=self.minimum_weight,
        )

        if resolution.status == ResolutionStatus.CLAUSE_DISAGREEMENT:
            return Prediction(
                benchmark_id=benchmark_id,
                status=PredictionStatus.CLAUSE_DISAGREEMENT,
                failure_reason=resolution.failure_reason,
                clause_selections=selections,
                resolution=resolution,
            )

        if not resolution.is_success:
            return Prediction(
                benchmark_id=benchmark_id,
                status=PredictionStatus.GROUNDING_FAILURE,
                failure_reason=resolution.failure_reason,
                clause_selections=selections,
                resolution=resolution,
            )

        # Coordinate sanity (master spec section 47). A prediction outside New
        # Zealand is a failure with a reason, never a plotted point.
        if self.enforce_boxes and self.boxes:
            box = containing_box(resolution.latitude, resolution.longitude, self.boxes)
            if box is None:
                return Prediction(
                    benchmark_id=benchmark_id,
                    status=PredictionStatus.VALIDATION_FAILURE,
                    failure_reason=(
                        f"predicted coordinate ({resolution.latitude:.5f}, "
                        f"{resolution.longitude:.5f}) falls outside every configured "
                        "New Zealand bounding box"
                    ),
                    clause_selections=selections,
                    resolution=resolution,
                )

        return Prediction(
            benchmark_id=benchmark_id,
            status=PredictionStatus.SUCCESS,
            latitude=resolution.latitude,
            longitude=resolution.longitude,
            clause_selections=selections,
            resolution=resolution,
        )
