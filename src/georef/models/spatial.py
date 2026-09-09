"""Typed spatial interpretation (master spec section 13).

A description can state more than one distance-and-direction relation:

    "54 km ESE of New Plymouth, 4 km W of Whangamommona"

Both clauses describe the SAME point. Capturing only the first throws away a
second, independent constraint -- and in that example the discarded one is the
better of the two, because a 4 km offset compounds far less error than a 54 km
one. So the model reports every clause it finds and the prediction stage decides
what to do with them. Two clauses that disagree are also an internal consistency
check that needs no ground truth.

Three models, deliberately separate:

* ``SpatialClause`` is one relation exactly as the description states it, with
  no computed values.
* ``ModelInterpretation`` is everything the LLM reported: the clauses plus the
  description-level fields.
* ``SpatialInterpretation`` adds what Python derived. Bearings come from a
  lookup table and distances from a conversion table, so they are identical on
  every run and a wrong bearing is a table bug rather than an invisible model
  error.

Any bearing the model volunteers is captured and never used. It is retained
only as a cross-check worth reporting in the error analysis.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from georef.spatial.units import (
    UnknownDirection,
    UnknownUnit,
    bearing_for,
    distance_in_metres,
)

__all__ = [
    "RELATION_TYPES",
    "ExtractionStatus",
    "SpatialClause",
    "DerivedClause",
    "ModelInterpretation",
    "SpatialInterpretation",
    "derive",
    "TOOL_NAME",
    "tool_parameters_schema",
]

RELATION_TYPES = (
    "DISTANCE_DIRECTION",
    "MULTI_ANCHOR",
    "EXACT_LOCATION",
    "RELATIVE_LOCATION",
    "TOPOLOGICAL",
    "AMBIGUOUS",
    "UNRESOLVED",
)


class ExtractionStatus:
    """Outcomes of the interpretation stage (master spec section 28)."""

    SUCCESS = "SUCCESS"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_NO_TOOL_CALL = "MODEL_NO_TOOL_CALL"
    MODEL_OUTPUT_TRUNCATED = "MODEL_OUTPUT_TRUNCATED"
    SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
    NO_USABLE_CLAUSE = "NO_USABLE_CLAUSE"
    API_FAILURE = "API_FAILURE"


# A number optionally preceded by an approximation hedge: "ca. 3", "c.70",
# "approx 12", "~2.5". Recovering the 3 from "ca. 3" is unambiguous parsing of
# what the model returned, not a guess about what it meant, so it is done here
# rather than discarding the record. The hedge itself is not thrown away: it
# sets distance_is_approximate.
_HEDGE = r"(?:ca|c|approx|approximately|about|around|~)?\.?\s*"
_NUM = r"-?\d+(?:[.,]\d+)?"

_HEDGED_NUMBER = re.compile(rf"^\s*{_HEDGE}(?P<number>{_NUM})\s*$", re.IGNORECASE)

# "1/2 mile" is half a mile. Unambiguous, and common in older labels.
_FRACTION = re.compile(
    rf"^\s*{_HEDGE}(?:(?P<whole>\d+)\s+)?(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*$", re.IGNORECASE
)

# "10-12 km", "1-1.5 km", with a hyphen, en dash or "to".
_RANGE = re.compile(
    rf"^\s*{_HEDGE}(?P<low>{_NUM})\s*(?:-|\u2013|\u2014|to)\s*(?P<high>{_NUM})\s*$",
    re.IGNORECASE,
)


class SpatialClause(BaseModel):
    """One distance-and-direction relation, exactly as written."""

    anchor_entity: str | None = None
    distance_value: float | None = None
    distance_unit: str | None = None
    direction_text: str | None = Field(
        default=None, description="The direction as written, e.g. 'SW', 'south-west', 'ESE'."
    )
    anchor_is_generic: bool = Field(
        default=False,
        description="True when the anchor is an unnamed feature such as 'the bridge'.",
    )
    distance_is_approximate: bool = Field(
        default=False,
        description="True when the description hedges the distance, e.g. 'ca. 8 km'.",
    )

    @field_validator("distance_value", mode="before")
    @classmethod
    def _parse_hedged_number(cls, value):
        """Accept 'ca. 3' and '~2.5' as numbers.

        A model told to copy a distance 'as written' will sometimes copy the
        hedge with it. That is a formatting quirk, not a missing value, and
        losing a whole record to it would be a worse error than parsing it.
        Anything that is not a bare hedged number is left alone and fails
        validation as before.
        """
        if not isinstance(value, str):
            return value

        match = _HEDGED_NUMBER.match(value)
        if match:
            return float(match.group("number").replace(",", "."))

        match = _FRACTION.match(value)
        if match:
            whole = float(match.group("whole") or 0)
            denominator = float(match.group("den"))
            if denominator:
                return whole + float(match.group("num")) / denominator

        # A range states genuine uncertainty about the distance. The midpoint
        # is taken, which minimises expected error if the true value is
        # anywhere in the interval. This is a fixed pre-benchmark policy, not a
        # tuned choice, and the alternative -- discarding the record -- would
        # lose information the description does contain.
        match = _RANGE.match(value)
        if match:
            low = float(match.group("low").replace(",", "."))
            high = float(match.group("high").replace(",", "."))
            return (low + high) / 2.0

        return value

    @field_validator("anchor_entity", "direction_text", "distance_unit")
    @classmethod
    def _blank_to_none(cls, value):
        if value is None:
            return None
        return str(value).strip() or None


class DerivedClause(BaseModel):
    """A clause plus the deterministically computed bearing and distance."""

    clause: SpatialClause
    bearing_degrees: float | None = None
    distance_metres: float | None = None
    failure_reason: str | None = None

    @property
    def is_usable(self) -> bool:
        return (
            self.bearing_degrees is not None
            and self.distance_metres is not None
            and bool(self.clause.anchor_entity)
        )


class ModelInterpretation(BaseModel):
    """Exactly what the model reported. No derived values."""

    relation_type: Literal[RELATION_TYPES]
    clauses: list[SpatialClause] = Field(
        default_factory=list,
        description="Every distance-and-direction relation stated, in the order written.",
    )
    geographic_context: str | None = Field(
        default=None,
        description="Region, district, county or island only. Not roads, reserves or landmarks.",
    )
    ambiguity_detected: bool = False
    confidence: float | None = None
    bearing_degrees: float | None = Field(
        default=None, description="Do not populate. Captured only if volunteered."
    )

    @field_validator("geographic_context")
    @classmethod
    def _blank_to_none(cls, value):
        if value is None:
            return None
        return str(value).strip() or None

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_in_range(cls, value):
        """Clamp to [0, 1]; drop non-numeric self-reports.

        The model sometimes answers "high". There is no defensible number to
        map that to, and inventing 0.9 would be fabrication, so it becomes
        null. The field is unused by the pipeline and exists for reporting
        only, which is why dropping it costs nothing.
        """
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, numeric))


class SpatialInterpretation(BaseModel):
    """Model report plus deterministically derived quantities."""

    status: str
    model_interpretation: ModelInterpretation | None = None
    derived_clauses: list[DerivedClause] = Field(default_factory=list)
    failure_reason: str | None = None

    @property
    def usable_clauses(self) -> list[DerivedClause]:
        return [clause for clause in self.derived_clauses if clause.is_usable]

    @property
    def is_usable(self) -> bool:
        return self.status == ExtractionStatus.SUCCESS and bool(self.usable_clauses)

    @property
    def bearing_disagreement(self) -> float | None:
        """Gap between any bearing the model volunteered and the derived one."""
        if not self.derived_clauses or self.model_interpretation is None:
            return None
        reported = self.model_interpretation.bearing_degrees
        derived = self.derived_clauses[0].bearing_degrees
        if reported is None or derived is None:
            return None
        gap = abs(float(reported) - derived) % 360.0
        return min(gap, 360.0 - gap)

    def to_provenance(self) -> dict:
        model = self.model_interpretation
        usable = self.usable_clauses
        first = usable[0] if usable else None
        return {
            "status": self.status,
            "relation_type": model.relation_type if model else None,
            "clause_count": len(self.derived_clauses),
            "usable_clause_count": len(usable),
            "clauses": [
                {
                    "anchor_entity": d.clause.anchor_entity,
                    "anchor_is_generic": d.clause.anchor_is_generic,
                    "distance_value": d.clause.distance_value,
                    "distance_unit": d.clause.distance_unit,
                    "direction_text": d.clause.direction_text,
                    "bearing_degrees": d.bearing_degrees,
                    "distance_metres": d.distance_metres,
                    "failure_reason": d.failure_reason,
                }
                for d in self.derived_clauses
            ],
            # Flattened first usable clause, so single-clause records stay easy
            # to read in a table without unpacking the list.
            "anchor_entity": first.clause.anchor_entity if first else None,
            "distance_value": first.clause.distance_value if first else None,
            "distance_unit": first.clause.distance_unit if first else None,
            "direction_text": first.clause.direction_text if first else None,
            "bearing_degrees": first.bearing_degrees if first else None,
            "distance_metres": first.distance_metres if first else None,
            "geographic_context": model.geographic_context if model else None,
            "ambiguity_detected": model.ambiguity_detected if model else None,
            "model_confidence": model.confidence if model else None,
            "model_reported_bearing": model.bearing_degrees if model else None,
            "bearing_disagreement_degrees": self.bearing_disagreement,
            "failure_reason": self.failure_reason,
        }


def _derive_clause(clause: SpatialClause) -> DerivedClause:
    if not clause.anchor_entity:
        return DerivedClause(clause=clause, failure_reason="no anchor entity")
    try:
        bearing = bearing_for(clause.direction_text)
    except UnknownDirection as error:
        return DerivedClause(clause=clause, failure_reason=str(error))
    try:
        metres = distance_in_metres(clause.distance_value, clause.distance_unit)
    except UnknownUnit as error:
        return DerivedClause(clause=clause, bearing_degrees=bearing, failure_reason=str(error))
    return DerivedClause(clause=clause, bearing_degrees=bearing, distance_metres=metres)


def derive(model_interpretation: ModelInterpretation) -> SpatialInterpretation:
    """Convert every clause, and decide whether anything usable came out.

    A clause that cannot be converted is kept with its reason rather than
    dropped: a description with one good clause and one unusable one is a
    different case from one with a single good clause, and the error analysis
    needs to tell them apart.
    """
    derived = [_derive_clause(clause) for clause in model_interpretation.clauses]

    if model_interpretation.relation_type == "AMBIGUOUS":
        return SpatialInterpretation(
            status=ExtractionStatus.AMBIGUOUS,
            model_interpretation=model_interpretation,
            derived_clauses=derived,
            failure_reason="model reported the description as ambiguous",
        )

    if not derived:
        return SpatialInterpretation(
            status=ExtractionStatus.UNRESOLVED,
            model_interpretation=model_interpretation,
            failure_reason=f"no clauses; relation_type={model_interpretation.relation_type}",
        )

    if not any(clause.is_usable for clause in derived):
        reasons = "; ".join(
            sorted({c.failure_reason for c in derived if c.failure_reason})
        )
        return SpatialInterpretation(
            status=ExtractionStatus.NO_USABLE_CLAUSE,
            model_interpretation=model_interpretation,
            derived_clauses=derived,
            failure_reason=reasons or "no usable clause",
        )

    return SpatialInterpretation(
        status=ExtractionStatus.SUCCESS,
        model_interpretation=model_interpretation,
        derived_clauses=derived,
    )


TOOL_NAME = "record_spatial_interpretation"


def _inline_refs(node, definitions: dict):
    """Replace every $ref with the definition it points at.

    Pydantic emits nested models as $defs plus $ref. Tool-parameter handling of
    $ref varies between OpenAI-compatible endpoints, and a schema the endpoint
    silently ignores is worse than one it rejects, so the schema is flattened
    before it is sent.
    """
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            resolved = dict(definitions.get(name, {}))
            extras = {k: v for k, v in node.items() if k != "$ref"}
            resolved.update(extras)
            return _inline_refs(resolved, definitions)
        return {key: _inline_refs(value, definitions) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, definitions) for item in node]
    return node


def tool_parameters_schema() -> dict:
    """JSON schema for the single forced tool call.

    Generated from the Pydantic model so the schema the endpoint enforces and
    the validator that checks the reply cannot drift apart.
    """
    schema = ModelInterpretation.model_json_schema()
    definitions = schema.pop("$defs", {})
    schema = _inline_refs(schema, definitions)
    schema.pop("title", None)
    return schema
