"""Semantic interpretation stage.

    description -> normalise -> LLM (forced tool call) -> Pydantic -> derive

This is the only place the LLM is consulted in Condition B, and it is consulted
for one thing: reading the description. It receives no geographic data, no
candidates, no coordinates and no tools beyond the schema-shaped one it is
forced to call.

Every failure mode is named rather than collapsed (master spec section 28), and
every failure preserves the raw response, so an interpretation error can be
diagnosed after the fact instead of inferred.

A note on PydanticAI: the specification names it for orchestration. It is not
used here, and the reason is recorded rather than glossed. This stage makes a
single non-agentic call whose raw response, exact attempt count and served
model identifier must all be preserved verbatim for the output contract
(sections 29 and 31). An agent framework owns the retry loop and normalises the
response, which would put those three values behind its abstraction. The
validation the specification actually asks for -- typed, Pydantic-checked
structured output -- is done, with the same models. The interface here is
narrow enough that swapping in PydanticAI later would touch one function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from georef.llm.client import LlmClient, LlmError
from georef.models.spatial import (
    TOOL_NAME,
    ExtractionStatus,
    ModelInterpretation,
    SpatialInterpretation,
    derive,
    tool_parameters_schema,
)
from georef.normalise import normalise

__all__ = ["ExtractionResult", "SpatialExtractor", "SYSTEM_PROMPT", "PROMPT_VERSION"]

PROMPT_VERSION = "extract-v5"

SYSTEM_PROMPT = """You extract spatial structure from New Zealand biological collection locality descriptions.

Call the record_spatial_interpretation tool exactly once. Report only what the description says. You do not know where anything is. You never estimate coordinates. You never calculate.

================================================================
FOLLOW THESE FIVE STEPS IN ORDER
================================================================

STEP 1. Read the WHOLE description first, start to finish.

STEP 2. List every place name the description contains, anywhere in it.
   Example: "Lindis Pass, Stream gully crossing Highway 8 about 200m west of Pass"
   Place names present: Lindis Pass, Highway 8

STEP 3. Find every phrase of the form <distance> <direction> of/from <place>.
   Each one is a CLAUSE. There may be none, one, or several.

STEP 4. For each clause, check whether the place it names is a BACK-REFERENCE.
   A back-reference is a bare or shortened way of pointing at a place already
   named earlier in the SAME description.
   If it is, replace it with the full name from your STEP 2 list.

STEP 5. Fill in the tool fields.

================================================================
STEP 4 IN DETAIL: BACK-REFERENCES. READ THIS TWICE.
================================================================

Collection labels name a place once, then refer back to it in short form.
The short form is NOT the anchor. The full name from STEP 2 is the anchor.

A place is a back-reference when it is:
  - a bare feature word:        "Pass", "Point", "Summit", "Trig", "Bridge"
  - "the" plus a feature word:  "the summit", "the point", "the bridge", "the hut"
  - "the" plus a feature word plus more words:
                                "the bridge across the stream"

WORKED EXAMPLES. Study each one.

  Description: "Lindis Pass, Stream gully crossing Highway 8 about 200m west of Pass"
  STEP 2 names: Lindis Pass, Highway 8
  The clause says "200m west of Pass".
  "Pass" is bare. "Lindis Pass" is in the list and ends with "Pass".
  anchor_entity = "Lindis Pass"          NOT "Pass"

  Description: "Mt. Maungatua south west of Dunedin, rock c. 500 m west of the summit"
  STEP 2 names: Mt. Maungatua, Dunedin
  The clause says "500 m west of the summit".
  "the summit" is a back-reference. A summit is a mountain. "Mt. Maungatua" is a mountain.
  anchor_entity = "Mt. Maungatua"        NOT "the summit"

  Description: "S.W. Nelson, Perpendicular Point by trig. 1/2 mile N of the point"
  STEP 2 names: Nelson, Perpendicular Point
  The clause says "1/2 mile N of the point".
  "the point" is a back-reference. "Perpendicular Point" ends with "Point".
  anchor_entity = "Perpendicular Point"  NOT "the point"

  Description: "bank of Mangahowhi Stream, between Mangahoe Road and stream. Approx 10 m NW from the bridge across the stream to Sutherlands Bush No. 2"
  STEP 2 names: Mangahowhi Stream, Mangahoe Road, Sutherlands Bush
  The clause says "10 m NW from the bridge across the stream".
  The bridge crosses a stream. The stream is named: Mangahowhi Stream.
  anchor_entity = "Mangahowhi Stream"    NOT "the bridge across the stream"

  Description: "Chatham Islands; South East Island (Rangatira); Northwest headland, 100m west of hut track"
  STEP 2 names: Chatham Islands, South East Island, Rangatira
  The clause says "100m west of hut track".
  No hut and no track is named anywhere. There is nothing to resolve to.
  anchor_entity = "hut track", anchor_is_generic = true

RULE: set anchor_is_generic = true ONLY when STEP 2 produced no matching place.
If a matching place name exists in the description, USE IT.
Never invent a name the description does not contain.

================================================================
CLAUSE FIELDS
================================================================

anchor_entity
  The place the distance is measured FROM, after applying STEP 4.
  Take the name only, not the surrounding words.
    "ca 40 km south-east of Auckland"          -> "Auckland"
  Drop bracketed alternative names.
    "Minaret Creek [Minaret Burn]"             -> "Minaret Creek"

distance_value
  The NUMBER ONLY. A number, not text. No words attached.
    "ca. 8 km"     -> 8
    "c. 70miles"   -> 70
    "1/2 mile"     -> 0.5
    "10-12 km"     -> 11
  NEVER convert between units. 3.5 miles stays 3.5.

distance_unit
  The unit as written: "km", "m", "miles", "mls".

distance_is_approximate
  true when the description hedges: "ca.", "c.", "about", "approximately", "~".

direction_text
  The direction EXACTLY as written. NEVER degrees. NEVER a number.
    "southwest" -> "southwest"      "S.E." -> "S.E."      "NNW" -> "NNW"

anchor_is_generic
  See the STEP 4 rule above.

================================================================
DESCRIPTION-LEVEL FIELDS
================================================================

relation_type
  DISTANCE_DIRECTION   exactly one clause
  MULTI_ANCHOR         two or more clauses
  RELATIVE_LOCATION    positioned relative to something, but not by both a
                       distance AND a direction
  EXACT_LOCATION       a named place, no offset
  TOPOLOGICAL          "in", "on", "along", "at the head of"
  AMBIGUOUS            you cannot tell which reading is meant
  UNRESOLVED           you cannot read it at all

geographic_context
  ONLY a wider administrative or physiographic area.
    YES: "Buller County", "Nelson", "Canterbury", "South Island", "Banks Peninsula", "Fiordland"
    NO:  roads, highways, reserves, parks, stations, streams, buildings, farms
  null if none.

ambiguity_detected
  true when two different places could be the anchor, or when a clause's anchor
  stayed generic after STEP 4.

confidence
  A number between 0 and 1, or null. NEVER a word like "high".

================================================================
WHAT IS NOT A CLAUSE
================================================================

  "14 miles from Glade House"      distance but no direction  -> no clause
  "east of the river"              direction but no distance  -> no clause
  "44.3 km upstream of Pipiriki"   upstream is not a compass direction -> no clause

================================================================
TWO FULL EXAMPLES
================================================================

Description: "North Island, Taranaki, c. 54 km ESE of New Plymouth, 4 km W of Whangamomona"
  relation_type: MULTI_ANCHOR
  clauses:
    1. anchor_entity "New Plymouth", distance_value 54, distance_unit "km",
       direction_text "ESE", distance_is_approximate true, anchor_is_generic false
    2. anchor_entity "Whangamomona", distance_value 4, distance_unit "km",
       direction_text "W", distance_is_approximate false, anchor_is_generic false
  geographic_context: "Taranaki"
  ambiguity_detected: false

Description: "Lindis Pass, Stream gully crossing Highway 8 about 200m west of Pass"
  relation_type: DISTANCE_DIRECTION
  clauses:
    1. anchor_entity "Lindis Pass", distance_value 200, distance_unit "m",
       direction_text "west", distance_is_approximate true, anchor_is_generic false
  geographic_context: null
  ambiguity_detected: false
"""


@dataclass
class ExtractionResult:
    """An interpretation plus the LLM provenance that produced it."""

    benchmark_id: str
    original_description: str
    normalised_description: str
    interpretation: SpatialInterpretation
    gazetteer_hints: list[str] = field(default_factory=list)
    raw_model_response: str | None = None
    structured_model_response: dict | None = None
    served_model: str | None = None
    requested_model: str | None = None
    provider: str | None = None
    profile: str | None = None
    attempt_count: int = 0
    processing_time_seconds: float = 0.0
    response_id: str | None = None
    prompt_version: str = PROMPT_VERSION

    def to_record(self) -> dict:
        record = {
            "benchmark_id": self.benchmark_id,
            "input_description": self.original_description,
            "normalised_description": self.normalised_description,
            "prompt_version": self.prompt_version,
            "gazetteer_hints": self.gazetteer_hints,
            "model": self.requested_model,
            "served_model": self.served_model,
            "provider": self.provider,
            "llm_profile": self.profile,
            "attempt_count": self.attempt_count,
            "processing_time": self.processing_time_seconds,
            "response_id": self.response_id,
            "raw_model_response": self.raw_model_response,
            "structured_model_response": self.structured_model_response,
        }
        record.update(self.interpretation.to_provenance())
        return record


class SpatialExtractor:
    def __init__(
        self,
        client: LlmClient,
        prompt: str = SYSTEM_PROMPT,
        name_spotter=None,
        max_hints: int = 12,
    ):
        self.client = client
        self.prompt = prompt
        # Optional pre-retrieval pass. Off unless a spotter is supplied, so the
        # two-pass variant is a measurable condition rather than a silent
        # change to the architecture.
        self.name_spotter = name_spotter
        self.max_hints = max_hints

    def _messages(self, description: str, hints: list[str] | None = None) -> list[dict]:
        content = description
        if hints:
            # Names only, never coordinates: the model still does not know where
            # anything is. This tells it which strings in front of it are real
            # gazetteer names, which is what STEP 2 of the prompt asks it to
            # work out for itself and what it gets wrong on misspellings and
            # back-references.
            listed = "\n".join(f"- {name}" for name in hints)
            content = (
                f"{description}\n\n"
                "New Zealand gazetteer names found in this description:\n"
                f"{listed}\n"
                "Use one of these as anchor_entity only if it is the place the "
                "distance is measured from. Ignore the rest."
            )
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": content},
        ]

    def extract(self, benchmark_id: str, description: str) -> ExtractionResult:
        original = description
        normalised = normalise(description)

        if not normalised:
            return ExtractionResult(
                benchmark_id=benchmark_id,
                original_description=original,
                normalised_description=normalised,
                interpretation=SpatialInterpretation(
                    status=ExtractionStatus.INVALID_INPUT,
                    failure_reason="empty description after normalisation",
                ),
            )

        # The model reads the ORIGINAL text. Normalisation exists for keying and
        # matching; feeding the model a lowercased, punctuation-stripped string
        # would discard capitalisation that helps it find proper nouns.
        hints = (
            self.name_spotter.spot(original, self.max_hints) if self.name_spotter else []
        )

        try:
            arguments, response = self.client.complete_structured(
                self._messages(original, hints), TOOL_NAME, tool_parameters_schema()
            )
        except LlmError as error:
            return ExtractionResult(
                benchmark_id=benchmark_id,
                original_description=original,
                normalised_description=normalised,
                interpretation=SpatialInterpretation(
                    status=ExtractionStatus.API_FAILURE, failure_reason=str(error)
                ),
                gazetteer_hints=hints,
            )

        result = ExtractionResult(
            benchmark_id=benchmark_id,
            original_description=original,
            normalised_description=normalised,
            interpretation=SpatialInterpretation(status=ExtractionStatus.UNRESOLVED),
            gazetteer_hints=hints,
            raw_model_response=response.text or None,
            structured_model_response=arguments,
            served_model=response.served_model,
            requested_model=response.requested_model,
            provider=response.provider,
            profile=response.profile,
            attempt_count=response.attempt_count,
            processing_time_seconds=response.processing_time_seconds,
            response_id=response.response_id,
        )

        # Truncation is checked BEFORE parsing. Reasoning tokens count against
        # max_tokens, so a long description can cut the tool call short; parsing
        # what survived would silently produce a partial interpretation.
        if response.finish_reason == "length":
            result.interpretation = SpatialInterpretation(
                status=ExtractionStatus.MODEL_OUTPUT_TRUNCATED,
                failure_reason="response hit the token limit before completing",
            )
            return result

        if arguments is None:
            result.interpretation = SpatialInterpretation(
                status=ExtractionStatus.MODEL_NO_TOOL_CALL,
                failure_reason="no parseable tool call in the response",
            )
            return result

        try:
            model_interpretation = ModelInterpretation.model_validate(arguments)
        except ValidationError as error:
            # Name the offending field and value. A bare count ("1 schema
            # violation") tells whoever reads the results nothing and makes a
            # systematic prompt problem look like random noise.
            details = "; ".join(
                f"{'.'.join(str(part) for part in issue['loc'])}: {issue['type']}"
                f" (got {issue.get('input')!r})"
                for issue in error.errors()[:5]
            )
            result.interpretation = SpatialInterpretation(
                status=ExtractionStatus.SCHEMA_VALIDATION_FAILURE,
                failure_reason=f"{error.error_count()} schema violation(s): {details}",
            )
            return result

        result.interpretation = derive(model_interpretation)
        return result
