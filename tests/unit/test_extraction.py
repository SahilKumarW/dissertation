"""UNIT TEST -- extraction stage. Fake LLM client; no network, no key.

Exercises the failure taxonomy, the derived-versus-reported split, and the
provenance record. NOT evidence that gpt-oss-20b interprets descriptions well:
that is measured on the dev set, separately.
"""

from __future__ import annotations

import pytest

from georef.llm.client import LlmError, LlmResponse
from georef.llm.extraction import SYSTEM_PROMPT, SpatialExtractor
from georef.models.spatial import (
    ExtractionStatus,
    ModelInterpretation,
    derive,
    tool_parameters_schema,
)


class FakeClient:
    """Returns a canned tool call, or raises, without touching a network."""

    def __init__(self, arguments=None, finish_reason="stop", raises=None, text=""):
        self.arguments = arguments
        self.finish_reason = finish_reason
        self.raises = raises
        self.text = text
        self.calls = []

    def complete_structured(self, messages, tool_name, schema):
        self.calls.append((messages, tool_name, schema))
        if self.raises:
            raise self.raises
        response = LlmResponse(
            text=self.text,
            requested_model="openai/gpt-oss-20b",
            served_model="openai/gpt-oss-20b",
            finish_reason=self.finish_reason,
            attempt_count=1,
            processing_time_seconds=1.2,
            response_id="cmpl-1",
            tool_calls=[{"name": tool_name, "arguments": self.arguments}],
        )
        return self.arguments, response


def clause(**overrides) -> dict:
    base = {
        "anchor_entity": "Waiouru",
        "distance_value": 19,
        "distance_unit": "km",
        "direction_text": "southwest",
        "anchor_is_generic": False,
    }
    base.update(overrides)
    return base


VALID = {
    "relation_type": "DISTANCE_DIRECTION",
    "clauses": [clause()],
    "geographic_context": None,
    "ambiguity_detected": False,
    "confidence": 0.9,
    "bearing_degrees": 225,
}

MULTI = {
    "relation_type": "MULTI_ANCHOR",
    "clauses": [
        clause(anchor_entity="New Plymouth", distance_value=54, direction_text="ESE"),
        clause(anchor_entity="Whangamommona", distance_value=4, direction_text="W"),
    ],
    "geographic_context": "Taranaki",
    "ambiguity_detected": True,
    "confidence": None,
    "bearing_degrees": None,
}


class TestDerivation:
    def test_bearing_and_metres_come_from_python(self):
        result = derive(ModelInterpretation.model_validate(VALID))
        assert result.status == ExtractionStatus.SUCCESS
        assert result.usable_clauses[0].bearing_degrees == 225.0
        assert result.usable_clauses[0].distance_metres == 19000.0
        assert result.is_usable

    def test_the_models_own_bearing_is_never_used(self):
        """Model says 200; the table says 225. The table wins, the gap is recorded."""
        result = derive(ModelInterpretation.model_validate({**VALID, "bearing_degrees": 200}))
        assert result.usable_clauses[0].bearing_degrees == 225.0
        assert result.bearing_disagreement == 25.0

    def test_no_disagreement_when_the_model_agrees(self):
        assert derive(ModelInterpretation.model_validate(VALID)).bearing_disagreement == 0.0

    def test_model_units_are_not_trusted_to_be_converted(self):
        """The model reports miles; Python converts, so 3.5 miles is not 3.5 km."""
        result = derive(
            ModelInterpretation.model_validate(
                {**VALID, "clauses": [clause(distance_value=3.5, distance_unit="miles")]}
            )
        )
        assert result.usable_clauses[0].distance_metres == pytest.approx(5632.704)

    def test_unmappable_direction_abstains(self):
        result = derive(
            ModelInterpretation.model_validate(
                {**VALID, "clauses": [clause(direction_text="upstream")]}
            )
        )
        assert result.status == ExtractionStatus.NO_USABLE_CLAUSE
        assert not result.is_usable
        assert "upstream" in result.failure_reason

    def test_unmappable_distance_abstains(self):
        result = derive(
            ModelInterpretation.model_validate(
                {**VALID, "clauses": [clause(distance_unit="leagues")]}
            )
        )
        assert result.status == ExtractionStatus.NO_USABLE_CLAUSE
        assert not result.is_usable

    def test_missing_anchor_abstains(self):
        result = derive(
            ModelInterpretation.model_validate({**VALID, "clauses": [clause(anchor_entity="  ")]})
        )
        assert result.status == ExtractionStatus.NO_USABLE_CLAUSE

    @pytest.mark.parametrize(
        "relation", ["TOPOLOGICAL", "EXACT_LOCATION", "RELATIVE_LOCATION", "UNRESOLVED"]
    )
    def test_relation_types_without_clauses_do_not_produce_a_prediction(self, relation):
        result = derive(
            ModelInterpretation.model_validate({**VALID, "relation_type": relation, "clauses": []})
        )
        assert result.status == ExtractionStatus.UNRESOLVED
        assert not result.is_usable

    def test_ambiguous_is_reported_as_ambiguous_not_unresolved(self):
        result = derive(ModelInterpretation.model_validate({**VALID, "relation_type": "AMBIGUOUS"}))
        assert result.status == ExtractionStatus.AMBIGUOUS
        assert not result.is_usable


class TestMultipleClauses:
    def test_both_clauses_are_derived(self):
        """Two clauses describe the same point and are two independent constraints."""
        result = derive(ModelInterpretation.model_validate(MULTI))
        assert result.status == ExtractionStatus.SUCCESS
        assert len(result.usable_clauses) == 2
        assert result.usable_clauses[0].distance_metres == 54000.0
        assert result.usable_clauses[1].distance_metres == 4000.0
        assert result.usable_clauses[1].bearing_degrees == 270.0

    def test_clause_order_is_preserved_as_written(self):
        result = derive(ModelInterpretation.model_validate(MULTI))
        anchors = [c.clause.anchor_entity for c in result.derived_clauses]
        assert anchors == ["New Plymouth", "Whangamommona"]

    def test_one_bad_clause_does_not_discard_the_good_one(self):
        payload = {
            **MULTI,
            "clauses": [clause(direction_text="upstream"), clause(anchor_entity="Akaroa")],
        }
        result = derive(ModelInterpretation.model_validate(payload))
        assert result.status == ExtractionStatus.SUCCESS
        assert len(result.usable_clauses) == 1
        assert result.usable_clauses[0].clause.anchor_entity == "Akaroa"

    def test_an_unusable_clause_is_kept_with_its_reason(self):
        """A record with one good and one bad clause differs from one with a
        single good clause; the error analysis must be able to tell them apart."""
        payload = {**MULTI, "clauses": [clause(direction_text="upstream"), clause()]}
        result = derive(ModelInterpretation.model_validate(payload))
        assert len(result.derived_clauses) == 2
        assert result.derived_clauses[0].failure_reason is not None

    def test_provenance_exposes_every_clause_and_flattens_the_first(self):
        provenance = derive(ModelInterpretation.model_validate(MULTI)).to_provenance()
        assert provenance["clause_count"] == 2
        assert provenance["usable_clause_count"] == 2
        assert provenance["anchor_entity"] == "New Plymouth"
        assert len(provenance["clauses"]) == 2

    def test_generic_anchor_flag_survives(self):
        payload = {**VALID, "clauses": [clause(anchor_entity="the bridge", anchor_is_generic=True)]}
        result = derive(ModelInterpretation.model_validate(payload))
        assert result.derived_clauses[0].clause.anchor_is_generic is True

    def test_confidence_is_clamped(self):
        assert ModelInterpretation.model_validate({**VALID, "confidence": 4.2}).confidence == 1.0


class TestExtractor:
    def test_happy_path(self):
        result = SpatialExtractor(FakeClient(VALID)).extract("NZDD-001", "19 km SW of Waiouru")
        assert result.interpretation.is_usable
        assert result.served_model == "openai/gpt-oss-20b"
        assert result.attempt_count == 1

    def test_the_model_sees_the_original_text_not_the_normalised_key(self):
        """Lowercasing would throw away capitalisation that marks proper nouns."""
        client = FakeClient(VALID)
        SpatialExtractor(client).extract("NZDD-001", "19 km SW of Waiouru")
        messages, _, _ = client.calls[0]
        assert messages[-1]["content"] == "19 km SW of Waiouru"

    def test_truncation_is_a_failure_not_a_partial_parse(self):
        """Reasoning tokens count against max_tokens, so this is a real risk."""
        result = SpatialExtractor(FakeClient(VALID, finish_reason="length")).extract("x", "19 km SW of Y")
        assert result.interpretation.status == ExtractionStatus.MODEL_OUTPUT_TRUNCATED
        assert not result.interpretation.is_usable

    def test_absent_tool_call_is_named(self):
        result = SpatialExtractor(FakeClient(None, text="I'm not sure")).extract("x", "somewhere")
        assert result.interpretation.status == ExtractionStatus.MODEL_NO_TOOL_CALL
        assert result.raw_model_response == "I'm not sure"

    def test_schema_violation_is_named(self):
        bad = {**VALID, "relation_type": "SOMETHING_INVENTED"}
        result = SpatialExtractor(FakeClient(bad)).extract("x", "19 km SW of Y")
        assert result.interpretation.status == ExtractionStatus.SCHEMA_VALIDATION_FAILURE

    def test_api_failure_is_named_and_does_not_raise(self):
        result = SpatialExtractor(FakeClient(raises=LlmError("gateway down"))).extract("x", "y")
        assert result.interpretation.status == ExtractionStatus.API_FAILURE
        assert "gateway down" in result.interpretation.failure_reason

    def test_empty_description_never_reaches_the_model(self):
        client = FakeClient(VALID)
        result = SpatialExtractor(client).extract("x", "   ")
        assert result.interpretation.status == ExtractionStatus.INVALID_INPUT
        assert client.calls == []

    def test_failures_still_preserve_the_raw_response(self):
        result = SpatialExtractor(FakeClient(None, text="waffle")).extract("x", "y")
        assert result.raw_model_response == "waffle"

    def test_record_carries_the_output_contract_fields(self):
        record = SpatialExtractor(FakeClient(VALID)).extract("NZDD-001", "19 km SW of Waiouru").to_record()
        for field in (
            "benchmark_id", "input_description", "normalised_description", "prompt_version",
            "model", "served_model", "attempt_count", "processing_time", "raw_model_response",
            "structured_model_response", "status", "relation_type", "anchor_entity",
            "distance_value", "distance_unit", "bearing_degrees", "distance_metres",
            "failure_reason", "clauses", "clause_count", "usable_clause_count",
        ):
            assert field in record, field


class TestToolSchema:
    def test_schema_is_generated_from_the_validator(self):
        """Hand-writing it separately would let the two drift apart."""
        schema = tool_parameters_schema()
        assert set(schema["properties"]) >= {
            "relation_type", "clauses", "geographic_context", "ambiguity_detected",
        }
        assert set(schema["properties"]["clauses"]["items"]["properties"]) >= {
            "anchor_entity", "distance_value", "distance_unit", "direction_text",
        }

    def test_schema_contains_no_unresolved_references(self):
        """Tool-parameter handling of $ref varies between endpoints, and a schema
        silently ignored is worse than one rejected."""
        import json

        rendered = json.dumps(tool_parameters_schema())
        assert "$ref" not in rendered and "$defs" not in rendered

    def test_relation_type_is_constrained_to_the_declared_set(self):
        schema = tool_parameters_schema()
        assert "SOMETHING_INVENTED" not in str(schema["properties"]["relation_type"])


class TestLenientNumberParsing:
    """The model copies distances 'as written' and sometimes brings the hedge
    with it. Losing a whole record to a formatting quirk is a worse error than
    parsing it, so unambiguous forms are recovered. Regression from a 100-record
    dev run where 29 records failed validation on exactly this."""

    def parse(self, raw):
        payload = {**VALID, "clauses": [clause(distance_value=raw)]}
        return ModelInterpretation.model_validate(payload).clauses[0].distance_value

    @pytest.mark.parametrize(
        "raw, expected",
        [("ca. 3", 3.0), ("c. 70", 70.0), ("ca.300", 300.0), ("approx 12", 12.0),
         ("about 4.5", 4.5), ("~2.5", 2.5), ("~2,5", 2.5), ("8", 8.0), ("1.5", 1.5)],
    )
    def test_hedged_numbers(self, raw, expected):
        assert self.parse(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw, expected", [("1/2", 0.5), ("3/4", 0.75), ("1 1/2", 1.5), ("ca. 1/2", 0.5)]
    )
    def test_fractions(self, raw, expected):
        """'1/2 mile' is common in older collection labels."""
        assert self.parse(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw, expected",
        [("10-12", 11.0), ("1\u20131.5", 1.25), ("2 to 4", 3.0), ("ca. 10-12", 11.0)],
    )
    def test_ranges_take_the_midpoint(self, raw, expected):
        """A range states real uncertainty; the midpoint minimises expected
        error if the true value lies anywhere in the interval."""
        assert self.parse(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["a few", "several km", "", "north", "1/0"])
    def test_genuinely_unparseable_values_still_fail(self, raw):
        """Leniency recovers formatting quirks, not missing information."""
        with pytest.raises(Exception):
            self.parse(raw)

    def test_numeric_input_is_untouched(self):
        assert self.parse(19) == 19.0

    def test_word_confidence_becomes_null_rather_than_a_made_up_number(self):
        """There is no defensible number for 'high'; inventing 0.9 is fabrication."""
        payload = {**VALID, "confidence": "high"}
        assert ModelInterpretation.model_validate(payload).confidence is None

    def test_numeric_confidence_is_still_clamped(self):
        assert ModelInterpretation.model_validate({**VALID, "confidence": 4.2}).confidence == 1.0


class TestFailureReasonsAreDiagnostic:
    def test_schema_failure_names_the_field_and_the_value(self):
        """A bare count makes a systematic prompt problem look like random noise."""
        bad = {**VALID, "clauses": [clause(distance_value="a few")]}
        result = SpatialExtractor(FakeClient(bad)).extract("x", "a few km N of Y")
        assert result.interpretation.status == ExtractionStatus.SCHEMA_VALIDATION_FAILURE
        reason = result.interpretation.failure_reason
        assert "distance_value" in reason and "a few" in reason


class TestPromptCoversTheKnownFailureModes:
    """The prompt is an experimental parameter, frozen before the benchmark. Each
    rule below was added because a dev run failed without it, so these assert the
    instruction is still present rather than that the model obeys it -- obedience
    is measured on the dev set, not in a unit test."""

    def test_back_reference_examples_are_present_and_worked(self):
        """v4 stated the rule and the model still ignored it; v5 walks each case."""
        for phrase in ("Lindis Pass", "the summit", "Mt. Maungatua", "Perpendicular Point",
                       "Mangahowhi Stream", "hut track"):
            assert phrase in SYSTEM_PROMPT, phrase

    def test_the_procedure_is_numbered(self):
        """A 20B model follows a procedure more reliably than a principle."""
        for step in ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"):
            assert step in SYSTEM_PROMPT, step

    def test_no_instruction_contradicts_the_back_reference_rule(self):
        """v4 failed because the anchor_entity field said 'exactly as written',
        which the model followed over the resolution rule."""
        assert "exactly as written, without surrounding words" not in SYSTEM_PROMPT

    def test_arithmetic_prohibitions_survive(self):
        assert "NEVER convert between units" in SYSTEM_PROMPT
        assert "NEVER degrees" in SYSTEM_PROMPT

    def test_hedged_and_fractional_distances_are_shown(self):
        for example in ("ca. 8 km", "1/2 mile", "10-12 km"):
            assert example in SYSTEM_PROMPT, example

    def test_non_clauses_are_shown_explicitly(self):
        assert "14 miles from Glade House" in SYSTEM_PROMPT
        assert "upstream" in SYSTEM_PROMPT

    def test_context_exclusions_are_explicit(self):
        assert "roads, highways, reserves" in SYSTEM_PROMPT

    def test_full_examples_show_both_single_and_multi_clause(self):
        assert "MULTI_ANCHOR" in SYSTEM_PROMPT
        assert "Whangamomona" in SYSTEM_PROMPT
