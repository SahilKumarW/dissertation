"""UNIT TEST -- Condition A baseline. Fake client; no network."""

from __future__ import annotations

import pytest

from georef.llm.client import LlmError, LlmResponse
from georef.llm.baseline import BaselineGeoreferencer, BaselineStatus


class FakeClient:
    def __init__(self, arguments=None, finish_reason="stop", raises=None, text=""):
        self.arguments, self.finish_reason = arguments, finish_reason
        self.raises, self.text, self.calls = raises, text, []

    def complete_structured(self, messages, tool_name, schema):
        self.calls.append((messages, tool_name, schema))
        if self.raises:
            raise self.raises
        response = LlmResponse(
            text=self.text, requested_model="m", served_model="m",
            finish_reason=self.finish_reason, attempt_count=1,
            processing_time_seconds=0.5, response_id="r",
            provider="nvidia", profile="nvidia_gpt_oss",
            tool_calls=[{"name": tool_name, "arguments": self.arguments}],
        )
        return self.arguments, response


VALID = {"latitude": -39.6, "longitude": 175.5, "confidence": 0.6}


class TestPrediction:
    def test_returns_the_models_coordinates_unaltered(self):
        result = BaselineGeoreferencer(FakeClient(VALID)).predict("B1", "19 km SW of Waiouru")
        assert result.status == BaselineStatus.SUCCESS
        assert (result.latitude, result.longitude) == (-39.6, 175.5)

    def test_the_model_gets_only_the_description(self):
        """No candidates, no gazetteer, no coordinates: that is the condition."""
        client = FakeClient(VALID)
        BaselineGeoreferencer(client).predict("B1", "19 km SW of Waiouru")
        messages, _, _ = client.calls[0]
        assert messages[-1]["content"] == "19 km SW of Waiouru"
        assert len(messages) == 2

    def test_a_wildly_wrong_coordinate_is_kept_not_corrected(self):
        """Condition A has no sanity check by design. Correcting it would be
        giving the baseline help the experiment says it does not get."""
        result = BaselineGeoreferencer(
            FakeClient({"latitude": 51.5, "longitude": -0.12})
        ).predict("B1", "somewhere")
        assert result.status == BaselineStatus.SUCCESS
        assert result.latitude == 51.5

    def test_provenance_matches_the_output_contract(self):
        record = BaselineGeoreferencer(FakeClient(VALID)).predict("B1", "x").to_record()
        for field in ("benchmark_id", "condition", "status", "predicted_latitude",
                      "predicted_longitude", "model", "served_model", "llm_profile",
                      "attempt_count", "processing_time", "raw_model_response",
                      "structured_model_response", "prompt_version"):
            assert field in record, field
        assert record["condition"] == "A"


class TestFailures:
    def test_truncation_is_named(self):
        result = BaselineGeoreferencer(FakeClient(VALID, finish_reason="length")).predict("B1", "x")
        assert result.status == BaselineStatus.MODEL_OUTPUT_TRUNCATED

    def test_absent_tool_call_is_named(self):
        result = BaselineGeoreferencer(FakeClient(None, text="not sure")).predict("B1", "x")
        assert result.status == BaselineStatus.MODEL_NO_TOOL_CALL
        assert result.raw_model_response == "not sure"

    def test_a_refusal_to_estimate_is_an_abstention_not_a_crash(self):
        result = BaselineGeoreferencer(
            FakeClient({"latitude": None, "longitude": None})
        ).predict("B1", "x")
        assert result.status == BaselineStatus.NO_COORDINATE_RETURNED

    def test_out_of_range_values_are_a_schema_failure(self):
        result = BaselineGeoreferencer(
            FakeClient({"latitude": 200.0, "longitude": 0.0})
        ).predict("B1", "x")
        assert result.status == BaselineStatus.SCHEMA_VALIDATION_FAILURE
        assert "latitude" in result.failure_reason

    def test_api_failure_is_named(self):
        result = BaselineGeoreferencer(FakeClient(raises=LlmError("down"))).predict("B1", "x")
        assert result.status == BaselineStatus.API_FAILURE

    def test_empty_description_never_reaches_the_model(self):
        client = FakeClient(VALID)
        result = BaselineGeoreferencer(client).predict("B1", "  ")
        assert result.status == BaselineStatus.INVALID_INPUT
        assert client.calls == []
