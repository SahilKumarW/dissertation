"""Condition A: the standalone LLM baseline (master spec sections 4 and 64).

    description -> LLM -> latitude, longitude

That is the whole pipeline. The model is asked for coordinates and its answer is
the prediction. Nothing corrects it, nothing grounds it, nothing recomputes it.

**What this module must NOT do**, and what the isolation tests enforce: no
retrieval, no gazetteer, no snapshot, no LINZ, no deterministic geodesy, no
Condition B output, no ground truth. It imports none of those packages, and a
test walks its imports to prove it. A baseline with a hidden geographic
fallback would flatter the hybrid and make the comparison meaningless.

The model is the SAME model Condition B uses, with the same decoding
parameters. The conditions differ in the scaffolding around the call and in
nothing else; that difference is the entire experiment.

The only validation applied is that the returned numbers are numbers in valid
coordinate range. That is parsing, not correction: a prediction in the Atlantic
is recorded as a prediction in the Atlantic and scored as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError, field_validator

from georef.llm.client import LlmClient, LlmError

__all__ = [
    "BASELINE_PROMPT",
    "BASELINE_PROMPT_VERSION",
    "BaselineStatus",
    "BaselineResult",
    "BaselineGeoreferencer",
]

BASELINE_PROMPT_VERSION = "baseline-v1"

BASELINE_PROMPT = """You are given a natural-language locality description from a New Zealand biological collection record.

Return your single best estimate of the coordinates the description refers to.

Call the record_coordinates tool exactly once, with:
- latitude: decimal degrees, negative in the southern hemisphere
- longitude: decimal degrees
- confidence: a number between 0 and 1, or null

New Zealand latitudes are roughly -34 to -47. Longitudes are roughly 166 to 179, except the Chatham Islands which are around -176.

Give your best estimate even when the description is vague. If the description is unusable, set latitude and longitude to null."""

TOOL_NAME = "record_coordinates"


class BaselineStatus:
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_NO_TOOL_CALL = "MODEL_NO_TOOL_CALL"
    MODEL_OUTPUT_TRUNCATED = "MODEL_OUTPUT_TRUNCATED"
    SCHEMA_VALIDATION_FAILURE = "SCHEMA_VALIDATION_FAILURE"
    NO_COORDINATE_RETURNED = "NO_COORDINATE_RETURNED"
    API_FAILURE = "API_FAILURE"


class BaselineCoordinates(BaseModel):
    latitude: float | None = None
    longitude: float | None = None
    confidence: float | None = None

    @field_validator("latitude")
    @classmethod
    def _valid_latitude(cls, value):
        if value is None:
            return None
        value = float(value)
        if not -90.0 <= value <= 90.0:
            raise ValueError("latitude out of range")
        return value

    @field_validator("longitude")
    @classmethod
    def _valid_longitude(cls, value):
        if value is None:
            return None
        value = float(value)
        if not -180.0 <= value <= 180.0:
            raise ValueError("longitude out of range")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _numeric_confidence(cls, value):
        if value is None:
            return None
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return None


def tool_parameters_schema() -> dict:
    schema = BaselineCoordinates.model_json_schema()
    schema.pop("title", None)
    return schema


@dataclass
class BaselineResult:
    benchmark_id: str
    input_description: str
    status: str
    latitude: float | None = None
    longitude: float | None = None
    model_confidence: float | None = None
    failure_reason: str | None = None
    raw_model_response: str | None = None
    structured_model_response: dict | None = None
    requested_model: str | None = None
    served_model: str | None = None
    provider: str | None = None
    profile: str | None = None
    attempt_count: int = 0
    processing_time_seconds: float = 0.0
    response_id: str | None = None
    prompt_version: str = BASELINE_PROMPT_VERSION
    condition: str = "A"
    extra: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "condition": self.condition,
            "status": self.status,
            "input_description": self.input_description,
            "predicted_latitude": self.latitude,
            "predicted_longitude": self.longitude,
            "model_confidence": self.model_confidence,
            "failure_reason": self.failure_reason,
            "prompt_version": self.prompt_version,
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


class BaselineGeoreferencer:
    def __init__(self, client: LlmClient, prompt: str = BASELINE_PROMPT):
        self.client = client
        self.prompt = prompt

    def predict(self, benchmark_id: str, description: str) -> BaselineResult:
        if not description or not description.strip():
            return BaselineResult(
                benchmark_id=benchmark_id,
                input_description=description,
                status=BaselineStatus.INVALID_INPUT,
                failure_reason="empty description",
            )

        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": description},
        ]

        try:
            arguments, response = self.client.complete_structured(
                messages, TOOL_NAME, tool_parameters_schema()
            )
        except LlmError as error:
            return BaselineResult(
                benchmark_id=benchmark_id,
                input_description=description,
                status=BaselineStatus.API_FAILURE,
                failure_reason=str(error),
            )

        result = BaselineResult(
            benchmark_id=benchmark_id,
            input_description=description,
            status=BaselineStatus.MODEL_NO_TOOL_CALL,
            raw_model_response=response.text or None,
            structured_model_response=arguments,
            requested_model=response.requested_model,
            served_model=response.served_model,
            provider=response.provider,
            profile=response.profile,
            attempt_count=response.attempt_count,
            processing_time_seconds=response.processing_time_seconds,
            response_id=response.response_id,
        )

        if response.finish_reason == "length":
            result.status = BaselineStatus.MODEL_OUTPUT_TRUNCATED
            result.failure_reason = "response hit the token limit"
            return result

        if arguments is None:
            result.failure_reason = "no parseable tool call in the response"
            return result

        try:
            coordinates = BaselineCoordinates.model_validate(arguments)
        except ValidationError as error:
            result.status = BaselineStatus.SCHEMA_VALIDATION_FAILURE
            result.failure_reason = "; ".join(
                f"{'.'.join(str(p) for p in issue['loc'])}: {issue['type']}"
                f" (got {issue.get('input')!r})"
                for issue in error.errors()[:5]
            )
            return result

        if coordinates.latitude is None or coordinates.longitude is None:
            result.status = BaselineStatus.NO_COORDINATE_RETURNED
            result.failure_reason = "model returned no coordinate"
            return result

        result.status = BaselineStatus.SUCCESS
        result.latitude = coordinates.latitude
        result.longitude = coordinates.longitude
        result.model_confidence = coordinates.confidence
        return result
