"""Condition B: the full hybrid pipeline (master spec section 65).

    description -> LLM interpretation -> retrieval -> disambiguation
                -> LINZ anchor -> deterministic WGS84 geodesy
                -> coordinate sanity -> prediction

The two halves already exist and are tested separately. This joins them and
produces one record carrying the provenance of both, so a prediction can be
traced from the description through every stage that shaped it.

The LLM appears once, at the start, reading the description. Everything after
it is deterministic. That division is the intervention the experiment measures,
so it is expressed here as structure rather than as a convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from georef.llm.extraction import SpatialExtractor
from georef.pipeline.grounding import Grounder, PredictionStatus

__all__ = ["ConditionBResult", "ConditionBPipeline"]


@dataclass
class ConditionBResult:
    benchmark_id: str
    input_description: str
    status: str
    latitude: float | None = None
    longitude: float | None = None
    failure_reason: str | None = None
    interpretation: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    condition: str = "B"

    def to_record(self) -> dict:
        record = {
            "benchmark_id": self.benchmark_id,
            "condition": self.condition,
            "status": self.status,
            "input_description": self.input_description,
            "predicted_latitude": self.latitude,
            "predicted_longitude": self.longitude,
            "failure_reason": self.failure_reason,
        }
        # Interpretation provenance first, then grounding, with the pipeline's
        # own verdict last so a stage-level status cannot overwrite it.
        record.update({f"interp_{k}": v for k, v in self.interpretation.items()
                       if k not in ("benchmark_id", "input_description")})
        record.update({k: v for k, v in self.grounding.items()
                       if k not in ("benchmark_id", "status", "predicted_latitude",
                                    "predicted_longitude", "failure_reason")})
        record["status"] = self.status
        record["predicted_latitude"] = self.latitude
        record["predicted_longitude"] = self.longitude
        record["failure_reason"] = self.failure_reason
        return record


class ConditionBPipeline:
    def __init__(self, extractor: SpatialExtractor, grounder: Grounder):
        self.extractor = extractor
        self.grounder = grounder

    def predict(self, benchmark_id: str, description: str) -> ConditionBResult:
        extraction = self.extractor.extract(benchmark_id, description)
        interpretation = extraction.to_record()

        # An interpretation failure is carried through as itself rather than
        # relabelled: "the model could not read this" and "no anchor could be
        # grounded" are different findings and the error analysis needs both.
        if extraction.interpretation.status != "SUCCESS":
            return ConditionBResult(
                benchmark_id=benchmark_id,
                input_description=description,
                status=PredictionStatus.NO_INTERPRETATION,
                failure_reason=(
                    f"{extraction.interpretation.status}: "
                    f"{extraction.interpretation.failure_reason}"
                ),
                interpretation=interpretation,
            )

        prediction = self.grounder.predict(interpretation)
        return ConditionBResult(
            benchmark_id=benchmark_id,
            input_description=description,
            status=prediction.status,
            latitude=prediction.latitude,
            longitude=prediction.longitude,
            failure_reason=prediction.failure_reason,
            interpretation=interpretation,
            grounding=prediction.to_record(),
        )
