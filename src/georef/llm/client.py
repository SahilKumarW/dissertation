"""LLM access through the official OpenAI SDK, pointed at an OpenAI-compatible
endpoint (NVIDIA NIM).

One client serves BOTH experimental conditions. The conditions differ in the
prompt and in what surrounds the call, never in the model or its decoding
parameters; a difference there would confound the experiment beyond repair.

Every call returns a `LlmResponse` carrying the provenance the output contract
requires (master spec section 29): the requested model, the model the API
actually reports serving, the raw text, token usage, attempt count and elapsed
time. The requested and served identifiers are kept separate on purpose — a
hosted provider can change served weights without notice, and a silent switch
must be visible in the results rather than invisible in them.

The API key is read from the environment and never logged, echoed or stored.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["LlmError", "LlmAuthError", "LlmResponse", "LlmClient"]


class LlmError(RuntimeError):
    """A completion failed after the configured retries."""


class LlmAuthError(LlmError):
    """Credentials were rejected or absent. Not retried."""


@dataclass
class LlmResponse:
    text: str
    requested_model: str
    served_model: str | None
    finish_reason: str | None
    attempt_count: int
    processing_time_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    response_id: str | None = None
    provider: str | None = None
    profile: str | None = None
    parameters: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def model_switched(self) -> bool:
        """True when the endpoint served something other than what was asked for."""
        return bool(self.served_model) and self.served_model != self.requested_model

    def to_provenance(self) -> dict:
        return asdict(self)


class LlmClient:
    """Thin, frozen-parameter wrapper around the OpenAI chat completions API.

    ``client_factory`` is injectable so retry behaviour, parameter freezing and
    provenance capture can be unit tested without a network or a key.
    """

    def __init__(
        self,
        models_config: dict,
        apis_config: dict,
        api_key: str | None = None,
        client: Any | None = None,
        sleep=time.sleep,
    ) -> None:
        self.llm_config = models_config["llm"]
        provider = self.llm_config["provider"]
        self.api_config = apis_config[provider]
        self.retry_policy = self.api_config["retry_policy"]
        self._sleep = sleep

        key = api_key if api_key is not None else os.environ.get(self.api_config["api_key_env"], "")
        if not key and client is None:
            raise LlmAuthError(
                f"{self.api_config['api_key_env']} is not set. Copy .env.example to "
                ".env and add your key."
            )

        if client is not None:
            self._client = client
        else:  # pragma: no cover -- requires the real SDK and a key
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.api_config["base_url"],
                api_key=key,
                timeout=self.api_config["timeout_seconds"],
                max_retries=0,  # retries are handled here so attempts are counted
            )

    # -- frozen parameters ---------------------------------------------------

    def request_parameters(self) -> dict:
        """Exactly the decoding parameters sent with every call, for provenance."""
        parameters = {
            "model": self.llm_config["model"],
            "temperature": self.llm_config["temperature"],
            "top_p": self.llm_config["top_p"],
            "max_tokens": self.llm_config["max_tokens"],
        }
        if self.llm_config.get("seed") is not None:
            parameters["seed"] = self.llm_config["seed"]
        return parameters

    # -- retry classification ------------------------------------------------

    def _status_of(self, error: Exception) -> int | None:
        return getattr(error, "status_code", None) or getattr(error, "status", None)

    def _is_retryable(self, error: Exception) -> bool:
        status = self._status_of(error)
        if status is None:
            return True  # transport-level failure
        return status in set(self.retry_policy.get("retry_on_status", []))

    # -- completion ----------------------------------------------------------

    # -- structured output ---------------------------------------------------
    #
    # Tool calling is used here as a SERIALISATION CONTRACT, not as agency. The
    # model is given exactly one tool, shaped like the target schema, and is
    # forced to call it, so the response arrives as validated arguments rather
    # than prose that has to be scraped for JSON.
    #
    # No geographic or computational tool is ever exposed. The model does not
    # get to search the gazetteer, ground a name, or compute a destination:
    # those are the deterministic components whose effect the experiment exists
    # to measure, and routing them through the model's own choices would make
    # that effect unattributable and non-reproducible (master spec sections 0,
    # 7, 20 and 26).

    def complete_structured(
        self, messages: list[dict], tool_name: str, parameters_schema: dict
    ) -> tuple[dict | None, LlmResponse]:
        """Force a single schema-shaped tool call and return its arguments.

        Returns ``(arguments, response)``. ``arguments`` is ``None`` when the
        model returned no tool call or returned unparseable arguments; the
        caller treats that as an interpretation failure rather than guessing,
        and the raw response is preserved either way.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Record the structured interpretation of the description.",
                    "parameters": parameters_schema,
                },
            }
        ]
        response = self.complete(
            messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        if not response.tool_calls:
            return None, response
        raw_arguments = response.tool_calls[0].get("arguments")
        if isinstance(raw_arguments, dict):
            return raw_arguments, response
        try:
            return json.loads(raw_arguments), response
        except (TypeError, ValueError):
            return None, response

    def _backoff_delay(self, attempt: int, backoff: float) -> float:
        """Exponential backoff plus jitter.

        Jitter matters once several workers run: throttled together, they would
        otherwise retry in lockstep and throttle each other again.
        """
        jitter = float(self.retry_policy.get("jitter_seconds", 0.0))
        return backoff**attempt + (random.uniform(0, jitter) if jitter else 0.0)

    def complete(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
    ) -> LlmResponse:
        parameters = self.request_parameters()
        payload: dict[str, Any] = {"messages": messages, **parameters}
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if self.llm_config.get("reasoning_effort"):
            payload["reasoning_effort"] = self.llm_config["reasoning_effort"]

        max_retries = int(self.retry_policy["max_retries"])
        backoff = float(self.retry_policy["backoff_factor"])
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                completion = self._client.chat.completions.create(**payload)
            except Exception as error:  # SDK raises typed errors; classify by status
                status = self._status_of(error)
                if status in (401, 403):
                    raise LlmAuthError(
                        f"The endpoint rejected the credentials (HTTP {status})."
                    ) from None
                if not self._is_retryable(error):
                    raise LlmError(
                        f"Request failed with HTTP {status} and was not retried: {error}"
                    ) from None
                last_error = error
                if attempt < max_retries:
                    self._sleep(self._backoff_delay(attempt, backoff))
                continue

            choice = completion.choices[0]
            usage = getattr(completion, "usage", None)
            tool_calls = []
            for call in getattr(choice.message, "tool_calls", None) or []:
                function = getattr(call, "function", None)
                tool_calls.append(
                    {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    }
                )
            return LlmResponse(
                text=choice.message.content or "",
                requested_model=parameters["model"],
                served_model=getattr(completion, "model", None),
                finish_reason=getattr(choice, "finish_reason", None),
                attempt_count=attempt + 1,
                processing_time_seconds=round(time.monotonic() - started, 4),
                prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
                response_id=getattr(completion, "id", None),
                provider=self.llm_config.get("provider"),
                profile=self.llm_config.get("profile"),
                tool_calls=tool_calls,
                parameters={
                    **parameters,
                    "reasoning_effort": self.llm_config.get("reasoning_effort"),
                },
            )

        raise LlmError(
            f"Request failed after {max_retries + 1} attempts: {last_error}"
        ) from None
