"""UNIT TEST -- LLM client. No network, no key, no live endpoint.

A fake stands in for the OpenAI SDK client so parameter freezing, retry
classification and provenance capture are exercised deterministically. This is
NOT evidence that the NVIDIA endpoint works; that is a LIVE API SMOKE TEST,
reported separately.
"""

from __future__ import annotations

import pytest

from georef.config.loader import load_yaml, repo_root, resolve_llm_profile
from georef.llm.client import LlmAuthError, LlmClient, LlmError

RAW_MODELS = load_yaml(repo_root() / "configs" / "models.yaml")
MODELS = resolve_llm_profile(RAW_MODELS)
APIS = load_yaml(repo_root() / "configs" / "apis.yaml")


class FakeError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **payload):
        self.outer.calls.append(payload)
        if self.outer.fail_times > 0:
            self.outer.fail_times -= 1
            raise self.outer.fail_with
        return FakeCompletion(self.outer.served_model, self.outer.text)


class FakeChat:
    def __init__(self, outer):
        self.completions = FakeCompletions(outer)


class FakeSdk:
    def __init__(self, text='{"ok": true}', served_model="openai/gpt-oss-20b",
                 fail_times=0, fail_with=None):
        self.calls: list[dict] = []
        self.text = text
        self.served_model = served_model
        self.fail_times = fail_times
        self.fail_with = fail_with or FakeError(503)
        self.chat = FakeChat(self)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)
        self.finish_reason = "stop"


class FakeUsage:
    prompt_tokens = 120
    completion_tokens = 45


class FakeCompletion:
    def __init__(self, model, text):
        self.id = "cmpl-abc123"
        self.model = model
        self.choices = [FakeChoice(text)]
        self.usage = FakeUsage()


def client_with(sdk: FakeSdk) -> LlmClient:
    return LlmClient(MODELS, APIS, api_key="k", client=sdk, sleep=lambda _: None)


MESSAGES = [{"role": "user", "content": "19 km southwest of Waiouru"}]


class TestFrozenParameters:
    def test_sends_the_configured_decoding_parameters(self):
        sdk = FakeSdk()
        client_with(sdk).complete(MESSAGES)
        payload = sdk.calls[0]
        assert payload["model"] == "openai/gpt-oss-20b"
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["seed"] == 20260908
        assert payload["reasoning_effort"] == "medium"

    def test_both_conditions_share_one_model(self):
        """Differing models across conditions would confound the experiment."""
        assert MODELS["llm"]["model"] == "openai/gpt-oss-20b"
        parameters = client_with(FakeSdk()).request_parameters()
        assert parameters["temperature"] == 0.0

    def test_response_format_is_passed_through_when_requested(self):
        sdk = FakeSdk()
        schema = {"type": "json_object"}
        client_with(sdk).complete(MESSAGES, response_format=schema)
        assert sdk.calls[0]["response_format"] == schema

    def test_missing_key_fails_before_any_request(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(LlmAuthError, match="NVIDIA_API_KEY"):
            LlmClient(MODELS, APIS, api_key=None)


class TestProvenance:
    def test_captures_what_the_output_contract_requires(self):
        response = client_with(FakeSdk()).complete(MESSAGES)
        assert response.text == '{"ok": true}'
        assert response.requested_model == "openai/gpt-oss-20b"
        assert response.served_model == "openai/gpt-oss-20b"
        assert response.attempt_count == 1
        assert response.prompt_tokens == 120
        assert response.response_id == "cmpl-abc123"
        assert response.processing_time_seconds >= 0

    def test_detects_a_silently_switched_model(self):
        """A hosted provider can change served weights without notice."""
        response = client_with(FakeSdk(served_model="openai/gpt-oss-120b")).complete(MESSAGES)
        assert response.model_switched is True
        assert response.requested_model != response.served_model

    def test_matching_models_are_not_flagged(self):
        assert client_with(FakeSdk()).complete(MESSAGES).model_switched is False

    def test_attempt_count_reflects_real_attempts(self):
        """Master spec section 31: do not claim retries that did not happen."""
        response = client_with(FakeSdk(fail_times=2)).complete(MESSAGES)
        assert response.attempt_count == 3


class TestRetries:
    def test_rate_limiting_is_retried(self):
        sdk = FakeSdk(fail_times=2, fail_with=FakeError(429))
        client_with(sdk).complete(MESSAGES)
        assert len(sdk.calls) == 3

    @pytest.mark.parametrize("status", [400, 422])
    def test_request_faults_are_not_retried(self, status):
        """A malformed request should fail loudly, not burn free-tier quota."""
        sdk = FakeSdk(fail_times=99, fail_with=FakeError(status))
        with pytest.raises(LlmError, match=f"HTTP {status}"):
            client_with(sdk).complete(MESSAGES)
        assert len(sdk.calls) == 1

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failures_are_not_retried(self, status):
        sdk = FakeSdk(fail_times=99, fail_with=FakeError(status))
        with pytest.raises(LlmAuthError):
            client_with(sdk).complete(MESSAGES)
        assert len(sdk.calls) == 1

    def test_transport_failures_are_retried_then_reported(self):
        sdk = FakeSdk(fail_times=99, fail_with=RuntimeError("connection reset"))
        expected = APIS["nvidia"]["retry_policy"]["max_retries"] + 1
        with pytest.raises(LlmError, match=f"after {expected} attempts"):
            client_with(sdk).complete(MESSAGES)
        assert len(sdk.calls) == APIS["nvidia"]["retry_policy"]["max_retries"] + 1


SCHEMA = {
    "type": "object",
    "properties": {
        "relation_type": {"type": "string"},
        "anchor_entity": {"type": "string"},
        "distance_value": {"type": "number"},
        "distance_unit": {"type": "string"},
        "direction_text": {"type": "string"},
    },
    "required": ["relation_type"],
}


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name, arguments):
        self.function = FakeFunction(name, arguments)


class ToolSdk(FakeSdk):
    """Returns a forced tool call instead of message content."""

    def __init__(self, arguments, name="record_spatial_interpretation"):
        super().__init__(text="")
        self._arguments = arguments
        self._name = name

    class _Completions(FakeCompletions):
        def create(self, **payload):
            self.outer.calls.append(payload)
            completion = FakeCompletion(self.outer.served_model, "")
            completion.choices[0].message.tool_calls = [
                FakeToolCall(self.outer._name, self.outer._arguments)
            ]
            return completion

    def __getattribute__(self, item):
        if item == "chat":
            chat = object.__new__(FakeChat)
            chat.completions = ToolSdk._Completions(self)
            return chat
        return object.__getattribute__(self, item)


class TestStructuredOutputViaToolCalling:
    def test_forces_a_single_schema_shaped_tool(self):
        sdk = ToolSdk('{"relation_type": "DISTANCE_DIRECTION", "anchor_entity": "Waiouru"}')
        arguments, response = client_with(sdk).complete_structured(
            MESSAGES, "record_spatial_interpretation", SCHEMA
        )
        payload = sdk.calls[0]
        assert len(payload["tools"]) == 1
        assert payload["tool_choice"]["function"]["name"] == "record_spatial_interpretation"
        assert arguments["anchor_entity"] == "Waiouru"
        assert response.tool_calls[0]["name"] == "record_spatial_interpretation"

    def test_no_geographic_or_computational_tool_is_exposed(self):
        """The model interprets. It does not retrieve, ground, or compute."""
        sdk = ToolSdk('{"relation_type": "DISTANCE_DIRECTION"}')
        client_with(sdk).complete_structured(MESSAGES, "record_spatial_interpretation", SCHEMA)
        names = {t["function"]["name"] for t in sdk.calls[0]["tools"]}
        assert names == {"record_spatial_interpretation"}
        forbidden = {"search", "geocode", "gazetteer", "geodesic", "distance", "linz", "google"}
        assert not any(any(f in name.lower() for f in forbidden) for name in names)

    def test_unparseable_arguments_become_a_failure_not_a_guess(self):
        arguments, response = client_with(ToolSdk("{not json at all")).complete_structured(
            MESSAGES, "record_spatial_interpretation", SCHEMA
        )
        assert arguments is None
        assert response.tool_calls  # raw response preserved for audit

    def test_absent_tool_call_becomes_a_failure(self):
        arguments, response = client_with(FakeSdk(text="I think it is near Waiouru")).complete_structured(
            MESSAGES, "record_spatial_interpretation", SCHEMA
        )
        assert arguments is None
        assert response.text == "I think it is near Waiouru"


class TestConnectionFailures:
    """The observed production failure was a dropped connection under
    concurrency, not a rejected request. 17 of 100 dev records were lost to it
    at 8 workers."""

    def test_connection_errors_are_retried(self):
        sdk = FakeSdk(fail_times=3, fail_with=ConnectionError("Connection error."))
        client_with(sdk).complete(MESSAGES)
        assert len(sdk.calls) == 4

    def test_the_retry_budget_survives_a_run_of_connection_errors(self):
        """A lost record costs a whole description; a retry costs seconds."""
        budget = APIS["nvidia"]["retry_policy"]["max_retries"]
        assert budget >= 6
        sdk = FakeSdk(fail_times=budget, fail_with=ConnectionError("Connection error."))
        response = client_with(sdk).complete(MESSAGES)
        assert response.attempt_count == budget + 1

    def test_backoff_grows_and_carries_jitter(self):
        """Without jitter, workers throttled together retry in lockstep."""
        client = client_with(FakeSdk())
        first = client._backoff_delay(0, 2.0)
        later = client._backoff_delay(3, 2.0)
        assert later > first
        jitter = APIS["nvidia"]["retry_policy"]["jitter_seconds"]
        assert 8.0 <= later <= 8.0 + jitter


class TestProfiles:
    """Profiles let development continue when one endpoint is throttling. They
    are not an experimental variable: a benchmark run must use exactly one."""

    def test_the_default_profile_is_the_active_one(self):
        assert MODELS["llm"]["profile"] == RAW_MODELS["active_llm_profile"]

    def test_a_gemini_profile_selects_the_google_endpoint(self):
        resolved = resolve_llm_profile(RAW_MODELS, "gemini_flash")
        assert resolved["llm"]["provider"] == "gemini"
        assert "generativelanguage.googleapis.com" in APIS["gemini"]["base_url"]

    def test_each_profile_has_an_api_block(self):
        for name, profile in RAW_MODELS["llm_profiles"].items():
            assert profile["provider"] in APIS, f"{name} names an unconfigured provider"

    def test_an_unknown_profile_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown LLM profile"):
            resolve_llm_profile(RAW_MODELS, "not_a_profile")

    def test_unsupported_parameters_are_absent_rather_than_sent(self):
        """Google's endpoint rejects parameters it does not implement, and a 400
        is deliberately not retried, so they must simply not be sent. Gemini has
        no seed; it does accept reasoning_effort, which the compatibility layer
        maps onto its own thinking_level."""
        gemini = resolve_llm_profile(RAW_MODELS, "gemini_flash")
        sdk = FakeSdk()
        LlmClient(gemini, APIS, api_key="k", client=sdk, sleep=lambda _: None).complete(MESSAGES)
        payload = sdk.calls[0]
        assert "seed" not in payload
        assert payload["reasoning_effort"] == "medium"

    def test_both_profiles_use_the_same_reasoning_depth(self):
        """Comparable settings across providers, so a mid-project switch changes
        the endpoint and not how hard the model thinks."""
        for name in ("nvidia_gpt_oss", "gemini_flash"):
            assert resolve_llm_profile(RAW_MODELS, name)["llm"]["reasoning_effort"] == "medium"

    def test_the_nvidia_profile_still_sends_its_own_parameters(self):
        sdk = FakeSdk()
        client_with(sdk).complete(MESSAGES)
        assert sdk.calls[0]["seed"] == 20260908
        assert sdk.calls[0]["reasoning_effort"] == "medium"

    def test_provider_and_profile_reach_provenance(self):
        """A mixed output file must be detectable after the fact."""
        response = client_with(FakeSdk()).complete(MESSAGES)
        assert response.provider == "nvidia"
        assert response.profile == "nvidia_gpt_oss"

    def test_switching_profile_changes_the_requested_model(self):
        gemini = resolve_llm_profile(RAW_MODELS, "gemini_flash")
        sdk = FakeSdk(served_model="gemini-3.8-flash")
        response = LlmClient(
            gemini, APIS, api_key="k", client=sdk, sleep=lambda _: None
        ).complete(MESSAGES)
        assert response.requested_model == "gemini-3.8-flash"
        assert response.model_switched is False

    @pytest.mark.parametrize(
        "shorthand, expected",
        [("nvidia", "nvidia_gpt_oss"), ("gemini", "gemini_flash")],
    )
    def test_provider_shorthand_resolves(self, shorthand, expected):
        """Switching endpoints is '--profile gemini', not a profile id to memorise."""
        assert resolve_llm_profile(RAW_MODELS, shorthand)["llm"]["profile"] == expected

    def test_an_explicit_profile_still_wins_over_shorthand(self):
        resolved = resolve_llm_profile(RAW_MODELS, "gemini_flash_lite")
        assert resolved["llm"]["model"] == "gemini-3.5-flash-lite"

    def test_ambiguous_shorthand_without_a_declared_default_is_refused(self):
        """Guessing which of several profiles was meant would silently change
        the model."""
        config = {
            "llm_profiles": {
                "a": {"provider": "x", "model": "m1"},
                "b": {"provider": "x", "model": "m2"},
            }
        }
        with pytest.raises(ValueError, match="matches several profiles"):
            resolve_llm_profile(config, "x")
