# LLM Implementation and Inspection Guide

This guide provides an academic supervisor or code auditor with direct, line-by-line navigation to the Large Language Model (LLM) implementation, prompts, API calls, structured-output validation, and error-handling mechanisms used in this dissertation research.

---

## 1. Executive Summary of LLM Usage

Both experimental conditions utilize the exact same underlying frontier LLM—**Gemini 3.8 Flash**—to ensure that any observed differences in positional error or coverage stem solely from the architectural intervention (neuro-symbolic retrieval and deterministic GIS) rather than confounding variations in model capability.

| Attribute | Standalone Baseline (Condition A) | Hybrid Pipeline (Condition B) |
|---|---|---|
| **Model** | `gemini-3.8-flash` | `gemini-3.8-flash` |
| **Decoding Temperature** | `0.0` (greedy) | `0.0` (greedy) |
| **Top P** | `1.0` | `1.0` |
| **Reasoning Effort** | `medium` | `medium` |
| **Max Output Tokens** | `32768` | `32768` |
| **Prompt Version** | `baseline-v1` | `extract-v5` |
| **Forced Tool Schema** | `record_coordinates` | `record_spatial_interpretation` |
| **Task Assigned to LLM** | Direct coordinate estimation | Semantic relation extraction only |
| **External Geographic Knowledge** | None (parametric memory only) | None during extraction; grounded via LINZ post-extraction |
| **Calculations by LLM** | Implicit | Prohibited (handled deterministically in Python) |

---

## 2. Configuration Locations

### 2.1 Model Profile Configuration
- **File Path:** [`configs/models.yaml`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/configs/models.yaml)
- **Active Profile Setting:** Line 18 (`active_llm_profile: nvidia_gpt_oss`) and Line 21 (`default_profile_for_gemini: gemini_flash`).
- **Gemini Flash Profile Definition:** Lines 37–65:
  ```yaml
  gemini_flash:
    provider: gemini
    model: gemini-3.8-flash
    temperature: 0.0
    top_p: 1.0
    max_tokens: 32768
    reasoning_effort: medium
    record_served_model: true
  ```
- **Alternative Fast/Cheaper Profile:** Lines 66–73 (`gemini_flash_lite`, using `gemini-3.5-flash-lite`).

### 2.2 API Endpoint & Concurrency Configuration
- **File Path:** [`configs/apis.yaml`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/configs/apis.yaml)
- **Gemini Service Configuration:** Lines 39–50:
  ```yaml
  gemini:
    base_url: https://generativelanguage.googleapis.com/v1beta/openai/
    api_key_env: GEMINI_API_KEY
    timeout_seconds: 240
    max_workers: 4
    retry_policy:
      max_retries: 6
      backoff_factor: 2
      retry_on_status: [408, 409, 429, 500, 502, 503, 504]
      jitter_seconds: 2.0
  ```
- **Rationale for Concurrency:** Evaluated on the 100-record dev set: 4 workers yielded zero dropped connections and reliable latency (~4.3 s/record for Gemini), whereas higher concurrency triggered endpoint throttling.

---

## 3. Client Initialization & API Execution

### 3.1 Client Architecture
- **File Path:** [`src/georef/llm/client.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/client.py)
- **Class:** `LlmClient` (Lines 63–262).
- **Initialization:** Lines 70–102 (`__init__`).
  - Accesses API credentials from the environment variable designated by `api_key_env` (`GEMINI_API_KEY`).
  - Initializes `openai.OpenAI` pointing to `base_url="https://generativelanguage.googleapis.com/v1beta/openai/"` with SDK-level retries disabled (`max_retries=0`) so that attempts, latency, and status codes are explicitly managed, counted, and recorded in the research provenance log.
  - Supports dependency injection of mock clients (`client` argument) for hermetic offline testing.

### 3.2 Where API Calls are Dispatched
- **Low-Level Chat Completion:** Lines 188–257 in `LlmClient.complete()`.
  - Dispatches `self._client.chat.completions.create(**payload)`.
  - Preserves token counts (`prompt_tokens`, `completion_tokens`), response ID, processing latency, and the server-reported model string (`completion.model`).
- **Structured Tool Calling:** Lines 144–178 in `LlmClient.complete_structured()`.
  - Employs OpenAI function calling as a **serialization contract** rather than an autonomous agentic loop.
  - Passes exactly one tool defined by `parameters_schema` and forces its invocation via `tool_choice={"type": "function", "function": {"name": tool_name}}`.
  - Extracts and parses JSON arguments from `response.tool_calls[0].arguments`.

---

## 4. Prompts and Templates

### 4.1 Condition A: Standalone Baseline Prompt
- **File Path:** [`src/georef/llm/baseline.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/baseline.py)
- **Prompt Version Constant:** Line 39 (`BASELINE_PROMPT_VERSION = "baseline-v1"`).
- **System Prompt Definition:** Lines 41–52:
  ```text
  You are given a natural-language locality description from a New Zealand biological collection record.

  Return your single best estimate of the coordinates the description refers to.

  Call the record_coordinates tool exactly once, with:
  - latitude: decimal degrees, negative in the southern hemisphere
  - longitude: decimal degrees
  - confidence: a number between 0 and 1, or null

  New Zealand latitudes are roughly -34 to -47. Longitudes are roughly 166 to 179, except the Chatham Islands which are around -176.

  Give your best estimate even when the description is vague. If the description is unusable, set latitude and longitude to null.
  ```
- **User Prompt Construction:** Lines 168–171:
  ```python
  messages = [
      {"role": "system", "content": self.prompt},
      {"role": "user", "content": description},
  ]
  ```

### 4.2 Condition B: Hybrid Interpretation Prompt
- **File Path:** [`src/georef/llm/extraction.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/extraction.py)
- **Prompt Version Constant:** Line 44 (`PROMPT_VERSION = "extract-v5"`).
- **System Prompt Definition:** Lines 46–207 (`SYSTEM_PROMPT`).
  - Instructs the LLM to call `record_spatial_interpretation` exactly once.
  - Enforces: *"Report only what the description says. You do not know where anything is. You never estimate coordinates. You never calculate."*
  - Establishes a rigorous **5-step procedure**:
    1. Read the whole description.
    2. List every place name mentioned anywhere.
    3. Identify distance-and-direction clauses (`<distance> <direction> of/from <place>`).
    4. Resolve back-references (e.g. mapping "the summit" back to "Mt. Maungatua", "Pass" to "Lindis Pass").
    5. Populate tool fields.
- **User Prompt Construction:** Lines 267–286 in `SpatialExtractor._messages()`:
  - Passes the raw description.
  - When optional `--hints` (pre-retrieval gazetteer spotting) is enabled, appends detected place names (names only, never coordinates) to guide proper-noun boundary detection.

---

## 5. Output Validation and Schema Enforcement

### 5.1 Condition A Validation
- **File Path:** [`src/georef/llm/baseline.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/baseline.py)
- **Schema Model:** `BaselineCoordinates` (Lines 67–101).
- **Validation Rules:**
  - Latitude must be between $-90.0$ and $+90.0$.
  - Longitude must be between $-180.0$ and $+180.0$.
  - Confidence clamped to $[0.0, 1.0]$ or set to `None`.
- **Parsing Flow:** Lines 209–229 in `BaselineGeoreferencer.predict()`:
  - Checks for truncation (`response.finish_reason == "length"`).
  - Validates via `BaselineCoordinates.model_validate(arguments)`.
  - Captures validation failures as `SCHEMA_VALIDATION_FAILURE` with field-level details.

### 5.2 Condition B Validation & Derivation
- **File Paths:**
  - Extraction: [`src/georef/llm/extraction.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/llm/extraction.py)
  - Models & derivation: [`src/georef/models/spatial.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/models/spatial.py)
- **Schema Models:**
  - `SpatialClause` (Lines 102–164 in `spatial.py`):
    - Robust validator `_parse_hedged_number`: parses hedged text like `"ca. 3"`, `"~2.5"`, fractions `"1/2 mile"` $\to 0.5$, and ranges `"10-12 km"` $\to 11.0$ (interval midpoint).
  - `ModelInterpretation` (Lines 183–226 in `spatial.py`):
    - Validates `relation_type` against allowed literals (`DISTANCE_DIRECTION`, `MULTI_ANCHOR`, etc.).
    - Captures `clauses: list[SpatialClause]`.
    - Captures `geographic_context`, `ambiguity_detected`, and `confidence`.
- **Deterministic Derivation:** Lines 294–337 in `spatial.py`:
  - `derive()` converts raw model output into `SpatialInterpretation`.
  - Converts compass direction text (e.g. `"ESE"`, `"south-west"`) into true bearings in degrees using static lookup table `BEARINGS` in [`src/georef/spatial/units.py`](file:///c:/Users/sahil/Desktop/dissertation-georef-clean/src/georef/spatial/units.py).
  - Converts distance values and units (`"km"`, `"miles"`, `"m"`) into metres using static conversion table `UNIT_METRES`.

---

## 6. Error and Failure Handling

Both conditions systematically classify and record errors without silent crashes:

```
                  +-----------------------------------+
                  |         LLM API Request           |
                  +-----------------------------------+
                                    |
              +---------------------+---------------------+
              |                                           |
      HTTP Error / Exception                           Success
              |                                           |
    +---------+---------+                       +---------+---------+
    |                   |                       |                   |
Unretryable         Retryable               Truncated           Complete
(401, 403,          (429, 500,              (finish_reason      (finish_reason
 400, 422)           503, etc.)              == "length")        == "stop")
    |                   |                       |                   |
LlmAuthError /      Exponential             Status:             Validate JSON
LlmError            Backoff + Jitter        MODEL_OUTPUT_       Tool Arguments
(Terminates /       (Up to 6 attempts)      TRUNCATED               |
 recorded)              |                                   +-------+-------+
                    Max Retries                             |               |
                    Exceeded                            Pydantic        Pydantic
                        |                               Failure         Success
                    Status:                             Status:             |
                    API_FAILURE                         SCHEMA_         Status:
                                                        VALIDATION_     SUCCESS
                                                        FAILURE
```

1. **Authentication Failures (401, 403):** Raised immediately as `LlmAuthError` without retrying.
2. **Client Errors (400, 422):** Raised immediately as `LlmError` without burning retries on invalid requests.
3. **Transient Server Errors & Throttling (408, 409, 429, 500, 502, 503, 504):** Handled by exponential backoff with jitter up to 6 retries (`configs/apis.yaml`).
4. **Token Truncation:** If `finish_reason == "length"`, parsing is skipped and status is flagged as `MODEL_OUTPUT_TRUNCATED` to avoid operating on partial JSON structures.
5. **Schema Violations:** Recorded with exact field paths and input values (e.g. `clauses.0.distance_value: float_parsing`).
