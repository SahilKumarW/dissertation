#!/usr/bin/env python
"""LIVE API SMOKE TEST: confirm the endpoint, key and served model.

    python scripts/check_llm.py

Sends one tiny structured request and reports what the endpoint actually
served, which is the identifier recorded in provenance for every benchmark
record. Run this before building anything on top of the LLM.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from georef.config.loader import load_yaml, repo_root, resolve_llm_profile  # noqa: E402
from georef.llm.client import LlmClient, LlmError  # noqa: E402
from georef.linz.ingest import _load_dotenv  # noqa: E402

PROMPT = (
    "Return only JSON with keys anchor_entity, distance_value, distance_unit, "
    "bearing_degrees for this description. No prose, no code fences.\n\n"
    "19 km southwest of Waiouru"
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check an LLM endpoint.")
    parser.add_argument("--profile", default=None, help="e.g. gemini_flash")
    args = parser.parse_args()

    root = repo_root()
    _load_dotenv(root / ".env")
    models = resolve_llm_profile(load_yaml(root / "configs" / "models.yaml"), args.profile)
    apis = load_yaml(root / "configs" / "apis.yaml")

    try:
        client = LlmClient(models, apis)
        response = client.complete([{"role": "user", "content": PROMPT}])
    except LlmError as error:
        print(f"LLM SMOKE TEST FAILED\n\n{error}", file=sys.stderr)
        return 2

    print("LLM smoke test\n")
    print(f"  profile          {response.profile}")
    print(f"  provider         {response.provider}")
    print(f"  requested model  {response.requested_model}")
    print(f"  served model     {response.served_model}")
    if response.model_switched:
        print("  WARNING: the endpoint served a different model than requested.")
    print(f"  finish reason    {response.finish_reason}")
    print(f"  attempts         {response.attempt_count}")
    print(f"  tokens           {response.prompt_tokens} in / {response.completion_tokens} out")
    print(f"  elapsed          {response.processing_time_seconds}s")
    print(f"\n  parameters       {json.dumps(response.parameters)}")
    print(f"\n  raw response:\n{response.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
