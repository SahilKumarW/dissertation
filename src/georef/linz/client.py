"""LINZ Data Service WFS client.

Responsibilities: build correct WFS 2.0 requests, page through a layer
deterministically, and retry transient failures a bounded number of times.

It does not validate, normalise or store records; that is `ingest.py`. It does
not decide anything geographic.

**API key handling.** The key is read from the environment, interpolated into
the request URL because LDS requires it in the path, and redacted from every
value this module returns or raises. `redact()` is applied to any URL before it
reaches a log, a manifest or an exception message. There is no code path that
writes the raw URL anywhere persistent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator

__all__ = [
    "LinzApiError",
    "LinzAuthError",
    "LinzClient",
    "Page",
    "redact",
]

_KEY_PATTERN = "key="


class LinzApiError(RuntimeError):
    """A LINZ request failed after the configured retries."""


class LinzAuthError(LinzApiError):
    """The key was rejected. Not retried: this is a configuration fault."""


def redact(text: str) -> str:
    """Replace any ``key=...`` value with ``key=REDACTED``.

    Applied to every URL before it is logged, stored or raised.
    """
    if _KEY_PATTERN not in text:
        return text
    head, _, tail = text.partition(_KEY_PATTERN)
    # The key runs to the next path separator, ampersand or whitespace.
    end = len(tail)
    for stop in ("/", "&", "?", " "):
        position = tail.find(stop)
        if position != -1:
            end = min(end, position)
    return f"{head}{_KEY_PATTERN}REDACTED{tail[end:]}"


@dataclass
class Page:
    """One WFS page."""

    features: list[dict]
    number_matched: int | None
    number_returned: int
    start_index: int


def _default_transport(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dissertation-georef/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class LinzClient:
    """Paged reader for a single LDS WFS layer.

    ``transport`` is injectable so the request-building, paging and retry logic
    can be unit tested without a network or a key.
    """

    def __init__(
        self,
        config: dict,
        api_key: str | None = None,
        transport: Callable[[str, int], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.service = config["service"]
        self.dataset = config["dataset"]
        self.retry_policy = config["retry_policy"]
        self._transport = transport or _default_transport
        self._sleep = sleep

        key = api_key if api_key is not None else os.environ.get("LINZ_API_KEY", "")
        if not key:
            raise LinzAuthError(
                "LINZ_API_KEY is not set. Copy .env.example to .env and add your "
                "LINZ Data Service key, or pass --api-key."
            )
        self._key = key

    # -- request construction ------------------------------------------------

    def build_url(self, start_index: int, count: int, result_type: str | None = None) -> str:
        base = self.service["base_url_template"].format(key=self._key)
        params = {
            "service": "WFS",
            "version": self.service["version"],
            "request": "GetFeature",
            "typeNames": self.dataset["layer_id"],
            "outputFormat": self.service["output_format"],
            "srsName": self.service["srs_name"],
            "count": count,
            "startIndex": start_index,
        }
        sort_by = self.service.get("sort_by")
        if sort_by:
            # Without a deterministic sort the server may return overlapping or
            # missing records between pages, producing a snapshot that is wrong
            # in a way no later stage can detect.
            params["sortBy"] = sort_by
        if result_type:
            params["resultType"] = result_type
        return f"{base}?{urllib.parse.urlencode(params)}"

    # -- transport with bounded retries --------------------------------------

    def _get(self, url: str) -> bytes:
        max_retries = int(self.retry_policy["max_retries"])
        backoff = float(self.retry_policy["backoff_factor"])
        retry_on = set(self.retry_policy.get("retry_on_status", []))
        timeout = int(self.service["timeout_seconds"])

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return self._transport(url, timeout)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise LinzAuthError(
                        f"LINZ rejected the API key (HTTP {error.code}). Check "
                        "LINZ_API_KEY and that the key has access to "
                        f"{self.dataset['layer_id']}."
                    ) from None
                if error.code not in retry_on:
                    raise LinzApiError(
                        f"LINZ returned HTTP {error.code} for {redact(url)}"
                    ) from None
                last_error = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error

            if attempt < max_retries:
                self._sleep(backoff**attempt)

        raise LinzApiError(
            f"LINZ request failed after {max_retries + 1} attempts: "
            f"{redact(url)} ({last_error})"
        ) from None

    # -- public reads --------------------------------------------------------

    def fetch_page(self, start_index: int, count: int) -> Page:
        payload = json.loads(self._get(self.build_url(start_index, count)))
        features = payload.get("features", [])
        return Page(
            features=features,
            number_matched=payload.get("numberMatched"),
            number_returned=payload.get("numberReturned", len(features)),
            start_index=start_index,
        )

    def describe(self, sample_size: int = 1) -> dict:
        """Fetch a tiny sample and report the field names actually returned.

        Run this before a full ingest: it confirms the layer, the key and the
        field mapping in configs/linz.yaml against the live service, at the cost
        of one small request.
        """
        page = self.fetch_page(0, sample_size)
        properties = page.features[0].get("properties", {}) if page.features else {}
        return {
            "layer_id": self.dataset["layer_id"],
            "endpoint": redact(self.build_url(0, sample_size)),
            "number_matched": page.number_matched,
            "field_names": sorted(properties),
            "sample_properties": properties,
        }

    def iter_features(self, page_size: int | None = None) -> Iterator[dict]:
        """Yield every feature in the layer, paging until the server runs out."""
        count = int(page_size or self.service["page_size"])
        start_index = 0
        seen = 0
        while True:
            page = self.fetch_page(start_index, count)
            if not page.features:
                return
            for feature in page.features:
                yield feature
            seen += len(page.features)
            if page.number_matched is not None and seen >= page.number_matched:
                return
            if len(page.features) < count:
                return
            start_index += count
