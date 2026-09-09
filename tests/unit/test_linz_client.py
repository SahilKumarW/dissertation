"""UNIT TEST -- LINZ WFS client. No network, no API key, no live service.

A fake transport stands in for HTTP so request construction, paging and retry
behaviour are exercised deterministically. This is NOT evidence that the live
service works; that requires `scripts/fetch_linz.py --describe`, which is a
LIVE API SMOKE TEST and is reported separately.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from georef.config.loader import load_yaml, repo_root
from georef.linz.client import LinzApiError, LinzAuthError, LinzClient, redact

CONFIG = load_yaml(repo_root() / "configs" / "linz.yaml")
KEY = "abcdefgh123456789"


def make_feature(index: int) -> dict:
    return {
        "type": "Feature",
        "properties": {
            "name_id": str(1000 + index),
            "name": f"Place {index}",
            "feat_type": "Locality",
            "land_district": "Otago",
            "crd_latitude": -45.0 - index / 1000,
            "crd_longitude": 170.0 + index / 1000,
            "status": "Official Assigned",
            "feat_id": str(2000 + index),
            "maori_name": "",
            "crd_datum": "NZGD2000",
        },
    }


class FakeTransport:
    """Serves pages from a fixed corpus and records the URLs it was given."""

    def __init__(self, total: int, fail_times: int = 0, fail_with: Exception | None = None):
        self.total = total
        self.calls: list[str] = []
        self.fail_times = fail_times
        self.fail_with = fail_with or urllib.error.URLError("temporary")

    def __call__(self, url: str, timeout: int) -> bytes:
        self.calls.append(url)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.fail_with
        params = dict(pair.split("=", 1) for pair in url.split("?", 1)[1].split("&"))
        start = int(params["startIndex"])
        count = int(params["count"])
        features = [make_feature(i) for i in range(start, min(start + count, self.total))]
        return json.dumps(
            {
                "type": "FeatureCollection",
                "features": features,
                "numberMatched": self.total,
                "numberReturned": len(features),
            }
        ).encode("utf-8")


def client_with(transport, config: dict | None = None) -> LinzClient:
    return LinzClient(config or CONFIG, api_key=KEY, transport=transport, sleep=lambda _: None)


class TestRedaction:
    def test_redacts_key_in_lds_path_form(self):
        url = f"https://data.linz.govt.nz/services;key={KEY}/wfs?service=WFS"
        assert KEY not in redact(url)
        assert "key=REDACTED" in redact(url)
        assert "/wfs?service=WFS" in redact(url)

    def test_redacts_key_in_query_form(self):
        url = f"https://example.test/wfs?key={KEY}&request=GetFeature"
        assert KEY not in redact(url)
        assert "request=GetFeature" in redact(url)

    def test_leaves_urls_without_a_key_alone(self):
        url = "https://data.linz.govt.nz/services/wfs?request=GetFeature"
        assert redact(url) == url

    def test_built_url_redacts_cleanly(self):
        url = client_with(FakeTransport(1)).build_url(0, 10)
        assert KEY in url  # LDS requires it in the path
        assert KEY not in redact(url)


class TestRequestConstruction:
    def test_includes_layer_version_and_paging(self):
        url = client_with(FakeTransport(1)).build_url(500, 250)
        assert "typeNames=layer-51681" in url
        assert "version=2.0.0" in url
        assert "startIndex=500" in url
        assert "count=250" in url
        assert "outputFormat=json" in url

    def test_includes_sort_by_for_stable_paging(self):
        """Without sortBy the server may overlap or drop records between pages."""
        assert "sortBy=name_id" in client_with(FakeTransport(1)).build_url(0, 10)

    def test_missing_key_fails_before_any_request(self, monkeypatch):
        monkeypatch.delenv("LINZ_API_KEY", raising=False)
        with pytest.raises(LinzAuthError, match="LINZ_API_KEY"):
            LinzClient(CONFIG, api_key=None, transport=FakeTransport(1))


class TestPaging:
    def test_yields_every_feature_exactly_once(self):
        transport = FakeTransport(total=12_345)
        features = list(client_with(transport).iter_features(page_size=1000))
        assert len(features) == 12_345
        ids = [f["properties"]["name_id"] for f in features]
        assert len(set(ids)) == len(ids)

    def test_stops_at_number_matched_without_an_extra_request(self):
        transport = FakeTransport(total=2000)
        list(client_with(transport).iter_features(page_size=1000))
        assert len(transport.calls) == 2

    def test_handles_a_partial_final_page(self):
        transport = FakeTransport(total=2500)
        assert len(list(client_with(transport).iter_features(page_size=1000))) == 2500

    def test_empty_layer_yields_nothing(self):
        assert list(client_with(FakeTransport(total=0)).iter_features(page_size=100)) == []


class TestRetries:
    def test_transient_failure_is_retried_then_succeeds(self):
        transport = FakeTransport(total=10, fail_times=2)
        page = client_with(transport).fetch_page(0, 10)
        assert page.number_returned == 10
        assert len(transport.calls) == 3

    def test_gives_up_after_the_configured_maximum(self):
        transport = FakeTransport(total=10, fail_times=99)
        with pytest.raises(LinzApiError, match="after 4 attempts"):
            client_with(transport).fetch_page(0, 10)
        assert len(transport.calls) == CONFIG["retry_policy"]["max_retries"] + 1

    def test_failure_message_does_not_leak_the_key(self):
        transport = FakeTransport(total=10, fail_times=99)
        with pytest.raises(LinzApiError) as caught:
            client_with(transport).fetch_page(0, 10)
        assert KEY not in str(caught.value)

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_are_not_retried(self, status):
        error = urllib.error.HTTPError("u", status, "denied", {}, None)
        transport = FakeTransport(total=10, fail_times=99, fail_with=error)
        with pytest.raises(LinzAuthError):
            client_with(transport).fetch_page(0, 10)
        assert len(transport.calls) == 1

    def test_client_errors_are_not_retried(self):
        error = urllib.error.HTTPError("u", 400, "bad request", {}, None)
        transport = FakeTransport(total=10, fail_times=99, fail_with=error)
        with pytest.raises(LinzApiError, match="HTTP 400"):
            client_with(transport).fetch_page(0, 10)
        assert len(transport.calls) == 1

    def test_server_errors_are_retried(self):
        error = urllib.error.HTTPError("u", 503, "unavailable", {}, None)
        transport = FakeTransport(total=10, fail_times=2, fail_with=error)
        client_with(transport).fetch_page(0, 10)
        assert len(transport.calls) == 3


class TestDescribe:
    def test_reports_field_names_and_redacted_endpoint(self):
        description = client_with(FakeTransport(total=5)).describe()
        assert "crd_latitude" in description["field_names"]
        assert description["number_matched"] == 5
        assert KEY not in description["endpoint"]
