"""UNIT TEST -- LINZ normalisation and snapshot writing. No network.

The client is replaced by a stub yielding fixed features, so validation,
rejection accounting, snapshot stability and manifest provenance are tested
without touching the live service.
"""

from __future__ import annotations

import json

import pytest

from georef.config.loader import load_yaml, repo_root
from georef.linz.client import LinzApiError
from georef.linz.ingest import ingest, normalise_feature
from georef.provenance.hashing import sha256_file

CONFIG = load_yaml(repo_root() / "configs" / "linz.yaml")
FIELDS = CONFIG["fields"]
REQUIRED = CONFIG["required_fields"]


def feature(**overrides) -> dict:
    properties = {
        "name_id": "15057",
        "name": "Abbey Caves",
        "feat_type": "Cave",
        "land_district": "North Auckland",
        "crd_latitude": -35.709389,
        "crd_longitude": 174.357583,
        "status": "Unofficial Recorded",
        "feat_id": "100017",
        "maori_name": "No",
        "crd_datum": "NZGD2000",
        "region": None,
        "territorial_authority": "Whangarei District",
        "geom_type": "POINT",
    }
    properties.update(overrides)
    return {"type": "Feature", "properties": properties}


class StubClient:
    """Stands in for LinzClient; yields fixed features."""

    def __init__(self, features: list[dict]):
        self._features = features
        self.dataset = CONFIG["dataset"]
        self.service = CONFIG["service"]

    def iter_features(self, page_size=None):
        return iter(self._features)

    def build_url(self, start_index: int, count: int) -> str:
        return (
            "https://data.linz.govt.nz/services;key=SECRETKEY123/wfs"
            f"?typeNames=layer-51681&startIndex={start_index}&count={count}"
        )


class TestNormalisation:
    def test_maps_fields_onto_the_internal_schema(self):
        record, reason = normalise_feature(feature(), FIELDS, REQUIRED)
        assert reason is None
        assert record["linz_id"] == "15057"
        assert record["name"] == "Abbey Caves"
        assert record["feature_type"] == "Cave"
        assert record["name_status"] == "Unofficial Recorded"
        assert record["geometry_type"] == "POINT"

    def test_keeps_all_three_administrative_vocabularies_separately(self):
        """land_district, region and territorial_authority are not interchangeable."""
        record, _ = normalise_feature(feature(), FIELDS, REQUIRED)
        assert record["land_district"] == "North Auckland"
        assert record["territorial_authority"] == "Whangarei District"
        assert record["region"] is None  # sparsely populated in this layer

    def test_maori_name_flag_is_not_treated_as_a_name(self):
        """The source column holds Yes/No, not an alternative name string."""
        record, _ = normalise_feature(feature(maori_name="Yes"), FIELDS, REQUIRED)
        assert record["is_maori_name"] == "Yes"
        assert "maori_name" not in record

    def test_stamps_the_authoritative_source(self):
        """Master spec section 17: every record used for prediction carries source=LINZ."""
        record, _ = normalise_feature(feature(), FIELDS, REQUIRED)
        assert record["source"] == "LINZ"
        assert record["country"] == "New Zealand"

    def test_coordinates_become_floats(self):
        record, _ = normalise_feature(
            feature(crd_latitude="-35.709389", crd_longitude="174.357583"), FIELDS, REQUIRED
        )
        assert record["latitude"] == pytest.approx(-35.709389)
        assert isinstance(record["longitude"], float)

    @pytest.mark.parametrize("missing", ["name_id", "name", "crd_latitude", "crd_longitude"])
    def test_missing_required_field_is_rejected_not_defaulted(self, missing):
        record, reason = normalise_feature(feature(**{missing: None}), FIELDS, REQUIRED)
        assert record is None
        assert reason == "MISSING_REQUIRED_FIELD"

    def test_blank_name_is_rejected(self):
        record, reason = normalise_feature(feature(name="   "), FIELDS, REQUIRED)
        assert record is None
        assert reason == "MISSING_REQUIRED_FIELD"

    def test_non_numeric_coordinate_is_rejected(self):
        record, reason = normalise_feature(feature(crd_latitude="not a number"), FIELDS, REQUIRED)
        assert record is None
        assert reason == "NON_NUMERIC_COORDINATE"

    def test_optional_fields_may_be_absent(self):
        record, reason = normalise_feature(feature(territorial_authority=""), FIELDS, REQUIRED)
        assert reason is None
        assert record["territorial_authority"] is None


class TestIngest:
    def test_writes_a_sorted_snapshot_with_a_matching_hash(self, tmp_path):
        features = [feature(name_id=str(i), name=f"Place {i}") for i in (30, 10, 20)]
        result = ingest(CONFIG, tmp_path, StubClient(features), dataset_version="test")

        lines = result.snapshot_path.read_text(encoding="utf-8").strip().split("\n")
        assert [json.loads(line)["linz_id"] for line in lines] == ["10", "20", "30"]
        assert result.snapshot_hash == sha256_file(result.snapshot_path)
        assert result.record_count == 3

    def test_snapshot_is_stable_regardless_of_page_order(self, tmp_path):
        ordered = [feature(name_id=str(i), name=f"Place {i}") for i in (10, 20, 30)]
        shuffled = [feature(name_id=str(i), name=f"Place {i}") for i in (20, 30, 10)]
        first = ingest(CONFIG, tmp_path / "a", StubClient(ordered), dataset_version="test")
        second = ingest(CONFIG, tmp_path / "b", StubClient(shuffled), dataset_version="test")
        assert first.snapshot_hash == second.snapshot_hash

    def test_chatham_islands_survive_the_bounding_boxes(self, tmp_path):
        """Negative longitude east of the antimeridian; a single naive box loses these."""
        chatham = feature(name_id="1", crd_latitude=-43.95, crd_longitude=-176.55)
        result = ingest(CONFIG, tmp_path, StubClient([chatham]), dataset_version="test")
        assert result.record_count == 1
        record = json.loads(result.snapshot_path.read_text().strip())
        assert record["bounding_box"] == "chatham_islands"

    def test_antarctic_records_are_excluded_and_counted(self, tmp_path):
        """The gazetteer covers the Ross Dependency; the benchmark does not."""
        ross = feature(name_id="2", crd_latitude=-77.5, crd_longitude=166.5)
        keep = feature(name_id="3")
        result = ingest(CONFIG, tmp_path, StubClient([ross, keep]), dataset_version="test")
        assert result.record_count == 1
        assert result.rejected["OUTSIDE_BOUNDING_BOXES"] == 1

    def test_duplicate_identifiers_are_counted(self, tmp_path):
        duplicated = [feature(name_id="7"), feature(name_id="7")]
        result = ingest(CONFIG, tmp_path, StubClient(duplicated), dataset_version="test")
        assert result.record_count == 1
        assert result.rejected["DUPLICATE_LINZ_ID"] == 1

    def test_zero_usable_records_is_a_failure_not_an_empty_snapshot(self, tmp_path):
        with pytest.raises(LinzApiError, match="zero usable records"):
            ingest(CONFIG, tmp_path, StubClient([]), dataset_version="test")


class TestManifest:
    @pytest.fixture
    def manifest(self, tmp_path):
        result = ingest(CONFIG, tmp_path, StubClient([feature()]), dataset_version="test")
        return json.loads(result.manifest_path.read_text(encoding="utf-8")), result

    def test_records_the_provenance_the_spec_requires(self, manifest):
        """Master spec section 16: source, timestamp, endpoint, dataset id, version, count, hash."""
        data, result = manifest
        assert data["source_authority"] == "LINZ"
        assert data["layer_id"] == "layer-51681"
        assert data["dataset_version"] == "test"
        assert data["record_count"] == 1
        assert data["snapshot_sha256"] == result.snapshot_hash
        assert data["retrieved_at_utc"].endswith("+00:00")

    def test_manifest_never_contains_the_api_key(self, manifest):
        data, _ = manifest
        assert "SECRETKEY123" not in json.dumps(data)
        assert "key=REDACTED" in data["endpoint"]

    def test_rejection_reasons_are_itemised_not_collapsed(self, manifest):
        """Master spec section 28: every failure has a reason."""
        data, _ = manifest
        assert set(data["rejected"]) == {
            "MISSING_REQUIRED_FIELD",
            "NON_NUMERIC_COORDINATE",
            "OUTSIDE_BOUNDING_BOXES",
            "DUPLICATE_LINZ_ID",
        }
