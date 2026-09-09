"""REAL OFFLINE INTEGRATION TEST -- properties of the built artefacts.

These run against the artefacts produced by scripts/build_benchmark.py from the
actual supplier files, not against fixtures.
"""

import json

import pandas as pd
import pytest

from georef.config.loader import repo_root
from georef.normalise import normalise

FORBIDDEN_IN_BENCHMARK = ("lat", "lon", "coord", "decimal", "georeference")


@pytest.fixture(scope="module")
def benchmark(artefacts):
    return pd.read_csv(artefacts["benchmark"])


@pytest.fixture(scope="module")
def ground_truth(artefacts):
    return pd.read_csv(artefacts["ground_truth"])


@pytest.fixture(scope="module")
def dev_pool(artefacts):
    return pd.read_csv(artefacts["dev_pool"])


def test_benchmark_carries_no_coordinate_column(benchmark):
    for column in benchmark.columns:
        assert not any(token in column.lower() for token in FORBIDDEN_IN_BENCHMARK), column
    assert list(benchmark.columns) == ["benchmark_id", "description"]


def test_benchmark_size_matches_policy(benchmark, benchmark_cfg):
    assert len(benchmark) == benchmark_cfg["benchmark_size"]


def test_benchmark_ids_unique_and_well_formed(benchmark, benchmark_cfg):
    assert benchmark.benchmark_id.is_unique
    prefix = benchmark_cfg["benchmark_id_prefix"]
    assert benchmark.benchmark_id.str.match(rf"^{prefix}-\d{{3}}$").all()


def test_descriptions_present_and_distinct(benchmark):
    assert benchmark.description.notna().all()
    assert (benchmark.description.str.strip() != "").all()
    assert benchmark.description.map(normalise).is_unique


def test_every_benchmark_id_has_exactly_one_ground_truth(benchmark, ground_truth):
    assert set(benchmark.benchmark_id) == set(ground_truth.benchmark_id)
    assert ground_truth.benchmark_id.is_unique


def test_ground_truth_coordinates_are_inside_a_configured_box(ground_truth, benchmark_cfg):
    boxes = benchmark_cfg["bounding_boxes"]
    for record in ground_truth.itertuples():
        inside = any(
            box["lat_min"] <= record.decimal_latitude <= box["lat_max"]
            and box["lon_min"] <= record.decimal_longitude <= box["lon_max"]
            for box in boxes.values()
        )
        assert inside, f"{record.benchmark_id} outside every bounding box"


def test_ground_truth_categories_are_eligible(ground_truth, benchmark_cfg):
    allowed = set(benchmark_cfg["eligible_audit_categories"])
    assert set(ground_truth.audit_category) <= allowed


def test_items_are_independent(ground_truth, benchmark_cfg):
    """No two benchmark records share a ground-truth coordinate."""
    if not benchmark_cfg.get("one_record_per_coordinate"):
        pytest.skip("independence rule disabled in policy")
    decimals = benchmark_cfg["coordinate_equality_decimals"]
    points = list(
        zip(
            ground_truth.decimal_latitude.round(decimals),
            ground_truth.decimal_longitude.round(decimals),
        )
    )
    assert len(set(points)) == len(points)


def test_benchmark_and_dev_pool_are_disjoint(benchmark, dev_pool):
    benchmark_keys = set(benchmark.description.map(normalise))
    dev_keys = set(dev_pool.description.map(normalise))
    assert benchmark_keys & dev_keys == set()


def test_dev_pool_carries_no_coordinates(dev_pool):
    assert list(dev_pool.columns) == ["dev_id", "description", "is_dd_candidate"]


def test_recorded_hashes_match_the_files_on_disk(artefacts, paths_cfg):
    from georef.provenance.hashing import sha256_file

    root = repo_root()
    manifest = json.loads(
        (root / paths_cfg["benchmark_dir"] / "benchmark_manifest.json").read_text()
    )
    assert manifest["artefact_hashes"]["benchmark.csv"] == sha256_file(artefacts["benchmark"])
    assert manifest["artefact_hashes"]["ground_truth.csv"] == sha256_file(artefacts["ground_truth"])
    assert manifest["artefact_hashes"]["dev_pool.csv"] == sha256_file(artefacts["dev_pool"])

    stated = (root / paths_cfg["benchmark_dir"] / "benchmark_hash.txt").read_text().strip()
    assert stated == sha256_file(artefacts["benchmark"])


def test_audit_report_totals_are_consistent(paths_cfg):
    root = repo_root()
    report = json.loads(
        (root / paths_cfg["ground_truth_dir"] / "audit_report.json").read_text()
    )
    counts = report["audit_category_counts"]
    assert sum(counts.values()) == report["counts"]["distance_direction_candidates"]
    assert sum(report["selected_category_counts"].values()) == report["counts"]["benchmark_size"]


class TestDevGroundTruth:
    """The declared tuning set. Its whole value depends on being disjoint from
    the benchmark, so that is asserted rather than assumed."""

    @pytest.fixture(scope="class")
    def dev_ground_truth(self, paths_cfg):
        path = repo_root() / paths_cfg["dev_dir"] / "dev_ground_truth.csv"
        if not path.is_file():
            pytest.skip("dev ground truth not built")
        return pd.read_csv(path)

    def test_size_matches_policy(self, dev_ground_truth, benchmark_cfg):
        assert len(dev_ground_truth) == benchmark_cfg["dev_ground_truth_size"]

    def test_no_description_appears_in_the_benchmark(self, dev_ground_truth, benchmark):
        overlap = set(dev_ground_truth.description.map(normalise)) & set(
            benchmark.description.map(normalise)
        )
        assert overlap == set()

    def test_no_coordinate_is_shared_with_the_benchmark(
        self, dev_ground_truth, ground_truth, benchmark_cfg
    ):
        """Independence spans both sets: a shared coordinate would leak an
        answer from the tuning set into the evaluation set."""
        decimals = benchmark_cfg["coordinate_equality_decimals"]
        dev_points = set(
            zip(
                dev_ground_truth.decimal_latitude.round(decimals),
                dev_ground_truth.decimal_longitude.round(decimals),
            )
        )
        benchmark_points = set(
            zip(
                ground_truth.decimal_latitude.round(decimals),
                ground_truth.decimal_longitude.round(decimals),
            )
        )
        assert dev_points & benchmark_points == set()

    def test_ids_are_unique_and_well_formed(self, dev_ground_truth, benchmark_cfg):
        assert dev_ground_truth.dev_gt_id.is_unique
        prefix = benchmark_cfg["dev_id_prefix"]
        assert dev_ground_truth.dev_gt_id.str.match(rf"^{prefix}-\d{{3}}$").all()

    def test_categories_are_eligible(self, dev_ground_truth, benchmark_cfg):
        assert set(dev_ground_truth.audit_category) <= set(
            benchmark_cfg["eligible_audit_categories"]
        )

    def test_adding_it_did_not_disturb_the_benchmark(self, artefacts, paths_cfg):
        """The benchmark draw happens first, so its published hash must hold."""
        from georef.provenance.hashing import sha256_file

        stated = (
            (repo_root() / paths_cfg["benchmark_dir"] / "benchmark_hash.txt")
            .read_text()
            .strip()
        )
        assert stated == sha256_file(artefacts["benchmark"])
