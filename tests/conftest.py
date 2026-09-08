"""Shared fixtures. Data-dependent tests skip cleanly when the supplier files
are absent, so the suite still runs on a fresh clone."""

from pathlib import Path

import pytest

from georef.config.loader import load_benchmark_config, load_paths_config, repo_root

ROOT = repo_root()


@pytest.fixture(scope="session")
def paths_cfg():
    return load_paths_config()


@pytest.fixture(scope="session")
def benchmark_cfg():
    return load_benchmark_config()


@pytest.fixture(scope="session")
def source_dir(paths_cfg):
    directory = ROOT / paths_cfg["source_dir"]
    for name in (paths_cfg["descriptions_file"], paths_cfg["coordinates_file"]):
        if not (directory / name).is_file():
            pytest.skip(f"supplier file not present: {name}")
    return directory


@pytest.fixture(scope="session")
def artefacts(paths_cfg):
    benchmark = ROOT / paths_cfg["benchmark_dir"] / "benchmark.csv"
    ground_truth = ROOT / paths_cfg["ground_truth_dir"] / "ground_truth.csv"
    dev_pool = ROOT / paths_cfg["dev_dir"] / "dev_pool.csv"
    for path in (benchmark, ground_truth, dev_pool):
        if not path.is_file():
            pytest.skip("artefacts not built; run scripts/build_benchmark.py first")
    return {"benchmark": benchmark, "ground_truth": ground_truth, "dev_pool": dev_pool}
