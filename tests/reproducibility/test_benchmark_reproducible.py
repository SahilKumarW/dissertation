"""REPRODUCIBILITY TEST -- a rebuild from the same inputs is byte-identical.

Slow: it re-reads the full coordinate corpus. Deselect with -m "not slow".
"""

import pytest

from georef.data.benchmark_builder import build_benchmark
from georef.provenance.hashing import sha256_file

pytestmark = pytest.mark.slow


def test_rebuild_reproduces_artefact_hashes(tmp_path, source_dir, artefacts, benchmark_cfg, paths_cfg):
    rebuilt = build_benchmark(source_dir, tmp_path, benchmark_cfg, paths_cfg)
    assert sha256_file(rebuilt.benchmark_path) == sha256_file(artefacts["benchmark"])
    assert sha256_file(rebuilt.ground_truth_path) == sha256_file(artefacts["ground_truth"])
    assert sha256_file(rebuilt.dev_pool_path) == sha256_file(artefacts["dev_pool"])


def test_a_different_seed_selects_a_different_sample(tmp_path, source_dir, benchmark_cfg, paths_cfg):
    """Guards against the seed being ignored, which would hide a selection bug."""
    altered = dict(benchmark_cfg)
    altered["random_seed"] = int(benchmark_cfg["random_seed"]) + 1
    rebuilt = build_benchmark(source_dir, tmp_path, altered, paths_cfg)
    import pandas as pd

    original = pd.read_csv(build_benchmark(source_dir, tmp_path / "base", benchmark_cfg, paths_cfg).benchmark_path)
    other = pd.read_csv(rebuilt.benchmark_path)
    assert set(original.description) != set(other.description)
