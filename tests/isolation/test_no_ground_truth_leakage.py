"""ISOLATION TEST -- prediction code must not be able to reach ground truth.

The description corpus was drawn from the coordinate CSV, so that CSV holds the
answer to every benchmark and dev record. Exactly one module is permitted to
read it: the benchmark builder, which runs before any prediction exists.

This test is deliberately written to keep working as the repository grows: it
scans every module under src/georef and fails on any new prediction-side file
that references the evaluation-only assets.
"""

from pathlib import Path

import pytest

from georef.config.loader import load_paths_config, repo_root

# Modules allowed to reference the evaluation-only assets.
ALLOWED = {
    "georef/data/benchmark_builder.py",
    "georef/evaluation",  # prefix: evaluation runs strictly after prediction
}

FORBIDDEN_TOKENS = (
    "ground_truth",
    "MWLRDataCHR",
    "decimalLatitude",
    "decimalLongitude",
    "verbatimLocality",
    "Jones",
)


def _is_allowed(relative: str) -> bool:
    return any(relative == entry or relative.startswith(entry + "/") for entry in ALLOWED)


def python_modules():
    source_root = repo_root() / "src"
    return sorted(source_root.rglob("*.py"))


@pytest.mark.parametrize("path", python_modules(), ids=lambda p: str(p))
def test_module_does_not_reference_evaluation_assets(path: Path):
    relative = path.relative_to(repo_root() / "src").as_posix()
    if _is_allowed(relative):
        pytest.skip(f"{relative} is an allowed evaluation-side module")
    text = path.read_text(encoding="utf-8")
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"{relative} references evaluation-only asset {token!r}"


def test_paths_config_marks_the_coordinate_file_as_evaluation_only():
    text = (repo_root() / "configs" / "paths.yaml").read_text(encoding="utf-8")
    assert "EVALUATION-ONLY" in text


def test_ground_truth_lives_outside_the_benchmark_directory():
    paths_cfg = load_paths_config()
    root = repo_root()
    benchmark_dir = (root / paths_cfg["benchmark_dir"]).resolve()
    ground_truth_dir = (root / paths_cfg["ground_truth_dir"]).resolve()
    assert benchmark_dir != ground_truth_dir
    assert not str(ground_truth_dir).startswith(str(benchmark_dir) + "/")
    for path in benchmark_dir.glob("*"):
        assert "ground_truth" not in path.name
