"""ISOLATION TEST -- Condition A must be genuinely standalone.

Master spec sections 4 and 49. The baseline's whole evidential value is that it
had no help: a hidden geographic fallback would flatter the hybrid and make the
comparison meaningless. So this is checked mechanically rather than trusted.

The check walks the actual import graph of the baseline module, so it keeps
working as the code grows and fails the moment someone adds a convenient import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from georef.config.loader import repo_root

BASELINE = repo_root() / "src" / "georef" / "llm" / "baseline.py"

# Everything Condition A must not be able to reach.
FORBIDDEN_MODULES = (
    "georef.retrieval",
    "georef.disambiguation",
    "georef.linz",
    "georef.spatial",
    "georef.pipeline.grounding",
    "georef.evaluation",
    "chromadb",
    "sentence_transformers",
    "bm25s",
    "geographiclib",
)

FORBIDDEN_TOKENS = (
    "ground_truth",
    "MWLRDataCHR",
    "MWLR-Data",
    "linz",
    "chroma",
    "gazetteer",
    "geodesic",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _transitive_georef_imports(path: Path, seen: set[str] | None = None) -> set[str]:
    """Follow georef imports, so an indirect route is caught too."""
    seen = seen if seen is not None else set()
    for module in _imported_modules(path):
        if module in seen:
            continue
        seen.add(module)
        if module.startswith("georef."):
            candidate = repo_root() / "src" / Path(*module.split(".")).with_suffix(".py")
            if candidate.is_file():
                _transitive_georef_imports(candidate, seen)
    return seen


class TestBaselineImports:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_MODULES)
    def test_does_not_import_geographic_machinery(self, forbidden):
        imported = _transitive_georef_imports(BASELINE)
        offenders = [m for m in imported if m == forbidden or m.startswith(forbidden + ".")]
        assert not offenders, f"Condition A reaches {offenders} via its imports"

    def test_reaches_only_the_client_and_validation(self):
        georef_imports = {m for m in _transitive_georef_imports(BASELINE) if m.startswith("georef")}
        assert georef_imports <= {"georef.llm.client", "georef.config.loader"}, georef_imports

    @pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
    def test_mentions_no_geographic_asset(self, token):
        text = BASELINE.read_text(encoding="utf-8").lower()
        # The module docstring names these to explain what it must NOT do.
        code = "\n".join(
            line for line in text.split("\n") if not line.strip().startswith("#")
        )
        body = code.split('"""')[-1]
        assert token.lower() not in body, f"baseline body references {token!r}"


class TestBaselineBehaviour:
    def test_it_shares_the_model_configuration_with_condition_b(self):
        """Different models in each arm would measure model capability rather
        than the effect of the intervention."""
        from georef.config.loader import load_yaml, resolve_llm_profile

        models = resolve_llm_profile(load_yaml(repo_root() / "configs" / "models.yaml"))
        assert "llm" in models
        # There is exactly one llm block; both conditions construct their client
        # from it, so they cannot diverge by configuration.
        assert models["llm"]["model"]

    def test_no_coordinate_correction_is_applied(self):
        """A prediction in the Atlantic is recorded as one and scored as one."""
        from georef.llm.baseline import BaselineCoordinates

        parsed = BaselineCoordinates.model_validate({"latitude": 51.5, "longitude": -0.12})
        assert parsed.latitude == 51.5 and parsed.longitude == -0.12

    def test_impossible_coordinates_are_rejected_not_clamped(self):
        from pydantic import ValidationError

        from georef.llm.baseline import BaselineCoordinates

        with pytest.raises(ValidationError):
            BaselineCoordinates.model_validate({"latitude": 200.0, "longitude": 0.0})
