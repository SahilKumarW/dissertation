"""UNIT TEST -- Condition C combination rules.

Pure arithmetic over two prediction records. No model, no network.
"""

from __future__ import annotations

import pytest

from georef.pipeline.condition_c import combine, combine_record

WAIOURU = {"predicted_latitude": -39.4783, "predicted_longitude": 175.6689}
NEARBY = {"predicted_latitude": -39.4800, "predicted_longitude": 175.6700}
FAR = {"predicted_latitude": -43.8103, "predicted_longitude": 172.9679}
ABSENT = {"predicted_latitude": None, "predicted_longitude": None}


class TestFallback:
    def test_uses_the_hybrid_where_it_answered(self):
        result = combine_record("1", WAIOURU, FAR, strategy="fallback")
        assert result.source_condition == "B"

    def test_uses_the_baseline_where_the_hybrid_abstained(self):
        """Recovers full coverage without discarding any hybrid prediction."""
        result = combine_record("1", WAIOURU, ABSENT, strategy="fallback")
        assert result.source_condition == "A"
        assert result.latitude == WAIOURU["predicted_latitude"]

    def test_neither_answering_is_reported_not_invented(self):
        result = combine_record("1", ABSENT, ABSENT, strategy="fallback")
        assert result.status == "NO_PREDICTION"
        assert result.latitude is None


class TestAgreement:
    def test_agreement_takes_the_hybrid(self):
        """Two methods sharing no machinery arriving at the same place is
        evidence neither can produce alone."""
        result = combine_record("1", WAIOURU, NEARBY, strategy="agreement", agreement_km=10)
        assert result.source_condition == "B"
        assert result.agreement_km < 1.0

    def test_disagreement_falls_back_to_the_baseline(self):
        result = combine_record("1", WAIOURU, FAR, strategy="agreement", agreement_km=10)
        assert result.source_condition == "A"
        assert result.agreement_km > 400

    def test_the_threshold_is_honoured(self):
        close = combine_record("1", WAIOURU, NEARBY, strategy="agreement", agreement_km=0.01)
        assert close.source_condition == "A"

    def test_one_sided_answers_still_produce_a_prediction(self):
        assert combine_record("1", ABSENT, FAR, strategy="agreement").source_condition == "B"
        assert combine_record("1", WAIOURU, ABSENT, strategy="agreement").source_condition == "A"

    def test_agreement_uses_no_threshold_from_ground_truth(self):
        """The rule reads only the two predictions, which is what makes it the
        least contaminated of the four."""
        result = combine_record("1", WAIOURU, NEARBY, strategy="agreement")
        assert "agree" in result.reason


class TestConfidence:
    def test_a_confident_hybrid_is_used(self):
        row_b = {**NEARBY, "disambiguation_score": 0.8}
        result = combine_record("1", WAIOURU, row_b, strategy="confidence", minimum_score=0.6)
        assert result.source_condition == "B"

    def test_an_unconfident_hybrid_defers_to_the_baseline(self):
        row_b = {**NEARBY, "disambiguation_score": 0.4}
        result = combine_record("1", WAIOURU, row_b, strategy="confidence", minimum_score=0.6)
        assert result.source_condition == "A"

    def test_a_missing_score_is_treated_as_unconfident(self):
        result = combine_record("1", WAIOURU, NEARBY, strategy="confidence")
        assert result.source_condition == "A"


class TestControl:
    def test_prefer_baseline_is_the_control(self):
        """Whatever a combination achieves must beat this, or it achieves nothing."""
        result = combine_record("1", WAIOURU, NEARBY, strategy="prefer_baseline")
        assert result.source_condition == "A"

    def test_it_still_recovers_coverage_from_the_hybrid(self):
        result = combine_record("1", ABSENT, NEARBY, strategy="prefer_baseline")
        assert result.source_condition == "B"


class TestSafety:
    def test_an_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown combination strategy"):
            combine_record("1", WAIOURU, NEARBY, strategy="whatever")

    def test_no_rule_invents_a_coordinate(self):
        """Every output must be one of the two inputs, unchanged."""
        for strategy in ("fallback", "agreement", "confidence", "prefer_baseline"):
            result = combine_record("1", WAIOURU, FAR, strategy=strategy)
            assert (result.latitude, result.longitude) in {
                (WAIOURU["predicted_latitude"], WAIOURU["predicted_longitude"]),
                (FAR["predicted_latitude"], FAR["predicted_longitude"]),
            }

    def test_every_record_records_where_it_came_from(self):
        combined = combine(
            {"1": WAIOURU, "2": ABSENT}, {"1": NEARBY, "2": FAR}, ["1", "2"], strategy="fallback"
        )
        assert {r.source_condition for r in combined.values()} == {"B"}
        assert all(r.reason for r in combined.values())


class TestTheFreezeCoversTheRule:
    """A selected rule that the freeze does not hash is not frozen. Caught when
    the v2 freeze reported 8 configs against 9 present."""

    def test_condition_c_config_is_hashed(self):
        from georef.provenance.freeze import CONFIG_FILES

        assert "condition_c.yaml" in CONFIG_FILES

    def test_every_config_file_present_is_hashed(self):
        from georef.config.loader import repo_root
        from georef.provenance.freeze import CONFIG_FILES

        present = {p.name for p in (repo_root() / "configs").glob("*.yaml")}
        assert present <= set(CONFIG_FILES), present - set(CONFIG_FILES)

    def test_the_held_out_set_is_hashed(self):
        """So a validation run can be matched to the data it read."""
        from georef.config.loader import repo_root
        from georef.provenance.freeze import build_freeze_manifest

        manifest = build_freeze_manifest(repo_root())
        assert any("validation" in key for key in manifest["data_hashes"])


class TestFreezeIsPlatformIndependent:
    """A manifest written on Linux and verified on Windows reported every source
    file as removed and re-added, because `str(Path)` yields backslashes there."""

    def test_code_hash_keys_use_posix_separators(self):
        from georef.config.loader import repo_root
        from georef.provenance.freeze import build_freeze_manifest

        keys = build_freeze_manifest(repo_root())["code_hashes"]
        assert keys, "no source files hashed"
        assert not any("\\" in key for key in keys)
        assert any("/" in key for key in keys)

    def test_verification_against_its_own_manifest_is_clean(self):
        from georef.config.loader import repo_root
        from georef.provenance.freeze import build_freeze_manifest, verify_against_freeze

        manifest = build_freeze_manifest(repo_root())
        assert verify_against_freeze(repo_root(), manifest) == []
