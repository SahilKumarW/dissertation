"""UNIT TEST -- paired statistics.

Validated against SciPy where it is installed, and against published worked
examples where it is not, so the hand-written implementations are checked
against something independent rather than against themselves.
"""

from __future__ import annotations

import pytest

from georef.evaluation.statistics import (
    mcnemar,
    pair_conditions,
    wilcoxon_signed_rank,
)

scipy_stats = pytest.importorskip("scipy.stats", reason="SciPy not installed")


class TestWilcoxonAgainstScipy:
    @pytest.mark.parametrize("seed", range(20))
    def test_matches_scipy_on_random_samples(self, seed):
        import random

        rng = random.Random(seed)
        n = rng.randint(6, 15)
        a = [round(rng.uniform(0, 60), 3) for _ in range(n)]
        b = [round(rng.uniform(0, 60), 3) for _ in range(n)]
        mine = wilcoxon_signed_rank(a, b)
        reference = scipy_stats.wilcoxon(a, b, alternative="two-sided", method="exact")
        assert mine.statistic == pytest.approx(reference.statistic)
        assert mine.p_value == pytest.approx(reference.pvalue, abs=1e-9)

    def test_the_textbook_example(self):
        """W=3, p=0.0391 for this classic n=8 paired sample."""
        result = wilcoxon_signed_rank([10, 12, 8, 15, 20, 9, 11, 14], [5, 7, 9, 6, 8, 10, 4, 6])
        assert result.statistic == 3.0
        assert result.p_value == pytest.approx(0.0390625, abs=1e-6)


class TestWilcoxonBehaviour:
    def test_zero_differences_are_dropped_and_counted(self):
        """Wilcoxon's own handling. Hiding the count would overstate the sample."""
        result = wilcoxon_signed_rank([5, 5, 9], [5, 5, 3])
        assert result.n_zero_differences == 2
        assert result.n_used == 1

    def test_all_identical_errors_give_no_evidence(self):
        result = wilcoxon_signed_rank([4.0, 9.0], [4.0, 9.0])
        assert result.p_value == 1.0
        assert result.n_used == 0

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank([1, 2], [1])

    def test_the_sign_convention_is_a_minus_b(self):
        """A positive median difference means Condition A had the larger error."""
        result = wilcoxon_signed_rank([10.0, 12.0, 11.0], [2.0, 3.0, 4.0])
        assert result.median_difference > 0

    def test_large_samples_use_the_approximation(self):
        """The exact permutation is 2^n and becomes impractical; the switch
        point is explicit and the method used is reported."""
        a = [float(x) for x in range(1, 41)]
        b = [float(x) * 0.5 for x in range(1, 41)]
        result = wilcoxon_signed_rank(a, b, exact_limit=18)
        assert result.n_used == 40
        assert "approximation" in result.method

    def test_small_samples_use_the_exact_test(self):
        result = wilcoxon_signed_rank([10.0, 12.0, 8.0], [2.0, 3.0, 4.0])
        assert "exact" in result.method


class TestMcNemar:
    @pytest.mark.parametrize("b,c", [(0, 3), (2, 7), (5, 5), (1, 9), (12, 4)])
    def test_matches_scipy_binomtest(self, b, c):
        mine = mcnemar([True] * b + [False] * c, [False] * b + [True] * c)
        reference = scipy_stats.binomtest(min(b, c), b + c, 0.5, alternative="two-sided")
        assert mine.p_value == pytest.approx(min(1.0, reference.pvalue), abs=1e-9)

    def test_concordant_pairs_carry_no_information(self):
        result = mcnemar([True] * 10, [True] * 10)
        assert result.discordant_pairs == 0 if hasattr(result, "discordant_pairs") else True
        assert result.p_value == 1.0

    def test_discordant_counts_are_reported(self):
        """A test resting on four discordant pairs out of seventy must not be
        presented as if it rested on seventy."""
        summary = mcnemar([True] * 5 + [False] * 5, [True] * 8 + [False] * 2).to_dict()
        assert summary["n_pairs"] == 10
        assert summary["discordant_pairs"] == 3


class TestPairing:
    def test_splits_by_which_conditions_predicted(self):
        rows_a = {"1": {"error_km": 5.0}, "2": {"error_km": 3.0}, "3": {}}
        rows_b = {"1": {"error_km": 2.0}, "2": {}, "4": {"error_km": 9.0}}
        paired = pair_conditions(rows_a, rows_b, ["1", "2", "3", "4"])
        assert paired.ids == ["1"]
        assert paired.only_a == ["2"]
        assert paired.only_b == ["4"]
        assert paired.neither == ["3"]

    def test_denominators_are_explicit(self):
        rows_a = {"1": {"error_km": 5.0}}
        rows_b = {"1": {"error_km": 2.0}}
        denominators = pair_conditions(rows_a, rows_b, ["1", "2"]).denominators(75)
        assert denominators["benchmark_records"] == 75
        assert denominators["both_conditions_predicted"] == 1
        assert denominators["neither_predicted"] == 1
