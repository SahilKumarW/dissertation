"""Paired statistical comparison of the two conditions (master spec section 40).

Paired, because the same benchmark record is processed by both conditions and
the pairing is the whole design. Comparing unrelated samples would throw away
the control the experiment was built to have.

Wilcoxon signed-rank on the paired geodesic errors, McNemar on paired
threshold outcomes. No SciPy dependency: an exact Wilcoxon and an exact McNemar
are short to write and the sample is small enough that the exact forms are
preferable to normal approximations anyway.

Every function states its denominator. A test run over the pairs where BOTH
conditions produced a coordinate is a different quantity from one run over all
75 records, and reporting one while implying the other is the easiest way to
mislead with a true number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

__all__ = ["WilcoxonResult", "McNemarResult", "wilcoxon_signed_rank", "mcnemar", "PairedSet"]


@dataclass
class PairedSet:
    """Records where both conditions produced a coordinate."""

    ids: list[str] = field(default_factory=list)
    errors_a: list[float] = field(default_factory=list)
    errors_b: list[float] = field(default_factory=list)
    only_a: list[str] = field(default_factory=list)
    only_b: list[str] = field(default_factory=list)
    neither: list[str] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.ids)

    def denominators(self, total_records: int) -> dict:
        return {
            "benchmark_records": total_records,
            "both_conditions_predicted": self.n_pairs,
            "only_condition_a_predicted": len(self.only_a),
            "only_condition_b_predicted": len(self.only_b),
            "neither_predicted": len(self.neither),
        }


def pair_conditions(rows_a: dict, rows_b: dict, all_ids: list[str]) -> PairedSet:
    """Split records by which conditions produced a usable coordinate."""
    paired = PairedSet()
    for identifier in all_ids:
        a = (rows_a.get(identifier) or {}).get("error_km")
        b = (rows_b.get(identifier) or {}).get("error_km")
        if a is not None and b is not None:
            paired.ids.append(identifier)
            paired.errors_a.append(float(a))
            paired.errors_b.append(float(b))
        elif a is not None:
            paired.only_a.append(identifier)
        elif b is not None:
            paired.only_b.append(identifier)
        else:
            paired.neither.append(identifier)
    return paired


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    n_used: int
    n_zero_differences: int
    median_difference: float
    effect_size_r: float | None
    method: str

    def to_dict(self) -> dict:
        return {
            "test": "Wilcoxon signed-rank (paired, two-sided)",
            "statistic_W": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "n_used": self.n_used,
            "n_zero_differences_dropped": self.n_zero_differences,
            "median_difference_km": round(self.median_difference, 4),
            "effect_size_r": None if self.effect_size_r is None else round(self.effect_size_r, 4),
            "method": self.method,
        }


def _rank_with_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _exact_two_sided_p(w_plus: float, n: int) -> float:
    """Exact permutation p-value over all 2^n sign assignments."""
    total = 0
    at_least_as_extreme = 0
    ranks = list(range(1, n + 1))
    expected = n * (n + 1) / 4.0
    observed_deviation = abs(w_plus - expected)
    for size in range(n + 1):
        for subset in combinations(ranks, size):
            total += 1
            if abs(sum(subset) - expected) >= observed_deviation - 1e-9:
                at_least_as_extreme += 1
    return min(1.0, at_least_as_extreme / total)


def _normal_two_sided_p(w_plus: float, n: int) -> float:
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    return min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def wilcoxon_signed_rank(
    errors_a: list[float], errors_b: list[float], exact_limit: int = 18
) -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank on paired errors (A minus B).

    Zero differences are dropped, which is Wilcoxon's original handling, and the
    number dropped is reported rather than hidden: a large count means the two
    conditions agreed exactly and the test is answering a smaller question than
    the sample size suggests.
    """
    if len(errors_a) != len(errors_b):
        raise ValueError("paired samples must be the same length")

    differences = [a - b for a, b in zip(errors_a, errors_b)]
    non_zero = [d for d in differences if d != 0]
    zeros = len(differences) - len(non_zero)
    n = len(non_zero)

    if n == 0:
        return WilcoxonResult(0.0, 1.0, 0, zeros, 0.0, None, "no non-zero differences")

    ranks = _rank_with_ties([abs(d) for d in non_zero])
    w_plus = sum(rank for difference, rank in zip(non_zero, ranks) if difference > 0)
    w_minus = sum(rank for difference, rank in zip(non_zero, ranks) if difference < 0)
    statistic = min(w_plus, w_minus)

    if n <= exact_limit:
        p_value = _exact_two_sided_p(w_plus, n)
        method = f"exact permutation over 2^{n} sign assignments"
    else:
        p_value = _normal_two_sided_p(w_plus, n)
        method = "normal approximation with continuity correction"

    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )

    # r = Z / sqrt(N), the conventional effect size for this test.
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    effect = abs(w_plus - mean) / math.sqrt(variance) / math.sqrt(n) if variance > 0 else None

    return WilcoxonResult(statistic, p_value, n, zeros, median, effect, method)


@dataclass
class McNemarResult:
    b: int
    c: int
    p_value: float
    n_pairs: int
    method: str

    def to_dict(self) -> dict:
        return {
            "test": "McNemar (exact binomial, two-sided)",
            "a_success_b_failure": self.b,
            "a_failure_b_success": self.c,
            "discordant_pairs": self.b + self.c,
            "n_pairs": self.n_pairs,
            "p_value": round(self.p_value, 6),
            "method": self.method,
        }


def mcnemar(outcomes_a: list[bool], outcomes_b: list[bool]) -> McNemarResult:
    """Exact two-sided McNemar on paired binary outcomes.

    Only discordant pairs carry information. Concordant pairs are counted and
    reported, because a test resting on four discordant pairs out of seventy
    should not be presented as if it rested on seventy.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired samples must be the same length")

    b = sum(1 for a, x in zip(outcomes_a, outcomes_b) if a and not x)
    c = sum(1 for a, x in zip(outcomes_a, outcomes_b) if not a and x)
    n = b + c

    if n == 0:
        return McNemarResult(b, c, 1.0, len(outcomes_a), "no discordant pairs")

    # Exact binomial: P(X <= min(b,c)) * 2 under p = 0.5.
    smaller = min(b, c)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return McNemarResult(b, c, min(1.0, 2 * tail), len(outcomes_a), "exact binomial")
