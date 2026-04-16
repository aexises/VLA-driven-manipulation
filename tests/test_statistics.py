from __future__ import annotations

from src.analysis.statistics import holm_bonferroni, mann_whitney_u, rank_biserial_effect_size


def test_mann_whitney_prefers_larger_sample() -> None:
    result = mann_whitney_u([5, 6, 7], [1, 2, 3])
    assert result.p_value < 0.1
    assert result.u_statistic >= 0.0


def test_holm_bonferroni_is_monotonic() -> None:
    corrected = holm_bonferroni([0.01, 0.02, 0.5])
    assert corrected[0] <= corrected[1] <= corrected[2]


def test_rank_biserial_positive_for_stronger_sample() -> None:
    effect = rank_biserial_effect_size([4, 5, 6], [1, 2, 3])
    assert effect > 0.0

