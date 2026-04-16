"""Simple statistical tests used for Phase 1 reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MannWhitneyResult:
    """Approximate two-sided Mann-Whitney U test result."""

    u_statistic: float
    p_value: float


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end + 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def mann_whitney_u(x: list[float], y: list[float]) -> MannWhitneyResult:
    """Compute a two-sided Mann-Whitney U test with a normal approximation."""

    sample_x = np.asarray(x, dtype=float)
    sample_y = np.asarray(y, dtype=float)
    if len(sample_x) == 0 or len(sample_y) == 0:
        raise ValueError("Both samples must be non-empty.")

    combined = np.concatenate([sample_x, sample_y])
    ranks = _average_ranks(combined)
    rank_x = np.sum(ranks[: len(sample_x)])
    u_x = rank_x - len(sample_x) * (len(sample_x) + 1) / 2.0
    u_y = len(sample_x) * len(sample_y) - u_x
    u_stat = min(u_x, u_y)

    mu_u = len(sample_x) * len(sample_y) / 2.0
    sigma_u = math.sqrt(len(sample_x) * len(sample_y) * (len(sample_x) + len(sample_y) + 1) / 12.0)
    if sigma_u == 0:
        return MannWhitneyResult(u_statistic=u_stat, p_value=1.0)
    z_score = (u_stat - mu_u) / sigma_u
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return MannWhitneyResult(u_statistic=u_stat, p_value=p_value)


def rank_biserial_effect_size(x: list[float], y: list[float]) -> float:
    """Compute rank-biserial correlation from the Mann-Whitney statistic."""

    result = mann_whitney_u(x, y)
    n_x = len(x)
    n_y = len(y)
    max_u = n_x * n_y
    return float(1.0 - 2.0 * result.u_statistic / max_u)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    corrected = [0.0] * len(p_values)
    total = len(p_values)
    running_max = 0.0
    for offset, (original_index, p_value) in enumerate(indexed):
        adjusted = (total - offset) * p_value
        running_max = max(running_max, adjusted)
        corrected[original_index] = min(1.0, running_max)
    return corrected
