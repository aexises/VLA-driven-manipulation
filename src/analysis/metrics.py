"""Metric computation for Phase 1 experiments."""

from __future__ import annotations

import numpy as np


def compute_sample_efficiency(success_rates: list[float], target_threshold: float = 0.8) -> int | None:
    """Return the first iteration reaching the target success rate."""

    for index, success_rate in enumerate(success_rates, start=1):
        if success_rate >= target_threshold:
            return index
    return None


def trajectory_smoothness(actions: list[np.ndarray]) -> float:
    """Compute E[||a_t - a_{t-1}||^2] for a sequence of actions."""

    if len(actions) < 2:
        return 0.0
    diffs = [np.square(np.asarray(curr) - np.asarray(prev)).mean() for prev, curr in zip(actions[:-1], actions[1:])]
    return float(np.mean(diffs))


def negative_reward_ratio(rewards: np.ndarray) -> float:
    """Compute the fraction of timesteps with negative reward."""

    if len(rewards) == 0:
        return 0.0
    return float(np.mean(np.asarray(rewards) < 0.0))

