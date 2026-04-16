"""Reward and advantage computations for GRPO.

This module implements the equations referenced in `hypothesis.md`.
"""

from __future__ import annotations

import numpy as np


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute Eq. (1) from `hypothesis.md`: R_{i,t} = Σ γ^{t'-t} r_{i,t'}."""

    returns = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def normalize_group_returns(
    trajectory_returns: list[np.ndarray],
    use_group_normalization: bool = True,
    epsilon: float = 1e-8,
) -> tuple[list[np.ndarray], float, float]:
    """Compute Eq. (2) from `hypothesis.md`: R̂ = (R - μ) / σ."""

    all_returns = np.concatenate(trajectory_returns, axis=0) if trajectory_returns else np.asarray([], dtype=np.float64)
    if not use_group_normalization or len(all_returns) == 0:
        return [np.array(values, copy=True) for values in trajectory_returns], 0.0, 1.0
    mean = float(np.mean(all_returns))
    std = float(np.std(all_returns))
    safe_std = std if std > epsilon else 1.0
    normalized = [(values - mean) / safe_std for values in trajectory_returns]
    return normalized, mean, safe_std


def compute_advantages(
    normalized_returns: np.ndarray,
    gamma: float,
    discounted_advantage: bool = False,
) -> np.ndarray:
    """Compute Eq. (3) from `hypothesis.md` with optional γ-discount ablation."""

    advantages = np.zeros_like(normalized_returns, dtype=np.float64)
    running = 0.0
    for index in range(len(normalized_returns) - 1, -1, -1):
        factor = gamma if discounted_advantage else 1.0
        running = float(normalized_returns[index]) + factor * running
        advantages[index] = running
    return advantages


def select_reward_track(reward_type: str, reward_result) -> np.ndarray:
    """Select the configured reward stream from an episode reward bundle."""

    if reward_type == "binary":
        return reward_result.binary_rewards
    if reward_type == "dense":
        return reward_result.dense_rewards
    if reward_type == "clipped_dense":
        return reward_result.clipped_dense_rewards
    raise KeyError(f"Unsupported reward type: {reward_type}")

