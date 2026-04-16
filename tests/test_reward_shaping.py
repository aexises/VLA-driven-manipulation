from __future__ import annotations

import numpy as np

from src.algorithms.reward_shaping import compute_advantages, compute_discounted_returns, normalize_group_returns


def test_discounted_returns_matches_manual_example() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    returns = compute_discounted_returns(rewards, gamma=0.5)
    assert np.allclose(returns, np.asarray([2.75, 3.5, 3.0]))


def test_group_normalization_and_advantages() -> None:
    group_returns = [np.asarray([1.0, 2.0]), np.asarray([3.0])]
    normalized, mean, std = normalize_group_returns(group_returns, use_group_normalization=True)
    assert np.isclose(mean, 2.0)
    assert np.isclose(std, np.std([1.0, 2.0, 3.0]))
    advantages = compute_advantages(normalized[0], gamma=0.99, discounted_advantage=False)
    assert np.allclose(normalized[0], np.asarray([-1.22474487, 0.0]), atol=1e-6)
    assert np.allclose(advantages, np.asarray([-1.22474487, 0.0]), atol=1e-6)


def test_discounted_advantage_variant_changes_values() -> None:
    normalized_returns = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    plain = compute_advantages(normalized_returns, gamma=0.5, discounted_advantage=False)
    discounted = compute_advantages(normalized_returns, gamma=0.5, discounted_advantage=True)
    assert np.allclose(plain, np.asarray([3.0, 2.0, 1.0]))
    assert np.allclose(discounted, np.asarray([1.75, 1.5, 1.0]))
