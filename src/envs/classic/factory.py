"""Gymnasium environment factory for classic RL tasks."""

from __future__ import annotations

from typing import Any

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    gym = None


def make_env(env_name: str, seed: int | None = None) -> Any:
    """Create a Gymnasium environment and seed it."""

    if gym is None:
        raise ImportError(
            "gymnasium is required to create environments. Install project dependencies "
            "from pyproject.toml before running training."
        )
    env = gym.make(env_name)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env

