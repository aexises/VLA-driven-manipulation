from __future__ import annotations

import numpy as np

from src.config.types import RewardConfig
from src.envs.classic.specs import Transition, get_classic_task_spec


def _reward_config(reward_type: str = "dense", tau_clip: float = 0.1, success_threshold: float | None = None) -> RewardConfig:
    return RewardConfig(reward_type=reward_type, tau_clip=tau_clip, success_threshold=success_threshold)


def test_cartpole_reward_protocol() -> None:
    spec = get_classic_task_spec("CartPole-v1")
    trajectory = [
        Transition(
            observation=np.asarray([0.0, 0.0, 0.0, 0.0]),
            action=1,
            next_observation=np.asarray([0.05, 0.0, 0.05, 0.0]),
            info={},
            terminated=False,
            truncated=False,
        )
    ]
    result = spec.compute_episode_rewards(trajectory, _reward_config())
    assert result.success is True
    assert result.binary_rewards[-1] == 1.0
    assert result.dense_rewards.shape == (1,)
    assert result.clipped_dense_rewards[-1] >= 0.1


def test_mountaincar_reward_protocol() -> None:
    spec = get_classic_task_spec("MountainCar-v0")
    trajectory = [
        Transition(
            observation=np.asarray([-0.6, 0.0]),
            action=2,
            next_observation=np.asarray([0.55, 0.02]),
            info={},
            terminated=True,
            truncated=False,
        )
    ]
    result = spec.compute_episode_rewards(trajectory, _reward_config())
    assert result.success is True
    assert result.binary_rewards[-1] == 1.0


def test_acrobot_reward_protocol() -> None:
    spec = get_classic_task_spec("Acrobot-v1")
    trajectory = [
        Transition(
            observation=np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            action=1,
            next_observation=np.asarray([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            info={},
            terminated=True,
            truncated=False,
        )
    ]
    result = spec.compute_episode_rewards(trajectory, _reward_config())
    assert result.success is True
    assert result.binary_rewards[-1] == 1.0


def test_halfcheetah_reward_protocol() -> None:
    spec = get_classic_task_spec("HalfCheetah-v4", success_threshold=3.0)
    trajectory = [
        Transition(
            observation=np.asarray([0.0, 0.0]),
            action=np.asarray([0.1, -0.1]),
            next_observation=np.asarray([0.0, 0.0]),
            info={"x_velocity": 3.2},
            terminated=False,
            truncated=False,
        ),
        Transition(
            observation=np.asarray([0.0, 0.0]),
            action=np.asarray([0.2, -0.2]),
            next_observation=np.asarray([0.0, 0.0]),
            info={"x_velocity": 3.3},
            terminated=False,
            truncated=True,
        ),
    ]
    result = spec.compute_episode_rewards(trajectory, _reward_config(success_threshold=3.0))
    assert result.success is True
    assert result.episode_mean_velocity is not None and result.episode_mean_velocity >= 3.0


def test_ant_reward_protocol() -> None:
    spec = get_classic_task_spec("Ant-v4", success_threshold=1.0)
    trajectory = [
        Transition(
            observation=np.asarray([0.75, 0.0]),
            action=np.asarray([0.1, 0.2]),
            next_observation=np.asarray([0.75, 0.0]),
            info={"x_velocity": 1.2, "z_position": 0.76},
            terminated=False,
            truncated=False,
        ),
        Transition(
            observation=np.asarray([0.75, 0.0]),
            action=np.asarray([0.1, 0.1]),
            next_observation=np.asarray([0.75, 0.0]),
            info={"x_velocity": 1.1, "z_position": 0.74},
            terminated=False,
            truncated=True,
        ),
    ]
    result = spec.compute_episode_rewards(trajectory, _reward_config(success_threshold=1.0))
    assert result.success is True
    assert result.binary_rewards[-1] == 1.0

