"""Phase 1 reward definitions for classic RL environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config.types import RewardConfig


def _clip_unit(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


@dataclass(slots=True)
class Transition:
    """Environment transition stored during rollout."""

    observation: np.ndarray
    action: np.ndarray | int | float
    next_observation: np.ndarray
    info: dict[str, Any]
    terminated: bool
    truncated: bool


@dataclass(slots=True)
class RewardComponents:
    """Normalized dense reward components."""

    main_progress: float
    auxiliary_progress: float
    smoothness: float
    terminal_success: float = 0.0


@dataclass(slots=True)
class EpisodeRewardResult:
    """Reward outputs for one trajectory under all reward variants."""

    binary_rewards: np.ndarray
    dense_rewards: np.ndarray
    clipped_dense_rewards: np.ndarray
    success: bool
    components: list[RewardComponents]
    negative_reward_ratio: float
    trajectory_smoothness: float
    episode_mean_velocity: float | None = None


class ClassicTaskSpec:
    """Environment-specific dense reward and binary success protocol."""

    def __init__(self, env_name: str, success_threshold: float | None = None) -> None:
        self.env_name = env_name
        self.success_threshold = success_threshold

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        raise NotImplementedError

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        raise NotImplementedError

    def compute_episode_rewards(
        self,
        trajectory: list[Transition],
        reward_config: RewardConfig,
    ) -> EpisodeRewardResult:
        """Compose binary, dense, and clipped-dense rewards for a trajectory."""

        weights = reward_config.weights
        components: list[RewardComponents] = []
        previous_action: np.ndarray | int | float | None = None
        for transition in trajectory:
            component = self.step_components(transition, previous_action)
            components.append(component)
            previous_action = transition.action

        success, episode_mean_velocity = self.compute_success(trajectory)
        success = bool(success)
        if components:
            components[-1].terminal_success = 1.0 if success else 0.0

        dense_rewards = np.asarray(
            [
                weights.main_progress * comp.main_progress
                + weights.auxiliary_progress * comp.auxiliary_progress
                + weights.smoothness * comp.smoothness
                + weights.terminal_success * comp.terminal_success
                for comp in components
            ],
            dtype=np.float64,
        )
        clipped_dense_rewards = np.maximum(dense_rewards, reward_config.tau_clip)
        binary_rewards = np.zeros(len(trajectory), dtype=np.float64)
        if len(binary_rewards) > 0:
            binary_rewards[-1] = float(success)

        smoothness_values = [comp.smoothness for comp in components]
        smoothness = float(-np.mean(smoothness_values)) if smoothness_values else 0.0
        negative_ratio = float(np.mean(dense_rewards < 0.0)) if len(dense_rewards) else 0.0
        return EpisodeRewardResult(
            binary_rewards=binary_rewards,
            dense_rewards=dense_rewards,
            clipped_dense_rewards=clipped_dense_rewards,
            success=success,
            components=components,
            negative_reward_ratio=negative_ratio,
            trajectory_smoothness=smoothness,
            episode_mean_velocity=episode_mean_velocity,
        )


class CartPoleSpec(ClassicTaskSpec):
    """CartPole reward shaping from agents.md Phase 1 table."""

    theta_limit = np.deg2rad(12.0)
    x_limit = 2.4

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        x, _, theta, _ = transition.next_observation
        alive_bonus = 1.0 if not transition.terminated else 0.0
        angle_term = -abs(theta) / self.theta_limit
        position_term = -abs(x) / self.x_limit
        return RewardComponents(
            main_progress=_clip_unit(alive_bonus + angle_term),
            auxiliary_progress=_clip_unit(position_term),
            smoothness=0.0,
        )

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        if not trajectory:
            return False, None
        final_obs = trajectory[-1].next_observation
        x, _, theta, _ = final_obs
        success = abs(theta) < self.theta_limit and abs(x) < self.x_limit
        return success, None


class MountainCarSpec(ClassicTaskSpec):
    """MountainCar reward shaping from agents.md Phase 1 table."""

    min_position = -1.2
    goal_position = 0.5
    max_speed = 0.07

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        position, velocity = transition.next_observation
        position_score = ((position - self.min_position) / (self.goal_position - self.min_position)) * 2 - 1
        velocity_score = velocity / self.max_speed
        return RewardComponents(
            main_progress=_clip_unit(position_score),
            auxiliary_progress=_clip_unit(velocity_score),
            smoothness=0.0,
        )

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        if not trajectory:
            return False, None
        position, _ = trajectory[-1].next_observation
        return position >= self.goal_position, None


class AcrobotSpec(ClassicTaskSpec):
    """Acrobot reward shaping from agents.md Phase 1 table."""

    target_height = 1.0

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        cos1, sin1, cos2, sin2, _, _ = transition.next_observation
        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)
        tip_height = -np.cos(theta1) - np.cos(theta1 + theta2)
        height_progress = tip_height / 2.0
        angle_distance = (abs(_wrap_angle(theta1 - np.pi)) + abs(_wrap_angle(theta2))) / (2 * np.pi)
        return RewardComponents(
            main_progress=_clip_unit(height_progress),
            auxiliary_progress=_clip_unit(-angle_distance),
            smoothness=0.0,
        )

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        if not trajectory:
            return False, None
        cos1, sin1, cos2, sin2, _, _ = trajectory[-1].next_observation
        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)
        tip_height = -np.cos(theta1) - np.cos(theta1 + theta2)
        return tip_height >= self.target_height, None


class HalfCheetahSpec(ClassicTaskSpec):
    """HalfCheetah reward shaping from agents.md Phase 1 table."""

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        velocity = float(transition.info.get("x_velocity", transition.next_observation[0]))
        if previous_action is None:
            smoothness = 0.0
        else:
            previous = np.asarray(previous_action, dtype=np.float64)
            current = np.asarray(transition.action, dtype=np.float64)
            smoothness = -float(np.mean(np.square(current - previous)))
        threshold = self.success_threshold or 3.0
        return RewardComponents(
            main_progress=_clip_unit(velocity / threshold),
            auxiliary_progress=0.0,
            smoothness=_clip_unit(smoothness),
        )

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        velocities = [float(step.info.get("x_velocity", step.next_observation[0])) for step in trajectory]
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        threshold = self.success_threshold or 3.0
        return mean_velocity >= threshold, mean_velocity


class AntSpec(ClassicTaskSpec):
    """Ant reward shaping from agents.md Phase 1 table."""

    healthy_z = 0.75

    def step_components(
        self,
        transition: Transition,
        previous_action: np.ndarray | int | float | None,
    ) -> RewardComponents:
        velocity = float(transition.info.get("x_velocity", transition.next_observation[0]))
        z_position = float(transition.info.get("z_position", transition.next_observation[0]))
        if previous_action is None:
            smoothness = 0.0
        else:
            previous = np.asarray(previous_action, dtype=np.float64)
            current = np.asarray(transition.action, dtype=np.float64)
            smoothness = -float(np.mean(np.square(current - previous)))
        threshold = self.success_threshold or 1.0
        stability = -abs(z_position - self.healthy_z) / self.healthy_z
        return RewardComponents(
            main_progress=_clip_unit(velocity / threshold),
            auxiliary_progress=_clip_unit(stability),
            smoothness=_clip_unit(smoothness),
        )

    def compute_success(self, trajectory: list[Transition]) -> tuple[bool, float | None]:
        velocities = [float(step.info.get("x_velocity", step.next_observation[0])) for step in trajectory]
        alive_flags = [not step.terminated for step in trajectory]
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        threshold = self.success_threshold or 1.0
        success = mean_velocity >= threshold and all(alive_flags)
        return success, mean_velocity


def get_classic_task_spec(env_name: str, success_threshold: float | None = None) -> ClassicTaskSpec:
    """Return the reward specification for a supported Phase 1 environment."""

    specs: dict[str, type[ClassicTaskSpec]] = {
        "CartPole-v1": CartPoleSpec,
        "MountainCar-v0": MountainCarSpec,
        "Acrobot-v1": AcrobotSpec,
        "HalfCheetah-v4": HalfCheetahSpec,
        "Ant-v4": AntSpec,
    }
    if env_name not in specs:
        raise KeyError(f"Unsupported Phase 1 environment: {env_name}")
    return specs[env_name](env_name=env_name, success_threshold=success_threshold)
