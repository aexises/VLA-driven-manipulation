"""Generic GRPO trainer for binary, dense, and clipped-dense rewards."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.algorithms.reward_shaping import (
    compute_advantages,
    compute_discounted_returns,
    normalize_group_returns,
    select_reward_track,
)
from src.analysis.logging import ExperimentLogger
from src.config.types import ExperimentConfig
from src.envs.classic import Transition, get_classic_task_spec, make_env
from src.models.policies import DiscreteMLPPolicy, GaussianMLPPolicy

try:
    import torch
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    torch = None


@dataclass(slots=True)
class EpisodeSample:
    """A trajectory collected from the environment."""

    observations: list[np.ndarray]
    actions: list[np.ndarray | int | float]
    old_log_probs: list[float]
    transitions: list[Transition]
    rewards: np.ndarray
    success: bool
    negative_reward_ratio: float
    trajectory_smoothness: float
    episode_mean_velocity: float | None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "torch is required to run the GRPO trainer. Install project dependencies "
            "from pyproject.toml before running experiments."
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


class GRPOTrainer:
    """Single trainer shared across all reward types."""

    def __init__(self, config: ExperimentConfig) -> None:
        _require_torch()
        self.config = config
        self.env = make_env(config.env.name, seed=config.env.seed)
        self.task_spec = get_classic_task_spec(
            config.env.name,
            success_threshold=config.reward.success_threshold,
        )
        self.logger = ExperimentLogger(config)
        self.device = torch.device(config.device)
        _seed_everything(config.env.seed)
        self.policy = self._build_policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.grpo.learning_rate)

    def _build_policy(self):
        observation_shape = self.env.observation_space.shape
        if observation_shape is None:
            raise ValueError("Unsupported observation space without shape.")
        obs_dim = int(np.prod(observation_shape))
        hidden_sizes = self.config.env.hidden_sizes
        if hasattr(self.env.action_space, "n"):
            return DiscreteMLPPolicy(obs_dim, self.env.action_space.n, hidden_sizes)
        if hasattr(self.env.action_space, "shape"):
            act_dim = int(np.prod(self.env.action_space.shape))
            return GaussianMLPPolicy(obs_dim, act_dim, hidden_sizes)
        raise ValueError(f"Unsupported action space for {self.config.env.name}")

    def collect_episode(self) -> EpisodeSample:
        """Roll out one episode and transform rewards using the configured protocol."""

        observation, _ = self.env.reset(seed=self.config.env.seed)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray | int | float] = []
        old_log_probs: list[float] = []
        transitions: list[Transition] = []
        step_limit = self.config.env.max_episode_steps or self.config.grpo.max_episode_steps
        for _ in range(step_limit):
            action, log_prob = self.policy.act(observation)
            env_action = int(action) if hasattr(self.env.action_space, "n") else action
            next_observation, _, terminated, truncated, info = self.env.step(env_action)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(env_action)
            old_log_probs.append(log_prob)
            transitions.append(
                Transition(
                    observation=np.asarray(observation, dtype=np.float32),
                    action=env_action,
                    next_observation=np.asarray(next_observation, dtype=np.float32),
                    info=dict(info),
                    terminated=terminated,
                    truncated=truncated,
                )
            )
            observation = next_observation
            if terminated or truncated:
                break

        reward_result = self.task_spec.compute_episode_rewards(transitions, self.config.reward)
        rewards = select_reward_track(self.config.reward.reward_type, reward_result)
        return EpisodeSample(
            observations=observations,
            actions=actions,
            old_log_probs=old_log_probs,
            transitions=transitions,
            rewards=rewards,
            success=reward_result.success,
            negative_reward_ratio=reward_result.negative_reward_ratio,
            trajectory_smoothness=reward_result.trajectory_smoothness,
            episode_mean_velocity=reward_result.episode_mean_velocity,
        )

    def _batch_tensors(self, episodes: list[EpisodeSample], advantages: list[np.ndarray]):
        obs_array = np.concatenate([np.asarray(episode.observations) for episode in episodes], axis=0)
        action_array = np.concatenate(
            [
                np.asarray(episode.actions if hasattr(self.env.action_space, "n") else np.vstack(episode.actions))
                for episode in episodes
            ],
            axis=0,
        )
        old_log_probs = np.concatenate([np.asarray(episode.old_log_probs, dtype=np.float32) for episode in episodes], axis=0)
        advantage_array = np.concatenate(advantages, axis=0).astype(np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        action_tensor = self.policy.action_tensor(action_array).to(self.device)
        old_log_prob_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantage_array, dtype=torch.float32, device=self.device)
        return obs_tensor, action_tensor, old_log_prob_tensor, advantage_tensor

    def _train_on_group(self, episodes: list[EpisodeSample]) -> dict[str, float]:
        group_returns = [
            compute_discounted_returns(episode.rewards, gamma=self.config.grpo.gamma)
            for episode in episodes
        ]
        normalized_returns, norm_mean, norm_std = normalize_group_returns(
            group_returns,
            use_group_normalization=self.config.grpo.use_group_normalization,
        )
        advantages = [
            compute_advantages(
                returns,
                gamma=self.config.grpo.gamma,
                discounted_advantage=self.config.grpo.discounted_advantage,
            )
            for returns in normalized_returns
        ]
        obs_tensor, action_tensor, old_log_prob_tensor, advantage_tensor = self._batch_tensors(episodes, advantages)
        reference_policy = copy.deepcopy(self.policy).to(self.device)

        final_policy_loss = 0.0
        final_kl_loss = 0.0
        grad_norm = 0.0
        for _ in range(self.config.grpo.update_epochs):
            log_probs, entropy = self.policy.evaluate_actions(obs_tensor, action_tensor)
            ratio = torch.exp(log_probs - old_log_prob_tensor)
            unclipped = ratio * advantage_tensor
            clipped = torch.clamp(ratio, 1.0 - self.config.grpo.clip_eps, 1.0 + self.config.grpo.clip_eps) * advantage_tensor
            surrogate = torch.minimum(unclipped, clipped)
            kl_loss = self.policy.kl_to(reference_policy, obs_tensor).mean()
            entropy_bonus = entropy.mean()
            policy_loss = -surrogate.mean()
            total_loss = policy_loss + self.config.grpo.beta_kl * kl_loss - self.config.grpo.entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grpo.grad_clip).item())
            self.optimizer.step()

            final_policy_loss = float(policy_loss.item())
            final_kl_loss = float(kl_loss.item())

        reward_values = np.concatenate([episode.rewards for episode in episodes], axis=0)
        success_rate = float(np.mean([episode.success for episode in episodes]))
        mean_episode_length = float(np.mean([len(episode.transitions) for episode in episodes]))
        cumulative_rewards = [float(np.sum(episode.rewards)) for episode in episodes]
        flattened_advantages = np.concatenate(advantages, axis=0)
        return {
            "reward_step_mean": float(np.mean(reward_values)) if len(reward_values) else 0.0,
            "reward_episode_cumulative": float(np.mean(cumulative_rewards)) if cumulative_rewards else 0.0,
            "reward_negative_ratio": float(np.mean([episode.negative_reward_ratio for episode in episodes])),
            "loss_policy": final_policy_loss,
            "loss_kl": final_kl_loss,
            "loss_total": final_policy_loss + self.config.grpo.beta_kl * final_kl_loss,
            "success_rate": success_rate,
            "episode_length": mean_episode_length,
            "grad_norm": grad_norm,
            "advantage_mean": float(np.mean(flattened_advantages)) if len(flattened_advantages) else 0.0,
            "advantage_std": float(np.std(flattened_advantages)) if len(flattened_advantages) else 0.0,
            "return_norm_mean": norm_mean,
            "return_norm_std": norm_std,
            "trajectory_smoothness": float(np.mean([episode.trajectory_smoothness for episode in episodes])),
        }

    def run(self) -> dict[str, Any]:
        """Train for the configured number of iterations and persist run outputs."""

        iteration_metrics: list[dict[str, float]] = []
        raw_trajectories: list[dict[str, Any]] = []
        for iteration in range(1, self.config.total_iterations + 1):
            episodes = [self.collect_episode() for _ in range(self.config.grpo.group_size)]
            metrics = self._train_on_group(episodes)
            self.logger.log_iteration(iteration, metrics)
            iteration_metrics.append(metrics)
            raw_trajectories.append(
                {
                    "iteration": iteration,
                    "successes": [episode.success for episode in episodes],
                    "cumulative_rewards": [float(np.sum(episode.rewards)) for episode in episodes],
                    "episode_lengths": [len(episode.transitions) for episode in episodes],
                }
            )

        summary = {
            "final_success_rate": iteration_metrics[-1]["success_rate"] if iteration_metrics else 0.0,
            "best_success_rate": max(metric["success_rate"] for metric in iteration_metrics) if iteration_metrics else 0.0,
            "mean_cumulative_reward": float(np.mean([metric["reward_episode_cumulative"] for metric in iteration_metrics]))
            if iteration_metrics
            else 0.0,
            "mean_gradient_variance_proxy": float(np.mean([metric["advantage_std"] for metric in iteration_metrics]))
            if iteration_metrics
            else 0.0,
        }
        self.logger.finalize(self.config, summary=summary, raw_trajectories=raw_trajectories)
        return summary

