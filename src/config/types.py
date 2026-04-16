"""Typed experiment configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RewardType = Literal["binary", "dense", "clipped_dense"]


@dataclass(slots=True)
class RewardWeights:
    """Dense reward weights frozen before multi-seed runs."""

    main_progress: float = 1.0
    auxiliary_progress: float = 0.25
    smoothness: float = 0.05
    terminal_success: float = 1.0


@dataclass(slots=True)
class RewardConfig:
    """Reward shaping configuration."""

    reward_type: RewardType = "binary"
    weights: RewardWeights = field(default_factory=RewardWeights)
    tau_clip: float = 0.0
    clip_candidates: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.5])
    normalize_components: bool = True
    success_threshold: float | None = None


@dataclass(slots=True)
class EnvConfig:
    """Environment-specific configuration."""

    name: str
    phase: str
    seed: int = 11
    max_episode_steps: int | None = None
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])


@dataclass(slots=True)
class GRPOConfig:
    """GRPO/PPO-style training parameters."""

    gamma: float = 0.99
    group_size: int = 32
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    learning_rate: float = 3e-4
    update_epochs: int = 4
    grad_clip: float = 1.0
    max_episode_steps: int = 500
    discounted_advantage: bool = False
    use_group_normalization: bool = True
    entropy_coef: float = 0.0


@dataclass(slots=True)
class LoggingConfig:
    """Filesystem and logging behavior."""

    experiment_name: str
    run_id: str = "phase1"
    runs_dir: str = "runs"
    results_dir: str = "results"
    write_raw_trajectories: bool = False
    log_interval: int = 1


@dataclass(slots=True)
class ExperimentConfig:
    """Full single-run experiment configuration."""

    env: EnvConfig
    reward: RewardConfig
    grpo: GRPOConfig
    logging: LoggingConfig
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    total_iterations: int = 10
    episodes_per_evaluation: int = 5
    device: str = "cpu"
    hypothesis_ref: str = "hypothesis.md#Формулировка-тестируемой-гипотезы"

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        """Build a config from a nested mapping."""

        env = EnvConfig(**payload["env"])
        reward_payload = dict(payload["reward"])
        reward_weights = RewardWeights(**reward_payload.pop("weights", {}))
        reward = RewardConfig(weights=reward_weights, **reward_payload)
        grpo = GRPOConfig(**payload["grpo"])
        logging = LoggingConfig(**payload["logging"])
        return cls(
            env=env,
            reward=reward,
            grpo=grpo,
            logging=logging,
            seeds=list(payload.get("seeds", [11, 22, 33, 44, 55])),
            total_iterations=payload.get("total_iterations", 10),
            episodes_per_evaluation=payload.get("episodes_per_evaluation", 5),
            device=payload.get("device", "cpu"),
            hypothesis_ref=payload.get(
                "hypothesis_ref",
                "hypothesis.md#Формулировка-тестируемой-гипотезы",
            ),
        )

