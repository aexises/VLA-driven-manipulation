"""Default Phase 1 experiment settings."""

from __future__ import annotations

from .types import EnvConfig, ExperimentConfig, GRPOConfig, LoggingConfig, RewardConfig

PHASE1A_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
PHASE1B_ENVS = ["HalfCheetah-v4", "Ant-v4"]

PHASE1_SEEDS = [11, 22, 33, 44, 55]
CONTINUOUS_SUCCESS_THRESHOLDS = {
    "HalfCheetah-v4": 3.0,
    "Ant-v4": 1.0,
}


def default_grpo_config() -> GRPOConfig:
    """Return the frozen GRPO defaults for all reward conditions."""

    return GRPOConfig(
        gamma=0.99,
        group_size=32,
        clip_eps=0.2,
        beta_kl=0.01,
        learning_rate=3e-4,
        update_epochs=4,
        grad_clip=1.0,
        max_episode_steps=500,
        discounted_advantage=False,
        use_group_normalization=True,
    )


def make_phase1_experiment_config(
    env_name: str,
    reward_type: str = "binary",
    seed: int = 11,
    tau_clip: float = 0.0,
    total_iterations: int = 10,
) -> ExperimentConfig:
    """Build a single-run experiment config with project defaults."""

    phase = "phase1a" if env_name in PHASE1A_ENVS else "phase1b"
    reward = RewardConfig(
        reward_type=reward_type,
        tau_clip=tau_clip,
        success_threshold=CONTINUOUS_SUCCESS_THRESHOLDS.get(env_name),
    )
    tau_suffix = f"-tau{tau_clip:.2f}".replace(".", "p") if reward_type == "clipped_dense" else ""
    experiment_name = f"{phase}-{env_name.lower().replace('-', '_')}-{reward_type}{tau_suffix}"
    return ExperimentConfig(
        env=EnvConfig(name=env_name, phase=phase, seed=seed),
        reward=reward,
        grpo=default_grpo_config(),
        logging=LoggingConfig(experiment_name=experiment_name, run_id=f"seed{seed}"),
        seeds=list(PHASE1_SEEDS),
        total_iterations=total_iterations,
    )
