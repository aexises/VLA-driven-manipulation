"""Dense-reward GRPO convenience helpers."""

from __future__ import annotations

from src.config.defaults import make_phase1_experiment_config


def build_dense_config(env_name: str, seed: int = 11):
    """Construct a dense-reward Phase 1 experiment config."""

    return make_phase1_experiment_config(env_name=env_name, reward_type="dense", seed=seed)

