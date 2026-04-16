"""Clipped dense-reward GRPO convenience helpers."""

from __future__ import annotations

from src.config.defaults import make_phase1_experiment_config


def build_clipped_dense_config(env_name: str, seed: int = 11, tau_clip: float = 0.0):
    """Construct a clipped-dense Phase 1 experiment config."""

    return make_phase1_experiment_config(
        env_name=env_name,
        reward_type="clipped_dense",
        seed=seed,
        tau_clip=tau_clip,
    )
