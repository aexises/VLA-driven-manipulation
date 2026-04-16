from __future__ import annotations

import importlib.util

import pytest

from src.config.defaults import make_phase1_experiment_config


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("gymnasium") is None,
    reason="torch and gymnasium are required for trainer smoke tests",
)
def test_cartpole_smoke_training_runs(tmp_path) -> None:
    from src.algorithms.grpo_trainer import GRPOTrainer

    config = make_phase1_experiment_config("CartPole-v1", reward_type="dense", seed=11, tau_clip=0.0, total_iterations=1)
    config.grpo.group_size = 2
    config.grpo.update_epochs = 1
    config.grpo.max_episode_steps = 25
    config.logging.runs_dir = str(tmp_path / "runs")
    config.logging.results_dir = str(tmp_path / "results")
    summary = GRPOTrainer(config).run()
    assert "final_success_rate" in summary
