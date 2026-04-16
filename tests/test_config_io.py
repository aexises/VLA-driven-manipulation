from __future__ import annotations

from pathlib import Path

from src.config import load_experiment_config, make_phase1_experiment_config, save_experiment_config


def test_config_round_trip(tmp_path: Path) -> None:
    config = make_phase1_experiment_config("CartPole-v1", reward_type="dense", seed=22, tau_clip=0.2)
    target = tmp_path / "config.yaml"
    save_experiment_config(config, target)

    loaded = load_experiment_config(target)
    assert loaded.env.name == "CartPole-v1"
    assert loaded.reward.reward_type == "dense"
    assert loaded.reward.tau_clip == 0.2
    assert loaded.env.seed == 22

