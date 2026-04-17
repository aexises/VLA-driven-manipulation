from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SimpleVLA-RL"))

from verl.trainer.ppo.core_algos import compute_grpo_dense_advantage
from verl.trainer.ppo.rob_reward import build_binary_reward_tensor, combine_dense_rewards, reward_spec_from_config, validate_reward_support
try:
    from verl.trainer.ppo.ray_trainer import validate_training_config
except Exception:  # pragma: no cover - optional in lightweight test envs
    validate_training_config = None


def _config(
    task_suite_name: str = "libero_10",
    reward_type: str = "dense",
    tau_clip: float = 0.1,
    normalize_components: bool = False,
    n_samples: int = 4,
):
    return SimpleNamespace(
        data=SimpleNamespace(task_suite_name=task_suite_name, n_samples=n_samples),
        reward=SimpleNamespace(
            type=reward_type,
            impl="libero_native_dense" if reward_type != "binary" else "baseline_terminal",
            tau_clip=tau_clip,
            normalize_components=normalize_components,
            log_components=True,
            weights=SimpleNamespace(subgoal=0.0, progress=1.0, smoothness=0.05, terminal=1.0),
        ),
        actor_rollout_ref=SimpleNamespace(
            model=SimpleNamespace(action_token_len=1, action_chunks_len=4, vla="openvla-oft")
        ),
        trainer=SimpleNamespace(seed=0, allow_single_sample_grpo_debug=False),
        algorithm=SimpleNamespace(
            adv_estimator="grpo",
            gamma=0.99,
            grpo=SimpleNamespace(discounted_advantage=True),
        ),
    )


def test_binary_reward_marks_last_valid_token() -> None:
    complete = torch.tensor([True, False], dtype=torch.bool)
    valid_lengths = torch.tensor([3, 2], dtype=torch.int64)
    reward = build_binary_reward_tensor(complete, valid_lengths, torch.Size([2, 1, 4]), device=torch.device("cpu"))
    assert reward.shape == (2, 4)
    assert reward[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert reward[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_dense_reward_combination_and_clipping() -> None:
    config = _config(reward_type="clipped_dense", tau_clip=0.1)
    reward_spec = reward_spec_from_config(config)
    batch = {
        "responses": torch.zeros((1, 1, 4), dtype=torch.int64),
        "finish_step": torch.tensor([4], dtype=torch.int64),
        "dense_progress_scores": torch.tensor([[[0.0, -0.2, 0.5, 0.1]]], dtype=torch.float32),
        "dense_smoothness_scores": torch.tensor([[[0.0, -1.0, 0.0, 0.0]]], dtype=torch.float32),
        "dense_terminal_scores": torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32),
        "dense_subgoal_scores": torch.zeros((1, 1, 4), dtype=torch.float32),
    }
    reward_tensor_dict, metrics = combine_dense_rewards(batch, reward_spec)
    assert reward_tensor_dict["all"].shape == (1, 4)
    assert torch.all(reward_tensor_dict["all"] >= 0.1)
    assert metrics["clip_activation_ratio"] > 0.0


def test_dense_reward_metrics_split_padded_valid_and_sequence_sum() -> None:
    config = _config(reward_type="dense", tau_clip=0.0)
    reward_spec = reward_spec_from_config(config)
    batch = {
        "responses": torch.zeros((1, 1, 4), dtype=torch.int64),
        "finish_step": torch.tensor([2], dtype=torch.int64),
        "dense_progress_scores": torch.tensor([[[1.0, 3.0, 5.0, 7.0]]], dtype=torch.float32),
        "dense_smoothness_scores": torch.zeros((1, 1, 4), dtype=torch.float32),
        "dense_terminal_scores": torch.zeros((1, 1, 4), dtype=torch.float32),
        "dense_subgoal_scores": torch.zeros((1, 1, 4), dtype=torch.float32),
    }
    _, metrics = combine_dense_rewards(batch, reward_spec)
    assert metrics["reward_mean_padded"] == pytest.approx(4.0)
    assert metrics["reward_mean_valid"] == pytest.approx(2.0)
    assert metrics["reward_sum_sequence_mean"] == pytest.approx(4.0)
    assert metrics["component_progress_mean_padded"] == pytest.approx(4.0)
    assert metrics["component_progress_mean_valid"] == pytest.approx(2.0)
    assert metrics["component_progress_sum_sequence_mean"] == pytest.approx(4.0)
    assert metrics["valid_token_count_mean"] == pytest.approx(2.0)
    assert metrics["finish_step_mean"] == pytest.approx(2.0)


def test_dense_grpo_advantage_is_masked_and_finite() -> None:
    rewards = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.3, 0.1, 0.4, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    index = ["prompt-a", "prompt-a"]
    advantages, returns = compute_grpo_dense_advantage(rewards, mask, index=index, gamma=0.9)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape
    assert torch.isfinite(advantages).all()
    assert torch.equal(advantages[~mask], torch.zeros_like(advantages[~mask]))


def test_discounted_dense_grpo_advantage_changes_scale_but_stays_finite() -> None:
    rewards = torch.tensor([[0.1, 0.2, 0.0, 0.0], [0.3, 0.1, 0.4, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    index = ["prompt-a", "prompt-a"]
    advantages_plain, _ = compute_grpo_dense_advantage(
        rewards, mask, index=index, gamma=0.9, discounted_advantage=False
    )
    advantages_discounted, _ = compute_grpo_dense_advantage(
        rewards, mask, index=index, gamma=0.9, discounted_advantage=True
    )
    assert torch.isfinite(advantages_discounted).all()
    assert not torch.allclose(advantages_plain, advantages_discounted)
    assert torch.equal(advantages_discounted[~mask], torch.zeros_like(advantages_discounted[~mask]))


def test_robotwin_dense_rejected() -> None:
    config = _config(task_suite_name="robotwin2_lift_pot", reward_type="dense")
    try:
        validate_reward_support(config)
    except ValueError as exc:
        assert "Robotwin currently supports only reward.type=binary" in str(exc)
    else:
        raise AssertionError("Expected validate_reward_support to reject Robotwin dense reward.")


def test_normalize_components_flag_fails_fast() -> None:
    config = _config(reward_type="dense", normalize_components=True)
    with pytest.raises(ValueError, match="normalize_components=True is not implemented"):
        validate_reward_support(config)


def test_grpo_single_sample_requires_debug_override() -> None:
    if validate_training_config is None:
        pytest.skip("validate_training_config import unavailable in this environment")
    config = _config(reward_type="dense", n_samples=1)
    with pytest.raises(ValueError, match="data.n_samples >= 2"):
        validate_training_config(config)
