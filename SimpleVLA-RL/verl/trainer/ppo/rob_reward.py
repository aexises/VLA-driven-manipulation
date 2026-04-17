"""Reward helpers for SimpleVLA-RL VLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class RewardSpec:
    """Normalized reward config used by rollout and trainer code."""

    reward_type: str
    reward_impl: str
    tau_clip: float
    normalize_components: bool
    log_components: bool
    action_token_len: int
    weights: dict[str, float]


def _cfg_get(config, key: str, default):
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def reward_spec_from_config(config) -> RewardSpec:
    reward_cfg = config.reward
    return RewardSpec(
        reward_type=str(reward_cfg.type),
        reward_impl=str(reward_cfg.impl),
        tau_clip=float(reward_cfg.tau_clip),
        normalize_components=bool(reward_cfg.normalize_components),
        log_components=bool(reward_cfg.log_components),
        action_token_len=int(config.actor_rollout_ref.model.action_token_len),
        weights={
            "subgoal": float(reward_cfg.weights.subgoal),
            "progress": float(reward_cfg.weights.progress),
            "smoothness": float(reward_cfg.weights.smoothness),
            "terminal": float(reward_cfg.weights.terminal),
        },
    )


def validate_reward_support(config) -> None:
    reward_spec = reward_spec_from_config(config)
    task_suite_name = str(config.data.task_suite_name)
    if reward_spec.normalize_components:
        raise ValueError(
            "reward.normalize_components=True is not implemented for the current SimpleVLA-RL dense reward path. "
            "Set reward.normalize_components=False."
        )
    if "robotwin" in task_suite_name and reward_spec.reward_type != "binary":
        raise ValueError(
            "Robotwin currently supports only reward.type=binary. "
            f"Got reward.type={reward_spec.reward_type!r} for {task_suite_name!r}."
        )
    if reward_spec.reward_type not in {"binary", "dense", "clipped_dense"}:
        raise ValueError(f"Unsupported reward.type={reward_spec.reward_type!r}.")
    if reward_spec.reward_type in {"dense", "clipped_dense"} and reward_spec.reward_impl != "libero_native_dense":
        raise ValueError(
            "Dense rewards currently support only reward.impl='libero_native_dense'. "
            f"Got {reward_spec.reward_impl!r}."
        )


def flatten_token_level_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.reshape(tensor.shape[0], -1)
    raise ValueError(f"Unsupported reward tensor rank: {tensor.ndim}")


def build_binary_reward_tensor(
    complete: torch.Tensor,
    valid_response_length: torch.Tensor,
    response_shape: torch.Size,
    *,
    device: torch.device,
) -> torch.Tensor:
    reward_tensor = torch.zeros(response_shape, dtype=torch.float32, device=device).reshape(response_shape[0], -1)
    for row_index in range(reward_tensor.shape[0]):
        if bool(complete[row_index]) and int(valid_response_length[row_index]) > 0:
            reward_tensor[row_index, int(valid_response_length[row_index]) - 1] = 1.0
    return reward_tensor


def build_valid_response_mask(
    finish_step: torch.Tensor,
    flattened_length: int,
    *,
    action_token_len: int,
    device: torch.device,
) -> torch.Tensor:
    valid_response_length = finish_step.to(dtype=torch.int64) * int(action_token_len)
    steps = torch.arange(flattened_length, device=device, dtype=torch.int64)
    return steps.unsqueeze(0) < valid_response_length.unsqueeze(1)


def _component_from_batch(batch: dict[str, torch.Tensor], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.zeros_like(reference)
    return flatten_token_level_tensor(value).to(dtype=torch.float32, device=reference.device)


def _masked_mean(tensor: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_values = tensor[valid_mask]
    if valid_values.numel() == 0:
        return 0.0
    return float(valid_values.mean().item())


def _masked_ratio(masked_condition: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_values = masked_condition[valid_mask]
    if valid_values.numel() == 0:
        return 0.0
    return float(valid_values.float().mean().item())


def _sequence_sum_mean(tensor: torch.Tensor, valid_mask: torch.Tensor) -> float:
    return float((tensor * valid_mask.to(dtype=tensor.dtype)).sum(dim=-1).mean().item())


def _add_component_metrics(metrics: dict[str, float], prefix: str, tensor: torch.Tensor, valid_mask: torch.Tensor) -> None:
    padded_mean = float(tensor.mean().item())
    valid_mean = _masked_mean(tensor, valid_mask)
    sequence_sum_mean = _sequence_sum_mean(tensor, valid_mask)

    metrics[f"{prefix}_mean"] = padded_mean
    metrics[f"{prefix}_mean_padded"] = padded_mean
    metrics[f"{prefix}_mean_valid"] = valid_mean
    metrics[f"{prefix}_sum_sequence_mean"] = sequence_sum_mean


def combine_dense_rewards(batch: dict[str, torch.Tensor], reward_spec: RewardSpec) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    responses = batch["responses"]
    reward_shape = responses.shape
    reference = torch.zeros(reward_shape[0], reward_shape[1] * reward_shape[2], dtype=torch.float32, device=responses.device)
    valid_mask = build_valid_response_mask(
        batch["finish_step"],
        reference.shape[1],
        action_token_len=reward_spec.action_token_len,
        device=responses.device,
    )

    progress = _component_from_batch(batch, "dense_progress_scores", reference)
    smoothness = _component_from_batch(batch, "dense_smoothness_scores", reference)
    terminal = _component_from_batch(batch, "dense_terminal_scores", reference)
    subgoal = _component_from_batch(batch, "dense_subgoal_scores", reference)

    dense_total = (
        reward_spec.weights["subgoal"] * subgoal
        + reward_spec.weights["progress"] * progress
        + reward_spec.weights["smoothness"] * smoothness
        + reward_spec.weights["terminal"] * terminal
    )
    clipped_total = torch.maximum(dense_total, torch.full_like(dense_total, reward_spec.tau_clip))
    reward_total = clipped_total if reward_spec.reward_type == "clipped_dense" else dense_total

    reward_tensor_dict = {
        "dense_subgoal_scores": subgoal,
        "dense_progress_scores": progress,
        "dense_smoothness_scores": smoothness,
        "dense_terminal_scores": terminal,
        "dense_total_scores": dense_total,
        "all": reward_total,
    }
    metrics = {
        "reward_all": _sequence_sum_mean(reward_total, valid_mask),
        "reward_mean": float(reward_total.mean().item()),
        "reward_mean_padded": float(reward_total.mean().item()),
        "reward_mean_valid": _masked_mean(reward_total, valid_mask),
        "reward_sum_sequence_mean": _sequence_sum_mean(reward_total, valid_mask),
        "reward_min": reward_total.min().item(),
        "reward_max": reward_total.max().item(),
        "negative_ratio": float((reward_total < 0).float().mean().item()),
        "negative_ratio_valid": _masked_ratio(reward_total < 0, valid_mask),
        "clip_active": 1.0 if reward_spec.reward_type == "clipped_dense" else 0.0,
        "valid_token_count_mean": float(valid_mask.sum(dim=-1).float().mean().item()),
        "finish_step_mean": float(batch["finish_step"].float().mean().item()),
        "clip_activation_ratio": float((dense_total < reward_spec.tau_clip).float().mean().item())
        if reward_spec.reward_type == "clipped_dense"
        else 0.0,
        "clip_activation_ratio_valid": _masked_ratio(dense_total < reward_spec.tau_clip, valid_mask)
        if reward_spec.reward_type == "clipped_dense"
        else 0.0,
    }
    _add_component_metrics(metrics, "component_subgoal", subgoal, valid_mask)
    _add_component_metrics(metrics, "component_progress", progress, valid_mask)
    _add_component_metrics(metrics, "component_smoothness", smoothness, valid_mask)
    _add_component_metrics(metrics, "component_terminal", terminal, valid_mask)
    return reward_tensor_dict, metrics


def git_hash(cwd: str | os.PathLike[str]) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return "unknown"


def ensure_json_parent(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    target = ensure_json_parent(path)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_symlink(target: str | os.PathLike[str], link_path: str | os.PathLike[str]) -> None:
    target_path = Path(target)
    link = Path(link_path)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target_path.resolve(), target_is_directory=True)
    except OSError:
        if link.exists():
            if link.is_dir():
                shutil.rmtree(link)
            else:
                link.unlink()
        shutil.copytree(target_path, link)
