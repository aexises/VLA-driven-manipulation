# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Callable, Type, Tuple, Union
import uuid
import json
from pathlib import Path
from omegaconf import OmegaConf, open_dict
import numpy as np
from codetiming import Timer

from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.rob_reward import (
    build_valid_response_mask,
    git_hash,
    reward_spec_from_config,
    safe_symlink,
    validate_reward_support,
    write_json,
)
from verl.utils.dataset.rob_dataset import BufferedDataLoader

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def _config_get(config, key: str, default):
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _build_response_mask(finish_step: torch.Tensor, flattened_length: int, action_token_len: int, device: torch.device) -> torch.Tensor:
    return build_valid_response_mask(
        finish_step,
        flattened_length,
        action_token_len=action_token_len,
        device=device,
    )


def _masked_std(tensor: torch.Tensor, mask: torch.Tensor) -> float:
    valid_values = tensor[mask.bool()]
    if valid_values.numel() == 0:
        return 0.0
    return float(valid_values.std(unbiased=False).detach().item())


def _masked_value(tensor: torch.Tensor, mask: torch.Tensor, fn: str) -> float:
    valid_values = tensor[mask.bool()]
    if valid_values.numel() == 0:
        return 0.0
    if fn == 'mean':
        return float(valid_values.mean().detach().item())
    if fn == 'max':
        return float(valid_values.max().detach().item())
    if fn == 'min':
        return float(valid_values.min().detach().item())
    raise ValueError(f"Unsupported masked reduction: {fn}")


def validate_training_config(config) -> None:
    validate_reward_support(config)
    if str(config.algorithm.adv_estimator) == 'grpo':
        n_samples = int(config.data.n_samples)
        allow_single_sample = bool(_config_get(config.trainer, 'allow_single_sample_grpo_debug', False))
        if n_samples < 2 and not allow_single_sample:
            raise ValueError(
                "GRPO reward-comparison runs require data.n_samples >= 2. "
                "Set trainer.allow_single_sample_grpo_debug=True only for debug-only single-sample runs."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl', action_token_len=7, action_chunks_len=8):
    responses = data.batch['responses']
    
    traj_length = responses.size(1) * action_chunks_len  
    action_length = action_token_len  # next fix
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    #attention_mask = data.batch['attention_mask']
    finish_step = data.batch['finish_step'] * action_length
    response_mask = _build_response_mask(
        data.batch['finish_step'],
        traj_length * action_length,
        action_token_len=action_length,
        device=data.batch['responses'].device,
    )

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        raw_kl = data.batch['old_log_probs'] - data.batch['ref_log_prob']
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
        ref_available = 1.0
    else:
        beta = 0
        raw_kl = torch.zeros_like(data.batch['old_log_probs'], dtype=torch.float32)
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
        ref_available = 0.0

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {
        'critic/kl': current_kl,
        'critic/kl_coeff': beta,
        'kl/enabled': 1.0 if beta > 0 else 0.0,
        'kl/ref_available': ref_available,
        'kl/old_ref_mean': _masked_value(raw_kl, response_mask, 'mean'),
        'kl/old_ref_std': _masked_std(raw_kl, response_mask),
    }

    return data, metrics


def compute_advantage(data: DataProto, gamma, lam, adv_estimator, config):

    responses = data.batch['responses']
    response_length = responses.size(1) *  responses.size(2)
    # attention_mask = data.batch['attention_mask']
    response_mask = _build_response_mask(
        data.batch['finish_step'],
        response_length,
        action_token_len=config.actor_rollout_ref.model.action_token_len,
        device=data.batch['responses'].device,
    )

    token_level_rewards = data.batch['token_level_rewards'] if 'token_level_rewards' in list(data.batch.keys()) else data.batch['token_level_scores']

    # TODO: add other ways to estimate advantages
    if adv_estimator == 'rloo':
        # prompt_ids = data.batch['prompts']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:,prompt_length:].sum(-1)
        advantages, returns = core_algos.compute_rloo_returns(data=data,
                                                eos_mask=response_mask,n_samples=config.data.n_samples, config=config)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(1) *  responses.size(2)
        response_mask = _build_response_mask(
            data.batch['finish_step'],
            response_length,
            action_token_len=config.actor_rollout_ref.model.action_token_len,
            device=data.batch['responses'].device,
        )
        reward_type = str(config.reward.type)
        if reward_type == 'binary':
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                            eos_mask=response_mask,
                                                                            index=index)
        else:
            discounted_advantage = bool(_config_get(_config_get(config.algorithm, 'grpo', {}), 'discounted_advantage', False))
            advantages, returns = core_algos.compute_grpo_dense_advantage(
                token_level_rewards=token_level_rewards,
                eos_mask=response_mask,
                index=index,
                gamma=config.algorithm.gamma,
                discounted_advantage=discounted_advantage,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def compute_data_metrics(batch,config):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    advantages = batch.batch['advantages']
    returns = batch.batch['returns']
    #add
    response_mask = _build_response_mask(
        batch.batch['finish_step'],
        batch.batch['responses'].size(1) * batch.batch['responses'].size(2),
        action_token_len=config.actor_rollout_ref.model.action_token_len,
        device=advantages.device,
    )
    #
    prefix = 'grpo' if str(config.algorithm.adv_estimator) == 'grpo' else 'critic'
    metrics = {
        # score
        f'{prefix}/score/mean': torch.mean(sequence_score).detach().item(),
        f'{prefix}/score/max': torch.max(sequence_score).detach().item(),
        f'{prefix}/score/min': torch.min(sequence_score).detach().item(),
        # reward
        f'{prefix}/rewards/mean': torch.mean(sequence_reward).detach().item(),
        f'{prefix}/rewards/max': torch.max(sequence_reward).detach().item(),
        f'{prefix}/rewards/min': torch.min(sequence_reward).detach().item(),
        # adv
        f'{prefix}/advantages/mean': masked_mean(advantages, response_mask).detach().item(),
        f'{prefix}/advantages/max': _masked_value(advantages, response_mask, 'max'),
        f'{prefix}/advantages/min': _masked_value(advantages, response_mask, 'min'),
        f'{prefix}/advantages/std': _masked_std(advantages, response_mask),
        # returns
        f'{prefix}/returns/mean': masked_mean(returns, response_mask).detach().item(),
        f'{prefix}/returns/max': _masked_value(returns, response_mask, 'max'),
        f'{prefix}/returns/min': _masked_value(returns, response_mask, 'min'),
        f'{prefix}/returns/std': _masked_std(returns, response_mask),
    }
    if prefix == 'grpo':
        for metric_name, value in list(metrics.items()):
            metrics['critic/' + metric_name.split('/', 1)[1]] = value
    return metrics

class RayTrainer(object):
   
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):


        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        validate_training_config(config)
        self.reward_spec = reward_spec_from_config(config)
        self._metric_history: list[dict[str, float]] = []
        self._metrics_file = None
        self._checkpoint_records: list[dict] = []
        self._best_checkpoint_record: dict | None = None
        self._current_metric_name = 'train_verify_score/all'
        self._current_metric_value = float('-inf')
        self._best_metric_name = 'val/test_score/all'
        self._best_metric_value = float('-inf')
        self._last_checkpoint_path: str | None = None
        self._resumed_from_path: str | None = None
        self._artifact_root = Path(self.config.trainer.default_local_dir)
        self._actor_root = self._artifact_root / 'actor'
        self._metadata_root = self._artifact_root / 'metadata'
        self._resolved_config_path = self._artifact_root / 'resolved_config.yaml'
        self._metrics_path = self._artifact_root / 'metrics.jsonl'
        self._summary_path = self._artifact_root / 'summary.json'
        self._checkpoints_index_path = self._metadata_root / 'checkpoints.json'
        self._best_checkpoint_path = self._metadata_root / 'best_checkpoint.json'

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping and config.algorithm.kl_ctrl.kl_coef > 0
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._resolve_resume_checkpoint()
        self._prepare_artifact_dirs()
        self._write_resolved_config()

    def _prepare_artifact_dirs(self) -> None:
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        self._actor_root.mkdir(parents=True, exist_ok=True)
        self._metadata_root.mkdir(parents=True, exist_ok=True)

    def _write_resolved_config(self) -> None:
        self._resolved_config_path.write_text(OmegaConf.to_yaml(self.config, resolve=True), encoding='utf-8')

    def _reward_metadata(self) -> dict[str, object]:
        return {
            'reward_type': self.reward_spec.reward_type,
            'reward_impl': self.reward_spec.reward_impl,
            'tau_clip': self.reward_spec.tau_clip,
            'weights': dict(self.reward_spec.weights),
            'task_suite_name': str(self.config.data.task_suite_name),
            'action_token_len': int(self.config.actor_rollout_ref.model.action_token_len),
            'action_chunks_len': int(self.config.actor_rollout_ref.model.action_chunks_len),
            'model_vla': str(self.config.actor_rollout_ref.model.vla),
            'seed': int(self.config.trainer.get('seed', 0)),
            'git_hash': git_hash(Path(__file__).resolve().parents[3]),
        }

    def _resolve_resume_checkpoint(self) -> None:
        if not bool(self.config.actor_rollout_ref.model.get('resume', False)):
            return
        resume_mode = str(self.config.trainer.get('resume_mode', 'none'))
        if resume_mode == 'none':
            raise ValueError("actor_rollout_ref.model.resume=True requires trainer.resume_mode to be 'last' or 'best'.")

        if resume_mode == 'last':
            checkpoint_path = self._actor_root / 'last'
        elif resume_mode == 'best':
            checkpoint_path = self._actor_root / 'best'
        else:
            raise ValueError(f"Unsupported trainer.resume_mode={resume_mode!r}.")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at {checkpoint_path}.")

        resolved_path = checkpoint_path.resolve()
        meta_path = resolved_path / 'checkpoint_meta.json'
        if not meta_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata missing at {meta_path}.")
        checkpoint_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        current_meta = self._reward_metadata()
        for key in ('reward_type', 'task_suite_name', 'action_token_len', 'action_chunks_len', 'model_vla'):
            if checkpoint_meta.get(key) != current_meta.get(key):
                raise ValueError(
                    f"Checkpoint mismatch for {key}: saved={checkpoint_meta.get(key)!r}, current={current_meta.get(key)!r}."
                )

        OmegaConf.set_struct(self.config, False)
        self.config.actor_rollout_ref.model.path = str(resolved_path)
        OmegaConf.set_struct(self.config, True)
        self._resumed_from_path = str(resolved_path)

    def _append_metrics_record(self, step: int, metrics: dict[str, float]) -> None:
        if self._metrics_file is None:
            return
        payload = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'step': step,
            'metrics': metrics,
        }
        self._metrics_file.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self._metrics_file.flush()

    def _select_monitor_metric(self, metrics: dict[str, float]) -> tuple[str, float]:
        if 'val/test_score/all' in metrics:
            return 'val/test_score/all', float(metrics['val/test_score/all'])
        if 'test_score/all' in metrics:
            return 'test_score/all', float(metrics['test_score/all'])
        if 'train_verify_score/all' in metrics:
            return 'train_verify_score/all', float(metrics['train_verify_score/all'])
        return 'train_verify_score/all', float('-inf')

    def _checkpoint_record(self, checkpoint_path: str, global_step: int, metric_name: str, metric_value: float, alias: str | None = None) -> dict:
        return {
            **self._reward_metadata(),
            'global_step': int(global_step),
            'checkpoint_path': checkpoint_path,
            'alias': alias,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        }

    def _persist_checkpoint_metadata(self, checkpoint_path: Path, record: dict) -> None:
        write_json(checkpoint_path / 'checkpoint_meta.json', record)

    def _update_checkpoint_indexes(self) -> None:
        write_json(self._checkpoints_index_path, self._checkpoint_records)
        if self._best_checkpoint_record is not None:
            write_json(self._best_checkpoint_path, self._best_checkpoint_record)

    def _save_periodic_checkpoint(self, global_step: int) -> None:
        checkpoint_dir = self._actor_root / f'global_step_{global_step}'
        self.actor_rollout_wg.save_checkpoint(str(checkpoint_dir), None)
        record = self._checkpoint_record(str(checkpoint_dir), global_step, self._current_metric_name, self._current_metric_value)
        self._checkpoint_records.append(record)
        self._persist_checkpoint_metadata(checkpoint_dir, record)
        safe_symlink(checkpoint_dir, self._actor_root / 'last')
        self._last_checkpoint_path = str((self._actor_root / 'last').resolve())
        self._update_checkpoint_indexes()

    def _save_best_checkpoint(self, global_step: int) -> None:
        best_dir = self._actor_root / 'best'
        self.actor_rollout_wg.save_checkpoint(str(best_dir), None)
        record = self._checkpoint_record(str(best_dir), global_step, self._current_metric_name, self._current_metric_value, alias='best')
        self._best_checkpoint_record = record
        self._best_metric_name = self._current_metric_name
        self._best_metric_value = self._current_metric_value
        self._persist_checkpoint_metadata(best_dir, record)
        self._update_checkpoint_indexes()

    def _write_summary(self, final_val_metrics: dict[str, float] | None = None) -> None:
        final_val_metrics = final_val_metrics or {}
        val_score = None
        if 'val/success_rate' in final_val_metrics:
            val_score = final_val_metrics['val/success_rate']
        elif 'val/test_score/all' in final_val_metrics:
            val_score = final_val_metrics['val/test_score/all']
        elif 'test_score/all' in final_val_metrics:
            val_score = final_val_metrics['test_score/all']

        reward_metric_name = 'grpo/rewards/mean' if str(self.config.algorithm.adv_estimator) == 'grpo' else 'critic/rewards/mean'
        advantage_metric_name = 'grpo/advantages/std' if str(self.config.algorithm.adv_estimator) == 'grpo' else 'critic/advantages/std'

        reward_means = [metrics[reward_metric_name] for metrics in self._metric_history if reward_metric_name in metrics]
        advantage_stds = [metrics[advantage_metric_name] for metrics in self._metric_history if advantage_metric_name in metrics]
        negative_ratios = [
            metrics['train_reward/negative_ratio_valid']
            if 'train_reward/negative_ratio_valid' in metrics
            else metrics['train_reward/negative_ratio']
            for metrics in self._metric_history
            if 'train_reward/negative_ratio_valid' in metrics or 'train_reward/negative_ratio' in metrics
        ]

        summary = {
            **self._reward_metadata(),
            'benchmark': 'robotwin' if 'robotwin' in str(self.config.data.task_suite_name) else 'libero',
            'best_validation_success': self._best_metric_value if self._best_metric_value != float('-inf') else None,
            'best_validation_metric_name': self._best_metric_name,
            'final_validation_success': val_score,
            'mean_cumulative_reward': float(np.mean(reward_means)) if reward_means else None,
            'mean_advantage_std': float(np.mean(advantage_stds)) if advantage_stds else None,
            'negative_reward_ratio': float(np.mean(negative_ratios)) if negative_ratios else None,
            'best_checkpoint_path': self._best_checkpoint_record['checkpoint_path'] if self._best_checkpoint_record else None,
            'last_checkpoint_path': self._last_checkpoint_path,
            'resumed_from_path': self._resumed_from_path,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        }
        write_json(self._summary_path, summary)

    def _create_dataloader(self):   # next fix
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rob_dataset import LIBERO_Dataset, Robotwin_Dataset, collate_fn
        if "libero" in self.config.data.task_suite_name:
            self.train_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                                num_trials_per_task=self.config.data.num_trials_per_task,
                                                train_val ="train")
            self.val_dataset = LIBERO_Dataset(self.config.data.task_suite_name,
                                            num_trials_per_task=self.config.data.num_trials_per_task,
                                            train_val ="valid")
        elif "robotwin" in self.config.data.task_suite_name:
            # (cjh) We assume here that data set names are "robotwin_{task_name}" or "robotwin_all"
            self.train_dataset = Robotwin_Dataset(self.config.data.task_suite_name,
                                                  num_trials_per_task=self.config.data.num_trials_per_task,train_val ="train")
            self.val_dataset = Robotwin_Dataset(self.config.data.task_suite_name,
                                                num_trials_per_task=self.config.data.num_trials_per_task,train_val ="valid")
        else:
            raise ValueError(f'Unsupported task suite name: {self.config.data.task_suite_name}')

        self.train_dataloader = BufferedDataLoader(DataLoader(dataset=self.train_dataset,
                                           batch_size=int(self.config.data.train_batch_size*self.config.data.oversample_factor),
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn))
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self, global_steps=0):
        reward_tensor_lst = []
        data_source_lst = []
        dense_reward_lst = []
        metric_dict = {}
        val_batches = 0
        for test_data in self.val_dataloader:
            val_batches += 1
            test_batch = DataProto.from_single_dict(test_data)
           
            test_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
                "global_steps":global_steps
            }

            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_batch)
            print('validation generation end')
            rollout_metrics = test_output_gen_batch.meta_info.get('metrics', {})
            for key, value in rollout_metrics.items():
                metric_dict[key] = metric_dict.get(key, 0.0) + float(value)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            verifier_score, reward_metrics, format_metrics, reward_format_metrics = self.val_reward_fn.verify(test_batch)
            reward_tensor=torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)

            for k, v in reward_metrics.items():
                metric_dict['test_reward/' + k] = metric_dict.get('test_reward/' + k, 0.0) + float(v)
                
            for k, v in format_metrics.items():
                metric_dict['format_acc/' + k] = metric_dict.get('format_acc/' + k, 0.0) + float(v)
                
            for k, v in reward_format_metrics.items():
                metric_dict['acc_wformat/' + k] = metric_dict.get('acc_wformat/' + k, 0.0) + float(v)
            reward_tensor_lst.append(reward_tensor)
            if str(self.reward_spec.reward_type) != 'binary':
                reward_tensor_dict, _ = self.reward_fn(test_batch)
                dense_reward_lst.append(reward_tensor_dict['all'].sum(-1).detach().cpu())
            #data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            #data_source_lst.append( [self.config.data.task_suite_name] * reward_tensor.shape[0])
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', [self.config.data.task_suite_name] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        rollout_metric_keys = [key for key in metric_dict.keys() if key.startswith('rollout/')]
        for key in rollout_metric_keys:
            metric_dict[key] = float(metric_dict[key]) / max(val_batches, 1)
        aggregate_metric_keys = [key for key in metric_dict.keys() if key.startswith('test_reward/') or key.startswith('format_acc/') or key.startswith('acc_wformat/')]
        for key in aggregate_metric_keys:
            metric_dict[key] = float(metric_dict[key]) / max(val_batches, 1)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'test_score/{data_source}'] = np.mean(rewards)

        metric_dict['success_rate'] = reward_tensor.mean().item()
        metric_dict[f'test_score/all'] = reward_tensor.mean().item()
        if dense_reward_lst:
            dense_reward_tensor = torch.cat(dense_reward_lst, dim=0)
            metric_dict['reward_dense_sum_mean'] = dense_reward_tensor.mean().item()

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in ['rloo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['grpo']:
            self.use_critic = False
        elif self.config.algorithm.adv_estimator in ['reinforce_plus_plus']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def fit(self):
        """
        The training loop of VLA-RL.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          local_dir=self.config.trainer.default_local_dir,
                          wandb_mode=self.config.trainer.wandb_mode,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self._metrics_file = self._metrics_path.open('a', encoding='utf-8')

        global_steps = 0
        dp_size = self.actor_rollout_wg.world_size // self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)
            self._append_metrics_record(global_steps, val_metrics)
            if self.config.trainer.get('val_only', False):
                self._write_summary(final_val_metrics=val_metrics)
                self._metrics_file.close()
                return

        for epoch in range(self.config.trainer.total_epochs):
            self.train_dataloader.start_new_epoch()
            while True:
                valid_batch = []
                buffer_batch = []

                if self.train_dataloader.buffer_size() > 0:
                    buffer_batch = self.train_dataloader.get_from_buffer(batch_size, self.actor_rollout_wg.world_size)
                metrics = defaultdict(list)
                metrics['timing/gen'] = 0
                metrics['timing/verify'] = 0
                metrics['timing/acc&trunc_filter'] = 0
                metrics['timing/filter_format_error'] = 0
                metrics['timing/compute_all_entropy'] = 0
                metrics['reward_model/enabled'] = 1.0 if bool(self.config.reward_model.enable) else 0.0
                metrics['reward_model/active_coef'] = float(self.config.reward_model.rm_coef)
                metrics['kl/enabled'] = 1.0 if self.use_reference_policy else 0.0

                while len(valid_batch) < batch_size * n_samples:
                    try:
                        batch_dict = self.train_dataloader.get_next_batch()
                    except StopIteration:
                        break

                    # generate a batch
                    with Timer(name='gen', text="{name}: {seconds:.1f} seconds") as timer:

                        newbatch: DataProto = DataProto.from_single_dict(batch_dict)

                        if len(buffer_batch) > 0:
                            newbatch = DataProto.concat([buffer_batch, newbatch])
                            buffer_batch = []

                        if "robotwin" in self.config.data.task_suite_name:
                            gen_batch = newbatch.select(batch_keys=['task_id', 'trial_id',"trial_seed"],
                                                        non_tensor_batch_keys={"task_suite_name"},
                                                        meta_info_keys={})
                        else:
                            gen_batch = newbatch.select(batch_keys=['task_id', 'trial_id'],
                                                        non_tensor_batch_keys={"task_suite_name"},
                                                        meta_info_keys={})
 
                        newbatch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                                                             dtype=object)

                        batch_lst = sum([[newbatch[i:i + 1] for _ in range(n_samples)] for i in range(len(newbatch))],
                                        [])

                        gen_batch.meta_info = {
                            'eos_token_id': self.tokenizer.eos_token_id,
                            'n_samples': n_samples,
                            'pad_token_id': self.tokenizer.pad_token_id,
                        }
                        
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(prompts=gen_batch)
                        rollout_metrics = gen_batch_output.meta_info.get('metrics', {})
                        for key, value in rollout_metrics.items():
                            metrics[key].append(float(value))
                        
                        roll_batch = DataProto.concat(batch_lst)
                        #roll_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                        roll_batch = roll_batch.union(gen_batch_output)

                    metrics['timing/gen'] += timer.last
                    
                    
                    # do accuracy filtering and score logging
                    with Timer(name='verify', text="{name}: {seconds:.1f} seconds") as timer:
                        scores_tensor, reward_metrics, format_metrics, reward_format_metrics = self.reward_fn.verify(roll_batch)
                        for k, v in reward_metrics.items():
                            metrics['train_verify_score/' + k].append(v)
                            
                        for k, v in format_metrics.items():
                            metrics['format_score/' + k].append(v)
                            
                        for k, v in reward_format_metrics.items():
                            metrics['train_verify_score_wo_format/' + k].append(v)    
                            
                    metrics['timing/verify'] += timer.last
                    
                    # do accuracy filtering and score logging
                    with Timer(name='acc&trunc_filter', text="{name}: {seconds:.1f} seconds") as timer:
                        filtered_roll_batch = roll_batch
                        if self.config.data.filter_accuracy or self.config.data.filter_truncated:
                            print(f"before filtering: {len(roll_batch)}")
                            pre_count = len(roll_batch)
                            pre_success_rate = float(roll_batch.batch['acc'].float().mean().item())
                            metrics['filter/pre_count'].append(float(pre_count))
                            metrics['filter/pre_success_rate'].append(pre_success_rate)
                            pre_reward_tensor_dict, pre_reward_metrics = self.reward_fn(roll_batch)
                            metrics['filter/pre_reward_sum_mean'].append(
                                float(pre_reward_metrics.get('reward_sum_sequence_mean', pre_reward_tensor_dict['all'].sum(-1).mean().item()))
                            )
                            filtered_roll_batch = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
                            print(f"after filtering: {len(filtered_roll_batch)}")
                            post_count = len(filtered_roll_batch)
                            metrics['filter/post_count'].append(float(post_count))
                            metrics['filter/drop_fraction'].append(float(max(pre_count - post_count, 0) / max(pre_count, 1)))
                            if post_count > 0:
                                post_success_rate = float(filtered_roll_batch.batch['acc'].float().mean().item())
                                metrics['filter/post_success_rate'].append(post_success_rate)
                                post_reward_tensor_dict, post_reward_metrics = self.reward_fn(filtered_roll_batch)
                                metrics['filter/post_reward_sum_mean'].append(
                                    float(post_reward_metrics.get('reward_sum_sequence_mean', post_reward_tensor_dict['all'].sum(-1).mean().item()))
                                )
                            else:
                                metrics['filter/post_success_rate'].append(0.0)
                                metrics['filter/post_reward_sum_mean'].append(0.0)
                    metrics['timing/acc&trunc_filter'] += timer.last

                    
                    if self.config.data.filter_warmup:
                        raise ValueError
                        roll_batch_to_add = filtered_roll_batch if len(filtered_roll_batch) > 0 else roll_batch
                    else:
                        roll_batch_to_add = filtered_roll_batch
                    
                    if len(valid_batch) == 0:
                        valid_batch = roll_batch_to_add
                    else:
                        valid_batch = DataProto.concat([valid_batch, roll_batch_to_add])
                    print(
                        f"collected {len(valid_batch)} / {batch_size * n_samples} rollouts and each prompt has {n_samples} responses")
                    
                if len(valid_batch) < batch_size * n_samples:
                    break
                elif len(valid_batch) > batch_size * n_samples:
                    valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)

                for k, v in reward_metrics.items():
                    metrics['train_verify_score/' + k] = np.mean(metrics['train_verify_score/' + k])
                    
                for k, v in format_metrics.items():
                    metrics['format_score/' + k] = np.mean(metrics['format_score/' + k])
                    
                for k, v in reward_format_metrics.items():
                    metrics['train_verify_score_wo_format/' + k] = np.mean(metrics['train_verify_score_wo_format/' + k])

                for key, value in list(metrics.items()):
                    if isinstance(value, list):
                        metrics[key] = float(np.mean(value)) if value else 0.0
                
                batch = valid_batch
                print(f'rollout batch size: {len(batch)}')
                
                if self.use_reference_policy:
                    # compute reference log_prob
                    with Timer(name='ref', text="{name}: {seconds:.1f} seconds") as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                        if 'ref_log_prob' not in batch.batch.keys():
                            raise RuntimeError(
                                'Reference policy is enabled (algorithm.kl_ctrl.kl_coef > 0) '
                                'but ref_log_prob was not produced by the ref worker.'
                            )
                    metrics['timing/ref'] = timer.last

                with Timer(name='reward', text="{name}: {seconds:.1f} seconds") as timer:
                    if self.use_rm:
                        print("Not implement yet")
                        raise ValueError
                        # batch.meta_info['n_samples'] = n_samples
                        # reward_model_tensor= self.rm_wg.compute_rm_score(batch)
                        # if 'metrics' in reward_model_tensor.meta_info:
                        #     reward_model_metrics = reduce_metrics(reward_model_tensor.meta_info.pop('metrics'))
                        #     metrics.update(reward_model_metrics)
                        # batch = batch.union(reward_model_tensor)

                metrics['timing/reward_model'] = timer.last

                with Timer(name='adv', text="{name}: {seconds:.1f} seconds") as timer:
                    # directly reuse previously computed rewards; but with reward shaping
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor_dict['all']
                    for k, v in reward_metrics.items():
                        metrics['train_reward/' + k] = v
                    # decomposed rewards:
                    for k,v in reward_tensor_dict.items():
                        batch.batch[k]=v

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(batch,
                                                         kl_ctrl=self.kl_ctrl,
                                                         kl_penalty=self.config.algorithm.kl_penalty,
                                                         action_token_len=self.config.actor_rollout_ref.model.action_token_len, 
                                                         action_chunks_len=self.config.actor_rollout_ref.model.action_chunks_len,)
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              self.config.algorithm.gamma,
                                              self.config.algorithm.lam,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              config = self.config)
                metrics['timing/adv'] = timer.last

                # critic is disabled

                # implement critic warmup
                if self.config.trainer.critic_warmup <= global_steps:
                    # update actor
                    with Timer(name='update_actor', text="{name}: {seconds:.1f} seconds") as timer:
                        batch.meta_info['is_filtered'] = bool(self.config.data.filter_accuracy or self.config.data.filter_truncated)
                        batch.meta_info['train_mode'] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        entropy_output = self.actor_rollout_wg.compute_entropy(data=batch)
                    metrics['timing/update_actor'] = timer.last
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    entropy_output_metrics = reduce_metrics(entropy_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)
                    metrics.update(entropy_output_metrics)
                # validate
                if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                    with Timer(name='testing', text="{name}: {seconds:.1f} seconds") as timer:
                        val_metrics: dict = self._validate(global_steps=global_steps+1)
                        val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                    metrics['timing/testing'] = timer.last
                    metrics.update(val_metrics)
                    logger.log(data=val_metrics, step=global_steps)

                # collect metrics
                with Timer(name='logging1', text="{name}: {seconds:.1f} seconds") as timer:
                    data_metrics = compute_data_metrics(batch=batch, config = self.config)
                with Timer(name='logging2', text="{name}: {seconds:.1f} seconds") as timer:
                    metrics.update(data_metrics)
                with Timer(name='logging3', text="{name}: {seconds:.1f} seconds") as timer:
                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=global_steps)
                    self._append_metrics_record(global_steps, dict(metrics))
                    self._metric_history.append(dict(metrics))
                    self._current_metric_name, self._current_metric_value = self._select_monitor_metric(metrics)
                    if self._current_metric_value >= self._best_metric_value:
                        self._save_best_checkpoint(global_steps)

                if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                    self._save_periodic_checkpoint(global_steps)

                global_steps += 1

        # perform validation after training
        final_val_metrics = {}
        if self.val_reward_fn is not None:
            val_metrics = self._validate(global_steps=global_steps)
            pprint(f'Final validation metrics: {val_metrics}')
            final_val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
            logger.log(data=final_val_metrics, step=global_steps)
            self._append_metrics_record(global_steps, final_val_metrics)
        self._write_summary(final_val_metrics=final_val_metrics)
        self._metrics_file.close()

    def filter_format(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        if self.config.data.filter_format:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Format distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= 1)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled

        # Combine both masks
        combined_mask = acc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered format batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        
        return filtered_batch

    def filter(self, reward_tensor, batch, n_samples):
        """
        Filter responses based on accuracy and truncation criteria.
        
        Args:
            reward_tensor: Tensor containing accuracy scores
            batch: DataProto batch containing responses
            n_samples: Number of responses per prompt
        
        Returns:
            DataProto: Filtered batch
        """
        # First do accuracy filtering if enabled
        if self.config.data.filter_accuracy:
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Accuracy distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= self.config.data.accuracy_lower_bound) & (
                        acc_tensor <= self.config.data.accuracy_upper_bound)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)
        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch['responses']
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -responses.size(1):]

            # Calculate response lengths
            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(-1, n_samples)  # (num_prompts, n_samples)

            # Get max possible length from config
            max_len = self.config.data.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            print("Truncation distribution:", 
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}")
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)

        # Apply the mask to the batch
        filtered_batch = batch.slice(final_mask)

        print(f"Filtered batch size: {len(filtered_batch)} (from original size: {len(batch)})")
        return filtered_batch

    def add_to_buffer(self, batch, batch_size, n_samples):
        buffer_length = len(batch) // n_samples - batch_size
        # buffer_batch = batch.slice(range(batch_size * n_samples, (buffer_length + batch_size) * n_samples, n_samples))
        # # notice that we only add prompts to buffer, and slicing strategy should be exactly consistent to what is in ray_trainer.py
        # buffer_batch = buffer_batch.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        # buffer_batch.slice_batch(start=0, length=self.config.data.max_prompt_length, dim=1)
        buffer_mask = torch.ones(buffer_length + batch_size, dtype=torch.bool)
        buffer_mask[batch_size:] = False
        buffer_mask = buffer_mask.repeat_interleave(n_samples)
        batch = batch.slice(buffer_mask)
        # self.train_dataloader.add_to_buffer(buffer_batch)
        return batch
