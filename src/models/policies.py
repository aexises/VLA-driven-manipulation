"""Policy network interfaces for discrete and continuous control."""

from __future__ import annotations

import abc
from typing import Iterable

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical, Independent, Normal, kl_divergence
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    torch = None
    nn = None
    Categorical = None
    Independent = None
    Normal = None
    kl_divergence = None

BaseModule = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "torch is required for policy construction and training. Install project "
            "dependencies from pyproject.toml before running GRPO experiments."
        )


def _build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> "nn.Sequential":
    _require_torch()
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(previous, hidden), nn.Tanh()])
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class Policy(abc.ABC, BaseModule):
    """Abstract base class for policies consumed by the GRPO trainer."""

    def __init__(self) -> None:
        _require_torch()
        super().__init__()

    @abc.abstractmethod
    def distribution(self, observations: "torch.Tensor"):
        """Return the action distribution for a batch of observations."""

    @abc.abstractmethod
    def action_tensor(self, actions):
        """Convert environment actions into the tensor shape expected by the policy."""

    def act(self, observation, deterministic: bool = False):
        """Sample an action and return its log-probability."""

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        dist = self.distribution(obs_tensor)
        action = dist.mean if deterministic and hasattr(dist, "mean") else dist.sample()
        log_prob = dist.log_prob(action)
        return action.squeeze(0).detach().cpu().numpy(), float(log_prob.item())

    def evaluate_actions(self, observations: "torch.Tensor", actions: "torch.Tensor"):
        """Compute log-probabilities and entropy for a batch of actions."""

        dist = self.distribution(observations)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        return log_prob, entropy

    def kl_to(self, reference_policy: "Policy", observations: "torch.Tensor") -> "torch.Tensor":
        """Compute D_KL(pi_theta || pi_ref) for each observation."""

        current = self.distribution(observations)
        reference = reference_policy.distribution(observations)
        kl_values = kl_divergence(current, reference)
        if kl_values.ndim > 1:
            kl_values = kl_values.sum(dim=-1)
        return kl_values


class DiscreteMLPPolicy(Policy):
    """Categorical MLP policy for discrete-control environments."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        self.network = _build_mlp(observation_dim, hidden_sizes, action_dim)

    def distribution(self, observations: "torch.Tensor"):
        logits = self.network(observations)
        return Categorical(logits=logits)

    def action_tensor(self, actions):
        return torch.as_tensor(actions, dtype=torch.int64)


class GaussianMLPPolicy(Policy):
    """Diagonal Gaussian MLP policy for continuous-control environments."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes: Iterable[int]) -> None:
        super().__init__()
        self.mean_network = _build_mlp(observation_dim, hidden_sizes, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def distribution(self, observations: "torch.Tensor"):
        mean = self.mean_network(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def action_tensor(self, actions):
        return torch.as_tensor(actions, dtype=torch.float32)
