"""Classic RL environment adapters and reward specifications."""

from .factory import make_env
from .specs import EpisodeRewardResult, Transition, get_classic_task_spec

__all__ = ["EpisodeRewardResult", "Transition", "get_classic_task_spec", "make_env"]

