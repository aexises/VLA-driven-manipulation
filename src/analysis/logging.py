"""Experiment logging and result persistence."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.io import save_experiment_config
from src.config.types import ExperimentConfig

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    SummaryWriter = None


class NullSummaryWriter:
    """Fallback writer used when TensorBoard is unavailable."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_hparams(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class LogPaths:
    """Resolved log directories for a single run."""

    run_dir: Path
    results_dir: Path
    config_path: Path
    metrics_csv_path: Path
    summary_json_path: Path
    raw_trajectories_path: Path


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:  # pragma: no cover - best effort only
        return "unknown"


class ExperimentLogger:
    """Persist scalars, configs, summaries, and experiment index entries."""

    def __init__(self, config: ExperimentConfig) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_label = f"{config.logging.run_id}_{timestamp}"
        run_dir = Path(config.logging.runs_dir) / config.logging.experiment_name / run_label
        results_dir = Path(config.logging.results_dir) / config.logging.experiment_name / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.paths = LogPaths(
            run_dir=run_dir,
            results_dir=results_dir,
            config_path=results_dir / "config.yaml",
            metrics_csv_path=results_dir / "metrics.csv",
            summary_json_path=results_dir / "summary.json",
            raw_trajectories_path=results_dir / "raw_trajectories.json",
        )
        self.writer = SummaryWriter(log_dir=str(run_dir)) if SummaryWriter is not None else NullSummaryWriter()
        self._metrics_file = self.paths.metrics_csv_path.open("w", encoding="utf-8", newline="")
        self._metrics_writer = csv.DictWriter(
            self._metrics_file,
            fieldnames=[
                "iteration",
                "reward_step_mean",
                "reward_episode_cumulative",
                "reward_negative_ratio",
                "loss_policy",
                "loss_kl",
                "loss_total",
                "metrics_success_rate",
                "metrics_episode_length",
                "grad_norm",
                "grad_advantage_mean",
                "grad_advantage_std",
            ],
        )
        self._metrics_writer.writeheader()
        save_experiment_config(config, self.paths.config_path)
        self.writer.add_hparams(
            {
                "learning_rate": config.grpo.learning_rate,
                "clip_threshold": config.grpo.clip_eps,
                "beta_kl": config.grpo.beta_kl,
                "reward_type": {"binary": 0, "dense": 1, "clipped_dense": 2}[config.reward.reward_type],
                "tau_clip": config.reward.tau_clip,
            },
            {},
        )

    def log_iteration(self, iteration: int, metrics: dict[str, float]) -> None:
        """Write iteration metrics to CSV and TensorBoard."""

        row = {
            "iteration": iteration,
            "reward_step_mean": metrics["reward_step_mean"],
            "reward_episode_cumulative": metrics["reward_episode_cumulative"],
            "reward_negative_ratio": metrics["reward_negative_ratio"],
            "loss_policy": metrics["loss_policy"],
            "loss_kl": metrics["loss_kl"],
            "loss_total": metrics["loss_total"],
            "metrics_success_rate": metrics["success_rate"],
            "metrics_episode_length": metrics["episode_length"],
            "grad_norm": metrics["grad_norm"],
            "grad_advantage_mean": metrics["advantage_mean"],
            "grad_advantage_std": metrics["advantage_std"],
        }
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()
        self.writer.add_scalar("reward/step_mean", metrics["reward_step_mean"], iteration)
        self.writer.add_scalar("reward/episode_cumulative", metrics["reward_episode_cumulative"], iteration)
        self.writer.add_scalar("reward/negative_ratio", metrics["reward_negative_ratio"], iteration)
        self.writer.add_scalar("loss/policy", metrics["loss_policy"], iteration)
        self.writer.add_scalar("loss/kl", metrics["loss_kl"], iteration)
        self.writer.add_scalar("loss/total", metrics["loss_total"], iteration)
        self.writer.add_scalar("metrics/success_rate", metrics["success_rate"], iteration)
        self.writer.add_scalar("metrics/episode_length", metrics["episode_length"], iteration)
        self.writer.add_scalar("grad/norm", metrics["grad_norm"], iteration)
        self.writer.add_scalar("grad/advantage_mean", metrics["advantage_mean"], iteration)
        self.writer.add_scalar("grad/advantage_std", metrics["advantage_std"], iteration)

    def finalize(
        self,
        config: ExperimentConfig,
        summary: dict[str, Any],
        raw_trajectories: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist the final summary and update the experiment index."""

        summary_with_metadata = {
            **summary,
            "git_hash": _git_hash(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "hypothesis_ref": config.hypothesis_ref,
            "env_name": config.env.name,
            "reward_type": config.reward.reward_type,
            "seed": config.env.seed,
        }
        self.paths.summary_json_path.write_text(
            json.dumps(summary_with_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if raw_trajectories is not None and config.logging.write_raw_trajectories:
            self.paths.raw_trajectories_path.write_text(
                json.dumps(raw_trajectories, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        index_path = Path(config.logging.results_dir) / "EXPERIMENTS.md"
        if index_path.exists():
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"- `{config.logging.experiment_name}` | env=`{config.env.name}` | "
                    f"reward=`{config.reward.reward_type}` | seed=`{config.env.seed}` | "
                    f"git=`{summary_with_metadata['git_hash']}` | "
                    f"success_rate=`{summary.get('final_success_rate', 'n/a')}`\n"
                )
        self._metrics_file.close()
        self.writer.close()

