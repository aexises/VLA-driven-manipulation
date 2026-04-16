"""Plot generation for single runs and aggregated Phase 1 comparisons."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "vla_phase1_matplotlib"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_run_artifacts(results_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load `metrics.csv` and `summary.json` for a single run directory."""

    run_dir = Path(results_dir)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return metrics, summary


def save_run_plots(results_dir: str | Path) -> list[Path]:
    """Create a standard training-curves figure for one experiment run."""

    run_dir = Path(results_dir)
    metrics, summary = load_run_artifacts(run_dir)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    iterations = metrics["iteration"]

    axes[0, 0].plot(iterations, metrics["metrics_success_rate"], label="success_rate")
    axes[0, 0].set_title("Success Rate")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Rate")

    axes[0, 1].plot(iterations, metrics["reward_episode_cumulative"], label="episode_reward", color="tab:green")
    axes[0, 1].set_title("Episode Reward")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Reward")

    axes[1, 0].plot(iterations, metrics["loss_total"], label="loss_total", color="tab:red")
    axes[1, 0].plot(iterations, metrics["loss_kl"], label="loss_kl", color="tab:orange")
    axes[1, 0].set_title("Loss Terms")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].legend()

    axes[1, 1].plot(iterations, metrics["grad_advantage_std"], label="advantage_std", color="tab:purple")
    axes[1, 1].plot(iterations, metrics["reward_negative_ratio"], label="negative_ratio", color="tab:brown")
    axes[1, 1].set_title("Variance Proxy and Negative Reward Ratio")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend()

    figure.suptitle(
        f"{summary['env_name']} | {summary['reward_type']} | seed {summary['seed']}",
        fontsize=14,
    )
    output_path = run_dir / "training_curves.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return [output_path]


def save_phase_comparison_plot(aggregate_df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Create mean-by-condition comparison plots for a matrix of runs."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    grouped = (
        aggregate_df.groupby(["env_name", "condition_label"], as_index=False)
        .agg(
            final_success_rate_mean=("final_success_rate", "mean"),
            final_success_rate_std=("final_success_rate", "std"),
            mean_cumulative_reward_mean=("mean_cumulative_reward", "mean"),
            gradient_variance_mean=("mean_gradient_variance_proxy", "mean"),
        )
        .fillna(0.0)
    )

    saved_paths: list[Path] = []
    for env_name, env_frame in grouped.groupby("env_name"):
        env_frame = env_frame.sort_values("condition_label")
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        axes[0].bar(
            env_frame["condition_label"],
            env_frame["final_success_rate_mean"],
            yerr=env_frame["final_success_rate_std"],
            color=["tab:blue", "tab:green", "tab:orange"][: len(env_frame)],
        )
        axes[0].set_title("Final Success Rate")
        axes[0].set_ylim(0.0, 1.05)

        axes[1].bar(env_frame["condition_label"], env_frame["mean_cumulative_reward_mean"], color="tab:green")
        axes[1].set_title("Mean Cumulative Reward")

        axes[2].bar(env_frame["condition_label"], env_frame["gradient_variance_mean"], color="tab:red")
        axes[2].set_title("Advantage Std Proxy")

        figure.suptitle(env_name, fontsize=14)
        path = target_dir / f"{env_name.lower().replace('-', '_')}_comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved_paths.append(path)
    return saved_paths
