"""Phase 1 orchestration for classic RL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.algorithms.grpo_trainer import GRPOTrainer
from src.analysis.metrics import compute_sample_efficiency
from src.config import (
    PHASE1A_ENVS,
    PHASE1B_ENVS,
    load_yaml_like,
    make_phase1_experiment_config,
    save_experiment_config,
)


def expand_phase_matrix(matrix_config: dict[str, Any]) -> list:
    """Expand a matrix config into single-run ExperimentConfig objects."""

    envs = list(matrix_config["envs"])
    reward_types = list(matrix_config["reward_types"])
    seeds = list(matrix_config["seeds"])
    tau_clip = float(matrix_config.get("tau_clip", 0.0))
    total_iterations = int(matrix_config.get("total_iterations", 10))
    configs = []
    for env_name in envs:
        for reward_type in reward_types:
            for seed in seeds:
                configs.append(
                    make_phase1_experiment_config(
                        env_name=env_name,
                        reward_type=reward_type,
                        seed=seed,
                        tau_clip=tau_clip,
                        total_iterations=total_iterations,
                    )
                )
    return configs


def select_tau_clip(sweep_results: dict[float, dict[str, list[float]]]) -> float:
    """Rank tau candidates by sample efficiency first and advantage variance second."""

    ranked = []
    for tau, metrics in sweep_results.items():
        sample_eff = compute_sample_efficiency(metrics.get("success_rates", []), target_threshold=0.8)
        variance_proxy = float(sum(metrics.get("advantage_stds", [])) / max(len(metrics.get("advantage_stds", [])), 1))
        ranked.append((sample_eff if sample_eff is not None else float("inf"), variance_proxy, tau))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return float(ranked[0][2])


def write_generated_configs(configs, output_dir: Path) -> None:
    """Write expanded single-run configs to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        file_name = (
            f"{config.env.phase}_{config.env.name.lower().replace('-', '_')}_"
            f"{config.reward.reward_type}_seed{config.env.seed}.yaml"
        )
        save_experiment_config(config, output_dir / file_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a phase matrix YAML/JSON config.")
    parser.add_argument(
        "--mode",
        choices=["generate-configs", "run-matrix"],
        default="generate-configs",
        help="Generate per-run configs or execute the matrix directly.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/generated-configs",
        help="Where to write generated per-run configs.",
    )
    args = parser.parse_args()

    matrix = load_yaml_like(args.config)
    configs = expand_phase_matrix(matrix)
    if args.mode == "generate-configs":
        write_generated_configs(configs, Path(args.output_dir))
        print(json.dumps({"generated_configs": len(configs), "output_dir": args.output_dir}, indent=2))
        return

    summaries = []
    for config in configs:
        summaries.append(
            {
                "env": config.env.name,
                "reward_type": config.reward.reward_type,
                "seed": config.env.seed,
                "summary": GRPOTrainer(config).run(),
            }
        )
    print(json.dumps({"runs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

