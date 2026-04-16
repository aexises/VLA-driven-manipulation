"""Run a single Phase 1 experiment from a config file."""

from __future__ import annotations

import argparse
import json

from src.algorithms.grpo_trainer import GRPOTrainer
from src.config import load_experiment_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a single-run YAML/JSON config.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = GRPOTrainer(config).run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

