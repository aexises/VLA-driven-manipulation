"""YAML-like config loading and saving utilities.

The fallback serializer writes JSON, which is valid YAML 1.2 and keeps the
project usable in lightweight environments where PyYAML is not installed yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ExperimentConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in environments without PyYAML
    yaml = None


def load_yaml_like(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON document into a dictionary."""

    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data)!r}.")
    return data


def save_yaml_like(payload: dict[str, Any], path: str | Path) -> None:
    """Persist a mapping as YAML when available, otherwise as JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    target.write_text(text, encoding="utf-8")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a single-run experiment config."""

    return ExperimentConfig.from_dict(load_yaml_like(path))


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    """Persist a single-run experiment config."""

    save_yaml_like(config.to_dict(), path)
