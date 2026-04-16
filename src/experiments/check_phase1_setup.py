"""Check whether the local environment is ready for Phase 1 experiments."""

from __future__ import annotations

import json

from src.config.defaults import PHASE1A_ENVS, PHASE1B_ENVS
from src.envs.classic.factory import make_env


def _check_env(env_name: str) -> str:
    try:
        env = make_env(env_name, seed=11)
        env.close()
        return "ok"
    except Exception as exc:  # pragma: no cover - integration-oriented helper
        return f"error: {exc}"


def main() -> None:
    phase1a = {env_name: _check_env(env_name) for env_name in PHASE1A_ENVS}
    phase1b = {env_name: _check_env(env_name) for env_name in PHASE1B_ENVS}
    print(
        json.dumps(
            {
                "phase1a_envs": phase1a,
                "phase1b_envs": phase1b,
                "note": "Phase 1B requires MuJoCo support in the current Gymnasium installation.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
