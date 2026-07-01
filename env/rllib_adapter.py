from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from env.pettingzoo_adapter import ColdChainParallelEnv


def make_rllib_env(config: dict[str, Any] | None = None) -> ParallelPettingZooEnv:
    """Wrap the canonical PettingZoo cold-chain env for RLlib's MultiAgentEnv API."""
    config = config or {}
    par_env = ColdChainParallelEnv(
        seed=config.get("seed"), max_steps=config.get("max_steps")
    )
    return ParallelPettingZooEnv(par_env)
