from __future__ import annotations

import numpy as np

from env.training_env import ColdChainTrainingEnv
from training.agents import Agent


def rollout(
    env: ColdChainTrainingEnv,
    agents: dict[str, Agent],
    primary: str,
    n_episodes: int,
    metric_key: str,
) -> tuple[float, float]:
    """Return (mean episode return, mean per-episode metric) for ``primary``, acting greedily."""
    returns: list[float] = []
    metrics: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_metric: list[float] = []
        while not done:
            actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            ep_return += rewards[primary]
            ep_metric.append(infos[primary][metric_key])
            done = terminated["__all__"] or truncated["__all__"]
        returns.append(ep_return)
        metrics.append(float(np.mean(ep_metric)))
    return float(np.mean(returns)), float(np.mean(metrics))
