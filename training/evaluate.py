from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule

from training.env import ColdChainTrainingEnv

ActionFn = Callable[[dict], dict]


def greedy_action_fn(module: RLModule, action_space: gym.spaces.Box, agent: str) -> ActionFn:
    """Deterministic (distribution-mean) action for one agent from a trained SAC module."""
    low = torch.as_tensor(action_space.low)
    high = torch.as_tensor(action_space.high)
    dist_cls = module.get_inference_action_dist_cls()

    def fn(obs: dict) -> dict:
        o = torch.as_tensor(np.asarray(obs[agent]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: o})
            action = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS]).to_deterministic().sample()
            action = torch.clamp(action, low, high)
        return {agent: action.squeeze(0).cpu().numpy().astype(np.float32)}

    return fn


def random_action_fn(action_space: gym.spaces.Box, agent: str) -> ActionFn:
    def fn(obs: dict) -> dict:
        return {agent: action_space.sample()}

    return fn


def rollout(
    env: ColdChainTrainingEnv,
    action_fn: ActionFn,
    n_episodes: int,
    agent: str,
    metric_key: str = "temp_deviation",
) -> tuple[float, float]:
    """Return (mean episode return, mean per-episode metric) for one agent.

    Only the acting agent's action is supplied; the frozen agents default to
    no-op and do not affect its shaped reward.
    """
    returns: list[float] = []
    metrics: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_metric: list[float] = []
        while not done:
            obs, rewards, terminated, truncated, infos = env.step(action_fn(obs))
            ep_return += rewards[agent]
            ep_metric.append(infos[agent][metric_key])
            done = terminated["__all__"] or truncated["__all__"]
        returns.append(ep_return)
        metrics.append(float(np.mean(ep_metric)))
    return float(np.mean(returns)), float(np.mean(metrics))
