from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import RLModule

from training.env import TemperatureTrainingEnv

ActionFn = Callable[[dict], dict]


def greedy_action_fn(module: RLModule, action_space: gym.spaces.Box) -> ActionFn:
    """Deterministic (distribution-mean) temperature action from a trained SAC module."""
    low = torch.as_tensor(action_space.low)
    high = torch.as_tensor(action_space.high)
    dist_cls = module.get_inference_action_dist_cls()

    def fn(obs: dict) -> dict:
        o = torch.as_tensor(np.asarray(obs["temperature"]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = module.forward_inference({Columns.OBS: o})
            action = dist_cls.from_logits(out[Columns.ACTION_DIST_INPUTS]).to_deterministic().sample()
            action = torch.clamp(action, low, high)
        return {"temperature": action.squeeze(0).cpu().numpy().astype(np.float32)}

    return fn


def random_action_fn(action_space: gym.spaces.Box) -> ActionFn:
    def fn(obs: dict) -> dict:
        return {"temperature": action_space.sample()}

    return fn


def rollout(env: TemperatureTrainingEnv, action_fn: ActionFn, n_episodes: int) -> tuple[float, float]:
    """Return (mean episode return, mean per-episode temperature deviation).

    Only the temperature action is supplied; the frozen agents default to no-op
    and do not affect the temperature reward.
    """
    returns: list[float] = []
    deviations: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_dev: list[float] = []
        while not done:
            obs, rewards, terminated, truncated, infos = env.step(action_fn(obs))
            ep_return += rewards["temperature"]
            ep_dev.append(infos["temperature"]["temp_deviation"])
            done = terminated["__all__"] or truncated["__all__"]
        returns.append(ep_return)
        deviations.append(float(np.mean(ep_dev)))
    return float(np.mean(returns)), float(np.mean(deviations))
