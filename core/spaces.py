from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from core import config
from core.config import OBS_FIELDS_BY_AGENT

ACTION_SPACES: dict[str, gym.Space] = {
    "routing": Discrete(config.N_NEXT_NODES),
    "temperature": Box(
        low=config.TEMPERATURE_ACTION_LOW_C,
        high=config.TEMPERATURE_ACTION_HIGH_C,
        shape=(1,),
        dtype=np.float32,
    ),
    "spoilage": Box(
        low=config.SPOILAGE_ACTION_LOW,
        high=config.SPOILAGE_ACTION_HIGH,
        shape=(1,),
        dtype=np.float32,
    ),
    "inventory": Box(
        low=config.INVENTORY_ACTION_LOW,
        high=config.INVENTORY_ACTION_HIGH,
        shape=(1,),
        dtype=np.float32,
    ),
    **{name: Discrete(config.N_DELIVERY_WINDOWS) for name in config.DELIVERY_AGENTS},
}


def make_action_spaces() -> dict[str, gym.Space]:
    return dict(ACTION_SPACES)


def make_observation_spaces() -> dict[str, gym.Space]:
    return {
        agent: Box(low=-np.inf, high=np.inf, shape=(len(fields),), dtype=np.float32)
        for agent, fields in OBS_FIELDS_BY_AGENT.items()
    }
