from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from core import config
from core.config import OBS_FIELDS_BY_AGENT


def make_action_spaces() -> dict[str, gym.Space]:
    """Fresh spaces per call: seeding one env's spaces must not leak to others."""
    return {
        **{name: Discrete(config.N_NEXT_NODES) for name in config.ROUTING_AGENTS},
        **{
            name: Box(
                low=config.TEMPERATURE_ACTION_LOW_C,
                high=config.TEMPERATURE_ACTION_HIGH_C,
                shape=(1,),
                dtype=np.float32,
            )
            for name in config.TEMPERATURE_AGENTS
        },
        **{
            name: Box(
                low=config.SPOILAGE_ACTION_LOW,
                high=config.SPOILAGE_ACTION_HIGH,
                shape=(1,),
                dtype=np.float32,
            )
            for name in config.SPOILAGE_AGENTS
        },
        **{
            name: Box(
                low=config.INVENTORY_ACTION_LOW,
                high=config.INVENTORY_ACTION_HIGH,
                shape=(1,),
                dtype=np.float32,
            )
            for name in config.INVENTORY_AGENTS
        },
        **{
            name: Discrete(config.N_DELIVERY_WINDOWS) for name in config.DELIVERY_AGENTS
        },
    }


def make_observation_spaces() -> dict[str, gym.Space]:
    return {
        agent: Box(low=-np.inf, high=np.inf, shape=(len(fields),), dtype=np.float32)
        for agent, fields in OBS_FIELDS_BY_AGENT.items()
    }
