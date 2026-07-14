from __future__ import annotations

from typing import Any

import gymnasium as gym
from pettingzoo import ParallelEnv

from core.config import FruitKey
from core.dynamics import step
from core.interfaces.observations import all_obs
from core.interfaces.spaces import make_action_spaces, make_observation_spaces
from core.state import GlobalState, init_state


class ColdChainParallelEnv(ParallelEnv):
    metadata = {"name": "coldchain_v0", "render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int | None = None,
        fruit: FruitKey | None = None,
    ) -> None:
        self._seed = seed
        self._max_steps = max_steps
        self._fruit = fruit
        self._state: GlobalState | None = None
        self.observation_spaces: dict[str, gym.Space] = make_observation_spaces()
        self.action_spaces: dict[str, gym.Space] = make_action_spaces()
        self.possible_agents: list[str] = list(self.observation_spaces.keys())
        self.agents: list[str] = list(self.possible_agents)

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        active_seed = seed if seed is not None else self._seed
        self._state = init_state(
            seed=active_seed, max_steps=self._max_steps, fruit=self._fruit
        )
        self.agents = list(self.possible_agents)
        observations = all_obs(self._state)
        infos = {agent: {} for agent in self.possible_agents}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        result = step(self._state, actions)
        if result.terminated.get("__all__", False):
            self.agents = []
        return (
            result.observations,
            result.rewards,
            result.terminated,
            result.truncated,
            result.infos,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._state = None
