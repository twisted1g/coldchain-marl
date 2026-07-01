from __future__ import annotations

from typing import Any

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.dynamics import make_action_spaces, step
from core.state import GlobalState, extract_all_obs, init_state, make_observation_spaces


class ColdChainMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        self._seed: int | None = config.get("seed")
        self._max_steps: int | None = config.get("max_steps")
        self._state: GlobalState | None = None

        self.observation_spaces: dict[str, gym.Space] = make_observation_spaces()
        self.action_spaces: dict[str, gym.Space] = make_action_spaces()
        self.possible_agents: list[str] = list(self.observation_spaces.keys())
        self.agents: list[str] = list(self.possible_agents)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        active_seed = seed if seed is not None else self._seed
        self._state = init_state(seed=active_seed, max_steps=self._max_steps)
        self.agents = list(self.possible_agents)
        observations = extract_all_obs(self._state)
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
