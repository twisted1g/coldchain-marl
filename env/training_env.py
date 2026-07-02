from __future__ import annotations

from typing import Any, Callable

from core.config import FruitKey
from core.fruits import get_params
from env.pettingzoo_adapter import ColdChainParallelEnv

DEFAULT_FRUIT = FruitKey.STRAWBERRY
DEFAULT_MAX_STEPS = 20
DEFAULT_BASE_SEED = 0
DEFAULT_LEARNERS = ("temperature",)

# Fixed reward weights. w2 is tuned so the interior optimum of the energy/spoilage
# trade-off lands on the fruit's ideal midpoint: the energy gradient is a constant
# -0.1*w1 while sensor < ambient, balanced against the Arrhenius spoilage gradient.
# PHASE 4: context-aware Pareto weights.
ENERGY_WEIGHT = 1.0
SPOILAGE_WEIGHT = 25.0
STEP_PENALTY = 0.01

# Routing (paper Algorithm 1): penalize travel time, emissions, spoilage risk
# accrued on the route. Fixed weights; PHASE 4: context-aware Pareto weights.
ROUTE_TIME_WEIGHT = 1.0
ROUTE_EMISSIONS_WEIGHT = 0.5
ROUTE_RISK_WEIGHT = 10.0

RewardMethod = Callable[[], "tuple[float, dict[str, float]]"]


class ColdChainTrainingEnv(ColdChainParallelEnv):
    """Cold-chain env that shapes rewards for the trainable ("learner") agents.

    Each learner's reward is computed by its own ``_<agent>_reward`` method; the
    frozen agents keep the core's zero reward. The fruit is fixed so the ideal
    band is stationary, and episodes follow a deterministic seed sequence for
    reproducible curves. To train a new agent: add its reward method, register it
    in ``self._reward_methods``, and list it in training.config.LEARNERS.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._base_seed = config.get("base_seed", DEFAULT_BASE_SEED)
        super().__init__(
            max_steps=config.get("max_steps", DEFAULT_MAX_STEPS),
            fruit=FruitKey(config.get("fruit", DEFAULT_FRUIT)),
        )
        supported: dict[str, RewardMethod] = {
            "temperature": self._temperature_reward,
            "routing": self._routing_reward,
        }
        learners = config.get("learners", DEFAULT_LEARNERS)
        self._reward_methods = {a: supported[a] for a in learners}
        self._episode_index = 0
        self._prev: dict[str, float] = {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, infos = super().reset(seed=self._base_seed + self._episode_index)
        self._episode_index += 1
        self._prev = {
            "spoilage_risk": self._state.shipment.spoilage_risk,
            "route_travel_time": self._state.route_travel_time,
            "route_emissions": self._state.route_emissions,
        }
        return obs, infos

    def step(self, actions: dict[str, Any]):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        for agent, reward_method in self._reward_methods.items():
            reward, metrics = reward_method()
            rewards[agent] = reward
            infos[agent].update(metrics)
        self._prev["spoilage_risk"] = self._state.shipment.spoilage_risk
        self._prev["route_travel_time"] = self._state.route_travel_time
        self._prev["route_emissions"] = self._state.route_emissions
        return obs, rewards, terminated, truncated, infos

    def _temperature_reward(self) -> tuple[float, dict[str, float]]:
        s = self._state.shipment
        energy = float(self._state.energy_usage)
        spoilage_delta = max(0.0, s.spoilage_risk - self._prev["spoilage_risk"])
        reward = -(ENERGY_WEIGHT * energy + SPOILAGE_WEIGHT * spoilage_delta) - STEP_PENALTY

        params = get_params(s.fruit_type)
        ideal = (params.optimal_temp_low_c + params.optimal_temp_high_c) / 2.0
        deviation = abs(s.sensor_temperature_c - ideal)
        return reward, {"temp_deviation": deviation}

    def _routing_reward(self) -> tuple[float, dict[str, float]]:
        dt_time = self._state.route_travel_time - self._prev["route_travel_time"]
        dt_emissions = self._state.route_emissions - self._prev["route_emissions"]
        risk = self._state.shipment.spoilage_risk
        cost = (
            ROUTE_TIME_WEIGHT * dt_time
            + ROUTE_EMISSIONS_WEIGHT * dt_emissions
            + ROUTE_RISK_WEIGHT * risk
        )
        return -cost, {"route_cost": cost}
