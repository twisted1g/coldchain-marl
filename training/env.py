from __future__ import annotations

from typing import Any

from core.config import FruitKey
from core.fruits import get_params
from env.pettingzoo_adapter import ColdChainParallelEnv

DEFAULT_FRUIT = FruitKey.STRAWBERRY
DEFAULT_MAX_STEPS = 20
DEFAULT_BASE_SEED = 0

# w2 is tuned so the interior optimum of the energy/spoilage trade-off lands on
# the fruit's ideal midpoint: the energy gradient is a constant -0.1*w1 while
# sensor < ambient, balanced against the Arrhenius spoilage gradient there.
ENERGY_WEIGHT = 1.0
SPOILAGE_WEIGHT = 25.0
STEP_PENALTY = 0.01


class TemperatureTrainingEnv(ColdChainParallelEnv):
    """PettingZoo env where only the temperature agent gets a shaped reward.

    The fruit is fixed so the ideal band is stationary and learnable, and
    episodes follow a deterministic seed sequence for reproducible curves.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._base_seed = config.get("base_seed", DEFAULT_BASE_SEED)
        self._energy_weight = config.get("energy_weight", ENERGY_WEIGHT)
        self._spoilage_weight = config.get("spoilage_weight", SPOILAGE_WEIGHT)
        self._step_penalty = config.get("step_penalty", STEP_PENALTY)
        super().__init__(
            max_steps=config.get("max_steps", DEFAULT_MAX_STEPS),
            fruit=FruitKey(config.get("fruit", DEFAULT_FRUIT)),
        )
        self._episode_index = 0
        self._prev_spoilage_risk = 0.0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, infos = super().reset(seed=self._base_seed + self._episode_index)
        self._episode_index += 1
        self._prev_spoilage_risk = self._state.shipment.spoilage_risk
        return obs, infos

    def step(self, actions: dict[str, Any]):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        energy = float(infos["temperature"]["energy_usage"])
        risk = self._state.shipment.spoilage_risk
        spoilage_delta = max(0.0, risk - self._prev_spoilage_risk)
        self._prev_spoilage_risk = risk

        # PHASE 4: context-aware Pareto weights
        rewards["temperature"] = -(
            self._energy_weight * energy + self._spoilage_weight * spoilage_delta
        ) - self._step_penalty
        infos["temperature"]["temp_deviation"] = self.temperature_deviation()
        return obs, rewards, terminated, truncated, infos

    def ideal_temperature_c(self) -> float:
        params = get_params(self._fruit)
        return (params.optimal_temp_low_c + params.optimal_temp_high_c) / 2.0

    def temperature_deviation(self) -> float:
        assert self._state is not None
        return abs(self._state.shipment.sensor_temperature_c - self.ideal_temperature_c())
