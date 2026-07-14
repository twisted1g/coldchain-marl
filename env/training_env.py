from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from core import config as core_config
from core.config import DELIVERY_AGENTS, FruitKey
from core.fruits import get_params
from core.observations import all_obs
from core.spoilage import risk_to_label
from env.pettingzoo_adapter import ColdChainParallelEnv

DEFAULT_FRUIT = FruitKey.STRAWBERRY
DEFAULT_MAX_STEPS = 20
DEFAULT_BASE_SEED = 0
DEFAULT_LEARNERS = ("temperature",)

ENERGY_WEIGHT = 1.0
SPOILAGE_WEIGHT = 25.0
DEVIATION_WEIGHT = 2.0
STEP_PENALTY = 0.01

ROUTE_TIME_WEIGHT = 1.0
ROUTE_EMISSIONS_WEIGHT = 0.1
ROUTE_RISK_WEIGHT = 10.0
DELIVERY_BONUS = 100.0

SPOILAGE_PRED_WEIGHT = 3.0
SPOILAGE_FN_WEIGHT = 5.0
SPOILAGE_INSPECTION_WEIGHT = 0.2

INVENTORY_SPOILAGE_WEIGHT = 5.0
INVENTORY_HOLDING_WEIGHT = 1.0
INVENTORY_EMISSIONS_WEIGHT = 1.0

DELIVERY_DELAY_WEIGHT = 1.0
DELIVERY_SLA_WEIGHT = 5.0
DELIVERY_EMISSIONS_WEIGHT = 0.05
DELIVERY_CONFLICT_PENALTY = 5.0

RewardMethod = Callable[[], "tuple[float, dict[str, float]]"]


def _dynamic_pareto(costs: list[tuple[float, float]]) -> float:
    """Paper Alg 1-5 context-aware weights w_j = a_j*c_j/sum_k(a_k*c_k): returns sum(w_j*c_j)."""
    total = sum(a * c for a, c in costs)
    if total <= 0.0:
        return 0.0
    return sum(a * c * c for a, c in costs) / total


class ColdChainTrainingEnv(ColdChainParallelEnv):
    """Cold-chain env that shapes rewards for the trainable ("learner") agents.

    Each learner's reward comes from its own ``_<agent>_reward`` method; frozen agents
    keep the core's zero reward. To add a learner: define its reward method, register it
    in ``self._reward_methods``, and list it in training.config.LEARNERS.

    With ``config["forecaster"]`` set to a checkpoint path, the frozen transformer
    fills ``state.demand_forecast`` from the rolling history each step; otherwise the
    stub constant stays (stub-vs-transformer ablation).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._base_seed = config.get("base_seed", DEFAULT_BASE_SEED)
        self._forecaster = None
        self._predict_next = None
        forecaster_path = config.get("forecaster")
        if forecaster_path is not None:
            from training.forecaster import load_forecaster, predict_next

            self._forecaster = load_forecaster(forecaster_path)
            self._predict_next = predict_next
        super().__init__(
            max_steps=config.get("max_steps", DEFAULT_MAX_STEPS),
            fruit=FruitKey(config.get("fruit", DEFAULT_FRUIT)),
        )
        supported: dict[str, RewardMethod] = {
            "temperature": self._temperature_reward,
            "routing": self._routing_reward,
            "spoilage": self._spoilage_reward,
            "inventory": self._inventory_reward,
            **{
                name: partial(self._delivery_reward, i)
                for i, name in enumerate(DELIVERY_AGENTS)
            },
        }
        learners = config.get("learners", DEFAULT_LEARNERS)
        self._reward_methods = {a: supported[a] for a in learners}
        self._episode_index = 0
        self._prev: dict[str, float] = {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        episode_seed = self._base_seed + self._episode_index if seed is None else seed
        obs, infos = super().reset(seed=episode_seed)
        obs = self._apply_forecast(obs)
        self._episode_index += 1
        self._prev = {
            "spoilage_risk": self._state.shipment.spoilage_risk,
            "route_travel_time": self._state.route_travel_time,
            "route_emissions": self._state.route_emissions,
        }
        return obs, infos

    def _apply_forecast(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Forecast tomorrow's demand from the history window; rebuild obs so
        agents act on the updated ``demand_forecast``."""
        if self._forecaster is None:
            return obs
        state = self._state
        state.demand_forecast = self._predict_next(
            self._forecaster,
            state.history,
            (state.day_of_year + 1) % core_config.DAYS_PER_YEAR,
            (state.weekday + 1) % core_config.DAYS_PER_WEEK,
            state.ambient_weather,
        )
        return all_obs(state)

    def step(self, actions: dict[str, Any]):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        obs = self._apply_forecast(obs)
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

        params = get_params(s.fruit_type)
        ideal = (params.optimal_temp_low_c + params.optimal_temp_high_c) / 2.0
        deviation = abs(s.sensor_temperature_c - ideal)

        weighted = _dynamic_pareto(
            [(ENERGY_WEIGHT, energy), (SPOILAGE_WEIGHT, spoilage_delta)]
        )
        reward = -(weighted + DEVIATION_WEIGHT * deviation) - STEP_PENALTY
        return reward, {"temp_deviation": deviation}

    def _routing_reward(self) -> tuple[float, dict[str, float]]:
        dt_time = self._state.route_travel_time - self._prev["route_travel_time"]
        dt_emissions = self._state.route_emissions - self._prev["route_emissions"]
        risk = self._state.shipment.spoilage_risk
        cost = _dynamic_pareto(
            [
                (ROUTE_TIME_WEIGHT, dt_time),
                (ROUTE_EMISSIONS_WEIGHT, dt_emissions),
                (ROUTE_RISK_WEIGHT, risk),
            ]
        )
        s = self._state.shipment
        delivered = s.current_node == s.target_node
        reward = -cost + (DELIVERY_BONUS if delivered else 0.0)
        return reward, {"route_cost": cost, "delivered": float(delivered)}

    def _spoilage_reward(self) -> tuple[float, dict[str, float]]:
        pred = self._state.spoilage_prediction
        label = float(risk_to_label(self._state.shipment.spoilage_risk))
        pred_error = (pred - label) ** 2
        false_negative = 1.0 if (label == 1.0 and pred < 0.5) else 0.0
        reward = -_dynamic_pareto(
            [
                (SPOILAGE_PRED_WEIGHT, pred_error),
                (SPOILAGE_FN_WEIGHT, false_negative),
                (SPOILAGE_INSPECTION_WEIGHT, pred),
            ]
        )
        return reward, {
            "fn_rate": false_negative,
            "y_pred": pred,
            "spoilage_label": label,
        }

    def _inventory_reward(self) -> tuple[float, dict[str, float]]:
        s = self._state
        spoilage_loss = s.inventory_level * s.shipment.spoilage_risk
        holding = s.inventory_level
        emissions = s.inventory_order
        cost = _dynamic_pareto(
            [
                (INVENTORY_SPOILAGE_WEIGHT, spoilage_loss),
                (INVENTORY_HOLDING_WEIGHT, holding),
                (INVENTORY_EMISSIONS_WEIGHT, emissions),
            ]
        )
        return -cost, {
            "inventory_cost": cost,
            "unmet_demand": s.unmet_demand,
            "inventory_level": s.inventory_level,
        }

    def _delivery_reward(self, i: int) -> tuple[float, dict[str, float]]:
        v = self._state.vehicles[i]
        conflict = DELIVERY_CONFLICT_PENALTY if v.conflict else 0.0
        weighted = _dynamic_pareto(
            [
                (DELIVERY_DELAY_WEIGHT, v.delay),
                (DELIVERY_SLA_WEIGHT, float(v.sla_violated)),
                (DELIVERY_EMISSIONS_WEIGHT, v.emissions),
            ]
        )
        cost = weighted + conflict
        return -cost, {
            "delivery_cost": cost,
            "delay": v.delay,
            "sla_violated": float(v.sla_violated),
            "emissions": v.emissions,
            "conflict": float(v.conflict),
        }
