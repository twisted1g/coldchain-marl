from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from core import config as core_config
from core.config import DELIVERY_AGENTS, FruitKey
from core.interfaces.observations import all_obs
from core.world.fruits import get_params
from core.world.graph_features import node_delay
from core.world.spoilage import risk_to_label
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
INVENTORY_STOCKOUT_WEIGHT = 5.0

DELIVERY_DELAY_WEIGHT = 1.0
DELIVERY_SLA_WEIGHT = 5.0
DELIVERY_EMISSIONS_WEIGHT = 0.05
DELIVERY_CONFLICT_PENALTY = 5.0

RewardMethod = Callable[[], "tuple[float, dict[str, float]]"]


def _dynamic_pareto(
    costs: list[tuple[float, float]], ctx: list[float] | None = None
) -> float:
    """Paper Alg 1-5 weights w_j = a_j*ctx_j/sum_k(a_k*ctx_k) applied to costs c_j.

    Boxes 2-5 define ctx as the cost components themselves (the default);
    Alg 1 computes weights from a separate context vector.
    """
    ctx = [c for _, c in costs] if ctx is None else ctx
    total = sum(a * x for (a, _), x in zip(costs, ctx, strict=True))
    if total <= 0.0:
        return 0.0
    return sum(a * x * c for (a, c), x in zip(costs, ctx, strict=True)) / total


class ColdChainTrainingEnv(ColdChainParallelEnv):
    """Cold-chain env that shapes rewards for the trainable ("learner") agents.

    Each learner's reward comes from its own ``_<agent>_reward`` method; frozen agents
    keep the core's zero reward. To add a learner: define its reward method, register it
    in ``self._reward_methods``, and list it in training.config.LEARNERS.

    With ``config["forecaster"]`` set to a checkpoint path, the frozen transformer
    fills ``state.demand_forecast`` from the rolling history each step; otherwise the
    stub constant stays (stub-vs-transformer ablation).

    With ``config["scenario_bank"]`` set to a bank path, episodes replay one
    LLM-generated disruption scenario (drawn via the episode rng, or fixed with
    ``options={"scenario_id": ...}`` on reset). ``config["scenario_prob"]``
    (default 1.0) gates the random draw: with 0.5 half the episodes stay clean,
    which preserves clean-regime behavior during robustness training.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self._base_seed = config.get("base_seed", DEFAULT_BASE_SEED)
        self._forecaster = None
        self._predict_next = None
        forecaster_path = config.get("forecaster")
        if forecaster_path is not None:
            from training.forecaster.model import load_forecaster, predict_next

            self._forecaster = load_forecaster(forecaster_path)
            self._predict_next = predict_next
        self._scenario_bank = None
        self._scenario_runner = None
        bank_path = config.get("scenario_bank")
        self._scenario_prob = float(config.get("scenario_prob", 1.0))
        if bank_path is not None:
            from llm.scenarios import load_bank

            self._scenario_bank = load_bank(bank_path)
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
        self._scenario_runner = self._pick_scenario(options)
        self._episode_index += 1
        self._snapshot_prev()
        return obs, infos

    def _snapshot_prev(self) -> None:
        """Remember the cumulative quantities whose deltas feed the rewards."""
        self._prev = {
            "spoilage_risk": self._state.shipment.spoilage_risk,
            "route_travel_time": self._state.route_travel_time,
            "route_emissions": self._state.route_emissions,
        }

    def _apply_forecast(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Forecast demand for the order-arrival day (lead time ahead) from the
        history window; rebuild obs so agents act on the updated
        ``demand_forecast``. An order placed now arrives k ticks later, so the
        demand that matters is day t+k, not tomorrow."""
        if self._forecaster is None:
            return obs
        state = self._state
        horizon = core_config.INVENTORY_LEAD_TIME_TICKS
        state.demand_forecast = self._predict_next(
            self._forecaster,
            state.history,
            (state.day_of_year + horizon) % core_config.DAYS_PER_YEAR,
            (state.weekday + horizon) % core_config.DAYS_PER_WEEK,
            state.ambient_weather,
        )
        return all_obs(state)

    def _pick_scenario(self, options: dict[str, Any] | None):
        if not self._scenario_bank:
            return None
        from env.scenarios import ScenarioRunner

        wanted = (options or {}).get("scenario_id")
        if wanted is not None:
            scenario = next(s for s in self._scenario_bank if s.id == wanted)
        else:
            if self._state.rng.random() >= self._scenario_prob:
                return None
            idx = int(self._state.rng.integers(0, len(self._scenario_bank)))
            scenario = self._scenario_bank[idx]
        return ScenarioRunner(scenario, self._state)

    def step(self, actions: dict[str, Any]):
        if self._scenario_runner is not None:
            self._scenario_runner.before_step(self._state)
        obs, rewards, terminated, truncated, infos = super().step(actions)
        obs = self._apply_forecast(obs)
        for agent, reward_method in self._reward_methods.items():
            reward, metrics = reward_method()
            rewards[agent] = reward
            infos[agent].update(metrics)
        self._snapshot_prev()
        return obs, rewards, terminated, truncated, infos

    def _temperature_reward(self) -> tuple[float, dict[str, float]]:
        s = self._state.shipment
        energy = float(self._state.energy_usage)
        spoilage_delta = max(0.0, s.spoilage_risk - self._prev["spoilage_risk"])

        deviation = abs(
            s.sensor_temperature_c - get_params(s.fruit_type).optimal_temp_c
        )

        weighted = _dynamic_pareto(
            [
                (DEVIATION_WEIGHT, deviation),
                (ENERGY_WEIGHT, energy),
                (SPOILAGE_WEIGHT, spoilage_delta),
            ]
        )
        reward = -weighted - STEP_PENALTY
        return reward, {"temp_deviation": deviation}

    def _routing_reward(self) -> tuple[float, dict[str, float]]:
        """Paper Alg 1: weights from ctx = [dT, traffic, SLA priority] applied to
        costs [t, sigma, e]. Traffic = disruption delay at the current node; SLA
        priority = episode-deadline pressure (paper leaves both undefined)."""
        dt_time = self._state.route_travel_time - self._prev["route_travel_time"]
        dt_emissions = self._state.route_emissions - self._prev["route_emissions"]
        risk = self._state.shipment.spoilage_risk

        s = self._state.shipment
        temp_deviation = abs(
            s.sensor_temperature_c - get_params(s.fruit_type).optimal_temp_c
        )
        traffic = node_delay(self._state, s.current_node)
        sla_priority = self._state.tick / self._state.max_steps

        cost = _dynamic_pareto(
            [
                (ROUTE_TIME_WEIGHT, dt_time),
                (ROUTE_RISK_WEIGHT, risk),
                (ROUTE_EMISSIONS_WEIGHT, dt_emissions),
            ],
            ctx=[temp_deviation, traffic, sla_priority],
        )
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
        """Paper Alg 4 reward (Lspoil, H, E) plus a stockout term: the box omits
        it, but Section 4.1 tasks the agent with "match demand" / "anticipate
        shortages"; without it order=0 is optimal and the forecast is dead."""
        s = self._state
        spoilage_loss = s.inventory_level * s.shipment.spoilage_risk
        holding = s.inventory_level
        emissions = s.inventory_order
        cost = _dynamic_pareto(
            [
                (INVENTORY_SPOILAGE_WEIGHT, spoilage_loss),
                (INVENTORY_HOLDING_WEIGHT, holding),
                (INVENTORY_EMISSIONS_WEIGHT, emissions),
                (INVENTORY_STOCKOUT_WEIGHT, s.unmet_demand),
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
        slot_cost = (
            DELIVERY_DELAY_WEIGHT * v.delay
            + DELIVERY_SLA_WEIGHT * float(v.sla_violated)
            + conflict
        )
        return -cost, {
            "delivery_cost": cost,
            "slot_cost": slot_cost,
            "delay": v.delay,
            "sla_violated": float(v.sla_violated),
            "emissions": v.emissions,
            "conflict": float(v.conflict),
        }
