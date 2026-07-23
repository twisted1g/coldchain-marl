from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from core import config as core_config
from core.config import (
    DELIVERY_AGENTS,
    INVENTORY_AGENTS,
    ROUTING_AGENTS,
    SPOILAGE_AGENTS,
    TEMPERATURE_AGENTS,
    FruitKey,
)
from core.dynamics import expected_lead_time
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
# A delivery trip's raw carbon (~50) dwarfs holding/spoilage (~0.1-0.5); the
# self-normalising Pareto is dominated by its largest term, so unscaled Et would
# make every arrival a ~50 cost spike and relocate the order=0 collapse to the
# arrival tick. Scale the delivery carbon onto the inventory-cost range instead.
INVENTORY_EMISSIONS_SCALE = 0.01
# Additive shortage penalty (outside the Pareto sum) on the unmet-demand
# fraction in [0,1]. At 1.0 the shortage cliff is steep and asymmetric (sharp
# below demand, gentle above), so DDPG settles on the safe over-ordering side
# (~0.78 vs the ~0.4 optimum) and ties random. 0.5 softens the cliff: optimum
# ~0.35, and the cost-optimal policy beats random by ~13% (vs ~6% at 1.0).
INVENTORY_STOCKOUT_WEIGHT = 0.5

DELIVERY_DELAY_WEIGHT = 1.0
DELIVERY_SLA_WEIGHT = 5.0
DELIVERY_EMISSIONS_WEIGHT = 0.05
# Honest-transit routes make a trip's raw carbon (~50) dominate the
# self-normalising Pareto (weight a_j*value_j), swamping the delay/SLA slot
# levers the agent actually controls — so it stops optimising them and loses to
# random on slot_cost. Scale the (largely route-fixed, uncontrollable) carbon
# onto the delay/SLA range so the Pareto reflects the slot decision, mirroring
# INVENTORY_EMISSIONS_SCALE. The routing agent (Alg 1) owns emissions.
DELIVERY_EMISSIONS_SCALE = 0.01
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
            rolling=bool(config.get("rolling", False)),
        )
        supported: dict[str, RewardMethod] = {
            **{
                name: partial(self._temperature_reward, i)
                for i, name in enumerate(TEMPERATURE_AGENTS)
            },
            **{
                name: partial(self._routing_reward, i)
                for i, name in enumerate(ROUTING_AGENTS)
            },
            **{
                name: partial(self._spoilage_reward, i)
                for i, name in enumerate(SPOILAGE_AGENTS)
            },
            **{
                name: partial(self._inventory_reward, i)
                for i, name in enumerate(INVENTORY_AGENTS)
            },
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

    def rollover(self) -> dict[str, Any]:
        """Live rolling inference: keep the same world running (inventory, vehicles,
        cargo, calendar persist) past a terminal slot-span boundary. Mirrors the obs
        pipeline of reset/step without a full reset and restores the agent list the
        terminal step cleared. Returns the fresh observations.
        """
        self.agents = list(self.possible_agents)
        obs = self._apply_forecast(all_obs(self._state))
        self._snapshot_prev()
        return obs

    def _snapshot_prev(self) -> None:
        """Per-vehicle cumulative quantities whose deltas feed the routing/
        temperature rewards (singleton eliminated — one subject per truck)."""
        vehicles = self._state.vehicles
        self._prev = {
            "route_transit": [v.route_transit for v in vehicles],
            "route_emissions": [v.route_emissions for v in vehicles],
            "spoilage_risk": [
                v.load.spoilage_risk if v.load is not None else 0.0 for v in vehicles
            ],
        }

    def _apply_forecast(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Forecast demand for each instance's order-arrival day (lead time
        ahead) from its history window; rebuild obs so agents act on the
        updated ``demand_forecast``. An order placed now rides a vehicle, so
        the day that matters is the expected arrival day, not tomorrow."""
        if self._forecaster is None:
            return obs
        state = self._state
        for i, history in enumerate(state.histories):
            horizon = expected_lead_time(state, i)
            state.demand_forecast[i] = self._predict_next(
                self._forecaster,
                history,
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

    def _delivered_this_tick(self, i: int) -> bool:
        """Truck i's crate arrived this tick (its cargo's arrival_tick was just set
        in ``_deliver_vehicle``); the crate is already unloaded, so read the cargo."""
        return any(
            c.vehicle == i and c.arrival_tick == self._state.tick
            for c in self._state.cargo
        )

    def _temperature_reward(self, i: int) -> tuple[float, dict[str, float]]:
        """Paper Alg 2, per crate. Idle truck (no crate) contributes no signal."""
        crate = self._state.vehicles[i].load
        if crate is None:
            return 0.0, {"temp_deviation": 0.0}
        energy = crate.energy
        spoilage_delta = max(0.0, crate.spoilage_risk - self._prev["spoilage_risk"][i])
        deviation = abs(
            crate.sensor_temperature_c - get_params(crate.fruit_type).optimal_temp_c
        )
        weighted = _dynamic_pareto(
            [
                (DEVIATION_WEIGHT, deviation),
                (ENERGY_WEIGHT, energy),
                (SPOILAGE_WEIGHT, spoilage_delta),
            ]
        )
        return -weighted - STEP_PENALTY, {"temp_deviation": deviation}

    def _routing_reward(self, i: int) -> tuple[float, dict[str, float]]:
        """Paper Alg 1, per crate: cost = Pareto over [hop time, spoilage risk, hop
        emissions] with ctx = [dT, traffic, SLA priority]; a big bonus on arrival."""
        if self._delivered_this_tick(i):
            return DELIVERY_BONUS, {"route_cost": 0.0, "delivered": 1.0}
        v = self._state.vehicles[i]
        crate = v.load
        if crate is None:
            return 0.0, {"route_cost": 0.0, "delivered": 0.0}
        dt_time = v.route_transit - self._prev["route_transit"][i]
        dt_emissions = v.route_emissions - self._prev["route_emissions"][i]
        temp_deviation = abs(
            crate.sensor_temperature_c - get_params(crate.fruit_type).optimal_temp_c
        )
        traffic = node_delay(self._state, v.current_node)
        sla_priority = self._state.tick / self._state.max_steps
        cost = _dynamic_pareto(
            [
                (ROUTE_TIME_WEIGHT, dt_time),
                (ROUTE_RISK_WEIGHT, crate.spoilage_risk),
                (ROUTE_EMISSIONS_WEIGHT, dt_emissions),
            ],
            ctx=[temp_deviation, traffic, sla_priority],
        )
        return -cost, {"route_cost": cost, "delivered": 0.0}

    def _spoilage_reward(self, i: int) -> tuple[float, dict[str, float]]:
        """Paper Alg 3, per crate: penalise prediction error, false negatives, and
        inspection cost. Idle truck contributes no signal."""
        crate = self._state.vehicles[i].load
        if crate is None:
            return 0.0, {"fn_rate": 0.0, "y_pred": 0.0, "spoilage_label": 0.0}
        pred = crate.spoilage_prediction
        label = float(risk_to_label(crate.spoilage_risk))
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

    def _inventory_reward(self, i: int) -> tuple[float, dict[str, float]]:
        """Paper Alg 4 reward (Lspoil, H, E) plus a stockout term: the box omits
        it, but Section 4.1 tasks the agent with "match demand" / "anticipate
        shortages"; without it order=0 is optimal and the forecast is dead.
        Supply contention (line 9) is resolved by the "reassign" branch — the
        order queue defers a contended order to the next free vehicle — so no
        explicit coordination penalty ρ is added on top."""
        s = self._state
        level = s.inventory_levels[i]
        # Post-sale level is the leftover that did not meet demand, i.e. the
        # overstock that spoils (Alg 4 line 15, "overstocked perishables"). Without
        # a singleton shipment, held stock decays at the fruit's baseline rate.
        spoil_rate = 1.0 / get_params(s.fruit).base_shelf_life_ticks
        spoilage_loss = level * spoil_rate
        # Et = emissions of the delivery that arrived this tick (Alg 4 line 16),
        # not the order magnitude at order time — the latter penalised ordering
        # instantly while its payoff lagged by the lead time, so order=0 won.
        emissions = s.inventory_arrival_emissions[i] * INVENTORY_EMISSIONS_SCALE
        # Alg 4 box: 3-term Pareto with dynamic weights over ctx = [Lspoil, H, E].
        sustainability = _dynamic_pareto(
            [
                (INVENTORY_SPOILAGE_WEIGHT, spoilage_loss),
                (INVENTORY_HOLDING_WEIGHT, level),
                (INVENTORY_EMISSIONS_WEIGHT, emissions),
            ]
        )
        # Shortage stays outside the Pareto sum (like rho): Section 4.1 tasks the
        # agent to "match demand" / "anticipate shortages", but the Alg 4 box
        # omits the term. Inside the self-normalising Pareto it is diluted to a
        # convex-combination weight and cannot outweigh holding; as an additive
        # penalty its gradient survives and ordering-to-demand becomes optimal.
        shortage = s.unmet_demand[i] / max(s.demand_today[i], 1e-6)
        cost = sustainability + INVENTORY_STOCKOUT_WEIGHT * shortage
        return -cost, {
            "inventory_cost": cost,
            "order": s.inventory_order[i],
            "unmet_demand": s.unmet_demand[i],
            "inventory_level": level,
        }

    def _delivery_reward(self, i: int) -> tuple[float, dict[str, float]]:
        v = self._state.vehicles[i]
        conflict = DELIVERY_CONFLICT_PENALTY if v.conflict else 0.0
        emissions = v.emissions * DELIVERY_EMISSIONS_SCALE
        weighted = _dynamic_pareto(
            [
                (DELIVERY_DELAY_WEIGHT, v.delay),
                (DELIVERY_SLA_WEIGHT, float(v.sla_violated)),
                (DELIVERY_EMISSIONS_WEIGHT, emissions),
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
