from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core import config
from core.config import OBS_FIELDS_BY_AGENT, DisruptionType
from core.interfaces.intention import IntentionBuffer
from core.interfaces.observations import all_obs
from core.state import Cargo, GlobalState
from core.world import demand
from core.world.graph_features import node_delay
from core.world.noise import NoiseModel
from core.world.spoilage import ArrheniusSpoilage, risk_to_label


@dataclass(slots=True)
class StepResult:
    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict[str, Any]]


_spoilage_model = ArrheniusSpoilage()


def step(state: GlobalState, actions: dict[str, Any]) -> StepResult:
    state.tick += 1
    _advance_calendar(state)

    buffer = IntentionBuffer()
    buffer.declare_all(actions)
    conflicts = buffer.detect()

    _apply_temperature_action(state, actions.get("temperature"))
    _apply_routing_action(state, actions.get("routing"))
    _apply_inventory_action(state, actions.get("inventory"))
    _apply_delivery_action(state, actions, conflicts)
    _apply_spoilage_action(state, actions.get("spoilage"))

    _advance_thermal_state(state)
    _advance_humidity(state)
    _advance_spoilage(state)
    _advance_cargo(state)
    _maybe_sample_disruption(state)
    _update_energy(state)

    delivered = state.shipment.current_node == state.shipment.target_node
    done = state.tick >= state.max_steps or delivered
    if done:
        state.shipment.ground_truth_label = risk_to_label(state.shipment.spoilage_risk)

    observations = all_obs(state)
    rewards = dict.fromkeys(OBS_FIELDS_BY_AGENT, 0.0)
    terminated = dict.fromkeys(OBS_FIELDS_BY_AGENT, done)
    terminated["__all__"] = done
    truncated = dict.fromkeys(OBS_FIELDS_BY_AGENT, False)
    truncated["__all__"] = False
    infos = _build_infos(state, delivered)

    return StepResult(
        observations=observations,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        infos=infos,
    )


def _apply_routing_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    edges = list(state.graph.out_edges(state.shipment.current_node, data=True))
    if not edges:
        return
    idx = int(action) % len(edges)
    _, target, data = edges[idx]
    blocked = any(
        d.type is DisruptionType.BLOCKED_NODE and d.target == target
        for d in state.active_disruptions
    )
    if blocked:
        return
    state.route_travel_time += float(data["base_transit_time"])
    state.route_emissions += float(data["base_emissions"])
    state.shipment.current_node = target


def _apply_temperature_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    value = float(np.asarray(action).flatten()[0])
    state.shipment.desired_temperature_c = float(
        np.clip(
            value, config.TEMPERATURE_ACTION_LOW_C, config.TEMPERATURE_ACTION_HIGH_C
        )
    )


def _apply_inventory_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    arrived = sum(c.qty for c in state.cargo if c.arrival_tick <= state.tick)
    state.cargo = [c for c in state.cargo if c.arrival_tick > state.tick]
    level = min(state.inventory_level + arrived, 1.0)

    order = float(
        np.clip(
            np.asarray(action).flatten()[0],
            config.INVENTORY_ACTION_LOW,
            config.INVENTORY_ACTION_HIGH,
        )
    )
    qty = order * config.INVENTORY_RESTOCK_SCALE
    if qty > config.INVENTORY_MIN_ORDER_QTY:
        state.order_queue.append(qty)

    sold = min(level, state.demand_today)
    state.unmet_demand = state.demand_today - sold
    state.inventory_order = order
    state.inventory_level = float(np.clip(level - sold, 0.0, 1.0))


def slot_deadline(slot: int, max_steps: int) -> float:
    """Tick by which a delivery in this slot must arrive (slots split the episode)."""
    return (slot + 1) / config.N_DELIVERY_WINDOWS * max_steps


def slot_start(slot: int, max_steps: int) -> float:
    """Tick at which this slot's delivery window opens."""
    return slot / config.N_DELIVERY_WINDOWS * max_steps


def _vehicle_free(state: GlobalState, i: int) -> bool:
    return state.tick >= state.vehicles[i].busy_until


def expected_lead_time(state: GlobalState) -> int:
    """Ticks until an order placed now would arrive: transit of the nearest
    free vehicle, or wait-until-free + transit if all are busy. Ignores slot
    scheduling and queue backlog — an estimate for the demand forecast horizon."""
    eta = min(
        max(0, v.busy_until - state.tick) + int(np.ceil(v.route_transit))
        for v in state.vehicles
    )
    return max(1, eta)


def _apply_delivery_action(
    state: GlobalState, actions: dict[str, Any], conflicts: dict[str, bool]
) -> None:
    for i, vehicle in enumerate(state.vehicles):
        action = actions.get(f"delivery_{i}")
        slot = 0 if action is None else int(action) % config.N_DELIVERY_WINDOWS
        vehicle.chosen_slot = slot
        vehicle.delay = 0.0
        vehicle.sla_violated = False
        vehicle.emissions = 0.0
        vehicle.conflict = conflicts.get(f"delivery_{i}", False)
    _dispatch_orders(state)


def _dispatch_orders(state: GlobalState) -> None:
    """Nearest free vehicle takes the queue head (paper Alg 5 line 15: distance
    heuristic); its chosen slot schedules the trip. Departure waits for the slot
    window; delay/SLA/emissions accrue on the dispatch tick (per trip, not per
    tick — an idle vehicle neither drives nor emits)."""
    while state.order_queue:
        free = [i for i in range(len(state.vehicles)) if _vehicle_free(state, i)]
        if not free:
            return
        i = min(free, key=lambda j: state.vehicles[j].route_transit)
        vehicle = state.vehicles[i]
        qty = state.order_queue.pop(0)

        departure = max(
            state.tick, int(np.ceil(slot_start(vehicle.chosen_slot, state.max_steps)))
        )
        arrival = departure + int(np.ceil(vehicle.route_transit))
        deadline = slot_deadline(vehicle.chosen_slot, state.max_steps)

        vehicle.busy_until = arrival
        vehicle.delay = max(0.0, arrival - deadline)
        vehicle.sla_violated = arrival > deadline
        vehicle.emissions = vehicle.route_emissions
        state.cargo.append(
            Cargo(vehicle=i, departure_tick=departure, arrival_tick=arrival, qty=qty)
        )


def _apply_spoilage_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    value = float(np.asarray(action).flatten()[0])
    state.spoilage_prediction = float(np.clip(value, 0.0, 1.0))


def _advance_thermal_state(state: GlobalState) -> None:
    s = state.shipment
    diff = s.desired_temperature_c - s.sensor_temperature_c
    ambient_pull = (state.ambient_temp_c - s.sensor_temperature_c) * 0.05
    s.sensor_temperature_c = s.sensor_temperature_c + 0.5 * diff + ambient_pull


def _advance_humidity(state: GlobalState) -> None:
    s = state.shipment
    pull = (state.ambient_humidity - s.sensor_humidity) * config.HUMIDITY_AMBIENT_PULL
    noise = float(state.rng.normal(0.0, config.HUMIDITY_NOISE_SIGMA))
    s.sensor_humidity = float(np.clip(s.sensor_humidity + pull + noise, 0.0, 1.0))


def _advance_spoilage(state: GlobalState) -> None:
    s = state.shipment
    delay = node_delay(state, s.current_node)
    delta = _spoilage_model.risk_delta(
        s.fruit_type, s.sensor_temperature_c, s.sensor_humidity, delay, dt_ticks=1.0
    )
    s.spoilage_risk = float(np.clip(s.spoilage_risk + delta, 0.0, 1.0))
    s.freshness_score = float(max(0.0, 1.0 - s.spoilage_risk))
    s.age_ticks += 1


def _advance_calendar(state: GlobalState) -> None:
    state.day_of_year = (state.day_of_year + 1) % config.DAYS_PER_YEAR
    state.weekday = (state.weekday + 1) % config.DAYS_PER_WEEK
    state.event_days_left, state.event_multiplier = demand.advance_event(
        state.inventory_rng, state.event_days_left, state.event_multiplier
    )
    state.demand_mean = state.demand_shock_mult * demand.demand_mean(
        state.day_of_year, state.weekday, state.ambient_weather, state.event_multiplier
    )
    state.demand_today = state.demand_shock_mult * demand.sample_demand(
        state.inventory_rng,
        state.day_of_year,
        state.weekday,
        state.ambient_weather,
        state.event_multiplier,
    )
    demand.push_history(
        state.history,
        state.day_of_year,
        state.weekday,
        state.ambient_weather,
        state.event_multiplier,
        state.demand_mean,
        state.demand_today,
    )


def _advance_cargo(state: GlobalState) -> None:
    """In-transit spoilage: moving cargo decays with the chain-wide spoilage
    risk (single-shipment proxy until multi-instance). Queued cargo and cargo
    waiting for its slot window sit in cold storage and do not decay."""
    decay = config.TRANSIT_SPOILAGE_RATE * state.shipment.spoilage_risk
    state.transit_loss = 0.0
    for c in state.cargo:
        if c.departure_tick <= state.tick < c.arrival_tick:
            lost = c.qty * decay
            c.qty -= lost
            state.transit_loss += lost


def _maybe_sample_disruption(state: GlobalState) -> None:
    noise = NoiseModel(state.rng)
    new_disruption = noise.sample_disruption(state.graph)
    if new_disruption is not None:
        state.active_disruptions.append(new_disruption)


def _update_energy(state: GlobalState) -> None:
    s = state.shipment
    diff = abs(s.sensor_temperature_c - state.ambient_temp_c)
    state.energy_usage = diff * 0.1


def _build_infos(
    state: GlobalState,
    delivered: bool,
) -> dict[str, dict[str, Any]]:
    infos = {
        "routing": {"delivered": delivered},
        "temperature": {"energy_usage": state.energy_usage},
        "spoilage": {
            "y_pred": state.spoilage_prediction,
            "ground_truth_label": state.shipment.ground_truth_label,
        },
        "inventory": {
            "inventory_level": state.inventory_level,
            "transit_loss": state.transit_loss,
        },
    }
    for i, vehicle in enumerate(state.vehicles):
        infos[f"delivery_{i}"] = {
            "delay": vehicle.delay,
            "sla_violated": vehicle.sla_violated,
            "emissions": vehicle.emissions,
            "conflict": vehicle.conflict,
        }
    return infos
