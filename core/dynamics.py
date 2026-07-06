from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core import config
from core.config import OBS_FIELDS_BY_AGENT, DisruptionType
from core.noise import NoiseModel
from core.observations import all_obs
from core.spoilage import ArrheniusSpoilage, risk_to_label
from core.state import GlobalState


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

    _apply_temperature_action(state, actions.get("temperature"))
    _apply_routing_action(state, actions.get("routing"))
    _apply_inventory_action(state, actions.get("inventory"))
    _apply_delivery_action(state, actions.get("delivery"))
    _apply_spoilage_action(state, actions.get("spoilage"))

    _advance_thermal_state(state)
    _advance_spoilage(state)
    _maybe_sample_disruption(state)
    _update_energy(state)

    delivered = state.shipment.current_node == state.shipment.target_node
    done = state.tick >= state.max_steps or delivered
    if done:
        state.shipment.ground_truth_label = risk_to_label(state.shipment.spoilage_risk)

    if state.customer_window_ticks > 0:
        state.customer_window_ticks -= 1

    observations = all_obs(state)
    rewards = {agent: 0.0 for agent in OBS_FIELDS_BY_AGENT}
    terminated = {agent: done for agent in OBS_FIELDS_BY_AGENT}
    terminated["__all__"] = done
    truncated = {agent: False for agent in OBS_FIELDS_BY_AGENT}
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
        np.clip(value, config.TEMPERATURE_ACTION_LOW_C, config.TEMPERATURE_ACTION_HIGH_C)
    )


def _apply_inventory_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    value = float(np.asarray(action).flatten()[0])
    state.inventory_level = float(np.clip(state.inventory_level + value, 0.0, 1.0))


def _apply_delivery_action(state: GlobalState, action: Any) -> None:
    if action is None:
        return
    idx = int(action) % config.N_DELIVERY_WINDOWS
    remaining_window = max(1, state.max_steps - state.tick)
    state.customer_window_ticks = int(remaining_window * (idx + 1) / config.N_DELIVERY_WINDOWS)


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


def _advance_spoilage(state: GlobalState) -> None:
    s = state.shipment
    delta = _spoilage_model.risk_delta(s.fruit_type, s.sensor_temperature_c, dt_ticks=1.0)
    s.spoilage_risk = float(np.clip(s.spoilage_risk + delta, 0.0, 1.0))
    s.freshness_score = float(max(0.0, 1.0 - s.spoilage_risk))
    s.age_ticks += 1


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
    return {
        "routing": {"delivered": delivered},
        "temperature": {"energy_usage": state.energy_usage},
        "spoilage": {
            "y_pred": state.spoilage_prediction,
            "ground_truth_label": state.shipment.ground_truth_label,
        },
        "inventory": {"inventory_level": state.inventory_level},
        "delivery": {
            "customer_window_ticks": state.customer_window_ticks,
            "delivered": delivered,
        },
    }
