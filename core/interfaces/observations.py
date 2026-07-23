from __future__ import annotations

import networkx as nx
import numpy as np

from core import config
from core.config import DisruptionType, Weather
from core.state import GlobalState
from core.world.fruits import get_params
from core.world.graph_features import (
    crate_spoilage_node_features,
    spoilage_node_features,
)

WEATHER_INDEX: dict[Weather, int] = {w: i for i, w in enumerate(Weather)}

_TRANSIT_SCALE = float(config.EDGE_BASE_TRANSIT_TIME_RANGE[1])
_EMISSIONS_SCALE = float(
    config.EDGE_DISTANCE_KM_RANGE[1] * config.EDGE_BASE_EMISSIONS_PER_KM
)
_EDGE_FEATURES = len(config.EDGE_FEATURE_NAMES)


def _traffic_status(state: GlobalState) -> float:
    n = sum(
        1
        for d in state.active_disruptions
        if d.type is DisruptionType.INCREASED_TRANSIT
    )
    return min(1.0, n / 5.0)


def _route_status_at(state: GlobalState, node: str) -> float:
    blocked = any(
        d.type is DisruptionType.BLOCKED_NODE and d.target == node
        for d in state.active_disruptions
    )
    return 1.0 if blocked else 0.0


def _route_status(state: GlobalState) -> float:
    return _route_status_at(state, state.shipment.current_node)


def _breakdown_alerts(state: GlobalState) -> float:
    return 1.0 if state.fault_signals > 0 else 0.0


def _node_index(state: GlobalState, node: str) -> float:
    nodes = list(state.graph.nodes)
    return nodes.index(node) / max(1, len(nodes) - 1)


def _routing_edge_features_at(
    state: GlobalState, current: str, dest: str
) -> list[float]:
    edges = list(state.graph.out_edges(current, data=True))
    feats: list[float] = []
    for _, target, data in edges[: config.N_NEXT_NODES]:
        reaches = target == dest or nx.has_path(state.graph, target, dest)
        feats.extend(
            [
                float(data["base_transit_time"]) / _TRANSIT_SCALE,
                float(data["base_emissions"]) / _EMISSIONS_SCALE,
                1.0 if reaches else 0.0,
                1.0 if target == dest else 0.0,
                1.0 if target == current else 0.0,
            ]
        )
    feats.extend([0.0] * _EDGE_FEATURES * (config.N_NEXT_NODES - len(edges)))
    return feats


def _routing_edge_features(state: GlobalState) -> list[float]:
    s = state.shipment
    return _routing_edge_features_at(state, s.current_node, s.target_node)


def routing_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            _traffic_status(state),
            float(WEATHER_INDEX[state.ambient_weather]),
            s.perishability_index,
            _route_status(state),
            s.spoilage_risk,
            _node_index(state, s.current_node),
            _node_index(state, s.target_node),
            *_routing_edge_features(state),
        ],
        dtype=np.float32,
    )


def crate_routing_obs(state: GlobalState, crate, current_node: str) -> np.ndarray:
    """Routing obs from a single crate's perspective at ``current_node`` (CTDE
    decentralised execution — the trained routing policy drives each crate toward
    its own target). Same layout as ``routing_obs``, subject is the crate."""
    return np.array(
        [
            _traffic_status(state),
            float(WEATHER_INDEX[state.ambient_weather]),
            crate.perishability_index,
            _route_status_at(state, current_node),
            crate.spoilage_risk,
            _node_index(state, current_node),
            _node_index(state, crate.target_node),
            *_routing_edge_features_at(state, current_node, crate.target_node),
        ],
        dtype=np.float32,
    )


def temperature_obs(state: GlobalState) -> np.ndarray:
    return _temperature_obs_fields(
        state,
        state.shipment.sensor_temperature_c,
        state.shipment.sensor_humidity,
        state.shipment.desired_temperature_c,
        state.energy_usage,
    )


def crate_temperature_obs(state: GlobalState, crate) -> np.ndarray:
    """Temperature obs built from a single truck-borne crate (CTDE decentralized
    execution — the trained temperature policy runs per crate at inference). Same
    layout as ``temperature_obs``; energy is the crate's |ΔT| against its OWN node
    climate, so a crate at a warm hub sees a different cooling load than one in a
    cold DC and the agent reacts to the local node, not the global outside air."""
    external = state.node_temp_c.get(crate.current_node, state.ambient_temp_c)
    energy = abs(crate.sensor_temperature_c - external) * 0.1
    return _temperature_obs_fields(
        state,
        crate.sensor_temperature_c,
        crate.sensor_humidity,
        crate.desired_temperature_c,
        energy,
    )


def _temperature_obs_fields(
    state: GlobalState,
    sensor_temp: float,
    humidity: float,
    desired_temp: float,
    energy: float,
) -> np.ndarray:
    return np.array(
        [
            sensor_temp,
            humidity,
            desired_temp,
            energy,
            float(state.fault_signals),
        ],
        dtype=np.float32,
    )


def spoilage_obs(state: GlobalState) -> np.ndarray:
    return spoilage_node_features(state).flatten()


def crate_spoilage_obs(state: GlobalState, crate) -> np.ndarray:
    """Spoilage GNN obs for one truck-borne crate (CTDE decentralised execution —
    same layout as ``spoilage_obs``, subject is the crate not the shipment)."""
    return crate_spoilage_node_features(state, crate).flatten()


def inventory_obs(state: GlobalState, i: int) -> np.ndarray:
    # shelf life of the stock actually inbound to retailer i: the crate riding to
    # it (freshest concern if several), else the fruit's full shelf as a fresh
    # baseline. Per-instance, not the singleton shipment's age.
    inbound = [
        v.load
        for v in state.vehicles
        if v.carrying is not None and v.carrying.instance == i and v.load is not None
    ]
    if inbound:
        shelf_remaining = min(
            max(0, get_params(c.fruit_type).base_shelf_life_ticks - c.age_ticks)
            for c in inbound
        )
    else:
        shelf_remaining = get_params(state.fruit).base_shelf_life_ticks
    on_order = sum(qty for inst, qty in state.order_queue if inst == i) + sum(
        c.qty for c in state.cargo if c.instance == i
    )
    return np.array(
        [
            state.inventory_levels[i],
            on_order,
            state.demand_forecast[i],
            float(shelf_remaining),
        ],
        dtype=np.float32,
    )


def delivery_obs(state: GlobalState, i: int) -> np.ndarray:
    v = state.vehicles[i]
    horizon = max(1, state.max_steps)
    # spoilage of THIS vehicle's own cargo (0 when idle) — each delivery agent
    # reacts to the crate it carries, not the singleton shipment's risk
    cargo_risk = v.load.spoilage_risk if v.load is not None else 0.0
    return np.array(
        [
            i / max(1, config.N_VEHICLES - 1),
            1.0 if state.tick >= v.busy_until else 0.0,
            float(v.sla_window_ticks) / horizon,
            cargo_risk,
            _breakdown_alerts(state),
            v.route_transit / horizon,
        ],
        dtype=np.float32,
    )


_ROUTING_DIM = len(config.ROUTING_OBS_FIELDS)
_TEMPERATURE_DIM = len(config.TEMPERATURE_OBS_FIELDS)
_SPOILAGE_DIM = len(config.SPOILAGE_OBS_FIELDS)


def vehicle_routing_obs(state: GlobalState, i: int) -> np.ndarray:
    """Routing obs for truck i's crate, or a zero vector when the truck is idle
    (no crate to route — the slot is masked)."""
    v = state.vehicles[i]
    if v.load is None:
        return np.zeros(_ROUTING_DIM, dtype=np.float32)
    return crate_routing_obs(state, v.load, v.current_node)


def vehicle_temperature_obs(state: GlobalState, i: int) -> np.ndarray:
    v = state.vehicles[i]
    if v.load is None:
        return np.zeros(_TEMPERATURE_DIM, dtype=np.float32)
    return crate_temperature_obs(state, v.load)


def vehicle_spoilage_obs(state: GlobalState, i: int) -> np.ndarray:
    v = state.vehicles[i]
    if v.load is None:
        return np.zeros(_SPOILAGE_DIM, dtype=np.float32)
    return crate_spoilage_obs(state, v.load)


def all_obs(state: GlobalState) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    for i, name in enumerate(config.ROUTING_AGENTS):
        obs[name] = vehicle_routing_obs(state, i)
    for i, name in enumerate(config.TEMPERATURE_AGENTS):
        obs[name] = vehicle_temperature_obs(state, i)
    for i, name in enumerate(config.SPOILAGE_AGENTS):
        obs[name] = vehicle_spoilage_obs(state, i)
    for i, name in enumerate(config.INVENTORY_AGENTS):
        obs[name] = inventory_obs(state, i)
    for i, name in enumerate(config.DELIVERY_AGENTS):
        obs[name] = delivery_obs(state, i)
    return obs
