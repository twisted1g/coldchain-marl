from __future__ import annotations

import networkx as nx
import numpy as np

from core import config
from core.config import DisruptionType, Weather
from core.state import GlobalState
from core.world.fruits import get_params
from core.world.graph_features import spoilage_node_features

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


def _route_status(state: GlobalState) -> float:
    blocked_at_current = any(
        d.type is DisruptionType.BLOCKED_NODE
        and d.target == state.shipment.current_node
        for d in state.active_disruptions
    )
    return 1.0 if blocked_at_current else 0.0


def _breakdown_alerts(state: GlobalState) -> float:
    return 1.0 if state.fault_signals > 0 else 0.0


def _node_index(state: GlobalState, node: str) -> float:
    nodes = list(state.graph.nodes)
    return nodes.index(node) / max(1, len(nodes) - 1)


def _routing_edge_features(state: GlobalState) -> list[float]:
    s = state.shipment
    edges = list(state.graph.out_edges(s.current_node, data=True))
    feats: list[float] = []
    for _, target, data in edges[: config.N_NEXT_NODES]:
        reaches = target == s.target_node or nx.has_path(
            state.graph, target, s.target_node
        )
        feats.extend(
            [
                float(data["base_transit_time"]) / _TRANSIT_SCALE,
                float(data["base_emissions"]) / _EMISSIONS_SCALE,
                1.0 if reaches else 0.0,
                1.0 if target == s.target_node else 0.0,
                1.0 if target == s.current_node else 0.0,
            ]
        )
    feats.extend([0.0] * _EDGE_FEATURES * (config.N_NEXT_NODES - len(edges)))
    return feats


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
    execution — the trained temperature policy runs per crate at inference).
    Same layout as ``temperature_obs``; energy is the crate's own |ΔT_ambient|."""
    energy = abs(crate.sensor_temperature_c - state.ambient_temp_c) * 0.1
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


def inventory_obs(state: GlobalState, i: int) -> np.ndarray:
    s = state.shipment
    shelf_remaining = max(
        0, get_params(s.fruit_type).base_shelf_life_ticks - s.age_ticks
    )
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
    s = state.shipment
    v = state.vehicles[i]
    horizon = max(1, state.max_steps)
    return np.array(
        [
            i / max(1, config.N_VEHICLES - 1),
            1.0 if state.tick >= v.busy_until else 0.0,
            float(v.sla_window_ticks) / horizon,
            s.spoilage_risk,
            _breakdown_alerts(state),
            v.route_transit / horizon,
        ],
        dtype=np.float32,
    )


def all_obs(state: GlobalState) -> dict[str, np.ndarray]:
    obs = {
        "routing": routing_obs(state),
        "temperature": temperature_obs(state),
        "spoilage": spoilage_obs(state),
    }
    for i, name in enumerate(config.INVENTORY_AGENTS):
        obs[name] = inventory_obs(state, i)
    for i, name in enumerate(config.DELIVERY_AGENTS):
        obs[name] = delivery_obs(state, i)
    return obs
