from __future__ import annotations

import networkx as nx
import numpy as np

from core import config
from core.config import DisruptionType, Weather
from core.fruits import get_params
from core.state import GlobalState

WEATHER_INDEX: dict[Weather, int] = {w: i for i, w in enumerate(Weather)}

_TRANSIT_SCALE = float(config.EDGE_BASE_TRANSIT_TIME_RANGE[1])
_EMISSIONS_SCALE = float(config.EDGE_DISTANCE_KM_RANGE[1] * config.EDGE_BASE_EMISSIONS_PER_KM)
_EDGE_FEATURES = 5


def _traffic_status(state: GlobalState) -> float:
    n = sum(
        1 for d in state.active_disruptions if d.type is DisruptionType.INCREASED_TRANSIT
    )
    return min(1.0, n / 5.0)


def _route_status(state: GlobalState) -> float:
    blocked_at_current = any(
        d.type is DisruptionType.BLOCKED_NODE and d.target == state.shipment.current_node
        for d in state.active_disruptions
    )
    return 1.0 if blocked_at_current else 0.0


def _inspection_alerts(state: GlobalState) -> float:
    n = sum(1 for d in state.active_disruptions if d.type is DisruptionType.RISK_FLAG)
    return min(1.0, n / 5.0)


def _breakdown_alerts(state: GlobalState) -> float:
    return 1.0 if state.fault_signals > 0 else 0.0


def _location_index(state: GlobalState) -> float:
    nodes = list(state.graph.nodes)
    denom = max(1, len(nodes) - 1)
    return nodes.index(state.shipment.current_node) / denom


def _target_index(state: GlobalState) -> float:
    nodes = list(state.graph.nodes)
    denom = max(1, len(nodes) - 1)
    return nodes.index(state.shipment.target_node) / denom


def _routing_edge_features(state: GlobalState) -> list[float]:
    """Per-action-slot edge features in the same order ``_apply_routing_action``
    indexes ``out_edges``. Each slot: [transit_norm, emissions_norm,
    reaches_target, is_target, is_wait]. Missing slots are zero-padded so the
    agent sees them as dead ends (reaches_target = 0)."""
    s = state.shipment
    edges = list(state.graph.out_edges(s.current_node, data=True))
    feats: list[float] = []
    for _, target, data in edges[: config.N_NEXT_NODES]:
        reaches = target == s.target_node or nx.has_path(state.graph, target, s.target_node)
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
            _location_index(state),
            _target_index(state),
            *_routing_edge_features(state),
        ],
        dtype=np.float32,
    )


def temperature_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            s.sensor_temperature_c,
            s.sensor_humidity,
            s.desired_temperature_c,
            state.energy_usage,
            float(state.fault_signals),
        ],
        dtype=np.float32,
    )


def spoilage_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            s.sensor_temperature_c,
            s.sensor_humidity,
            _location_index(state),
            s.freshness_score,
            s.spoilage_risk,
            _inspection_alerts(state),
        ],
        dtype=np.float32,
    )


def inventory_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    shelf_remaining = max(0, get_params(s.fruit_type).base_shelf_life_ticks - s.age_ticks)
    return np.array(
        [
            state.inventory_level,
            state.demand_forecast,
            float(shelf_remaining),
            state.predicted_demand,
            state.energy_usage,
        ],
        dtype=np.float32,
    )


def delivery_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            1.0 if state.vehicle_available else 0.0,
            float(state.customer_window_ticks),
            s.spoilage_risk,
            _breakdown_alerts(state),
            _traffic_status(state),
        ],
        dtype=np.float32,
    )


def all_obs(state: GlobalState) -> dict[str, np.ndarray]:
    return {
        "routing": routing_obs(state),
        "temperature": temperature_obs(state),
        "spoilage": spoilage_obs(state),
        "inventory": inventory_obs(state),
        "delivery": delivery_obs(state),
    }
