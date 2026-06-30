from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium.spaces import Box

from core import config
from core.config import (
    DELIVERY_OBS_FIELDS,
    INVENTORY_OBS_FIELDS,
    ROUTING_OBS_FIELDS,
    SPOILAGE_OBS_FIELDS,
    TEMPERATURE_OBS_FIELDS,
    AmbientTempBucket,
    DisruptionType,
    FruitKey,
    Weather,
)
from core.fruits import get_params
from core.graph import build_supply_chain, sink_nodes, source_nodes
from core.noise import Disruption


WEATHER_INDEX: dict[Weather, int] = {w: i for i, w in enumerate(Weather)}
AMBIENT_BUCKET_INDEX: dict[AmbientTempBucket, int] = {b: i for i, b in enumerate(AmbientTempBucket)}
FRUIT_INDEX: dict[FruitKey, int] = {f: i for i, f in enumerate(FruitKey)}

OBS_FIELDS_BY_AGENT: dict[str, tuple[str, ...]] = {
    "routing": ROUTING_OBS_FIELDS,
    "temperature": TEMPERATURE_OBS_FIELDS,
    "spoilage": SPOILAGE_OBS_FIELDS,
    "inventory": INVENTORY_OBS_FIELDS,
    "delivery": DELIVERY_OBS_FIELDS,
}

_AMBIENT_BASE_TEMP_C: dict[Weather, float] = {
    Weather.SUNNY: 25.0,
    Weather.CLOUDY: 18.0,
    Weather.RAINY: 14.0,
    Weather.STORMY: 10.0,
}


@dataclass(slots=True)
class Shipment:
    fruit_type: FruitKey
    current_node: str
    target_node: str
    spoilage_risk: float
    ground_truth_label: int
    age_ticks: int
    perishability_index: float
    sensor_temperature_c: float
    sensor_humidity: float
    desired_temperature_c: float
    freshness_score: float


@dataclass(slots=True)
class GlobalState:
    tick: int
    max_steps: int
    rng: np.random.Generator
    graph: nx.DiGraph
    shipment: Shipment
    active_disruptions: list[Disruption]
    ambient_weather: Weather
    ambient_temp_c: float
    inventory_level: float
    demand_forecast: float
    predicted_demand: float
    vehicle_available: bool
    customer_window_ticks: int
    energy_usage: float
    cooling_status: bool
    fault_signals: int


def init_state(seed: int | None = None, max_steps: int | None = None) -> GlobalState:
    rng = np.random.default_rng(config.DEFAULT_SEED if seed is None else seed)
    n_steps = (
        max_steps
        if max_steps is not None
        else int(rng.integers(config.EPISODE_LEN_MIN, config.EPISODE_LEN_MAX + 1))
    )
    graph = build_supply_chain(rng)

    fruits = list(FruitKey)
    fruit = fruits[int(rng.integers(0, len(fruits)))]
    source = str(rng.choice(source_nodes(graph)))
    target = str(rng.choice(sink_nodes(graph)))

    params = get_params(fruit)
    desired_temp = (params.optimal_temp_low_c + params.optimal_temp_high_c) / 2.0

    shipment = Shipment(
        fruit_type=fruit,
        current_node=source,
        target_node=target,
        spoilage_risk=0.0,
        ground_truth_label=0,
        age_ticks=0,
        perishability_index=1.0 / params.base_shelf_life_ticks,
        sensor_temperature_c=desired_temp,
        sensor_humidity=0.85,
        desired_temperature_c=desired_temp,
        freshness_score=1.0,
    )

    weather = _sample_weather(rng)
    ambient_temp = _sample_ambient_temp(rng, weather)

    return GlobalState(
        tick=0,
        max_steps=n_steps,
        rng=rng,
        graph=graph,
        shipment=shipment,
        active_disruptions=[],
        ambient_weather=weather,
        ambient_temp_c=ambient_temp,
        inventory_level=1.0,
        demand_forecast=1.0,
        predicted_demand=1.0,
        vehicle_available=True,
        customer_window_ticks=n_steps,
        energy_usage=0.0,
        cooling_status=True,
        fault_signals=0,
    )


def _sample_weather(rng: np.random.Generator) -> Weather:
    weathers = list(Weather)
    weights = np.array([config.WEATHER_PRIORS[w] for w in weathers], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(weathers), p=weights))
    return weathers[idx]


def _sample_ambient_temp(rng: np.random.Generator, weather: Weather) -> float:
    return _AMBIENT_BASE_TEMP_C[weather] + float(rng.normal(0.0, 3.0))


def ambient_bucket(temp_c: float) -> AmbientTempBucket:
    edges = config.AMBIENT_TEMP_BUCKET_EDGES_C
    if temp_c < edges[0]:
        return AmbientTempBucket.COLD
    if temp_c < edges[1]:
        return AmbientTempBucket.MILD
    if temp_c < edges[2]:
        return AmbientTempBucket.WARM
    return AmbientTempBucket.HOT


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


def _route_delays(state: GlobalState) -> float:
    return _traffic_status(state)


def _location_index(state: GlobalState) -> float:
    nodes = list(state.graph.nodes)
    denom = max(1, len(nodes) - 1)
    return nodes.index(state.shipment.current_node) / denom


def extract_routing_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            _traffic_status(state),
            float(WEATHER_INDEX[state.ambient_weather]),
            s.perishability_index,
            _route_status(state),
            s.spoilage_risk,
        ],
        dtype=np.float32,
    )


def extract_temperature_obs(state: GlobalState) -> np.ndarray:
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


def extract_spoilage_obs(state: GlobalState) -> np.ndarray:
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


def extract_inventory_obs(state: GlobalState) -> np.ndarray:
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


def extract_delivery_obs(state: GlobalState) -> np.ndarray:
    s = state.shipment
    return np.array(
        [
            1.0 if state.vehicle_available else 0.0,
            float(state.customer_window_ticks),
            s.spoilage_risk,
            _breakdown_alerts(state),
            _route_delays(state),
        ],
        dtype=np.float32,
    )


def extract_all_obs(state: GlobalState) -> dict[str, np.ndarray]:
    return {
        "routing": extract_routing_obs(state),
        "temperature": extract_temperature_obs(state),
        "spoilage": extract_spoilage_obs(state),
        "inventory": extract_inventory_obs(state),
        "delivery": extract_delivery_obs(state),
    }


def make_observation_space(agent_id: str) -> gym.Space:
    n = len(OBS_FIELDS_BY_AGENT[agent_id])
    return Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)


def make_observation_spaces() -> dict[str, gym.Space]:
    return {agent: make_observation_space(agent) for agent in OBS_FIELDS_BY_AGENT}
