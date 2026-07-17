from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from core import config
from core.config import FruitKey, Weather
from core.world import demand
from core.world.fruits import get_params
from core.world.graph import build_supply_chain, sink_nodes, source_nodes
from core.world.noise import Disruption

_AMBIENT_BASE_TEMP_C: dict[Weather, float] = {
    Weather.SUNNY: 25.0,
    Weather.CLOUDY: 18.0,
    Weather.RAINY: 14.0,
    Weather.STORMY: 10.0,
}

_AMBIENT_HUMIDITY_BY_WEATHER: dict[Weather, float] = {
    Weather.SUNNY: 0.45,
    Weather.CLOUDY: 0.65,
    Weather.RAINY: 0.85,
    Weather.STORMY: 0.92,
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
class VehicleState:
    assigned_node: str
    route_transit: float
    route_emissions: float
    sla_window_ticks: int
    chosen_slot: int
    delay: float
    emissions: float
    sla_violated: bool
    conflict: bool


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
    ambient_humidity: float
    inventory_level: float
    inventory_rng: np.random.Generator
    unmet_demand: float
    inventory_order: float
    pending_orders: list[tuple[int, float]]
    demand_mean: float
    day_of_year: int
    weekday: int
    event_days_left: int
    event_multiplier: float
    demand_today: float
    demand_shock_mult: float
    history: demand.DemandSeries
    demand_forecast: float
    energy_usage: float
    fault_signals: int
    route_travel_time: float
    route_emissions: float
    spoilage_prediction: float
    vehicles: list[VehicleState]


def init_state(
    seed: int | None = None,
    max_steps: int | None = None,
    fruit: FruitKey | None = None,
) -> GlobalState:
    base_seed = config.DEFAULT_SEED if seed is None else seed
    rng = np.random.default_rng(base_seed)
    n_steps = (
        max_steps
        if max_steps is not None
        else int(rng.integers(config.EPISODE_LEN_MIN, config.EPISODE_LEN_MAX + 1))
    )
    graph = build_supply_chain(rng)

    fruits = list(FruitKey)
    fruit = fruit if fruit is not None else fruits[int(rng.integers(0, len(fruits)))]
    source = str(rng.choice(source_nodes(graph)))
    target = str(rng.choice(sink_nodes(graph)))

    params = get_params(fruit)
    desired_temp = params.optimal_temp_c

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

    weather = demand.sample_weather(rng)
    ambient_temp = _sample_ambient_temp(rng, weather)
    ambient_humidity = _sample_ambient_humidity(rng, weather)
    shipment.sensor_humidity = ambient_humidity

    inventory_rng = np.random.default_rng(base_seed + config.INVENTORY_RNG_OFFSET)
    day_of_year = int(inventory_rng.integers(0, config.DAYS_PER_YEAR))
    weekday = int(inventory_rng.integers(0, config.DAYS_PER_WEEK))

    history, event_days_left, event_multiplier = demand.backfill_history(
        inventory_rng,
        (day_of_year - 1) % config.DAYS_PER_YEAR,
        (weekday - 1) % config.DAYS_PER_WEEK,
        config.DEMAND_HISTORY_DAYS,
    )
    event_days_left, event_multiplier = demand.advance_event(
        inventory_rng, event_days_left, event_multiplier
    )
    mean_today = demand.demand_mean(day_of_year, weekday, weather, event_multiplier)
    demand_today = demand.sample_demand(
        inventory_rng, day_of_year, weekday, weather, event_multiplier
    )
    demand.push_history(
        history,
        day_of_year,
        weekday,
        weather,
        event_multiplier,
        mean_today,
        demand_today,
    )

    return GlobalState(
        tick=0,
        max_steps=n_steps,
        rng=rng,
        graph=graph,
        shipment=shipment,
        active_disruptions=[],
        ambient_weather=weather,
        ambient_temp_c=ambient_temp,
        ambient_humidity=ambient_humidity,
        inventory_level=config.INVENTORY_INIT_LEVEL,
        inventory_rng=inventory_rng,
        unmet_demand=0.0,
        inventory_order=0.0,
        pending_orders=[],
        demand_mean=mean_today,
        day_of_year=day_of_year,
        weekday=weekday,
        event_days_left=event_days_left,
        event_multiplier=event_multiplier,
        demand_today=demand_today,
        demand_shock_mult=1.0,
        history=history,
        demand_forecast=config.INVENTORY_DEMAND_MEAN,
        energy_usage=0.0,
        fault_signals=0,
        route_travel_time=0.0,
        route_emissions=0.0,
        spoilage_prediction=0.0,
        vehicles=_init_vehicles(graph, source, n_steps),
    )


def _init_vehicles(graph: nx.DiGraph, source: str, n_steps: int) -> list[VehicleState]:
    retailers = sink_nodes(graph)
    vehicles: list[VehicleState] = []
    for i in range(config.N_VEHICLES):
        assigned = retailers[i % len(retailers)]
        transit, emissions = _route_cost(graph, source, assigned)
        sla_window = min(n_steps, int(np.ceil(transit)) + 1)
        vehicles.append(
            VehicleState(
                assigned_node=assigned,
                route_transit=transit,
                route_emissions=emissions,
                sla_window_ticks=sla_window,
                chosen_slot=0,
                delay=0.0,
                emissions=0.0,
                sla_violated=False,
                conflict=False,
            )
        )
    return vehicles


def _route_cost(graph: nx.DiGraph, source: str, target: str) -> tuple[float, float]:
    try:
        path = nx.shortest_path(graph, source, target, weight="base_transit_time")
    except nx.NetworkXNoPath:
        return float(config.EPISODE_LEN_MAX), 0.0
    transit = 0.0
    emissions = 0.0
    for u, v in zip(path[:-1], path[1:], strict=True):
        edge = graph.edges[u, v]
        transit += float(edge["base_transit_time"])
        emissions += float(edge["base_emissions"])
    return transit, emissions


def _sample_ambient_temp(rng: np.random.Generator, weather: Weather) -> float:
    return _AMBIENT_BASE_TEMP_C[weather] + float(rng.normal(0.0, 3.0))


def _sample_ambient_humidity(rng: np.random.Generator, weather: Weather) -> float:
    base = _AMBIENT_HUMIDITY_BY_WEATHER[weather]
    return float(np.clip(base + rng.normal(0.0, 0.05), 0.0, 1.0))
