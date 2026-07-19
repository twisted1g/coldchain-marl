from __future__ import annotations

from enum import StrEnum
from typing import Final

EPISODE_LEN_MIN: Final[int] = 10
EPISODE_LEN_MAX: Final[int] = 20

DEFAULT_SEED: Final[int] = 0

N_FARMS: Final[int] = 2
N_HUBS: Final[int] = 3
N_DCS: Final[int] = 2
N_RETAILERS: Final[int] = 4
N_NODES: Final[int] = N_FARMS + N_HUBS + N_DCS + N_RETAILERS

EDGE_DISTANCE_KM_RANGE: Final[tuple[float, float]] = (20.0, 300.0)
EDGE_BASE_TRANSIT_TIME_RANGE: Final[tuple[int, int]] = (1, 3)
EDGE_BASE_EMISSIONS_PER_KM: Final[float] = 0.12

WAIT_EDGE_TRANSIT_TIME: Final[int] = 1
WAIT_EDGE_EMISSIONS: Final[float] = 0.0

N_NEXT_NODES: Final[int] = 5
N_RISK_LEVELS: Final[int] = 3
N_DELIVERY_WINDOWS: Final[int] = 4
N_VEHICLES: Final[int] = 3
SPOILAGE_ACTION_LOW: Final[float] = 0.0
SPOILAGE_ACTION_HIGH: Final[float] = 1.0
TEMPERATURE_ACTION_LOW_C: Final[float] = -30.0
TEMPERATURE_ACTION_HIGH_C: Final[float] = 30.0
INVENTORY_ACTION_LOW: Final[float] = 0.0
INVENTORY_ACTION_HIGH: Final[float] = 1.0

N_INVENTORY_INSTANCES: Final[int] = N_RETAILERS

INVENTORY_INIT_LEVEL: Final[float] = 1.0
INVENTORY_DEMAND_MEAN: Final[float] = 0.15
INVENTORY_RESTOCK_SCALE: Final[float] = 1.0
INVENTORY_MIN_ORDER_QTY: Final[float] = 0.05
TRANSIT_SPOILAGE_RATE: Final[float] = 0.05
INVENTORY_RNG_OFFSET: Final[int] = 90_001

DAYS_PER_YEAR: Final[int] = 365
DAYS_PER_WEEK: Final[int] = 7

DEMAND_SEASON_AMP: Final[float] = 0.3
DEMAND_WEEKEND_MULT: Final[float] = 1.2
DEMAND_EVENT_PROB: Final[float] = 0.02
DEMAND_EVENT_DURATION_RANGE: Final[tuple[int, int]] = (1, 3)
DEMAND_EVENT_MULT_RANGE: Final[tuple[float, float]] = (1.5, 2.5)
DEMAND_NOISE_SIGMA: Final[float] = 0.1
DEMAND_HISTORY_DAYS: Final[int] = 28


class FruitKey(StrEnum):
    STRAWBERRY = "strawberry"
    BANANA = "banana"
    ORANGE = "orange"


R_GAS: Final[float] = 8.314462618
KELVIN_OFFSET: Final[float] = 273.15

RISK_LABEL_THRESHOLD: Final[float] = 0.5

HUMIDITY_AMBIENT_PULL: Final[float] = 0.15
HUMIDITY_NOISE_SIGMA: Final[float] = 0.03
HUMIDITY_SEVERITY_SCALE: Final[float] = 0.3
DELAY_RISK_FACTOR: Final[float] = 0.5


class DisruptionType(StrEnum):
    BLOCKED_NODE = "blocked_node"
    INCREASED_TRANSIT = "increased_transit_time"
    RISK_FLAG = "risk_flag"


DISRUPTION_PROB_PER_TICK: Final[float] = 0.05

DISRUPTION_TYPE_WEIGHTS: Final[dict[DisruptionType, float]] = {
    DisruptionType.BLOCKED_NODE: 0.30,
    DisruptionType.INCREASED_TRANSIT: 0.50,
    DisruptionType.RISK_FLAG: 0.20,
}

DISRUPTION_TRANSIT_DELTA_RANGE: Final[tuple[int, int]] = (1, 3)


class Weather(StrEnum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"


WEATHER_PRIORS: Final[dict[Weather, float]] = {
    Weather.SUNNY: 0.55,
    Weather.CLOUDY: 0.25,
    Weather.RAINY: 0.15,
    Weather.STORMY: 0.05,
}

DEMAND_WEATHER_MULT: Final[dict[Weather, float]] = {
    Weather.SUNNY: 1.3,
    Weather.CLOUDY: 1.0,
    Weather.RAINY: 0.85,
    Weather.STORMY: 0.6,
}


EDGE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "transit",
    "emissions",
    "reaches_target",
    "is_target",
    "is_wait",
)

ROUTING_OBS_FIELDS: Final[tuple[str, ...]] = (
    "traffic_status",
    "weather",
    "perishability_index",
    "route_status",
    "fruit_degradation_risk",
    "location_index",
    "target_index",
    *tuple(
        f"edge{i}_{name}" for i in range(N_NEXT_NODES) for name in EDGE_FEATURE_NAMES
    ),
)

TEMPERATURE_OBS_FIELDS: Final[tuple[str, ...]] = (
    "current_temperature",
    "current_humidity",
    "desired_temperature",
    "energy_usage",
    "fault_signals",
)

SPOILAGE_OBS_FIELDS: Final[tuple[str, ...]] = tuple(
    f"node{i}_{name}"
    for i in range(N_NODES)
    for name in ("temperature", "humidity", "delay", "fruit_type")
)

INVENTORY_OBS_FIELDS: Final[tuple[str, ...]] = (
    "inventory_level",
    "on_order",
    "demand_forecast",
    "shelf_life",
    "zone_energy_usage",
    "peer_stock",
)

INVENTORY_AGENTS: Final[tuple[str, ...]] = tuple(
    f"inventory_{i}" for i in range(N_INVENTORY_INSTANCES)
)

DELIVERY_OBS_FIELDS: Final[tuple[str, ...]] = (
    "vehicle_id",
    "vehicle_availability",
    "customer_window",
    "spoilage_risk",
    "breakdown_alerts",
    "route_delays",
)

DELIVERY_AGENTS: Final[tuple[str, ...]] = tuple(
    f"delivery_{i}" for i in range(N_VEHICLES)
)

OBS_FIELDS_BY_AGENT: Final[dict[str, tuple[str, ...]]] = {
    "routing": ROUTING_OBS_FIELDS,
    "temperature": TEMPERATURE_OBS_FIELDS,
    "spoilage": SPOILAGE_OBS_FIELDS,
    **dict.fromkeys(INVENTORY_AGENTS, INVENTORY_OBS_FIELDS),
    **dict.fromkeys(DELIVERY_AGENTS, DELIVERY_OBS_FIELDS),
}
