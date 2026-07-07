from __future__ import annotations

from enum import Enum
from typing import Final


N_EPISODES_FULL: Final[int] = 10_000
N_EPISODES_DEFAULT: Final[int] = 200

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
SPOILAGE_ACTION_LOW: Final[float] = 0.0
SPOILAGE_ACTION_HIGH: Final[float] = 1.0
TEMPERATURE_ACTION_LOW_C: Final[float] = -30.0
TEMPERATURE_ACTION_HIGH_C: Final[float] = 30.0
INVENTORY_ACTION_LOW: Final[float] = 0.0
INVENTORY_ACTION_HIGH: Final[float] = 1.0

INVENTORY_INIT_LEVEL: Final[float] = 1.0
INVENTORY_DEMAND_MEAN: Final[float] = 0.15
INVENTORY_DEMAND_SIGMA: Final[float] = 0.05
INVENTORY_RESTOCK_SCALE: Final[float] = 0.25
INVENTORY_RNG_OFFSET: Final[int] = 90_001


class FruitKey(str, Enum):
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


class DisruptionType(str, Enum):
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


class Weather(str, Enum):
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


ROUTING_OBS_FIELDS: Final[tuple[str, ...]] = (
    "traffic_status",
    "weather",
    "perishability_index",
    "route_status",
    "fruit_degradation_risk",
    "location_index",
    "target_index",
    *tuple(
        f"edge{i}_{name}"
        for i in range(N_NEXT_NODES)
        for name in ("transit", "emissions", "reaches_target", "is_target", "is_wait")
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
    "demand_forecast",
    "shelf_life",
    "predicted_demand",
    "zone_energy_usage",
)

DELIVERY_OBS_FIELDS: Final[tuple[str, ...]] = (
    "vehicle_availability",
    "customer_window",
    "spoilage_risk",
    "breakdown_alerts",
    "route_delays",
)

OBS_FIELDS_BY_AGENT: Final[dict[str, tuple[str, ...]]] = {
    "routing": ROUTING_OBS_FIELDS,
    "temperature": TEMPERATURE_OBS_FIELDS,
    "spoilage": SPOILAGE_OBS_FIELDS,
    "inventory": INVENTORY_OBS_FIELDS,
    "delivery": DELIVERY_OBS_FIELDS,
}
