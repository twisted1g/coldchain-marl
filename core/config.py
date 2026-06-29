from __future__ import annotations

from enum import Enum
from typing import Final


N_EPISODES_FULL: Final[int] = 10_000
N_EPISODES_DEFAULT: Final[int] = 200

EPISODE_LEN_MIN: Final[int] = 10
EPISODE_LEN_MAX: Final[int] = 20

DEFAULT_SEED: Final[int] = 0

N_FARMS: Final[int] = 2
N_HUBS: Final[int] = 2
N_DCS: Final[int] = 1
N_RETAILERS: Final[int] = 3

EDGE_DISTANCE_KM_RANGE: Final[tuple[float, float]] = (20.0, 300.0)
EDGE_BASE_TRANSIT_TIME_RANGE: Final[tuple[int, int]] = (1, 3)
EDGE_BASE_EMISSIONS_PER_KM: Final[float] = 0.12

WAIT_EDGE_TRANSIT_TIME: Final[int] = 1
WAIT_EDGE_EMISSIONS: Final[float] = 0.0


class FruitKey(str, Enum):
    STRAWBERRY = "strawberry"
    BANANA = "banana"
    ORANGE = "orange"


R_GAS: Final[float] = 8.314462618
KELVIN_OFFSET: Final[float] = 273.15

RISK_LABEL_THRESHOLD: Final[float] = 0.5


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

DEMAND_NOISE_SIGMA: Final[float] = 0.10
TRANSIT_DELAY_NOISE_SIGMA: Final[float] = 0.25


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


class AmbientTempBucket(str, Enum):
    COLD = "cold"
    MILD = "mild"
    WARM = "warm"
    HOT = "hot"


AMBIENT_TEMP_BUCKET_EDGES_C: Final[tuple[float, ...]] = (5.0, 18.0, 30.0)


ROUTING_OBS_FIELDS: Final[tuple[str, ...]] = (
    "traffic_status",
    "weather",
    "perishability_index",
    "route_status",
    "fruit_degradation_risk",
)

TEMPERATURE_OBS_FIELDS: Final[tuple[str, ...]] = (
    "current_temperature",
    "current_humidity",
    "desired_temperature",
    "energy_usage",
    "fault_signals",
)

SPOILAGE_OBS_FIELDS: Final[tuple[str, ...]] = (
    "sensor_temperature",
    "sensor_humidity",
    "location_index",
    "freshness_score",
    "spoilage_risk",
    "inspection_alerts",
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
