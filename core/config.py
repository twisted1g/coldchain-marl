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

# Retailers start near-empty so restocking is the agent's job: a full shelf
# (1.0) buffers ~4 ticks of demand and flattens the cost landscape until even
# never-order sits within ~5% of optimal, so no learned policy can separate
# from a random baseline. At 0.3 restock is essential — the optimal policy
# beats never-order ~38% and random ~26% (vs +16%/+12% at 1.0).
INVENTORY_INIT_LEVEL: Final[float] = 0.3
INVENTORY_DEMAND_MEAN: Final[float] = 0.15
INVENTORY_RESTOCK_SCALE: Final[float] = 1.0
INVENTORY_MIN_ORDER_QTY: Final[float] = 0.05
TRANSIT_SPOILAGE_RATE: Final[float] = 0.05

# Delivery/inventory learn on a rolling world (the shipment respawns on
# delivery instead of ending the episode), so restock trucks drive the real
# multi-hop path in honest transit time rather than a scaled-down proxy.
ROLLING_HORIZON: Final[int] = 40
# Vehicles allowed to occupy one delivery slot before it is a conflict (Alg 5
# "resource overuse"). 1 => any two trucks sharing a slot must negotiate.
SLOT_CAPACITY: Final[int] = 1
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

# Per-node micro-climate. Each node holds its own storage temperature/humidity,
# mean-reverting toward a kind-specific setpoint and pulled toward ambient with a
# kind-specific strength (cold rooms are well-refrigerated, so their ambient pull
# is tiny), hard-clamped to a physically plausible band. This is the external
# environment a load fights: node_climate[current_node] drives crate/shipment
# thermal + energy, and feeds the spoilage GNN so its "risk across nodes" reasons
# over real per-node conditions (paper Table 1: spoilage input is per-node T/H +
# location) instead of flat ambient copies.
NODE_CLIMATE_SETPOINT_C: Final[dict[str, float]] = {
    "farm": 20.0,  # open-air handling, tracks ambient
    "hub": 11.0,  # cool cross-dock
    "dc": 3.0,  # refrigerated cold room
    "retail": 5.0,  # display chiller
}
NODE_CLIMATE_BAND_C: Final[dict[str, tuple[float, float]]] = {
    "farm": (5.0, 30.0),
    "hub": (8.0, 14.0),
    "dc": (0.0, 6.0),
    "retail": (2.0, 8.0),
}
NODE_CLIMATE_HUMIDITY: Final[dict[str, float]] = {
    "farm": 0.70,
    "hub": 0.80,
    "dc": 0.90,
    "retail": 0.85,
}
NODE_CLIMATE_AMBIENT_PULL: Final[dict[str, float]] = {
    "farm": 0.15,  # open air, follows outside temperature
    "hub": 0.05,
    "dc": 0.01,  # sealed cold room, barely feels ambient
    "retail": 0.02,
}
NODE_CLIMATE_REVERSION: Final[float] = 0.2  # pull toward the kind setpoint
NODE_CLIMATE_TEMP_SIGMA: Final[float] = 0.25  # mean-reverting temp noise (deg C)
NODE_CLIMATE_HUMIDITY_SIGMA: Final[float] = 0.02
NODE_CLIMATE_RNG_OFFSET: Final[int] = 777  # isolate climate noise from other streams


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

# Day-to-day weather evolution (one tick = one day). Weather is sticky — the
# diagonal dominates — so a spell of sun or a passing storm plays out over several
# days instead of a single frozen sample per episode (paper Sec 3: genAI simulates
# changing/extreme weather). Rows are the current weather, values the next-day
# distribution; each row sums to 1.
WEATHER_TRANSITION: Final[dict[Weather, dict[Weather, float]]] = {
    Weather.SUNNY: {
        Weather.SUNNY: 0.70,
        Weather.CLOUDY: 0.22,
        Weather.RAINY: 0.06,
        Weather.STORMY: 0.02,
    },
    Weather.CLOUDY: {
        Weather.SUNNY: 0.30,
        Weather.CLOUDY: 0.45,
        Weather.RAINY: 0.20,
        Weather.STORMY: 0.05,
    },
    Weather.RAINY: {
        Weather.SUNNY: 0.15,
        Weather.CLOUDY: 0.30,
        Weather.RAINY: 0.40,
        Weather.STORMY: 0.15,
    },
    Weather.STORMY: {
        Weather.SUNNY: 0.10,
        Weather.CLOUDY: 0.25,
        Weather.RAINY: 0.35,
        Weather.STORMY: 0.30,
    },
}
# Ambient temperature is resampled each day: weather base + annual seasonal swing
# + daily noise, clamped to a plausible outdoor range.
AMBIENT_SEASONAL_AMPLITUDE_C: Final[float] = 6.0
AMBIENT_DAILY_NOISE_SIGMA_C: Final[float] = 2.0
AMBIENT_TEMP_RANGE_C: Final[tuple[float, float]] = (-5.0, 42.0)
WEATHER_RNG_OFFSET: Final[int] = 555  # isolate weather noise from other streams

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

# Paper Alg 4 state = [stock level, demand forecast, shelf life, carbon]; we add
# on_order (pipeline) so the policy sees its inventory position. zone_energy_usage
# and peer_stock were shipment-global noise irrelevant to a per-instance order
# decision — the policy latched onto them and learned harmful state-dependence
# (ordered more when it should not), losing to a random baseline. Dropped.
INVENTORY_OBS_FIELDS: Final[tuple[str, ...]] = (
    "inventory_level",
    "on_order",
    "demand_forecast",
    "shelf_life",
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
