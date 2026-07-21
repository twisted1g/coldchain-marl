from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from core import config
from core.config import OBS_FIELDS_BY_AGENT, DisruptionType
from core.interfaces.intention import IntentionBuffer
from core.interfaces.observations import all_obs
from core.state import Cargo, Consignment, GlobalState, VehicleState
from core.world import demand
from core.world.fruits import get_params
from core.world.graph import sink_nodes, source_nodes
from core.world.graph_features import node_delay
from core.world.noise import NoiseModel
from core.world.spoilage import ArrheniusSpoilage, risk_to_label


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
    _advance_calendar(state)

    buffer = IntentionBuffer()
    buffer.declare_all(actions)
    free = {i for i in range(len(state.vehicles)) if _vehicle_free(state, i)}
    conflicts = buffer.detect(free)

    _apply_temperature_action(state, actions.get("temperature"))
    _apply_routing_action(state, actions.get("routing"))
    _apply_inventory_action(state, actions)
    _apply_delivery_action(state, actions, conflicts)
    _apply_spoilage_action(state, actions.get("spoilage"))

    _advance_vehicles(state)
    _advance_thermal_state(state)
    _advance_humidity(state)
    _advance_spoilage(state)
    _advance_loads(state)
    _advance_cargo(state)
    _maybe_sample_disruption(state)
    _update_energy(state)

    delivered = state.shipment.current_node == state.shipment.target_node
    if delivered:
        state.shipment.ground_truth_label = risk_to_label(state.shipment.spoilage_risk)
    # In the rolling world a delivered shipment respawns; the episode ends only
    # at the horizon, so delivery/inventory get a long run of real restock trips.
    done = state.tick >= state.max_steps or (delivered and not state.rolling)
    infos = _build_infos(state, delivered)
    if delivered and state.rolling and not done:
        respawn_shipment(state)

    observations = all_obs(state)
    rewards = dict.fromkeys(OBS_FIELDS_BY_AGENT, 0.0)
    terminated = dict.fromkeys(OBS_FIELDS_BY_AGENT, done)
    terminated["__all__"] = done
    truncated = dict.fromkeys(OBS_FIELDS_BY_AGENT, False)
    truncated["__all__"] = False

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
        np.clip(
            value, config.TEMPERATURE_ACTION_LOW_C, config.TEMPERATURE_ACTION_HIGH_C
        )
    )


def _apply_inventory_action(state: GlobalState, actions: dict[str, Any]) -> None:
    for i in range(config.N_INVENTORY_INSTANCES):
        agent = f"inventory_{i}"
        arrivals = [
            c
            for c in state.cargo
            if c.instance == i and c.arrival_tick <= state.tick
        ]
        arrived = sum(c.qty for c in arrivals)
        # Emissions Et are charged on the delivery that lands the goods (Alg 4
        # line 16, "from delivery"), not on the order action: the cost lands
        # with the benefit, so ordering carries no instant penalty.
        state.inventory_arrival_emissions[i] = sum(c.emissions for c in arrivals)
        level = min(state.inventory_levels[i] + arrived, 1.0)

        action = actions.get(agent)
        order = (
            0.0
            if action is None
            else float(
                np.clip(
                    np.asarray(action).flatten()[0],
                    config.INVENTORY_ACTION_LOW,
                    config.INVENTORY_ACTION_HIGH,
                )
            )
        )
        qty = order * config.INVENTORY_RESTOCK_SCALE
        if qty > config.INVENTORY_MIN_ORDER_QTY:
            state.order_queue.append((i, qty))

        sold = min(level, state.demand_today[i])
        state.unmet_demand[i] = state.demand_today[i] - sold
        state.inventory_order[i] = order
        state.inventory_levels[i] = float(np.clip(level - sold, 0.0, 1.0))
    state.cargo = [c for c in state.cargo if c.arrival_tick > state.tick]


def slot_deadline(slot: int, max_steps: int) -> float:
    """Tick by which a delivery in this slot must arrive (slots split the episode)."""
    return (slot + 1) / config.N_DELIVERY_WINDOWS * max_steps


def slot_start(slot: int, max_steps: int) -> float:
    """Tick at which this slot's delivery window opens."""
    return slot / config.N_DELIVERY_WINDOWS * max_steps


def _vehicle_free(state: GlobalState, i: int) -> bool:
    """A vehicle is available for a new order only when idle at the depot (not
    already carrying / waiting for a slot window / mid-route)."""
    return state.vehicles[i].carrying is None


def expected_lead_time(state: GlobalState, instance: int) -> int:
    """Ticks until an order placed now would arrive at this retailer: transit
    to it plus the soonest-free vehicle's wait. Ignores slot scheduling and
    queue backlog — an estimate for the demand forecast horizon."""
    wait = min(max(0, v.busy_until - state.tick) for v in state.vehicles)
    eta = wait + int(np.ceil(state.retailer_transit[instance]))
    return max(1, eta)


def _apply_delivery_action(
    state: GlobalState, actions: dict[str, Any], conflicts: dict[str, bool]
) -> None:
    for i, vehicle in enumerate(state.vehicles):
        action = actions.get(f"delivery_{i}")
        slot = 0 if action is None else int(action) % config.N_DELIVERY_WINDOWS
        vehicle.chosen_slot = slot
        vehicle.delay = 0.0
        vehicle.sla_violated = False
        vehicle.emissions = 0.0
        vehicle.conflict = conflicts.get(f"delivery_{i}", False)
    _dispatch_orders(state)


def _delivery_path(state: GlobalState, retail: str) -> list[str]:
    """Weighted shortest route the truck actually drives, depot -> ... -> retail."""
    try:
        return nx.shortest_path(
            state.graph, state.depot, retail, weight="base_transit_time"
        )
    except nx.NetworkXNoPath:
        return [state.depot, retail]


def _path_cost(state: GlobalState, path: list[str]) -> tuple[int, float]:
    transit = 0
    emissions = 0.0
    for u, v in zip(path[:-1], path[1:], strict=True):
        edge = state.graph.edges[u, v]
        transit += int(np.ceil(edge["base_transit_time"]))
        emissions += float(edge["base_emissions"])
    return transit, emissions


def _dispatch_orders(state: GlobalState) -> None:
    """First idle vehicle takes the queue head and drives the real graph path to
    the ordering retailer (all vehicles stage at the shared depot, so the paper's
    distance heuristic — Alg 5 line 15 — reduces to first-free); its chosen slot
    schedules the departure window. The trip is honest summed edge time; the
    truck then advances one hop per tick in ``_advance_vehicles``. delay/SLA/
    emissions are the trip's predicted cost, set on the dispatch tick."""
    while state.order_queue:
        free = [i for i in range(len(state.vehicles)) if _vehicle_free(state, i)]
        if not free:
            return
        i = free[0]
        vehicle = state.vehicles[i]
        instance, qty = state.order_queue.pop(0)

        path = _delivery_path(state, f"retail_{instance}")
        transit, emissions = _path_cost(state, path)
        departure = max(
            state.tick, int(np.ceil(slot_start(vehicle.chosen_slot, state.max_steps)))
        )
        arrival = departure + transit
        deadline = slot_deadline(vehicle.chosen_slot, state.max_steps)

        vehicle.route_transit = float(transit)
        vehicle.route_emissions = emissions
        vehicle.busy_until = arrival
        vehicle.delay = max(0.0, arrival - deadline)
        vehicle.sla_violated = arrival > deadline
        vehicle.emissions = emissions
        cargo = Cargo(
            vehicle=i,
            instance=instance,
            departure_tick=departure,
            # Filled with the actual delivery tick when the truck reaches the
            # retailer; inventory credits it then. A large sentinel keeps it
            # in-transit (spoilage decay applies) until then.
            arrival_tick=state.max_steps + transit + 1,
            qty=qty,
            emissions=emissions,
        )
        state.cargo.append(cargo)
        vehicle.carrying = cargo
        vehicle.load = _new_crate(state, f"retail_{instance}")
        vehicle.current_node = state.depot
        vehicle.route = path[1:]
        vehicle.edge_ticks_left = 0


def _new_crate(state: GlobalState, target: str) -> Consignment:
    """A fresh crate loaded onto a dispatched truck: cold at the depot, bound for
    its retailer. Carries its own thermal + spoilage state so goods spoil by
    their own temperature (multi-instance redesign). The temperature agent is
    wired to it in Phase 3; until then the setpoint stays at the fruit optimum."""
    params = get_params(state.shipment.fruit_type)
    return Consignment(
        fruit_type=state.shipment.fruit_type,
        current_node=state.depot,
        target_node=target,
        spoilage_risk=0.0,
        ground_truth_label=0,
        age_ticks=0,
        perishability_index=1.0 / params.base_shelf_life_ticks,
        sensor_temperature_c=params.optimal_temp_c,
        sensor_humidity=state.ambient_humidity,
        desired_temperature_c=params.optimal_temp_c,
        freshness_score=1.0,
    )


def _advance_vehicles(state: GlobalState) -> None:
    """Move each in-transit truck one graph edge at a time. A truck waits at the
    depot until its slot window opens (``departure_tick``), then consumes each
    edge's ``base_transit_time`` in ticks; on reaching the retailer it delivers
    (crediting inventory) and returns to the depot."""
    for vehicle in state.vehicles:
        cargo = vehicle.carrying
        if cargo is None or state.tick < cargo.departure_tick or not vehicle.route:
            continue
        if vehicle.edge_ticks_left <= 0:
            vehicle.edge_ticks_left = int(
                np.ceil(state.graph.edges[vehicle.current_node, vehicle.route[0]][
                    "base_transit_time"
                ])
            )
        vehicle.edge_ticks_left -= 1
        if vehicle.edge_ticks_left <= 0:
            vehicle.current_node = vehicle.route.pop(0)
            if not vehicle.route:
                _deliver_vehicle(state, vehicle)


def _deliver_vehicle(state: GlobalState, vehicle: VehicleState) -> None:
    cargo = vehicle.carrying
    if cargo is not None:
        cargo.arrival_tick = state.tick  # inventory credits it next tick
    vehicle.carrying = None
    vehicle.load = None  # crate delivered
    vehicle.busy_until = state.tick
    vehicle.current_node = state.depot  # deferred/instant return trip
    vehicle.route = []
    vehicle.edge_ticks_left = 0


def respawn_shipment(state: GlobalState) -> None:
    """Replace a delivered shipment with a fresh one entering the network;
    everything else (inventory, vehicles, cargo, calendar) carries over."""
    s = state.shipment
    params = get_params(s.fruit_type)
    s.current_node = str(state.rng.choice(source_nodes(state.graph)))
    s.target_node = str(state.rng.choice(sink_nodes(state.graph)))
    s.spoilage_risk = 0.0
    s.freshness_score = 1.0
    s.ground_truth_label = 0
    s.age_ticks = 0
    s.sensor_temperature_c = params.optimal_temp_c
    s.desired_temperature_c = params.optimal_temp_c
    s.sensor_humidity = state.ambient_humidity
    state.route_travel_time = 0.0
    state.route_emissions = 0.0


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


def _advance_humidity(state: GlobalState) -> None:
    s = state.shipment
    pull = (state.ambient_humidity - s.sensor_humidity) * config.HUMIDITY_AMBIENT_PULL
    noise = float(state.rng.normal(0.0, config.HUMIDITY_NOISE_SIGMA))
    s.sensor_humidity = float(np.clip(s.sensor_humidity + pull + noise, 0.0, 1.0))


def _advance_spoilage(state: GlobalState) -> None:
    s = state.shipment
    delay = node_delay(state, s.current_node)
    delta = _spoilage_model.risk_delta(
        s.fruit_type, s.sensor_temperature_c, s.sensor_humidity, delay, dt_ticks=1.0
    )
    s.spoilage_risk = float(np.clip(s.spoilage_risk + delta, 0.0, 1.0))
    s.freshness_score = float(max(0.0, 1.0 - s.spoilage_risk))
    s.age_ticks += 1


def _advance_calendar(state: GlobalState) -> None:
    state.day_of_year = (state.day_of_year + 1) % config.DAYS_PER_YEAR
    state.weekday = (state.weekday + 1) % config.DAYS_PER_WEEK
    state.event_days_left, state.event_multiplier = demand.advance_event(
        state.inventory_rng, state.event_days_left, state.event_multiplier
    )
    state.demand_mean = state.demand_shock_mult * demand.demand_mean(
        state.day_of_year, state.weekday, state.ambient_weather, state.event_multiplier
    )
    state.demand_today = [
        state.demand_mean * demand.demand_noise(state.inventory_rng)
        for _ in range(config.N_INVENTORY_INSTANCES)
    ]
    for history, today in zip(state.histories, state.demand_today, strict=True):
        demand.push_history(
            history,
            state.day_of_year,
            state.weekday,
            state.ambient_weather,
            state.event_multiplier,
            state.demand_mean,
            today,
        )


def _advance_loads(state: GlobalState) -> None:
    """Per-crate thermal + spoilage for every in-transit truck (multi-instance
    redesign): each crate advances by its OWN sensor temperature, not the global
    shipment's. Deterministic — no ``state.rng`` draws — so the main noise stream
    (disruptions, demand) stays bit-exact and only cargo spoilage shifts. The
    temperature agent is wired to ``desired_temperature_c`` in Phase 3; here the
    setpoint is the fruit optimum, so crates ride cold and barely decay."""
    for vehicle in state.vehicles:
        crate = vehicle.load
        cargo = vehicle.carrying
        if crate is None or cargo is None:
            continue
        if not (cargo.departure_tick <= state.tick < cargo.arrival_tick):
            continue
        crate.current_node = vehicle.current_node
        # thermal relaxation toward the setpoint plus ambient pull (mirrors the
        # singleton ``_advance_thermal_state``, deterministic)
        diff = crate.desired_temperature_c - crate.sensor_temperature_c
        ambient_pull = (state.ambient_temp_c - crate.sensor_temperature_c) * 0.05
        crate.sensor_temperature_c += 0.5 * diff + ambient_pull
        # humidity relaxes to ambient without the rng noise term (noise would
        # desync the shared stream)
        crate.sensor_humidity = float(
            np.clip(
                crate.sensor_humidity
                + (state.ambient_humidity - crate.sensor_humidity)
                * config.HUMIDITY_AMBIENT_PULL,
                0.0,
                1.0,
            )
        )
        delay = node_delay(state, crate.current_node)
        delta = _spoilage_model.risk_delta(
            crate.fruit_type,
            crate.sensor_temperature_c,
            crate.sensor_humidity,
            delay,
            dt_ticks=1.0,
        )
        crate.spoilage_risk = float(np.clip(crate.spoilage_risk + delta, 0.0, 1.0))
        crate.freshness_score = float(max(0.0, 1.0 - crate.spoilage_risk))
        crate.age_ticks += 1


def _advance_cargo(state: GlobalState) -> None:
    """In-transit spoilage: each truck's cargo decays with ITS crate's own
    spoilage risk (per-crate, multi-instance redesign). Queued cargo and cargo
    waiting for its slot window sit in cold storage and do not decay."""
    state.transit_loss = [0.0] * config.N_INVENTORY_INSTANCES
    for vehicle in state.vehicles:
        cargo = vehicle.carrying
        crate = vehicle.load
        if cargo is None or crate is None:
            continue
        if not (cargo.departure_tick <= state.tick < cargo.arrival_tick):
            continue
        lost = cargo.qty * config.TRANSIT_SPOILAGE_RATE * crate.spoilage_risk
        cargo.qty -= lost
        state.transit_loss[cargo.instance] += lost


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
    infos = {
        "routing": {"delivered": delivered},
        "temperature": {"energy_usage": state.energy_usage},
        "spoilage": {
            "y_pred": state.spoilage_prediction,
            "ground_truth_label": state.shipment.ground_truth_label,
        },
    }
    for i in range(config.N_INVENTORY_INSTANCES):
        infos[f"inventory_{i}"] = {
            "inventory_level": state.inventory_levels[i],
            "transit_loss": state.transit_loss[i],
        }
    for i, vehicle in enumerate(state.vehicles):
        infos[f"delivery_{i}"] = {
            "delay": vehicle.delay,
            "sla_violated": vehicle.sla_violated,
            "emissions": vehicle.emissions,
            "conflict": vehicle.conflict,
        }
    return infos
