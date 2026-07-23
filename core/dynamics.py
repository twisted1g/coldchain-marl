from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from core import config
from core.config import OBS_FIELDS_BY_AGENT, DisruptionType
from core.interfaces.intention import IntentionBuffer
from core.interfaces.observations import all_obs
from core.state import (
    Cargo,
    Consignment,
    GlobalState,
    VehicleState,
    _sample_ambient_humidity,
    _sample_ambient_temp,
)
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
    _advance_weather(state)
    _advance_calendar(state)

    buffer = IntentionBuffer()
    buffer.declare_all(actions)
    free = {i for i in range(len(state.vehicles)) if _vehicle_free(state, i)}
    conflicts = buffer.detect(free)

    # Per-crate first-class agents (singleton eliminated): routing/temperature/
    # spoilage each run one action per truck's crate.
    _apply_temperature_actions(state, actions)
    _apply_routing_actions(state, actions)
    _apply_inventory_action(state, actions)
    _apply_delivery_action(state, actions, conflicts)
    _apply_spoilage_actions(state, actions)

    _advance_vehicles(state)
    _advance_node_climate(state)
    _advance_loads(state)
    _advance_cargo(state)
    _maybe_sample_disruption(state)

    # The world runs to the horizon; there is no singleton delivery to end it.
    done = state.tick >= state.max_steps
    infos = _build_infos(state)

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


def _apply_routing_actions(state: GlobalState, actions: dict[str, Any]) -> None:
    """One routing agent per truck (paper Alg 1): a loaded truck that sits at a node
    (not mid-edge) and has departed picks its next hop toward its crate's target.
    The chosen edge becomes the truck's next step in ``_advance_vehicles``; a wait
    (self-edge) or a blocked node leaves it in place. Idle trucks are no-ops."""
    for i, vehicle in enumerate(state.vehicles):
        crate = vehicle.load
        cargo = vehicle.carrying
        if crate is None or cargo is None:
            continue
        # only choose a hop when standing at a node, departed, and target not reached
        if vehicle.edge_ticks_left > 0 or vehicle.route:
            continue
        if state.tick < cargo.departure_tick or vehicle.current_node == crate.target_node:
            continue
        action = actions.get(f"routing_{i}")
        if action is None:
            continue
        edges = list(state.graph.out_edges(vehicle.current_node, data=True))
        if not edges:
            continue
        _, target, data = edges[int(action) % len(edges)]
        if target == vehicle.current_node:  # wait action — stay put
            continue
        blocked = any(
            d.type is DisruptionType.BLOCKED_NODE and d.target == target
            for d in state.active_disruptions
        )
        if blocked:
            continue
        vehicle.route = [target]
        vehicle.route_transit += float(data["base_transit_time"])
        vehicle.route_emissions += float(data["base_emissions"])
        vehicle.emissions += float(data["base_emissions"])


def _apply_temperature_actions(state: GlobalState, actions: dict[str, Any]) -> None:
    """One temperature agent per truck's crate (paper Alg 2): set the reefer
    setpoint, applied by ``_advance_loads``. Idle trucks are no-ops."""
    for i, vehicle in enumerate(state.vehicles):
        crate = vehicle.load
        if crate is None:
            continue
        action = actions.get(f"temperature_{i}")
        if action is None:
            continue
        value = float(np.asarray(action).flatten()[0])
        crate.desired_temperature_c = float(
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

        # Routing now drives the (variable) path, so no route is precomputed. The
        # slot sets the departure window and the SLA deadline; delay / SLA and the
        # real transit/emissions accrue hop-by-hop and settle when the truck arrives.
        departure = max(
            state.tick, int(np.ceil(slot_start(vehicle.chosen_slot, state.max_steps)))
        )
        deadline = slot_deadline(vehicle.chosen_slot, state.max_steps)

        vehicle.route_transit = 0.0
        vehicle.route_emissions = 0.0
        vehicle.sla_deadline = deadline
        # optimistic until arrival: a large sentinel keeps busy_until in the future
        vehicle.busy_until = state.max_steps * 2
        vehicle.delay = 0.0
        vehicle.sla_violated = False
        vehicle.emissions = 0.0
        cargo = Cargo(
            vehicle=i,
            instance=instance,
            departure_tick=departure,
            # Filled with the actual delivery tick when the truck arrives; a large
            # sentinel keeps it in-transit (spoilage decay applies) until then.
            arrival_tick=state.max_steps * 3,
            qty=qty,
            emissions=0.0,
        )
        state.cargo.append(cargo)
        vehicle.carrying = cargo
        vehicle.load = _new_crate(state, f"retail_{instance}")
        vehicle.current_node = state.depot
        vehicle.route = []
        vehicle.edge_ticks_left = 0


def _new_crate(state: GlobalState, target: str) -> Consignment:
    """A fresh crate loaded onto a dispatched truck: cold at the depot, bound for
    its retailer. Carries its own thermal + spoilage state so goods spoil by
    their own temperature (multi-instance redesign). The temperature agent is
    wired to it in Phase 3; until then the setpoint stays at the fruit optimum."""
    params = get_params(state.fruit)
    return Consignment(
        fruit_type=state.fruit,
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
    (crediting inventory), is shown parked at the retailer for that tick, and
    snaps back to the depot on the next tick (the deferred/instant return)."""
    # Deferred return: an idle truck left at its retailer last tick goes home now
    # so the arrival is visible for one frame. ``current_node`` is viz-only (no
    # obs reads it), so this never touches training.
    for vehicle in state.vehicles:
        if vehicle.carrying is None and vehicle.current_node != state.depot:
            vehicle.current_node = state.depot
    for vehicle in state.vehicles:
        cargo = vehicle.carrying
        # ``route`` holds the single next hop the routing agent chose this tick;
        # empty means the truck waits (at the depot before departure, or between
        # hops until routing picks again).
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
            crate = vehicle.load
            if crate is not None and vehicle.current_node == crate.target_node:
                _deliver_vehicle(state, vehicle)


def _deliver_vehicle(state: GlobalState, vehicle: VehicleState) -> None:
    cargo = vehicle.carrying
    crate = vehicle.load
    # Delay / SLA settle against the real (routing-driven) arrival; emissions are
    # the summed hops the routing agent actually drove.
    vehicle.delay = max(0.0, state.tick - vehicle.sla_deadline)
    vehicle.sla_violated = state.tick > vehicle.sla_deadline
    if cargo is not None:
        cargo.arrival_tick = state.tick  # inventory credits it next tick
        cargo.emissions = vehicle.route_emissions
    if crate is not None:
        crate.ground_truth_label = risk_to_label(crate.spoilage_risk)
    vehicle.carrying = None
    vehicle.load = None  # crate delivered
    vehicle.busy_until = state.tick
    # Leave the truck parked at the retailer this tick so the arrival is visible;
    # ``_advance_vehicles`` returns it to the depot next tick.
    vehicle.route = []
    vehicle.edge_ticks_left = 0


def respawn_shipment(state: GlobalState) -> None:
    """Deprecated (singleton elimination): the world no longer cycles a singleton
    shipment. Kept as a no-op so rolling callers that still invoke it don't break
    until they are repointed. See docs/singleton_elimination.md."""
    return


def _apply_spoilage_actions(state: GlobalState, actions: dict[str, Any]) -> None:
    """One spoilage agent per truck's crate (paper Alg 3): predict this crate's
    spoilage probability. Idle trucks are no-ops."""
    for i, vehicle in enumerate(state.vehicles):
        crate = vehicle.load
        if crate is None:
            continue
        action = actions.get(f"spoilage_{i}")
        if action is None:
            continue
        value = float(np.asarray(action).flatten()[0])
        crate.spoilage_prediction = float(np.clip(value, 0.0, 1.0))


def _advance_node_climate(state: GlobalState) -> None:
    """Drift each node's storage temperature/humidity: mean-revert toward the kind
    setpoint, pull weakly toward ambient (kind-specific strength), add small noise,
    and hard-clamp to the kind band. Uses the isolated climate rng so demand and
    disruption streams stay bit-identical."""
    rng = state.node_climate_rng
    for node, data in state.graph.nodes(data=True):
        kind = data["kind"]
        set_c = config.NODE_CLIMATE_SETPOINT_C[kind]
        lo, hi = config.NODE_CLIMATE_BAND_C[kind]
        pull = config.NODE_CLIMATE_AMBIENT_PULL[kind]
        t = state.node_temp_c[node]
        t += config.NODE_CLIMATE_REVERSION * (set_c - t)
        t += pull * (state.ambient_temp_c - t)
        t += float(rng.normal(0.0, config.NODE_CLIMATE_TEMP_SIGMA))
        state.node_temp_c[node] = float(np.clip(t, lo, hi))

        h_set = config.NODE_CLIMATE_HUMIDITY[kind]
        h = state.node_humidity[node]
        h += config.NODE_CLIMATE_REVERSION * (h_set - h)
        h += float(rng.normal(0.0, config.NODE_CLIMATE_HUMIDITY_SIGMA))
        state.node_humidity[node] = float(np.clip(h, 0.0, 1.0))


def _node_external_temp(state: GlobalState, node: str) -> float:
    """External temperature a load at ``node`` fights — the node's own storage
    climate (Design F). Falls back to ambient if the node has no climate entry."""
    return state.node_temp_c.get(node, state.ambient_temp_c)


def _advance_thermal_state(state: GlobalState) -> None:
    s = state.shipment
    diff = s.desired_temperature_c - s.sensor_temperature_c
    external = _node_external_temp(state, s.current_node)
    ambient_pull = (external - s.sensor_temperature_c) * 0.05
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


def _advance_weather(state: GlobalState) -> None:
    """Evolve the day's weather via the sticky Markov chain, then re-roll ambient
    temperature (weather base + seasonal + noise) and humidity. Node micro-climate
    pulls toward this fresh ambient, so a passing storm or heat spell propagates
    into the network. Uses the isolated weather rng."""
    rng = state.weather_rng
    state.ambient_weather = demand.advance_weather(rng, state.ambient_weather)
    state.ambient_temp_c = _sample_ambient_temp(
        rng, state.ambient_weather, state.day_of_year
    )
    state.ambient_humidity = _sample_ambient_humidity(rng, state.ambient_weather)


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
        # thermal relaxation toward the setpoint plus a pull toward the crate's
        # local node climate (Design F — the facility it sits in), deterministic
        external = _node_external_temp(state, crate.current_node)
        diff = crate.desired_temperature_c - crate.sensor_temperature_c
        ambient_pull = (external - crate.sensor_temperature_c) * 0.05
        crate.sensor_temperature_c += 0.5 * diff + ambient_pull
        # reefer energy = fighting the local node climate (per-crate, Design F)
        crate.energy = abs(crate.sensor_temperature_c - external) * 0.1
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
    external = _node_external_temp(state, s.current_node)
    diff = abs(s.sensor_temperature_c - external)
    state.energy_usage = diff * 0.1


def _build_infos(state: GlobalState) -> dict[str, dict[str, Any]]:
    infos: dict[str, dict[str, Any]] = {}
    # Per-crate routing / temperature / spoilage (empty-ish when the truck is idle).
    for i, vehicle in enumerate(state.vehicles):
        crate = vehicle.load
        loaded = crate is not None
        infos[f"routing_{i}"] = {
            "loaded": float(loaded),
            "at_target": float(loaded and vehicle.current_node == crate.target_node),
        }
        infos[f"temperature_{i}"] = {
            "energy_usage": crate.energy if loaded else 0.0,
        }
        infos[f"spoilage_{i}"] = {
            "y_pred": crate.spoilage_prediction if loaded else 0.0,
            "ground_truth_label": crate.ground_truth_label if loaded else 0,
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
