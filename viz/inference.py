"""Rolling live inference — the single engine behind the dashboard.

A trained-policy world is kept running past a single delivery: the training
episode ends the moment the routing shipment reaches its target (~3-4 ticks),
far too short for restock vehicles to leave their slot windows or for
disruptions to accumulate. ``run_inference`` keeps the same world alive — when a
shipment is delivered it is *shown arriving*, then a fresh one is rolled onto the
network (``env.rollover``) while inventory, vehicles, cargo and the calendar
persist. Orders queue, vehicles dispatch and problems (slot conflicts, SLA
breaches, stockouts, disruptions) build up over a long horizon.

Delivery-slot conflicts (paper Alg 5 line 14) are resolved before each step by
the ``mediator`` (paper Alg 6): "off"/"greedy"/"llm". Each tick carries the
negotiation events under ``negotiations`` so the dashboard can show the
mediation.

The generator yields a ``meta`` record first, then one ``tick`` record per step
— the exact stream ``viz.record`` writes to JSONL and ``viz.server`` pushes over
SSE, so the frontend consumes both identically. ``viz.record`` collects the
stream into a file; ``viz.live`` streams it live.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from core import config
from core.world.fruits import get_params
from env.training_env import ColdChainTrainingEnv
from llm.mediation import build_mediator
from training.config import LEARNERS, env_config, load_agents

DEFAULT_HORIZON = 60
# The delivery slots split ``max_steps`` into departure windows; keeping this
# short compresses the windows so free vehicles dispatch densely and the fleet
# cycles continuously rather than leaking out one truck every several ticks.
DEFAULT_SLOT_SPAN = 10


def _graph_meta(state: Any) -> dict[str, Any]:
    nodes = [
        {
            "name": n,
            "kind": data["kind"],
            # static per-kind storage climate target + band (Design F)
            "climate_setpoint": config.NODE_CLIMATE_SETPOINT_C[data["kind"]],
            "climate_band": list(config.NODE_CLIMATE_BAND_C[data["kind"]]),
        }
        for n, data in state.graph.nodes(data=True)
    ]
    edges = [
        [u, v]
        for u, v, data in state.graph.edges(data=True)
        if not data["wait"]
    ]
    params = get_params(state.shipment.fruit_type)
    # The restock fleet's transit cost is the weighted shortest path from the
    # source farm to each retailer. Ship that exact route so the dashboard draws
    # trucks farm->hub->dc->retail along the real path.
    source = state.shipment.current_node
    retailers = sorted(
        n for n, data in state.graph.nodes(data=True) if data["kind"] == "retail"
    )
    restock_paths = [
        nx.shortest_path(state.graph, source, r, weight="base_transit_time")
        for r in retailers
    ]
    return {
        "type": "meta",
        "fruit": str(state.shipment.fruit_type),
        "source": source,
        "target": state.shipment.target_node,
        "max_steps": state.max_steps,
        "nodes": nodes,
        "edges": edges,
        "restock_paths": restock_paths,
        "thresholds": {
            "optimal_temp_low": params.optimal_temp_low_c,
            "optimal_temp_high": params.optimal_temp_high_c,
            "chill_injury": params.chilling_injury_threshold_c,
            "optimal_humidity_low": params.optimal_humidity_low,
            "optimal_humidity_high": params.optimal_humidity_high,
        },
        "n_windows": config.N_DELIVERY_WINDOWS,
    }


def _jsonify(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, bool, str)):
        return value
    return float(value)


def _tick_record(
    state: Any,
    rewards: dict[str, float],
    actions: dict[str, Any],
    infos: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    s = state.shipment
    return {
        "type": "tick",
        "tick": state.tick,
        "actions": {a: _jsonify(v) for a, v in actions.items()},
        "infos": {
            a: {k: _jsonify(v) for k, v in info.items()}
            for a, info in infos.items()
        },
        "shipment": {
            "current_node": s.current_node,
            "target_node": s.target_node,
            "age_ticks": s.age_ticks,
            "spoilage_risk": s.spoilage_risk,
            "freshness_score": s.freshness_score,
            "sensor_temp": s.sensor_temperature_c,
            "desired_temp": s.desired_temperature_c,
            "sensor_humidity": s.sensor_humidity,
        },
        "ambient": {
            "weather": str(state.ambient_weather),
            "temp": state.ambient_temp_c,
            "humidity": state.ambient_humidity,
        },
        # per-node micro-climate (Design F): each node's live storage temp/humidity
        "node_climate": {
            n: {
                "temp": float(state.node_temp_c[n]),
                "humidity": float(state.node_humidity[n]),
            }
            for n in state.node_temp_c
        },
        "calendar": {
            "day_of_year": state.day_of_year,
            "weekday": state.weekday,
            "event_multiplier": state.event_multiplier,
        },
        "inventory": {
            "levels": [float(x) for x in state.inventory_levels],
            "order": [float(x) for x in state.inventory_order],
            "unmet": [float(x) for x in state.unmet_demand],
            "demand_today": [float(x) for x in state.demand_today],
            "forecast": [float(x) for x in state.demand_forecast],
        },
        "cargo": [
            {
                "vehicle": c.vehicle,
                "instance": c.instance,
                "departure_tick": c.departure_tick,
                "arrival_tick": c.arrival_tick,
                "qty": float(c.qty),
            }
            for c in state.cargo
        ],
        "order_queue": [[int(i), float(q)] for i, q in state.order_queue],
        "vehicles": [
            {
                "assigned_node": v.assigned_node,
                "chosen_slot": v.chosen_slot,
                "busy_until": v.busy_until,
                "delay": float(v.delay),
                "sla_violated": bool(v.sla_violated),
                "conflict": bool(v.conflict),
                "current_node": v.current_node,
                "carrying": (v.carrying.instance if v.carrying is not None else None),
                "route_transit": float(v.route_transit),
                "route_emissions": float(v.route_emissions),
                "sla_window_ticks": int(v.sla_window_ticks),
                "emissions": float(v.emissions),
                # Per-crate cold-chain state (multi-instance redesign): each
                # truck's goods carry their own temperature / spoilage, driven by
                # the temperature policy per crate. ``null`` when the truck is idle.
                "crate": (
                    {
                        "sensor_temp": float(v.load.sensor_temperature_c),
                        "desired_temp": float(v.load.desired_temperature_c),
                        "sensor_humidity": float(v.load.sensor_humidity),
                        "spoilage_risk": float(v.load.spoilage_risk),
                        "freshness_score": float(v.load.freshness_score),
                    }
                    if v.load is not None
                    else None
                ),
            }
            for v in state.vehicles
        ],
        "disruptions": [
            {"type": str(d.type), "target": d.target} for d in state.active_disruptions
        ],
        "spoilage_prediction": float(state.spoilage_prediction),
        "energy_usage": float(state.energy_usage),
        "rewards": {k: float(v) for k, v in rewards.items()},
    }


def build_env(
    seed: int, tag: str | None, slot_span: int, forecaster: Path | None = None
) -> tuple[ColdChainTrainingEnv, dict[str, Any]]:
    """Frozen backdrop with trained learner modules loaded. ``slot_span`` sizes
    the delivery windows (env ``max_steps``); the rolling horizon is driven by
    the caller, not the env's terminal flag."""
    cfg = env_config(base_seed=seed, learners=LEARNERS, forecaster=forecaster)
    cfg["max_steps"] = slot_span
    env = ColdChainTrainingEnv(cfg)
    agents = load_agents(env, LEARNERS, tag)
    return env, agents


def _drive_crate_setpoints(state: Any, agents: dict[str, Any]) -> None:
    """CTDE decentralized execution: run the one trained temperature policy on
    each truck-borne crate and set its reefer setpoint, so every crate holds its
    own temperature (paper Section 4.2 — deploy the edge policy per crate/truck).
    ``_advance_loads`` applies the setpoint on the next tick."""
    policy = agents.get("temperature")
    if policy is None:
        return
    from core.config import TEMPERATURE_ACTION_HIGH_C, TEMPERATURE_ACTION_LOW_C
    from core.interfaces.observations import crate_temperature_obs

    for vehicle in state.vehicles:
        crate = vehicle.load
        if crate is None:
            continue
        action = policy.act(crate_temperature_obs(state, crate), explore=False)
        value = float(np.asarray(action).flatten()[0])
        crate.desired_temperature_c = float(
            min(max(value, TEMPERATURE_ACTION_LOW_C), TEMPERATURE_ACTION_HIGH_C)
        )


def run_inference(
    seed: int,
    tag: str | None = None,
    *,
    mediator: str | None = "llm",
    horizon: int = DEFAULT_HORIZON,
    slot_span: int = DEFAULT_SLOT_SPAN,
    forecaster: Path | None = None,
    scenario_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield ``meta`` then one ``tick`` record per step, for ``horizon`` ticks of
    rolling multi-shipment inference. See the module docstring."""
    env, agents = build_env(seed, tag, slot_span, forecaster)
    mediate = build_mediator(mediator)
    options = {"scenario_id": scenario_id} if scenario_id else None
    obs, _ = env.reset(options=options)

    meta = _graph_meta(env.world_state)
    meta["horizon"] = horizon
    meta["mediator"] = mediator if mediate is not None else "off"
    yield meta
    yield _tick_record(env.world_state, {}, {}, {})

    shipment_no = 1
    try:
        while env.world_state.tick < horizon:
            actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
            if mediate is not None:
                actions = mediate.resolve(actions, env.world_state)
            obs, rewards, _terminated, _truncated, infos = env.step(actions)
            state = env.world_state
            _drive_crate_setpoints(state, agents)

            rec = _tick_record(state, rewards, actions, infos)
            rec["shipment_no"] = shipment_no
            if mediate is not None:
                rec["negotiations"] = mediate.last_events
            yield rec

            # The routing shipment reached its target this tick: the record above
            # shows the arrival, now roll a fresh shipment onto the network and
            # carry on (env.step left the world otherwise terminal).
            if state.shipment.current_node == state.shipment.target_node:
                obs = env.rollover()
                shipment_no += 1
    finally:
        if mediate is not None:
            mediate.close()
