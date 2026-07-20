"""Roll out one episode with trained policies and serialize per-tick state.

Output is a JSONL stream: the first line is a ``meta`` record (graph layout,
fruit thresholds, episode length), each following line is a ``tick`` record
(shipment, inventory, vehicles, disruptions, rewards). The dashboard renderer
consumes this file; nothing here imports matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core import config
from core.world.fruits import get_params
from env.training_env import ColdChainTrainingEnv
from training.config import ARTIFACTS, LEARNERS, env_config, load_agents


def _graph_meta(state: Any) -> dict[str, Any]:
    nodes = [
        {"name": n, "kind": data["kind"]}
        for n, data in state.graph.nodes(data=True)
    ]
    edges = [
        [u, v]
        for u, v, data in state.graph.edges(data=True)
        if not data["wait"]
    ]
    params = get_params(state.shipment.fruit_type)
    return {
        "type": "meta",
        "fruit": str(state.shipment.fruit_type),
        "source": state.shipment.current_node,
        "target": state.shipment.target_node,
        "max_steps": state.max_steps,
        "nodes": nodes,
        "edges": edges,
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
                "arrival_tick": c.arrival_tick,
                "qty": float(c.qty),
            }
            for c in state.cargo
        ],
        "vehicles": [
            {
                "assigned_node": v.assigned_node,
                "chosen_slot": v.chosen_slot,
                "busy_until": v.busy_until,
                "delay": float(v.delay),
                "sla_violated": bool(v.sla_violated),
                "conflict": bool(v.conflict),
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


def record_episode(
    seed: int,
    tag: str | None,
    scenario_id: str | None,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    cfg = env_config(base_seed=seed, learners=LEARNERS)
    if max_steps is not None:
        cfg["max_steps"] = max_steps
    env = ColdChainTrainingEnv(cfg)
    agents = load_agents(env, LEARNERS, tag)

    options = {"scenario_id": scenario_id} if scenario_id else None
    obs, _ = env.reset(options=options)

    records: list[dict[str, Any]] = [_graph_meta(env.world_state)]
    records.append(_tick_record(env.world_state, {}, {}, {}))

    done = False
    while not done:
        actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        records.append(_tick_record(env.world_state, rewards, actions, infos))
        done = terminated["__all__"] or truncated["__all__"]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--tag", default=None, help="module variant, e.g. scn05")
    parser.add_argument("--scenario", default=None, help="LLM scenario id (needs bank)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL (default artifacts/episodes/<seed>.jsonl)",
    )
    args = parser.parse_args()

    records = record_episode(args.seed, args.tag, args.scenario, args.max_steps)

    out = args.out or ARTIFACTS / "episodes" / f"episode_{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    ticks = sum(1 for r in records if r["type"] == "tick")
    print(f"wrote {ticks} ticks -> {out}")


if __name__ == "__main__":
    main()
