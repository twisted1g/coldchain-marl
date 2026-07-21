"""Continuous live inference: keep the world running past a single delivery.

The training episode terminates the moment the routing shipment reaches its
target (~3-4 ticks), which is far too short for restock vehicles to leave their
slot windows or for disruptions to accumulate. For the live dashboard we keep
the same world alive: when a shipment is delivered we respawn a fresh one on the
network (new source/target, reset thermal/spoilage state) and carry on. Inventory
levels, vehicles, cargo and the calendar persist, so orders queue, vehicles
dispatch and problems (conflicts, SLA breaches, stockouts, disruptions) build up
over a long horizon — a genuine rolling inference rather than a canned clip.

``live_stream`` yields the same records as ``viz.record`` (meta first, then one
per tick) so the frontend and JSONL renderer consume it unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from core.dynamics import respawn_shipment
from core.interfaces.observations import all_obs
from env.training_env import ColdChainTrainingEnv
from llm.mediation import build_mediator
from training.config import LEARNERS, env_config, load_agents
from viz.record import _graph_meta, _tick_record

DEFAULT_HORIZON = 60
# Delivery slots split ``max_steps`` into windows; a large value pushes the
# higher slots' departures deep into the future, so with the trained policy
# (which fixes vehicles to slots 1/2/3) the fleet leaves one-by-one over ~20
# ticks. A short span compresses the windows: departures fire early, and once
# the rolling tick passes it every free vehicle dispatches at once — the fleet
# cycles continuously with all three trucks moving in parallel.
DEFAULT_LIVE_MAX_STEPS = 10


def build_live_env(
    seed: int, tag: str | None, max_steps: int
) -> tuple[ColdChainTrainingEnv, dict[str, Any]]:
    cfg = env_config(base_seed=seed, learners=LEARNERS)
    cfg["max_steps"] = max_steps
    env = ColdChainTrainingEnv(cfg)
    agents = load_agents(env, LEARNERS, tag)
    return env, agents


def live_stream(
    seed: int,
    tag: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    max_steps: int | None = None,
    mediator: str | None = "llm",
) -> Iterator[dict[str, Any]]:
    """Yield meta then per-tick records for a rolling multi-shipment inference.

    ``mediator`` ("off"/"greedy"/"llm") resolves delivery-slot conflicts with
    the Alg 6 protocol before each step; each tick carries the negotiation
    events under ``negotiations`` so the dashboard can show the mediation.
    """
    max_steps = max_steps or DEFAULT_LIVE_MAX_STEPS
    env, agents = build_live_env(seed, tag, max_steps)
    mediate = build_mediator(mediator)
    obs, _ = env.reset(options=None)

    meta = _graph_meta(env.world_state)
    meta["horizon"] = horizon
    meta["mediator"] = mediator if mediate is not None else "off"
    yield meta
    yield _tick_record(env.world_state, {}, {}, {})

    shipment_no = 1
    # ``max_steps`` sizes the delivery slot windows (kept short so vehicles
    # dispatch densely); it also flips the env's ``terminated`` flag every tick
    # once ``tick`` passes it, which we deliberately ignore. The world keeps
    # rolling: the routing shipment is respawned only when it is actually
    # delivered, so it routes normally instead of resetting every tick.
    try:
        while env.world_state.tick < horizon:
            actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
            if mediate is not None:
                actions = mediate.resolve(actions, env.world_state)
            obs, rewards, _, _, infos = env.step(actions)
            state = env.world_state
            rec = _tick_record(state, rewards, actions, infos)
            rec["shipment_no"] = shipment_no
            if mediate is not None:
                rec["negotiations"] = mediate.last_events
            yield rec

            if state.shipment.current_node == state.shipment.target_node:
                respawn_shipment(state)
                env.agents = list(env.possible_agents)
                obs = env._apply_forecast(all_obs(state))
                env._snapshot_prev()
                shipment_no += 1
    finally:
        if mediate is not None:
            mediate.close()
