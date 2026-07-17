"""Evaluate paper Alg 6 (LLM-mediated negotiation) on delivery slot conflicts.

Rolls out the trained delivery block on held-out seeds twice: as-is
(conflicting slot claims pay the env penalty) and with a mediator that
intercepts conflicting claims before env.step and replaces them with the
negotiated assignment (failed negotiations leave the conflict standing).

Usage:
    uv run python -m training.marl.negotiation_eval --greedy
    uv run python -m training.marl.negotiation_eval --model <lm-studio-model>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from core.config import DELIVERY_AGENTS, N_DELIVERY_WINDOWS
from core.state import GlobalState
from env.training_env import (
    DELIVERY_CONFLICT_PENALTY,
    DELIVERY_DELAY_WEIGHT,
    DELIVERY_SLA_WEIGHT,
    ColdChainTrainingEnv,
)
from llm.client import LLMConfig, OpenAICompatClient
from llm.negotiation import NegotiationError, SlotParty, negotiate
from training.config import FORECASTER_PATH, SEED, env_config, load_agents

NEGO_SEED = 800_000
DEFAULT_EPISODES = 30
DEFAULT_ROUNDS = 3

ARM_METRICS = ("conflict", "slot_cost", "delay", "sla_violated")


class SlotMediator:
    """Detects delivery slot collisions and runs the Alg 6 protocol on them.

    Identical conflicts (same claims, costs, and free slots) are resolved
    once and cached — negotiation is deterministic given its inputs.
    """

    def __init__(self, client: OpenAICompatClient | None, max_rounds: int) -> None:
        self._client = client
        self._max_rounds = max_rounds
        self._cache: dict[Any, Any] = {}
        self.events = 0
        self.agreements = 0
        self.failures = 0
        self.errors = 0
        self.cache_hits = 0
        self.negotiations = 0
        self.rounds: list[int] = []

    def resolve(self, actions: dict[str, Any], state: GlobalState) -> dict[str, Any]:
        claims = {
            a: int(actions[a]) % N_DELIVERY_WINDOWS
            for a in DELIVERY_AGENTS
            if a in actions
        }
        by_slot: dict[int, list[str]] = defaultdict(list)
        for agent, slot in claims.items():
            by_slot[slot].append(agent)
        for slot, group in by_slot.items():
            if len(group) < 2:
                continue
            self.events += 1
            forbidden = frozenset(s for s in by_slot if s != slot)
            parties = [self._party(a, claims[a], state) for a in group]
            agreement = self._negotiate_cached(parties, forbidden)
            if agreement is None:
                self.failures += 1
                continue
            self.agreements += 1
            self.rounds.append(agreement.rounds)
            for name, agreed in agreement.assignment.items():
                actions[name] = agreed
        return actions

    def _party(self, name: str, slot: int, state: GlobalState) -> SlotParty:
        vehicle = state.vehicles[int(name.rsplit("_", 1)[1])]
        costs = tuple(
            _slot_cost(vehicle.route_transit, s, state.max_steps)
            for s in range(N_DELIVERY_WINDOWS)
        )
        return SlotParty(name, slot, costs, DELIVERY_CONFLICT_PENALTY)

    def _negotiate_cached(self, parties: list[SlotParty], forbidden: frozenset[int]):
        key = (
            tuple(
                sorted(
                    (p.name, p.initial_slot, tuple(round(c, 2) for c in p.slot_costs))
                    for p in parties
                )
            ),
            forbidden,
        )
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.negotiations += 1
        try:
            agreement = negotiate(parties, self._client, self._max_rounds, forbidden)
        except NegotiationError:
            self.errors += 1
            agreement = None
        self._cache[key] = agreement
        return agreement


def _slot_cost(transit: float, slot: int, max_steps: int) -> float:
    """Conflict-free part of the env's slot_cost (core dynamics deadline)."""
    deadline = (slot + 1) / N_DELIVERY_WINDOWS * max_steps
    delay = max(0.0, transit - deadline)
    return DELIVERY_DELAY_WEIGHT * delay + DELIVERY_SLA_WEIGHT * float(
        transit > deadline
    )


def _run_arm(
    mediator: SlotMediator | None,
    episodes: int,
    forecaster,
    tag: str | None,
) -> dict[str, float]:
    block = list(DELIVERY_AGENTS)
    env = ColdChainTrainingEnv(env_config(NEGO_SEED, block, forecaster))
    agents = load_agents(env, block, tag)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {a: agents[a].act(obs[a], explore=False) for a in agents}
            if mediator is not None:
                actions = mediator.resolve(actions, env._state)
            obs, _, terminated, truncated, infos = env.step(actions)
            for a in block:
                for key in ARM_METRICS:
                    values[key].append(float(infos[a][key]))
            done = terminated["__all__"] or truncated["__all__"]
    return {key: float(np.mean(values[key])) for key in ARM_METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="negotiation limit T"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="rule-based mediator instead of the LLM (no server needed)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--forecaster", action="store_true")
    parser.add_argument(
        "--tag", default=None, help="load suffixed delivery module variant"
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    forecaster = FORECASTER_PATH if args.forecaster else None

    client = None
    if not args.greedy:
        overrides: dict[str, Any] = {"temperature": 0.2}
        if args.model:
            overrides["model"] = args.model
        if args.base_url:
            overrides["base_url"] = args.base_url
        client = OpenAICompatClient(LLMConfig.from_env(**overrides))

    mediator = SlotMediator(client, args.rounds)
    baseline = _run_arm(None, args.episodes, forecaster, args.tag)
    mediated = _run_arm(mediator, args.episodes, forecaster, args.tag)
    if client is not None:
        client.close()

    label = "greedy" if args.greedy else "llm"
    print(
        f"\ndelivery slot negotiation ({label} mediator, T={args.rounds}, "
        f"{args.episodes} episodes, seed {NEGO_SEED})"
    )
    print(f"{'metric':<14}{'baseline':>10}{'mediated':>10}{'delta':>9}")
    for key in ARM_METRICS:
        base, med = baseline[key], mediated[key]
        delta = f"{(med - base) / abs(base):+.0%}" if abs(base) >= 1e-9 else "   --"
        print(f"{key:<14}{base:>10.3f}{med:>10.3f}{delta:>9}")

    mean_rounds = float(np.mean(mediator.rounds)) if mediator.rounds else 0.0
    print(
        f"\nnegotiation: {mediator.events} conflicts, "
        f"{mediator.agreements} agreements (mean {mean_rounds:.1f} rounds), "
        f"{mediator.failures} failures ({mediator.errors} mediator errors), "
        f"{mediator.negotiations} negotiations run, "
        f"{mediator.cache_hits} cache hits"
    )


if __name__ == "__main__":
    main()
