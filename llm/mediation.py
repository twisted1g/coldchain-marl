"""Runtime slot-conflict mediator (paper Alg 6, applied live).

Detects delivery-slot collisions in a set of policy actions and rewrites
the conflicting claims with the negotiated, conflict-free assignment
*before* ``env.step`` runs — so the mediation actually changes the world
(SLA, delay, conflict penalty), not just a report. ``client=None`` runs
the greedy mediator; an ``OpenAICompatClient`` runs the LLM protocol.

The eval harness (``training.marl.negotiation_eval``) and the live
dashboard (``viz``) share this one implementation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from core.config import DELIVERY_AGENTS, N_DELIVERY_WINDOWS
from core.dynamics import slot_deadline
from core.state import GlobalState
from env.training_env import (
    DELIVERY_CONFLICT_PENALTY,
    DELIVERY_DELAY_WEIGHT,
    DELIVERY_SLA_WEIGHT,
)
from llm.client import LLMConfig, OpenAICompatClient
from llm.negotiation import Agreement, NegotiationError, SlotParty, negotiate


def slot_cost(transit: float, slot: int, max_steps: int) -> float:
    """Conflict-free part of the env's slot_cost (core dynamics deadline)."""
    deadline = slot_deadline(slot, max_steps)
    delay = max(0.0, transit - deadline)
    return DELIVERY_DELAY_WEIGHT * delay + DELIVERY_SLA_WEIGHT * float(
        transit > deadline
    )


class SlotMediator:
    """Detects delivery slot collisions and runs the Alg 6 protocol on them.

    Identical conflicts (same claims, costs, and free slots) are resolved
    once and cached — negotiation is deterministic given its inputs.
    ``resolve`` also records per-event detail in ``last_events`` so a UI
    can show the negotiation (offers, rounds, summary, outcome).
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
        self.last_events: list[dict[str, Any]] = []

    @property
    def uses_llm(self) -> bool:
        return self._client is not None

    def resolve(self, actions: dict[str, Any], state: GlobalState) -> dict[str, Any]:
        self.last_events = []
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
            if agreement is not None:
                self.agreements += 1
                self.rounds.append(agreement.rounds)
                for name, agreed in agreement.assignment.items():
                    actions[name] = agreed
            else:
                self.failures += 1
            self.last_events.append(self._event(slot, parties, claims, agreement))
        return actions

    def _event(
        self,
        slot: int,
        parties: list[SlotParty],
        claims: dict[str, int],
        agreement: Agreement | None,
    ) -> dict[str, Any]:
        group = [p.name for p in parties]
        final = agreement.assignment if agreement else {a: claims[a] for a in group}
        return {
            "slot": slot,
            "agents": group,
            "mediator": "llm" if self.uses_llm else "greedy",
            "initial": {a: claims[a] for a in group},
            "final": {a: int(final[a]) for a in group},
            "costs": {p.name: [round(c, 3) for c in p.slot_costs] for p in parties},
            "resolved": agreement is not None,
            "rounds": agreement.rounds if agreement else 0,
            "summary": agreement.summary if agreement else "",
        }

    def _party(self, name: str, slot: int, state: GlobalState) -> SlotParty:
        vehicle = state.vehicles[int(name.rsplit("_", 1)[1])]
        costs = tuple(
            slot_cost(vehicle.route_transit, s, state.max_steps)
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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def build_mediator(mode: str | None, rounds: int = 3) -> SlotMediator | None:
    """``mode`` in {"off"/None, "greedy", "llm"}. ``llm`` reads LLM_* env."""
    if not mode or mode == "off":
        return None
    client = (
        None
        if mode == "greedy"
        else OpenAICompatClient(LLMConfig.from_env(temperature=0.2))
    )
    return SlotMediator(client, rounds)
