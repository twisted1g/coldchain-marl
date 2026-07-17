"""Paper Alg 6: LLM-Mediated Agent Negotiation Protocol.

The paper box is two-party; a delivery-slot conflict in our world can
involve up to N_VEHICLES claimants, so the protocol is implemented
n-party (the box is the n=2 case). Mapping onto the box:

- offer O_it = a slot claim grounded in the party's local state (its
  per-slot cost table, derived from route transit time);
- S_t = L(H) = a structured summary with a proposed conflict-free slot
  assignment (the "summary of intents and preferences");
- utility U_i = -cost_i(assigned slot), computed from S_t;
- threshold tau_i = the status-quo utility of keeping the conflicted
  slot and paying the conflict penalty, so an agreement must leave
  every party no worse off than the unresolved conflict;
- "update proposal strategy based on feedback or regret" = each party
  concedes to its cheapest slot not claimed by the others in S_t.

The LLM proposal is validated (each party assigned exactly once,
distinct in-range slots, forbidden slots untouched) with reject/retry,
mirroring the scenario-bank pipeline: grammar-constrained decoding
guarantees structure, not semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.client import ChatClient, LLMError

MAX_SUMMARY_ATTEMPTS = 3

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "one-paragraph summary of the parties' intents",
        },
        "assignment": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "slot": {"type": "integer", "minimum": 0},
                },
                "required": ["agent", "slot"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "assignment"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a neutral mediator for delivery vehicles negotiating over "
    "delivery time slots. Several vehicles claimed the same slot. Each "
    "vehicle reports its cost for every slot (lower is better; costs "
    "reflect lateness against the slot deadline). Read the offer history, "
    "summarize each vehicle's intent, and propose a conflict-free "
    "assignment: every vehicle gets exactly one slot, all slots distinct, "
    "never use a forbidden slot. Prefer the assignment with the lowest "
    "total cost; break ties in favor of the vehicle with the most to lose."
)


class NegotiationError(LLMError):
    """LLM produced no valid assignment within the attempt budget."""


@dataclass(frozen=True)
class SlotParty:
    """One negotiating agent: its policy-chosen slot and per-slot costs."""

    name: str
    initial_slot: int
    slot_costs: tuple[float, ...]
    conflict_penalty: float

    @property
    def threshold(self) -> float:
        return -(self.slot_costs[self.initial_slot] + self.conflict_penalty)

    def utility(self, slot: int) -> float:
        return -self.slot_costs[slot]

    def best_slot(self, forbidden: frozenset[int]) -> int:
        candidates = [s for s in range(len(self.slot_costs)) if s not in forbidden]
        if not candidates:
            return self.initial_slot
        return min(candidates, key=self.slot_costs.__getitem__)


@dataclass(frozen=True)
class Agreement:
    assignment: dict[str, int]
    summary: str
    rounds: int


def negotiate(
    parties: list[SlotParty],
    client: ChatClient | None,
    max_rounds: int = 3,
    forbidden: frozenset[int] = frozenset(),
) -> Agreement | None:
    """Run the Alg 6 loop; ``client=None`` uses the greedy mediator baseline.

    Returns the agreement, or ``None`` after ``max_rounds`` failed rounds
    (the conflict then stands and the env penalty applies).
    """
    history: list[dict[str, int]] = []
    offers = {p.name: p.initial_slot for p in parties}
    for t in range(1, max_rounds + 1):
        history.append(dict(offers))
        if client is None:
            summary, assignment = _greedy_summary(parties, forbidden)
        else:
            summary, assignment = _llm_summary(client, parties, history, forbidden)
        if all(p.utility(assignment[p.name]) >= p.threshold for p in parties):
            return Agreement(assignment, summary, t)
        offers = {
            p.name: p.best_slot(
                forbidden | {s for n, s in assignment.items() if n != p.name}
            )
            for p in parties
        }
    return None


def _greedy_summary(
    parties: list[SlotParty], forbidden: frozenset[int]
) -> tuple[str, dict[str, int]]:
    """Regret-order greedy: parties with the most to lose pick first."""

    def regret(p: SlotParty) -> float:
        ranked = sorted(
            p.slot_costs[s] for s in range(len(p.slot_costs)) if s not in forbidden
        )
        return ranked[1] - ranked[0] if len(ranked) > 1 else float("inf")

    taken = set(forbidden)
    assignment: dict[str, int] = {}
    for p in sorted(parties, key=regret, reverse=True):
        slot = p.best_slot(frozenset(taken))
        assignment[p.name] = slot
        taken.add(slot)
    return "greedy mediator (no LLM)", assignment


def _llm_summary(
    client: ChatClient,
    parties: list[SlotParty],
    history: list[dict[str, int]],
    forbidden: frozenset[int],
) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_state(parties, history, forbidden)},
    ]
    last_error = ""
    for _ in range(MAX_SUMMARY_ATTEMPTS):
        raw = client.complete_json(messages, SUMMARY_SCHEMA, schema_name="mediation")
        try:
            return str(raw.get("summary", "")), _parse_assignment(
                raw, parties, forbidden
            )
        except NegotiationError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": str(raw)})
            messages.append(
                {"role": "user", "content": f"Invalid assignment: {exc}. Try again."}
            )
    raise NegotiationError(f"no valid assignment: {last_error}")


def _render_state(
    parties: list[SlotParty], history: list[dict[str, int]], forbidden: frozenset[int]
) -> str:
    n_slots = len(parties[0].slot_costs)
    lines = [f"Slots: 0..{n_slots - 1}."]
    if forbidden:
        lines.append(f"Forbidden slots (taken by others): {sorted(forbidden)}.")
    lines.append("Vehicles and their per-slot costs:")
    for p in parties:
        costs = ", ".join(f"slot {s}: {c:.2f}" for s, c in enumerate(p.slot_costs))
        lines.append(f"- {p.name}: wants slot {p.initial_slot}; costs: {costs}")
    lines.append("Offer history (round: vehicle -> claimed slot):")
    for t, offers in enumerate(history, start=1):
        claims = ", ".join(f"{name} -> {slot}" for name, slot in offers.items())
        lines.append(f"- round {t}: {claims}")
    lines.append(
        "Propose the assignment (each vehicle exactly one distinct slot) "
        "and summarize the negotiation."
    )
    return "\n".join(lines)


def _parse_assignment(
    raw: dict[str, Any], parties: list[SlotParty], forbidden: frozenset[int]
) -> dict[str, int]:
    names = {p.name for p in parties}
    n_slots = len(parties[0].slot_costs)
    entries = raw.get("assignment")
    if not isinstance(entries, list):
        raise NegotiationError("assignment is not a list")
    assignment: dict[str, int] = {}
    for entry in entries:
        agent, slot = entry.get("agent"), entry.get("slot")
        if agent not in names:
            raise NegotiationError(f"unknown agent {agent!r}")
        if agent in assignment:
            raise NegotiationError(f"agent {agent!r} assigned twice")
        if not isinstance(slot, int) or not 0 <= slot < n_slots:
            raise NegotiationError(f"slot {slot!r} out of range for {agent!r}")
        if slot in forbidden:
            raise NegotiationError(f"slot {slot} is forbidden")
        assignment[agent] = slot
    if set(assignment) != names:
        raise NegotiationError(f"missing agents: {sorted(names - set(assignment))}")
    if len(set(assignment.values())) != len(assignment):
        raise NegotiationError("assignment reuses a slot")
    return assignment
