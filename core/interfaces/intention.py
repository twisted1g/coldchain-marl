from __future__ import annotations

from typing import Any

from core import config


class IntentionBuffer:
    """Shared intention buffer B (paper Alg 5): declare, detect conflicting
    vehicle assignments / delivery-slot resource overuse. Inventory supply
    contention is left to the order queue's reassignment (Alg 4 line 10), so no
    ρ penalty is raised here."""

    def __init__(self) -> None:
        self._intentions: dict[str, Any] = {}

    def declare_all(self, actions: dict[str, Any]) -> None:
        self._intentions.update(actions)

    def detect(
        self, free: set[int] | None = None, capacity: int = config.SLOT_CAPACITY
    ) -> dict[str, bool]:
        """Flag a conflict when more than ``capacity`` *available* vehicles
        (indices in ``free``; ``None`` = all) claim the same delivery slot —
        the paper's "conflicting vehicle assignments or resource overuse". A
        vehicle already mid-trip is not being assigned, so it is excluded."""
        conflicts = dict.fromkeys(self._intentions, False)
        self._detect_delivery_slot_conflicts(conflicts, free, capacity)
        return conflicts

    def _detect_delivery_slot_conflicts(
        self, conflicts: dict[str, bool], free: set[int] | None, capacity: int
    ) -> None:
        slots: dict[int, list[str]] = {}
        for agent, action in self._intentions.items():
            if agent not in config.DELIVERY_AGENTS or action is None:
                continue
            idx = int(agent.rsplit("_", 1)[1])
            if free is not None and idx not in free:
                continue
            slot = int(action) % config.N_DELIVERY_WINDOWS
            slots.setdefault(slot, []).append(agent)
        for agents in slots.values():
            if len(agents) > capacity:
                for agent in agents:
                    conflicts[agent] = True
