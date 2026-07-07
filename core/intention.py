from __future__ import annotations

from typing import Any

from core import config


class IntentionBuffer:
    """Shared intention buffer B (paper Alg 1-5): agents declare (state, action)
    intentions each step, conflicts are detected across same-type instances, and a
    coordination penalty ρ is flagged (paper's alternative tie-break/reassign is not
    needed here — episodes terminate by delivery or max_steps regardless).

    In the scoped single-shipment world only the delivery role is multi-instance
    (N vehicles), so it is the sole live conflict source; the other four agents are
    single-instance and declare into B without producing conflicts.
    """

    def __init__(self) -> None:
        self._intentions: dict[str, Any] = {}

    def declare(self, agent: str, action: Any) -> None:
        self._intentions[agent] = action

    def declare_all(self, actions: dict[str, Any]) -> None:
        for agent, action in actions.items():
            self.declare(agent, action)

    def detect(self) -> dict[str, bool]:
        conflicts = {agent: False for agent in self._intentions}
        self._detect_delivery_slot_conflicts(conflicts)
        return conflicts

    def _detect_delivery_slot_conflicts(self, conflicts: dict[str, bool]) -> None:
        slots: dict[int, list[str]] = {}
        for agent, action in self._intentions.items():
            if agent in config.DELIVERY_AGENTS and action is not None:
                slot = int(action) % config.N_DELIVERY_WINDOWS
                slots.setdefault(slot, []).append(agent)
        for agents in slots.values():
            if len(agents) > 1:
                for agent in agents:
                    conflicts[agent] = True

    def clear(self) -> None:
        self._intentions = {}
