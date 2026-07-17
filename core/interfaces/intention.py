from __future__ import annotations

from typing import Any

from core import config


class IntentionBuffer:
    """Shared intention buffer B (paper Alg 1-5): declare, detect conflicts (ρ)."""

    def __init__(self) -> None:
        self._intentions: dict[str, Any] = {}

    def declare_all(self, actions: dict[str, Any]) -> None:
        self._intentions.update(actions)

    def detect(self) -> dict[str, bool]:
        conflicts = dict.fromkeys(self._intentions, False)
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
