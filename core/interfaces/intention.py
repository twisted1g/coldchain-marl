from __future__ import annotations

from typing import Any

import numpy as np

from core import config


class IntentionBuffer:
    """Shared intention buffer B (paper Alg 1-5): declare, detect conflicts (ρ)."""

    def __init__(self) -> None:
        self._intentions: dict[str, Any] = {}

    def declare_all(self, actions: dict[str, Any]) -> None:
        self._intentions.update(actions)

    def detect(self, free_vehicles: int | None = None) -> dict[str, bool]:
        conflicts = dict.fromkeys(self._intentions, False)
        self._detect_delivery_slot_conflicts(conflicts)
        if free_vehicles is not None:
            self._detect_supply_contention(conflicts, free_vehicles)
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

    def _detect_supply_contention(
        self, conflicts: dict[str, bool], free_vehicles: int
    ) -> None:
        """Paper Alg 4 line 9: supply contention — more instances ordering this
        tick than there are free vehicles to carry the orders."""
        orderers = [
            agent
            for agent, action in self._intentions.items()
            if agent in config.INVENTORY_AGENTS
            and action is not None
            and float(np.asarray(action).flatten()[0]) * config.INVENTORY_RESTOCK_SCALE
            > config.INVENTORY_MIN_ORDER_QTY
        ]
        if len(orderers) > free_vehicles:
            for agent in orderers:
                conflicts[agent] = True
