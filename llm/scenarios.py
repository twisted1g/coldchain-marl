"""Disruption scenario schema for the LLM-generated bank.

Paper (Section 5): an LLM produces disruption scenarios (labor strikes,
route closures, equipment malfunctions, regulatory actions) which a parser
turns into structured variables — blocked nodes, longer transit times,
risk flags. Extra effect kinds cover the stress axes of Figs. 37-39
(temperature, humidity, prolonged transport, power outages).

Effects use one flat shape ``{kind, magnitude, target_role}`` because the
GBNF grammar backing structured output handles flat objects and enums
reliably but not tagged unions. Numeric ranges are NOT enforced by the
grammar and are validated here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from core import config


class ScenarioCategory(StrEnum):
    LABOR_STRIKE = "labor_strike"
    ROUTE_CLOSURE = "route_closure"
    EQUIPMENT_MALFUNCTION = "equipment_malfunction"
    REGULATORY_ACTION = "regulatory_action"
    HEATWAVE = "heatwave"
    COLD_SNAP = "cold_snap"
    POWER_OUTAGE = "power_outage"


class EffectKind(StrEnum):
    BLOCK_NODES = "block_nodes"
    TRANSIT_DELAY = "transit_delay"
    RISK_FLAG = "risk_flag"
    TEMP_OFFSET = "temp_offset"
    HUMIDITY_OFFSET = "humidity_offset"
    DEMAND_SHOCK = "demand_shock"


class TargetRole(StrEnum):
    FARM = "farm"
    HUB = "hub"
    DC = "dc"
    RETAILER = "retailer"
    ANY = "any"


# Interpretation of ``magnitude`` per kind: block_nodes -> node count,
# transit_delay -> extra transit steps (matches DISRUPTION_TRANSIT_DELTA_RANGE),
# risk_flag -> risk level, temp_offset -> delta degrees C,
# humidity_offset -> delta fraction, demand_shock -> demand multiplier.
MAGNITUDE_RANGES: dict[EffectKind, tuple[float, float]] = {
    EffectKind.BLOCK_NODES: (1, 2),
    EffectKind.TRANSIT_DELAY: (
        config.DISRUPTION_TRANSIT_DELTA_RANGE[0],
        config.DISRUPTION_TRANSIT_DELTA_RANGE[1],
    ),
    EffectKind.RISK_FLAG: (1, config.N_RISK_LEVELS - 1),
    EffectKind.TEMP_OFFSET: (-15.0, 15.0),
    EffectKind.HUMIDITY_OFFSET: (-0.5, 0.5),
    EffectKind.DEMAND_SHOCK: (0.3, 2.5),
}

INTEGER_KINDS = frozenset(
    {EffectKind.BLOCK_NODES, EffectKind.TRANSIT_DELAY, EffectKind.RISK_FLAG}
)

TARGETED_KINDS = frozenset({EffectKind.BLOCK_NODES, EffectKind.RISK_FLAG})

CATEGORY_EFFECTS: dict[ScenarioCategory, frozenset[EffectKind]] = {
    ScenarioCategory.LABOR_STRIKE: frozenset(
        {EffectKind.BLOCK_NODES, EffectKind.TRANSIT_DELAY, EffectKind.DEMAND_SHOCK}
    ),
    ScenarioCategory.ROUTE_CLOSURE: frozenset(
        {EffectKind.BLOCK_NODES, EffectKind.TRANSIT_DELAY}
    ),
    ScenarioCategory.EQUIPMENT_MALFUNCTION: frozenset(
        {EffectKind.TEMP_OFFSET, EffectKind.HUMIDITY_OFFSET, EffectKind.RISK_FLAG}
    ),
    ScenarioCategory.REGULATORY_ACTION: frozenset(
        {EffectKind.BLOCK_NODES, EffectKind.DEMAND_SHOCK, EffectKind.RISK_FLAG}
    ),
    ScenarioCategory.HEATWAVE: frozenset(
        {EffectKind.TEMP_OFFSET, EffectKind.HUMIDITY_OFFSET, EffectKind.DEMAND_SHOCK}
    ),
    ScenarioCategory.COLD_SNAP: frozenset(
        {EffectKind.TEMP_OFFSET, EffectKind.DEMAND_SHOCK}
    ),
    ScenarioCategory.POWER_OUTAGE: frozenset(
        {EffectKind.TEMP_OFFSET, EffectKind.RISK_FLAG}
    ),
}

DURATION_RANGE: tuple[int, int] = (1, config.EPISODE_LEN_MIN)


class ScenarioValidationError(ValueError):
    """Raised when an LLM payload violates the scenario contract."""


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    magnitude: float
    target_role: TargetRole = TargetRole.ANY


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    category: ScenarioCategory
    description: str
    severity: float
    duration_steps: int
    effects: tuple[Effect, ...]

    def signature(self) -> tuple[Any, ...]:
        """Dedup key: category plus the ordered effect contents."""
        return (
            self.category,
            tuple(sorted((e.kind, e.magnitude, e.target_role) for e in self.effects)),
        )


LLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 20},
        "severity": {"type": "number"},
        "duration_steps": {"type": "integer"},
        "effects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": [k.value for k in EffectKind]},
                    "magnitude": {"type": "number"},
                    "target_role": {
                        "type": "string",
                        "enum": [r.value for r in TargetRole],
                    },
                },
                "required": ["kind", "magnitude", "target_role"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["description", "severity", "duration_steps", "effects"],
    "additionalProperties": False,
}


def parse_scenario(
    payload: dict[str, Any], scenario_id: str, category: ScenarioCategory
) -> Scenario:
    """Validate an LLM payload and build a Scenario, or raise."""
    description = str(payload.get("description", "")).strip()
    if len(description) < 20:
        raise ScenarioValidationError("description too short")

    severity = _check_number("severity", payload.get("severity"), 0.0, 1.0)
    duration = int(
        _check_number("duration_steps", payload.get("duration_steps"), *DURATION_RANGE)
    )

    raw_effects = payload.get("effects")
    if not isinstance(raw_effects, list) or not raw_effects:
        raise ScenarioValidationError("effects must be a non-empty list")

    allowed = CATEGORY_EFFECTS[category]
    effects: list[Effect] = []
    seen_kinds: set[EffectKind] = set()
    for raw in raw_effects:
        effect = _parse_effect(raw, allowed)
        if effect.kind in seen_kinds:
            raise ScenarioValidationError(f"duplicate effect kind {effect.kind}")
        seen_kinds.add(effect.kind)
        effects.append(effect)

    return Scenario(
        id=scenario_id,
        category=category,
        description=description,
        severity=severity,
        duration_steps=duration,
        effects=tuple(effects),
    )


def _parse_effect(raw: Any, allowed: frozenset[EffectKind]) -> Effect:
    if not isinstance(raw, dict):
        raise ScenarioValidationError("effect must be an object")
    try:
        kind = EffectKind(raw.get("kind"))
        role = TargetRole(raw.get("target_role", TargetRole.ANY))
    except ValueError as exc:
        raise ScenarioValidationError(str(exc)) from exc
    if kind not in allowed:
        raise ScenarioValidationError(f"effect {kind} not allowed for this category")
    if role is not TargetRole.ANY and kind not in TARGETED_KINDS:
        raise ScenarioValidationError(f"effect {kind} does not take a target role")

    magnitude = _check_number(f"{kind} magnitude", raw.get("magnitude"), *MAGNITUDE_RANGES[kind])
    if kind in INTEGER_KINDS:
        if magnitude != int(magnitude):
            raise ScenarioValidationError(f"{kind} magnitude must be an integer")
        magnitude = int(magnitude)
    return Effect(kind=kind, magnitude=magnitude, target_role=role)


def _check_number(name: str, value: Any, low: float, high: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScenarioValidationError(f"{name} must be a number")
    if not low <= value <= high:
        raise ScenarioValidationError(f"{name}={value} outside [{low}, {high}]")
    return float(value)


def save_bank(scenarios: list[Scenario], path: str, manifest: dict[str, Any]) -> None:
    payload = {
        "manifest": manifest,
        "scenarios": [asdict(s) for s in scenarios],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def load_bank(path: str) -> list[Scenario]:
    with open(path) as fh:
        payload = json.load(fh)
    scenarios = []
    for raw in payload["scenarios"]:
        scenarios.append(
            Scenario(
                id=raw["id"],
                category=ScenarioCategory(raw["category"]),
                description=raw["description"],
                severity=float(raw["severity"]),
                duration_steps=int(raw["duration_steps"]),
                effects=tuple(
                    Effect(
                        kind=EffectKind(e["kind"]),
                        magnitude=e["magnitude"],
                        target_role=TargetRole(e["target_role"]),
                    )
                    for e in raw["effects"]
                ),
            )
        )
    return scenarios
