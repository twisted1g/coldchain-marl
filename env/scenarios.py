"""Apply LLM-bank disruption scenarios to episodes.

A ScenarioRunner resolves one abstract Scenario (roles, magnitudes) into
concrete world perturbations for one episode: the onset tick and target
nodes/edges are drawn from the episode rng, effects are applied for
``duration_steps`` ticks and then reverted. Runs on top of the per-tick
random NoiseModel disruptions, which stay untouched.
"""

from __future__ import annotations

from core.config import DisruptionType
from core.state import GlobalState
from core.world.graph import nodes_by_kind
from core.world.noise import Disruption
from llm.scenarios import Effect, EffectKind, Scenario, TargetRole

_ROLE_TO_KIND = {
    TargetRole.FARM: "farm",
    TargetRole.HUB: "hub",
    TargetRole.DC: "dc",
    TargetRole.RETAILER: "retail",
}


def _nodes_by_role(state: GlobalState, role: TargetRole) -> list[str]:
    """Schema roles map to graph node kinds; the graph names the sink kind "retail"."""
    if role is TargetRole.ANY:
        return list(state.graph.nodes)
    return nodes_by_kind(state.graph, _ROLE_TO_KIND[role])


class ScenarioRunner:
    """Drives one scenario over one episode; call ``before_step`` every tick."""

    def __init__(self, scenario: Scenario, state: GlobalState) -> None:
        self.scenario = scenario
        duration = min(scenario.duration_steps, state.max_steps)
        self.onset = int(state.rng.integers(1, state.max_steps - duration + 2))
        self.end = self.onset + duration
        self._injected: list[Disruption] = []
        self._saved_temp: float | None = None
        self._saved_humidity: float | None = None

    def before_step(self, state: GlobalState) -> None:
        upcoming_tick = state.tick + 1
        if upcoming_tick == self.onset:
            self._activate(state)
        elif upcoming_tick == self.end:
            self._deactivate(state)

    def _activate(self, state: GlobalState) -> None:
        for effect in self.scenario.effects:
            self._apply(state, effect)

    def _apply(self, state: GlobalState, effect: Effect) -> None:
        kind = effect.kind
        if kind is EffectKind.BLOCK_NODES:
            candidates = _nodes_by_role(state, effect.target_role)
            count = min(int(effect.magnitude), len(candidates))
            picked = state.rng.choice(candidates, size=count, replace=False)
            for node in picked:
                self._inject(state, Disruption(DisruptionType.BLOCKED_NODE, str(node)))
        elif kind is EffectKind.TRANSIT_DELAY:
            edges = [
                (u, v)
                for u, v, data in state.graph.edges(data=True)
                if not data["wait"]
            ]
            u, v = edges[int(state.rng.integers(0, len(edges)))]
            self._inject(
                state,
                Disruption(
                    DisruptionType.INCREASED_TRANSIT,
                    f"{u}->{v}",
                    transit_delta=int(effect.magnitude),
                ),
            )
        elif kind is EffectKind.RISK_FLAG:
            candidates = _nodes_by_role(state, effect.target_role)
            node = str(state.rng.choice(candidates))
            self._inject(state, Disruption(DisruptionType.RISK_FLAG, node))
        elif kind is EffectKind.TEMP_OFFSET:
            self._saved_temp = state.ambient_temp_c
            state.ambient_temp_c += effect.magnitude
        elif kind is EffectKind.HUMIDITY_OFFSET:
            self._saved_humidity = state.ambient_humidity
            state.ambient_humidity = float(
                min(1.0, max(0.0, state.ambient_humidity + effect.magnitude))
            )
        elif kind is EffectKind.DEMAND_SHOCK:
            state.demand_shock_mult = effect.magnitude

    def _inject(self, state: GlobalState, disruption: Disruption) -> None:
        state.active_disruptions.append(disruption)
        self._injected.append(disruption)

    def _deactivate(self, state: GlobalState) -> None:
        for disruption in self._injected:
            state.active_disruptions.remove(disruption)
        self._injected.clear()
        if self._saved_temp is not None:
            state.ambient_temp_c = self._saved_temp
            self._saved_temp = None
        if self._saved_humidity is not None:
            state.ambient_humidity = self._saved_humidity
            self._saved_humidity = None
        state.demand_shock_mult = 1.0
