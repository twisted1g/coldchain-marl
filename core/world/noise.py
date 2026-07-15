from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from core import config
from core.config import DisruptionType


@dataclass(frozen=True, slots=True)
class Disruption:
    type: DisruptionType
    target: str
    transit_delta: int = 0


class NoiseModel:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def sample_disruption(self, graph: nx.DiGraph) -> Disruption | None:
        if self._rng.random() > config.DISRUPTION_PROB_PER_TICK:
            return None
        chosen_type = self._pick_type()
        if chosen_type is DisruptionType.BLOCKED_NODE:
            return self._sample_blocked_node(graph)
        if chosen_type is DisruptionType.INCREASED_TRANSIT:
            return self._sample_increased_transit(graph)
        return self._sample_risk_flag(graph)

    def _pick_type(self) -> DisruptionType:
        types = list(config.DISRUPTION_TYPE_WEIGHTS.keys())
        weights = np.array(list(config.DISRUPTION_TYPE_WEIGHTS.values()), dtype=float)
        weights /= weights.sum()
        idx = int(self._rng.choice(len(types), p=weights))
        return types[idx]

    def _sample_blocked_node(self, graph: nx.DiGraph) -> Disruption:
        candidates = [
            n for n, data in graph.nodes(data=True) if data["kind"] in ("hub", "dc")
        ]
        target = str(self._rng.choice(candidates)) if candidates else ""
        return Disruption(type=DisruptionType.BLOCKED_NODE, target=target)

    def _sample_increased_transit(self, graph: nx.DiGraph) -> Disruption:
        transport_edges = [
            (u, v) for u, v, data in graph.edges(data=True) if not data["wait"]
        ]
        idx = int(self._rng.integers(0, len(transport_edges)))
        u, v = transport_edges[idx]
        d_min, d_max = config.DISRUPTION_TRANSIT_DELTA_RANGE
        delta = int(self._rng.integers(d_min, d_max + 1))
        return Disruption(
            type=DisruptionType.INCREASED_TRANSIT,
            target=f"{u}->{v}",
            transit_delta=delta,
        )

    def _sample_risk_flag(self, graph: nx.DiGraph) -> Disruption:
        nodes = list(graph.nodes)
        target = str(self._rng.choice(nodes))
        return Disruption(type=DisruptionType.RISK_FLAG, target=target)
