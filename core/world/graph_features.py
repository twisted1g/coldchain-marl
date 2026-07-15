from __future__ import annotations

import networkx as nx
import numpy as np

from core.config import DisruptionType, FruitKey
from core.state import GlobalState

SPOILAGE_NODE_FEATURES = 4

_FRUIT_INDEX: dict[FruitKey, int] = {f: i for i, f in enumerate(FruitKey)}

_DELAY_SCALE = 3.0


def node_order(graph: nx.DiGraph) -> list[str]:
    return list(graph.nodes)


def static_edge_index(graph: nx.DiGraph) -> np.ndarray:
    index = {n: i for i, n in enumerate(node_order(graph))}
    edges = [(index[u], index[v]) for u, v in graph.edges]
    return np.array(edges, dtype=np.int64).T.reshape(2, -1)


def node_delay(state: GlobalState, node: str) -> float:
    total = 0.0
    for d in state.active_disruptions:
        if d.type is DisruptionType.BLOCKED_NODE and d.target == node:
            total += 1.0
        elif (
            d.type is DisruptionType.INCREASED_TRANSIT
            and d.target.split("->")[-1] == node
        ):
            total += float(d.transit_delta)
    return min(1.0, total / _DELAY_SCALE)


def spoilage_node_features(state: GlobalState) -> np.ndarray:
    s = state.shipment
    fruit = float(_FRUIT_INDEX[s.fruit_type])
    rows: list[list[float]] = []
    for node in node_order(state.graph):
        if node == s.current_node:
            temp, hum = s.sensor_temperature_c, s.sensor_humidity
        else:
            temp, hum = state.ambient_temp_c, state.ambient_humidity
        rows.append([temp, hum, node_delay(state, node), fruit])
    return np.array(rows, dtype=np.float32)
