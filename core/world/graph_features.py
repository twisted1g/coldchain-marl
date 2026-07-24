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


def _node_features(
    state: GlobalState,
    subject_node: str,
    sensor_temp: float,
    sensor_humidity: float,
    fruit_type,
) -> np.ndarray:
    """GNN node grid for one spoilage subject: the subject's sensor reading sits at
    its current node, every other node shows its own storage micro-climate (Design
    F). The subject is the truck-borne crate (per-crate decentralised execution)."""
    fruit = float(_FRUIT_INDEX[fruit_type])
    rows: list[list[float]] = []
    for node in node_order(state.graph):
        if node == subject_node:
            temp, hum = sensor_temp, sensor_humidity
        else:
            temp = state.node_temp_c.get(node, state.ambient_temp_c)
            hum = state.node_humidity.get(node, state.ambient_humidity)
        rows.append([temp, hum, node_delay(state, node), fruit])
    return np.array(rows, dtype=np.float32)


def crate_spoilage_node_features(state: GlobalState, crate) -> np.ndarray:
    """Node grid for a single truck-borne crate (CTDE decentralised execution — the
    trained spoilage GNN policy predicts each crate at inference)."""
    return _node_features(
        state,
        crate.current_node,
        crate.sensor_temperature_c,
        crate.sensor_humidity,
        crate.fruit_type,
    )
