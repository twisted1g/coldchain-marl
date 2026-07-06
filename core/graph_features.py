from __future__ import annotations

import networkx as nx
import numpy as np

from core.config import DisruptionType, FruitKey
from core.state import GlobalState

# Node feature layout for the spoilage GNN (paper Alg 3): X_v = [T, H, delay, fruit_type].
SPOILAGE_NODE_FEATURES = 4

_FRUIT_INDEX: dict[FruitKey, int] = {f: i for i, f in enumerate(FruitKey)}

# Normalizers: BLOCKED_NODE contributes 1.0, INCREASED_TRANSIT up to ~3 ticks of delta.
_DELAY_SCALE = 3.0
_DEFAULT_NODE_HUMIDITY = 0.85


def node_order(graph: nx.DiGraph) -> list[str]:
    """Deterministic node ordering. ``build_supply_chain`` inserts farms, hubs, dcs,
    retailers in a fixed sequence and networkx preserves insertion order, so this is
    stable across episodes — the spoilage agent relies on it to reuse a static edge index."""
    return list(graph.nodes)


def static_edge_index(graph: nx.DiGraph) -> np.ndarray:
    """COO edge index [2, E] over ``node_order`` indices (includes wait self-loops).

    Topology is identical across episodes (only edge weights differ), so the agent
    computes this once and reuses it for every graph embedding."""
    index = {n: i for i, n in enumerate(node_order(graph))}
    edges = [(index[u], index[v]) for u, v in graph.edges]
    return np.array(edges, dtype=np.int64).T.reshape(2, -1)


def _node_delay(state: GlobalState, node: str) -> float:
    total = 0.0
    for d in state.active_disruptions:
        if d.type is DisruptionType.BLOCKED_NODE and d.target == node:
            total += 1.0
        elif d.type is DisruptionType.INCREASED_TRANSIT and d.target.split("->")[-1] == node:
            total += float(d.transit_delta)
    return min(1.0, total / _DELAY_SCALE)


def spoilage_node_features(state: GlobalState) -> np.ndarray:
    """Per-node feature matrix X [N, 4] = [T, H, delay, fruit_type] for the GNN encoder.

    The shipment's current node carries its real sensor T/H; other nodes fall back to
    ambient temperature and a default humidity (the sim tracks one shipment, so
    off-shipment node signals are scenario-derived — a justified reconstruction of the
    paper's multi-node feature graph). ``delay`` is per-node transit disruption; fruit
    type is a constant index across nodes."""
    s = state.shipment
    fruit = float(_FRUIT_INDEX[s.fruit_type])
    rows: list[list[float]] = []
    for node in node_order(state.graph):
        if node == s.current_node:
            temp, hum = s.sensor_temperature_c, s.sensor_humidity
        else:
            temp, hum = state.ambient_temp_c, _DEFAULT_NODE_HUMIDITY
        rows.append([temp, hum, _node_delay(state, node), fruit])
    return np.array(rows, dtype=np.float32)
