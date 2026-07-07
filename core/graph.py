from __future__ import annotations

import networkx as nx
import numpy as np

from core import config

NodeKind = str


def build_supply_chain(rng: np.random.Generator) -> nx.DiGraph:
    graph: nx.DiGraph = nx.DiGraph()

    farms = [f"farm_{i}" for i in range(config.N_FARMS)]
    hubs = [f"hub_{i}" for i in range(config.N_HUBS)]
    dcs = [f"dc_{i}" for i in range(config.N_DCS)]
    retailers = [f"retail_{i}" for i in range(config.N_RETAILERS)]

    for node in farms:
        graph.add_node(node, kind="farm")
    for node in hubs:
        graph.add_node(node, kind="hub")
    for node in dcs:
        graph.add_node(node, kind="dc")
    for node in retailers:
        graph.add_node(node, kind="retail")

    for upstream, downstream in [(farms, hubs), (hubs, dcs), (dcs, retailers)]:
        for u in upstream:
            for v in downstream:
                _add_transport_edge(graph, u, v, rng)

    for node in graph.nodes:
        _add_wait_edge(graph, node)

    return graph


def _add_transport_edge(
    graph: nx.DiGraph,
    u: str,
    v: str,
    rng: np.random.Generator,
) -> None:
    d_min, d_max = config.EDGE_DISTANCE_KM_RANGE
    t_min, t_max = config.EDGE_BASE_TRANSIT_TIME_RANGE
    distance = float(rng.uniform(d_min, d_max))
    transit = int(rng.integers(t_min, t_max + 1))
    emissions = distance * config.EDGE_BASE_EMISSIONS_PER_KM
    graph.add_edge(
        u,
        v,
        distance_km=distance,
        base_transit_time=transit,
        base_emissions=emissions,
        wait=False,
    )


def _add_wait_edge(graph: nx.DiGraph, node: str) -> None:
    graph.add_edge(
        node,
        node,
        distance_km=0.0,
        base_transit_time=config.WAIT_EDGE_TRANSIT_TIME,
        base_emissions=config.WAIT_EDGE_EMISSIONS,
        wait=True,
    )


def nodes_by_kind(graph: nx.DiGraph, kind: NodeKind) -> list[str]:
    return [n for n, data in graph.nodes(data=True) if data["kind"] == kind]


def source_nodes(graph: nx.DiGraph) -> list[str]:
    return nodes_by_kind(graph, "farm")


def sink_nodes(graph: nx.DiGraph) -> list[str]:
    return nodes_by_kind(graph, "retail")
