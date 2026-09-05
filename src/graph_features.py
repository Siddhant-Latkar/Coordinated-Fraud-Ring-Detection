"""
Offline ring-visualization helper for the dashboard's "Ring Explorer"
panel only -- not used by the trained model (see src/features.py for
the graph_risk_score the model actually trains and scores on).

A static, all-time graph over user/device/ip/merchant edges collapses
into one giant connected component, since normal users end up linked
through shared merchants. Dropping merchant edges and keeping only
user<->device and user<->ip fixes that for visual exploration, but
this still looks at the whole dataset at once rather than a trailing
window, so it's for investigating historical data, not live scoring.
"""

import pandas as pd
import networkx as nx


def build_ring_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Builds a user<->device / user<->ip graph (merchant edges excluded
    on purpose -- see module docstring) for visual ring exploration.
    """
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        user_node = f"user:{row.user_id}"
        device_node = f"device:{row.device_id}"
        ip_node = f"ip:{row.ip_id}"

        graph.add_node(user_node, node_type="user")
        graph.add_node(device_node, node_type="device")
        graph.add_node(ip_node, node_type="ip")

        graph.add_edge(user_node, device_node, relation="used_device")
        graph.add_edge(user_node, ip_node, relation="used_ip")

    return graph


def find_suspicious_rings(df: pd.DataFrame, min_size: int = 4, max_size: int = 200):
    """
    Returns connected components (excluding the one dominant "everyone
    is loosely connected" component) sized between min_size and
    max_size -- i.e. plausible ring clusters, for a human to inspect in
    the dashboard. This is exploratory tooling, not a scoring feature.
    """
    graph = build_ring_graph(df)
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)

    # The largest component in this kind of bipartite identity graph is
    # almost always "everyone, loosely", not a ring -- skip it.
    candidate_components = components[1:] if len(components) > 1 else []

    rings = [c for c in candidate_components if min_size <= len(c) <= max_size]
    return rings
