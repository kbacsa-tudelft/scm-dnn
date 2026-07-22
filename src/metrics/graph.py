"""Graph-comparison metrics against a ground-truth causal graph.

`structural_hamming_distance` follows CDT paper Eq. 29:
    SHD = |E_true Δ E_discovered| + |R_true Δ R_discovered|
i.e. skeleton (undirected) symmetric difference, plus orientation
disagreements on edges present in both skeletons. Comparison is restricted
to nodes present in both graphs (see caveat in `data/hai.py` re: the boiler
graph using physical-component ids rather than dataset tag names).
"""
from __future__ import annotations

import networkx as nx


def _skeleton(g: nx.DiGraph) -> set[frozenset]:
    return {frozenset((u, v)) for u, v in g.edges()}


def _restrict_to_shared_nodes(true_graph: nx.DiGraph, pred_graph: nx.DiGraph) -> tuple[nx.DiGraph, nx.DiGraph]:
    shared = set(true_graph.nodes()) & set(pred_graph.nodes())
    return true_graph.subgraph(shared), pred_graph.subgraph(shared)


def structural_hamming_distance(true_graph: nx.DiGraph, pred_graph: nx.DiGraph) -> int:
    t, p = _restrict_to_shared_nodes(true_graph, pred_graph)
    skel_t, skel_p = _skeleton(t), _skeleton(p)
    skeleton_diff = len(skel_t ^ skel_p)

    orientation_diff = 0
    for edge in skel_t & skel_p:
        u, v = tuple(edge)
        if ((u, v) in t.edges()) != ((u, v) in p.edges()):
            orientation_diff += 1

    return skeleton_diff + orientation_diff


def edge_precision_recall_f1(true_graph: nx.DiGraph, pred_graph: nx.DiGraph) -> tuple[float, float, float]:
    """Exact directed-edge match precision/recall/F1, restricted to shared nodes."""
    t, p = _restrict_to_shared_nodes(true_graph, pred_graph)
    true_edges, pred_edges = set(t.edges()), set(p.edges())
    tp = len(true_edges & pred_edges)
    precision = tp / len(pred_edges) if pred_edges else 0.0
    recall = tp / len(true_edges) if true_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
