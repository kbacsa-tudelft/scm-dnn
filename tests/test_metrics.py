import networkx as nx
import numpy as np

from metrics.detection import (
    conflict_index_factor,
    detection_rate,
    false_alarm_rate,
    point_adjust,
    precision_recall_f1,
    summarize,
)
from metrics.graph import edge_precision_recall_f1, structural_hamming_distance


def test_precision_recall_f1_perfect():
    y = np.array([0, 0, 1, 1])
    assert precision_recall_f1(y, y) == (1.0, 1.0, 1.0)


def test_precision_recall_f1_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_detection_rate_and_far():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert detection_rate(y_true, y_pred) == 0.5
    assert false_alarm_rate(y_true, y_pred) == 0.5


def test_conflict_index_factor_weights():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert conflict_index_factor(y_true, y_pred) == 0.4  # Dr=1, Fr=0 -> 0.4*1 - 0.6*0


def test_point_adjust_fills_segment():
    y_true = np.array([0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])
    adjusted = point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(adjusted, [0, 1, 1, 1, 0])


def test_point_adjust_no_hit_stays_zero():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    adjusted = point_adjust(y_true, y_pred)
    np.testing.assert_array_equal(adjusted, [0, 0, 0, 0])


def test_summarize_has_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = summarize(y_true, scores, threshold=0.5)
    assert result["f1"] == 1.0
    assert "conflict_index_factor" in result


def test_shd_identical_graphs_is_zero():
    g = nx.DiGraph([("a", "b"), ("b", "c")])
    assert structural_hamming_distance(g, g) == 0


def test_shd_penalizes_missing_and_flipped_edges():
    true_graph = nx.DiGraph([("a", "b"), ("b", "c")])
    pred_graph = nx.DiGraph([("b", "a")])
    pred_graph.add_node("c")  # node known but with no discovered edges to it
    # orientation disagreement on shared skeleton edge a-b (+1), missing b-c skeleton edge (+1)
    assert structural_hamming_distance(true_graph, pred_graph) == 2


def test_edge_precision_recall_exact_match():
    true_graph = nx.DiGraph([("a", "b"), ("b", "c")])
    pred_graph = nx.DiGraph([("a", "b")])
    pred_graph.add_node("c")
    precision, recall, f1 = edge_precision_recall_f1(true_graph, pred_graph)
    assert precision == 1.0
    assert recall == 0.5
