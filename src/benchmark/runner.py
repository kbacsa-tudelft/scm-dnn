"""Runs a set of AnomalyDetectionMethod instances against a set of ICSDataset
loaders and collects metrics for each (method, dataset) pair."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from data.base import ICSDataset
from methods.base import AnomalyDetectionMethod
from metrics.detection import summarize
from metrics.graph import edge_precision_recall_f1, structural_hamming_distance


@dataclass
class RunResult:
    method: str
    dataset: str
    metrics: dict
    fit_seconds: float
    score_seconds: float
    graph_metrics: dict | None = None
    error: str | None = None


def run_one(method_name: str, method: AnomalyDetectionMethod, dataset: ICSDataset) -> RunResult:
    t0 = time.time()
    method.fit(dataset.train)
    fit_seconds = time.time() - t0

    t0 = time.time()
    scores = method.score(dataset.test)
    score_seconds = time.time() - t0

    threshold = getattr(method, "threshold_", np.quantile(scores, 0.99))
    metrics = summarize(dataset.test_labels, scores, threshold)

    graph_metrics = None
    pred_graph = method.causal_graph()
    if pred_graph is not None and dataset.ground_truth_graph is not None:
        precision, recall, f1 = edge_precision_recall_f1(dataset.ground_truth_graph, pred_graph)
        graph_metrics = {
            "shd": structural_hamming_distance(dataset.ground_truth_graph, pred_graph),
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
        }

    return RunResult(
        method=method_name,
        dataset=dataset.name,
        metrics=metrics,
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
        graph_metrics=graph_metrics,
    )


def run_benchmark(
    methods: dict[str, Callable[[], AnomalyDetectionMethod]],
    datasets: dict[str, ICSDataset],
) -> list[RunResult]:
    """`methods` maps a display name to a zero-arg factory (fresh instance per dataset)."""
    results = []
    for dataset_name, dataset in datasets.items():
        for method_name, make_method in methods.items():
            try:
                result = run_one(method_name, make_method(), dataset)
            except Exception as exc:  # noqa: BLE001 -- surfaced in the report, run continues
                result = RunResult(
                    method=method_name,
                    dataset=dataset_name,
                    metrics={},
                    fit_seconds=0.0,
                    score_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
    return results
