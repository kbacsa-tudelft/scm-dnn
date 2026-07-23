"""End-to-end: run both methods through the actual benchmark runner (not just
in isolation) on a small SWaT sample, and assert the results are structurally
sound. Full-scale, multi-dataset comparisons are run via `run_benchmark.py`;
this just guards the wiring between data loaders / methods / runner / metrics.
"""
import math

from data.swat import load_swat
from methods.cdt.model import CDT
from methods.pbnn.model import PbNN
from methods.cnn1d.model import CNN1D
from benchmark.runner import run_benchmark

_NROWS = 8000


def test_benchmark_runs_all_methods_on_swat():
    dataset = load_swat(nrows=_NROWS)
    methods = {"cdt": lambda: CDT(), "pbnn": lambda: PbNN(), "cnn1d": lambda: CNN1D()}

    results = run_benchmark(methods, {"swat": dataset})

    assert len(results) == 3
    for result in results:
        assert result.error is None, f"{result.method} failed: {result.error}"
        assert 0.0 <= result.metrics["f1"] <= 1.0
        assert 0.0 <= result.metrics["f1_pa"] <= 1.0
        assert result.metrics["recall_pa"] >= result.metrics["recall"]
        assert math.isfinite(result.metrics["conflict_index_factor"])
        assert result.fit_seconds > 0
        assert result.config, f"{result.method} produced no config snapshot"

    cdt_config = next(r.config for r in results if r.method == "cdt")
    assert cdt_config["discovery_subsample_rows"] == 5000  # CDT's default, unless overridden

    cnn1d_config = next(r.config for r in results if r.method == "cnn1d")
    assert cnn1d_config["ensemble"] == "combined"  # CNN1D's default


def test_cdt_exposes_causal_graph_and_root_cause():
    dataset = load_swat(nrows=_NROWS)
    method = CDT()
    method.fit(dataset.train)

    graph = method.causal_graph()
    assert graph.number_of_nodes() == len(dataset.columns)

    attack_rows = dataset.test_labels.nonzero()[0]
    root_cause = method.root_cause(dataset.test, int(attack_rows[0]))
    assert root_cause is not None
