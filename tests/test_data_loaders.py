import numpy as np

from data.swat import load_swat
from data.wadi import load_wadi
from data.hai import load_hai, load_hai_boiler_graph

_NROWS = 20000


def _check_dataset(ds):
    assert len(ds.columns) > 0
    assert ds.train.shape[1] == len(ds.columns)
    assert ds.test.shape[1] == len(ds.columns)
    assert len(ds.test_labels) == len(ds.test)
    assert set(np.unique(ds.test_labels)).issubset({0, 1})
    assert not ds.train.isna().any().any()
    assert not ds.test.isna().any().any()


def test_load_swat():
    _check_dataset(load_swat(nrows=_NROWS))


def test_load_wadi():
    _check_dataset(load_wadi(nrows=_NROWS))


def test_load_hai():
    ds = load_hai(nrows=_NROWS)
    _check_dataset(ds)
    assert ds.ground_truth_graph is not None


def test_hai_boiler_graph():
    g = load_hai_boiler_graph()
    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0
    assert g.is_directed()
