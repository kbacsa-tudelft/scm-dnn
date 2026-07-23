import numpy as np

from data.swat import load_swat
from data.wadi import load_wadi
from data.hai import load_hai, load_hai_boiler_graph
from data.batadal import load_batadal
from data.tep import load_tep

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


def test_load_batadal():
    ds = load_batadal()  # small enough to load in full, no nrows cap needed
    _check_dataset(ds)
    assert 0.0 < ds.test_labels.mean() < 1.0  # confirmed attacks exist, not all/none


def test_load_tep():
    ds = load_tep(nrows=_NROWS)
    _check_dataset(ds)
    assert 0.0 < ds.test_labels.mean() < 1.0


def test_tep_fault_introduced_sample_labeling():
    # nrows=20000 lands entirely within faulty_testing's first simulation
    # run (960 samples/run), which starts at sample 1 -- so test rows
    # [0:159] are pre-fault (label 0) and [159:] are post-fault (label 1)
    # for that run, per the dataset's documented 8h/160-sample fault
    # introduction point. Loads fault_free_testing at the same small cap
    # (all label 0) followed by faulty_testing, so this checks the boundary
    # lands exactly where the source data says it should.
    ds = load_tep(nrows=_NROWS)
    fault_free_len = _NROWS  # fault_free_testing capped at the same nrows
    faulty_start = fault_free_len
    assert ds.test_labels[faulty_start : faulty_start + 159].sum() == 0
    assert ds.test_labels[faulty_start + 160] == 1
