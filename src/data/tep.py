"""Loader for the TEP (Tennessee Eastman Process) dataset, Rieth et al. 2017
release (Harvard Dataverse), RData format converted to parquet by
`scripts/build_tep_parquet.py` (run that first).

Structurally different from SWaT/WADI/HAI/BATADAL: rather than one
continuous recording, TEP is organized as many independent short simulation
runs -- 500 runs of 500 samples (training, 25h @ 3-min intervals) or 960
samples (testing, 48h @ 3-min intervals) each, across 21 conditions
(`faultNumber` 0 = normal, 1-20 = distinct fault types). Per the user's
choice, all 20 fault types are merged into one binary normal/anomaly
dataset here (like SWaT/WADI/HAI/BATADAL), rather than kept as 20 separate
per-fault benchmarks (an option for later, analogous to Z24's per-scenario
pairing, if finer-grained per-fault detectability is ever wanted).

Concatenating many independent runs end-to-end (in `simulationRun` order)
means there are artificial "seams" between runs, similar in spirit to Z24's
setup-array structure -- not a real continuous process, just a convenient
flat table. This is inherent to the data's shape, not something a loader
choice can avoid.

Labeling within `faulty_testing`: per the dataset's own documentation,
faults are introduced 8 hours (= 160 samples at 3-minute intervals) into
each faulty testing run -- rows before that point are still genuinely
fault-free even though the run's `faultNumber` is nonzero, so they're
labeled 0, not 1.

`faulty_training` is never used (not even extracted, see
`scripts/build_tep_parquet.py`): every method in this harness trains on
fault-free data only.

Only the process variables (`xmeas_*`, `xmv_*`, 52 total) are used as
columns; `faultNumber`/`simulationRun`/`sample` are metadata, not features.

Scale warning: the full faulty-testing file is ~9.6M rows (20 faults x 500
runs x 960 samples); combined with fault-free testing's ~480k rows, a
full-scale `test` set here is ~10x bigger than SWaT's. Use `nrows` while
iterating (caps rows read per source file, same convention as every other
loader in this repo) -- full-scale runs on this dataset will be slow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns

FAULT_INTRODUCED_SAMPLE = 160  # 8 hours at 3-minute sampling intervals
_METADATA_COLS = ("faultNumber", "simulationRun", "sample")


def _read_parquet(path: str, nrows: int | None) -> pd.DataFrame:
    """Full read when nrows is None; otherwise reads only the row groups
    needed to cover `nrows` rows via pyarrow directly, rather than
    `pd.read_parquet(...).iloc[:nrows]` -- the faulty-testing file alone is
    ~9.6M rows/~800MB on disk, several GB once decompressed in memory, and
    reading it whole just to immediately truncate it would defeat the point
    of `nrows`-capped dev/smoke runs."""
    if nrows is None:
        return pd.read_parquet(path)
    batches = pq.ParquetFile(path).iter_batches(batch_size=nrows)
    return next(batches).to_pandas()


def load_tep(root: str = "datasets/raw/tep/tep_files", nrows: int | None = None) -> ICSDataset:
    fault_free_train = _read_parquet(f"{root}/fault_free_training.parquet", nrows)
    fault_free_test = _read_parquet(f"{root}/fault_free_testing.parquet", nrows)
    faulty_test = _read_parquet(f"{root}/faulty_testing.parquet", nrows)

    columns = [c for c in fault_free_train.columns if c not in _METADATA_COLS]

    faulty_labels = (faulty_test["sample"] >= FAULT_INTRODUCED_SAMPLE).astype(int).to_numpy()
    test_raw = pd.concat([fault_free_test, faulty_test], ignore_index=True)
    test_labels = np.concatenate([np.zeros(len(fault_free_test), dtype=int), faulty_labels])

    train = clean_numeric_frame(fault_free_train, columns)
    test = clean_numeric_frame(test_raw, columns)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name="tep",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=None,
    )
