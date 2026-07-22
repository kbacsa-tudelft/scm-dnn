"""Turns the Z24 PDT scenarios (see z24.py) into ICSDataset-shaped binary
anomaly-detection problems, so CDT/PbNN can run on them exactly like
SWaT/WADI/HAI: pair a healthy (reference) scenario with a damaged scenario,
concatenate them in time, and label the damaged portion as anomalous.

This is a fundamentally synthetic construction -- unlike SWaT/WADI/HAI's
attack windows (a real anomaly onset inside one continuous recording), Z24's
"anomaly" here is two independently-recorded ~655s experiments stitched
together. There is no real transition dynamics at the seam; the label marks
which experiment each row came from, not an observed developing fault.

Per the user's choice: train and the test's healthy baseline come from two
*different* reference scenarios (default 01 for train, 02 for the test's
healthy portion) rather than splitting one scenario's recording in half --
this keeps full-length training data and uses genuinely separate recordings
for fit vs. eval, mirroring (in spirit, not mechanism) SWaT's leak-free split.

Column handling: different scenarios carry slightly different channel sets
(documented in z24.py -- e.g. DP1V lost in scenario 01's setups 01 & 05).
`build_z24_binary_dataset` restricts every pairing to `shared_columns()`,
the intersection across *all* 17 scenarios (300 of a possible 309 columns),
so every pairing has an identical, comparable column set rather than a
pairing-dependent one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns
from .z24 import SCENARIOS, Z24Dataset, load_z24_pdt

REFERENCE_SCENARIOS = ("01", "02", "08")
DAMAGED_SCENARIOS = tuple(s for s in sorted(SCENARIOS) if s not in REFERENCE_SCENARIOS)


def shared_columns(ds: Z24Dataset) -> list[str]:
    """Intersection of columns present in every one of the 17 scenarios."""
    col_sets = [set(ds.load_scenario(s).columns) for s in sorted(SCENARIOS)]
    shared = set.intersection(*col_sets)
    return sorted(shared)


def build_z24_binary_dataset(
    damaged: str,
    healthy_train: str = "01",
    healthy_test: str = "02",
    root: str = "datasets/raw/z24",
) -> ICSDataset:
    """One damaged scenario vs. the fixed healthy baseline, ICSDataset-shaped:
    train = healthy_train scenario (label-free, attack-free by construction);
    test  = healthy_test scenario (label 0) followed by `damaged` (label 1).
    """
    if damaged not in DAMAGED_SCENARIOS:
        raise ValueError(f"{damaged!r} is a reference scenario, not a damaged one: {REFERENCE_SCENARIOS}")

    ds = load_z24_pdt(root)
    cols = shared_columns(ds)

    train_raw = ds.load_scenario(healthy_train)[cols]
    healthy_test_raw = ds.load_scenario(healthy_test)[cols]
    damaged_raw = ds.load_scenario(damaged)[cols]

    test_raw = pd.concat([healthy_test_raw, damaged_raw], ignore_index=True)
    test_labels = np.concatenate([
        np.zeros(len(healthy_test_raw), dtype=int),
        np.ones(len(damaged_raw), dtype=int),
    ])

    train = clean_numeric_frame(train_raw, cols)
    test = clean_numeric_frame(test_raw, cols)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name=f"z24_{damaged}",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=None,
    )
