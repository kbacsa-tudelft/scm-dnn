"""Loader for the BATADAL dataset (BATtle of the Attack Detection ALgorithms,
iTrust/University of Cyprus C-Town water distribution network).

Two files, closely analogous to SWaT's normal.csv/attack.csv split:
  - `BATADAL_dataset03.csv`: ~1 year, hourly, entirely normal operation
    (train).
  - `BATADAL_dataset04.csv`: ~6 months, hourly, contains 7 documented
    attacks. Its `ATT_FLAG` column is `1` for confirmed-attack rows and
    `-999` for everything else (there is no explicit `0`/normal label in
    this file) -- `-999` is treated as normal (0) here, the standard
    convention for this dataset: only the confirmed attack windows are
    marked, everything else is implicitly normal.

`BATADAL_test_dataset.zip` (the original competition's held-out test set)
is NOT used: it ships with no label column at all (unlabeled by design, for
the original competition's blind scoring), so there's no ground truth to
evaluate against here.

`CTOWN.INP` (the EPANET network topology model) is extracted but not parsed
-- unlike HAI's boiler graph, it isn't wired in as a ground-truth causal
graph, since node ids would need a translation layer to the SCADA tag names
below (same category of gap as HAI's, not attempted here).
"""
from __future__ import annotations

import pandas as pd

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns

_LABEL_COL = "ATT_FLAG"
_TIME_COL = "DATETIME"


def _load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_batadal(root: str = "datasets/raw/batadal/batadal", nrows: int | None = None) -> ICSDataset:
    train_raw = _load_raw(f"{root}/BATADAL_dataset03.csv", nrows=nrows)
    test_raw = _load_raw(f"{root}/BATADAL_dataset04.csv", nrows=nrows)

    columns = [c for c in train_raw.columns if c not in (_TIME_COL, _LABEL_COL)]
    test_labels = (test_raw[_LABEL_COL] == 1).astype(int).to_numpy()

    train = clean_numeric_frame(train_raw, columns)
    test = clean_numeric_frame(test_raw, columns)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name="batadal",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=None,
    )
