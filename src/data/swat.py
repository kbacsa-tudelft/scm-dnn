"""Loader for the SWaT dataset.

Notes on this particular repackaging (see datasets/raw/swat/):
  - `normal.csv` (~1.39M rows) is entirely labeled "Normal".
  - `attack.csv` (~54.6k rows) is entirely labeled "Attack" -- this is a
    paper-specific extraction of just the attack windows, not the full
    canonical ~4-day mixed test period.
  - `merged.csv` is exactly `normal.csv` followed by `attack.csv`.

Using the shipped `merged.csv` directly as a test set would mean testing on
rows the model was also trained on (all of `normal.csv`). To get a fair,
leak-free split we instead hold out a tail slice of `normal.csv` for testing
and combine it with `attack.csv`:
    train = normal.csv[:train_frac]
    test  = normal.csv[train_frac:] + attack.csv

Read with `encoding="latin-1"` rather than pandas' default UTF-8 assumption:
these ICS dataset exports (SWaT/WADI/HAI/BATADAL alike) carry stray non-UTF-8
bytes on some machines/pandas/locale combinations, raising
`UnicodeDecodeError` under strict UTF-8 decoding even where a given dev copy
happens not to trip over it. Latin-1 maps every byte 0x00-0xFF to a
character 1:1 -- it never raises a decode error, and is identical to
ASCII/UTF-8 for the tag names and numeric data actually used here.
"""
from __future__ import annotations

import pandas as pd

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns

_LABEL_COL = "Normal/Attack"
_TIME_COL = "Timestamp"


def _load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_swat(
    root: str = "datasets/raw/swat",
    train_frac: float = 0.8,
    nrows: int | None = None,
) -> ICSDataset:
    normal = _load_raw(f"{root}/normal.csv", nrows=nrows)
    attack = _load_raw(f"{root}/attack.csv", nrows=nrows)

    columns = [c for c in normal.columns if c not in (_TIME_COL, _LABEL_COL)]

    split = int(len(normal) * train_frac)
    train_raw = normal.iloc[:split]
    held_out_normal = normal.iloc[split:]

    test_raw = pd.concat([held_out_normal, attack], ignore_index=True)
    test_labels = (test_raw[_LABEL_COL].str.strip() == "Attack").astype(int).to_numpy()

    train = clean_numeric_frame(train_raw, columns)
    test = clean_numeric_frame(test_raw, columns)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name="swat",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=None,
    )
