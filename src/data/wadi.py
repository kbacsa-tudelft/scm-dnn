"""Loader for the WADI dataset.

`WADI_14days_new.csv` is 14 days of normal operation (train). The attack file
has a spurious leading row of bare column indices ("0,1,2,...") before the
real header, and a label column named
`"Attack LABLE (1:No Attack, -1:Attack)"` whose value is 1 (no attack) / -1
(attack); pandas handles the embedded newline in that quoted header fine.

Both files are read with `encoding="latin-1"` rather than pandas' default
UTF-8 assumption: they carry stray non-UTF-8 bytes somewhere (a known quirk
of this dataset's Windows/Excel-originated export) that raise
`UnicodeDecodeError` under strict UTF-8 decoding on some machines/pandas/
locale combinations, even though this repo's own dev environment happened
not to trip over it. Latin-1 maps every byte 0x00-0xFF to a character 1:1 --
it never raises a decode error, and is identical to ASCII/UTF-8 for the tag
names and numeric data this loader actually uses, so this only changes how
the rare offending byte(s) get interpreted, not the columns/values relied on.
"""
from __future__ import annotations

import pandas as pd

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns

_META_COLS = ("Row", "Date", "Time")


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_wadi(root: str = "datasets/raw/wadi", nrows: int | None = None) -> ICSDataset:
    train_raw = _strip_cols(pd.read_csv(f"{root}/WADI_14days_new.csv", nrows=nrows, encoding="latin-1"))
    test_raw = _strip_cols(
        pd.read_csv(f"{root}/WADI_attackdataLABLE.csv", skiprows=1, nrows=nrows, encoding="latin-1")
    )

    label_col = test_raw.columns[-1]
    test_labels = (test_raw[label_col] == -1).astype(int).to_numpy()

    columns = [
        c
        for c in train_raw.columns
        if c not in _META_COLS and c in set(test_raw.columns)
    ]

    train = clean_numeric_frame(train_raw, columns)
    test = clean_numeric_frame(test_raw, columns)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name="wadi",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=None,
    )
