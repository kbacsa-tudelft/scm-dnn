"""Loader for the HAI dataset.

HAI ships several versions (20.07, 21.03, 22.04, 23.05) with different label
conventions. We default to hai-22.04, which has an inline 0/1 `Attack` column
in every file (train*.csv are all-normal, test*.csv are mixed) -- unlike
23.05, which needs a separate label file joined on timestamp. All train*.csv
/ test*.csv files within a version share the same 88-column schema and are
simply concatenated.

A ground-truth causal graph is only available for the HAI *boiler*
subsystem (datasets/raw/hai/graph/boiler/phy_boiler.json), not for the full
85-tag process. Its node attributes (`type`/`device`/`dynamics`) are a
serialization artifact -- the same tuple is duplicated on every node -- so
only graph topology (nodes, directed edges, edge-level `dynamics` code) is
trustworthy and used here.

CSVs are read with `encoding="latin-1"` rather than pandas' default UTF-8
assumption: these ICS dataset exports (SWaT/WADI/HAI/BATADAL alike) carry
stray non-UTF-8 bytes on some machines/pandas/locale combinations, raising
`UnicodeDecodeError` under strict UTF-8 decoding even where a given dev copy
happens not to trip over it. Latin-1 maps every byte 0x00-0xFF to a
character 1:1 -- it never raises a decode error, and is identical to
ASCII/UTF-8 for the tag names and numeric data actually used here.
"""
from __future__ import annotations

import glob
import json

import networkx as nx
import pandas as pd

from .base import ICSDataset, clean_numeric_frame, drop_constant_columns

_LABEL_COL = "Attack"
_TIME_COL = "timestamp"


def _load_concat(pattern: str, nrows: int | None) -> pd.DataFrame:
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"No files matched {pattern!r} -- have you extracted the HAI dataset yet? "
            f"See README.md 'Setup' for the unzip commands."
        )
    frames = [pd.read_csv(p, nrows=nrows, encoding="latin-1") for p in sorted(paths)]
    return pd.concat(frames, ignore_index=True)


def load_hai(
    root: str = "datasets/raw/hai",
    version: str = "hai-22.04",
    nrows: int | None = None,
) -> ICSDataset:
    version_dir = f"{root}/{version}"
    train_raw = _load_concat(f"{version_dir}/train*.csv", nrows)
    test_raw = _load_concat(f"{version_dir}/test*.csv", nrows)

    columns = [c for c in train_raw.columns if c not in (_TIME_COL, _LABEL_COL)]

    test_labels = test_raw[_LABEL_COL].astype(int).to_numpy()

    train = clean_numeric_frame(train_raw, columns)
    test = clean_numeric_frame(test_raw, columns)

    keep = drop_constant_columns(train, test)
    train, test = train[keep], test[keep]

    return ICSDataset(
        name=version,
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_labels=test_labels,
        columns=keep,
        ground_truth_graph=load_hai_boiler_graph(root),
    )


def load_hai_boiler_graph(root: str = "datasets/raw/hai") -> nx.DiGraph:
    """Ground-truth causal/process graph for the HAI boiler subsystem only.

    Node ids (e.g. "TK01", "PP01A") are physical-component tags, not the
    dataset's sensor column names, so this graph cannot be directly compared
    to a discovered graph over HAI's `P1_.../P2_...` columns -- it is
    provided for structural (topology-only) comparisons scoped to the boiler
    component set.
    """
    with open(f"{root}/graph/boiler/phy_boiler.json") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="links")
