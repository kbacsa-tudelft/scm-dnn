"""Common dataset representation shared by SWaT, WADI and HAI loaders."""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class ICSDataset:
    """A loaded industrial-control-system dataset, normalized to a common shape.

    `train` is assumed attack-free (used to fit methods). `test` contains both
    normal and attack periods, with `test_labels[i] == 1` meaning row i of
    `test` is an attack. Both frames contain only the numeric sensor/actuator
    columns listed in `columns`, in the same column order, already coerced to
    float and forward/back-filled for NaNs.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_labels: np.ndarray
    columns: list[str]
    ground_truth_graph: nx.DiGraph | None = field(default=None)

    def __post_init__(self) -> None:
        assert list(self.train.columns) == self.columns
        assert list(self.test.columns) == self.columns
        assert len(self.test_labels) == len(self.test)


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce `columns` of `df` to float, fill gaps, drop constant columns."""
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    out = out.ffill().bfill()
    out = out.fillna(0.0)
    return out.astype(np.float64)


def drop_constant_columns(*frames: pd.DataFrame) -> list[str]:
    """Return column names that are non-constant in every given frame."""
    keep = set(frames[0].columns)
    for frame in frames:
        nunique = frame.nunique(dropna=False)
        keep &= set(nunique[nunique > 1].index)
    return [c for c in frames[0].columns if c in keep]
