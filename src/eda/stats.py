"""Generic, dataset-agnostic statistics helpers shared by every EDA report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISCRETE_MAX_CARDINALITY = 5  # matches src/methods/cdt/scm.py's convention


@dataclass
class Segment:
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def classify_columns(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Returns (continuous_cols, discrete_cols) by cardinality, same heuristic
    the CDT/PbNN implementations already use to tell sensors from actuators."""
    continuous, discrete = [], []
    for c in cols:
        if df[c].nunique(dropna=True) <= DISCRETE_MAX_CARDINALITY:
            discrete.append(c)
        else:
            continuous.append(c)
    return continuous, discrete


def missingness(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """% missing per column, descending."""
    pct = (df[cols].isna().mean() * 100).sort_values(ascending=False)
    return pct


def constant_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    nunique = df[cols].nunique(dropna=False)
    return list(nunique[nunique <= 1].index)


def describe_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """mean/std/min/max/missing% per column."""
    desc = df[cols].describe().T[["mean", "std", "min", "max"]]
    desc["missing_pct"] = df[cols].isna().mean() * 100
    return desc


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr()


def top_correlated_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    corr_abs = corr.abs()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pairs.append((a, b, corr.loc[a, b]))
    out = pd.DataFrame(pairs, columns=["var_a", "var_b", "correlation"])
    return out.reindex(out.correlation.abs().sort_values(ascending=False).index).head(n)


def label_segments(labels: np.ndarray, value: int = 1) -> list[Segment]:
    """Contiguous runs of `labels == value` as (start, end) index segments."""
    labels = np.asarray(labels)
    is_val = labels == value
    if not is_val.any():
        return []
    edges = np.diff(is_val.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if is_val[0]:
        starts = [0] + starts
    if is_val[-1]:
        ends = ends + [len(labels)]
    return [Segment(s, e) for s, e in zip(starts, ends)]


def class_balance(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    n = len(labels)
    n_attack = int((labels == 1).sum())
    segments = label_segments(labels, value=1)
    lengths = [s.length for s in segments]
    return {
        "n_total": n,
        "n_normal": n - n_attack,
        "n_attack": n_attack,
        "attack_pct": 100 * n_attack / n if n else 0.0,
        "n_segments": len(segments),
        "segment_lengths": lengths,
        "mean_segment_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_segment_length": float(np.median(lengths)) if lengths else 0.0,
        "max_segment_length": int(np.max(lengths)) if lengths else 0,
    }


def downsample_for_plot(df: pd.DataFrame, max_points: int = 8000) -> pd.DataFrame:
    """Evenly-spaced row subsample for time-series plotting (not a random
    sample -- preserves temporal order and overall shape)."""
    if len(df) <= max_points:
        return df
    step = len(df) // max_points
    return df.iloc[::step]
