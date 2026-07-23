"""Anomaly scoring (Kravchik & Shabtai, Eqs 1-4).

Eq 1: e_t = |y_t - yhat_t| (per-feature absolute prediction error).
Eq 2: z_e_t = |e_t - mu_e| / sigma_e, mu_e/sigma_e the error's mean/std.
      The paper says these are "calculated over all of the data", ambiguous
      whether that means train-only or train+test -- train-only is used
      here (the standard, non-leaking interpretation). More specifically
      (see model.py::fit), they're computed from the model's *held-out
      validation* split, not the data it was directly fit on: in-sample
      error is systematically smaller than out-of-sample error for a network
      this size relative to typical dataset sizes here, so calibrating
      against in-sample error makes ordinary held-out data -- including
      normal test rows -- look anomalous purely from the generalization
      gap, not genuine deviation.
Eq 3: max_features(z_e_t) > T -- per-timestep breach flag.
Eq 4: alert at t only if EVERY step in the trailing window [t-W, t] breached
      T (an AND rule -- stricter than PbNN's CUSUM count-based partial-
      violation rule, and CDT's majority-vote rule).

T and W are never given final numeric values, only grid-search ranges
(T: 1.8-3.0, W: 50-300 "seconds", i.e. samples at ICS datasets' ~1Hz rate) --
defaults below (T=2.4, W=100) are documented midpoints, not paper quotes.

**Deviation, found during smoke-testing on real SWaT data**: the paper
scales/scores continuous and discrete/actuator columns identically ("no
special treatment"). In practice this makes `max_features(z_e_t)` collapse
to near-constant "flag everything": near-binary actuator columns (valves,
pumps) are trivially easy to predict during calibration, so their error std
is tiny, and any single state change during test then produces an enormous
z-score that dominates the max regardless of what the continuous sensors
are doing (confirmed empirically -- one such column alone accounted for the
argmax on ~49% of test rows in a real run). `per_timestep_score` therefore
restricts the max to *continuous* columns only (the same
`nunique() <= DISCRETE_MAX_CARDINALITY` heuristic CDT's SCM already uses to
tell sensors from actuators, see `model.py`) -- a deviation from a literal
reading of the paper, not from its formulas, which are otherwise implemented
as given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_T = 2.4
DEFAULT_W = 100


@dataclass
class ErrorStats:
    mean: np.ndarray  # mu_e, per feature
    std: np.ndarray   # sigma_e, per feature


def fit_error_stats(train_actual: np.ndarray, train_pred: np.ndarray) -> ErrorStats:
    e = np.abs(train_actual - train_pred)
    return ErrorStats(mean=e.mean(axis=0), std=e.std(axis=0) + 1e-9)


def per_timestep_score(
    actual: np.ndarray, pred: np.ndarray, stats: ErrorStats, score_mask: np.ndarray | None = None
) -> np.ndarray:
    """Eq 1-3's continuous quantity: max_features(z_e_t) per row -- this is
    what `CNN1D.score()` returns for the harness's threshold-free metrics.
    `score_mask` (boolean, one per feature) restricts the max to continuous
    columns only, see module docstring -- if None, all columns are used
    (the literal paper reading)."""
    e = np.abs(actual - pred)
    z = np.abs(e - stats.mean) / stats.std
    if score_mask is not None:
        z = z[:, score_mask]
    return z.max(axis=1)


def windowed_and_alerts(scores: np.ndarray, threshold: float = DEFAULT_T, window: int = DEFAULT_W) -> np.ndarray:
    """Eq 4: binary alert at t iff every step in the trailing window
    [t-window, t] has score > threshold."""
    breach = (scores > threshold).astype(int)
    # rolling "all breached" over `window`: min of a 0/1 series over the
    # window is 1 only if every element in it is 1.
    return (pd.Series(breach).rolling(window=window, min_periods=1).min().to_numpy() > 0).astype(int)
