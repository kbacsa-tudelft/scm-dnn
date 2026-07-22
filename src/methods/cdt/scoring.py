"""Anomaly scoring: interventional score (Eq 19), Multi-Component Anomaly
Index (Eq 20), and multi-scale ensemble (Eqs 30-31)."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .scm import FittedSCM

MULTISCALE_WINDOWS = (5, 10, 20)
MULTISCALE_WEIGHTS = {5: 0.5, 10: 0.3, 20: 0.2}


def centrality_weights(graph: nx.DiGraph) -> dict[str, float]:
    """Eq 20: alpha_i = (|Descendants(V_i)| + |Ancestors(V_i)|) / (2n)."""
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    return {
        node: (len(nx.descendants(graph, node)) + len(nx.ancestors(graph, node))) / (2 * n)
        for node in graph.nodes()
    }


def per_variable_scores(scm: FittedSCM, df: pd.DataFrame) -> pd.DataFrame:
    """Eq 19 for every variable, over rows `global_lag:` of `df` (earlier
    rows lack enough history to evaluate lagged parents and are dropped;
    callers pad the result back to `len(df)` rows -- see `model.py`)."""
    n = len(df)
    scores = {}
    for var, unit in scm.units.items():
        values = df[var].to_numpy()[scm.global_lag :]
        if not unit.parents:
            pred = np.full(len(values), unit.mean)
        else:
            parent_matrix = np.column_stack(
                [df[p].to_numpy()[scm.global_lag - lag : n - lag] for p, lag in unit.parents]
            )
            pred = unit.predict(parent_matrix)
        scores[var] = np.abs(values - pred) / unit.std
    return pd.DataFrame(scores)


def mcai(scores: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Eq 20."""
    w = np.array([weights.get(c, 0.0) for c in scores.columns])
    return (scores.to_numpy() * w).sum(axis=1)


def _windowed_cumulative(mcai_series: np.ndarray, tau: int) -> np.ndarray:
    """Eq 30: S_t^cumul(tau) = sum_{i=t-tau}^{t} MCAI(i)."""
    cumsum = np.cumsum(mcai_series)
    return cumsum - np.concatenate([np.zeros(tau), cumsum[:-tau]])


def fit_window_norm_stats(train_mcai: np.ndarray) -> dict[int, tuple[float, float]]:
    """Windowed cumulative sums over different tau aren't comparable on a raw
    scale, so each is z-normalized before combining. Stats are fit on train
    and reused for test to avoid normalizing test against its own
    (possibly attack-contaminated) distribution."""
    stats = {}
    for tau in MULTISCALE_WINDOWS:
        w = _windowed_cumulative(train_mcai, tau)
        stats[tau] = (float(w.mean()), float(w.std() + 1e-9))
    return stats


def multiscale_ensemble_score(mcai_series: np.ndarray, norm_stats: dict[int, tuple[float, float]]) -> np.ndarray:
    """Eq 31, continuous form: weighted combination (w5=0.5, w10=0.3,
    w20=0.2, exact from the paper) of the three z-normalized windowed
    scores. Used as `score()`'s output for threshold-free comparison."""
    combined = np.zeros(len(mcai_series))
    for tau in MULTISCALE_WINDOWS:
        mean, std = norm_stats[tau]
        combined += MULTISCALE_WEIGHTS[tau] * (_windowed_cumulative(mcai_series, tau) - mean) / std
    return combined


def multiscale_votes(mcai_series: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    """The paper's literal Eq 31 alert rule (weighted sum of 0/1 indicators
    >= 2) is dimensionally inconsistent as written -- weights sum to 1.0, so
    a weighted indicator sum can never reach 2 -- and no numeric per-scale
    theta_r is ever given. We use an unweighted majority vote instead:
    alert if at least 2 of the 3 per-scale detectors independently exceed
    their own train-calibrated threshold."""
    votes = np.zeros(len(mcai_series))
    for tau in MULTISCALE_WINDOWS:
        votes += (_windowed_cumulative(mcai_series, tau) > thresholds[tau]).astype(int)
    return (votes >= 2).astype(int)
