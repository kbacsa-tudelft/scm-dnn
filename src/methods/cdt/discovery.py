"""Phase I of CDT: temporal-lag causal graph discovery (Algorithm 1, Eqs 10-13).

Deviations from the paper, documented:
- TDMI candidate screening (Eq 11) runs in two stages for tractability: a
  cheap vectorized marginal-correlation prefilter narrows every variable's
  `(n_vars-1)*tau` lagged candidates down to a shortlist, then sklearn's
  k-NN mutual-information estimator ranks the shortlist and keeps the top
  `k` (the paper cites Fraser & Swinney 1986 for TDMI but gives no closed
  form, and never gives a numeric value for the "> delta" threshold in Eq
  11 -- since `k` already caps candidates, no separate delta is applied).
- Candidate parents are restricted to *other* variables (j != i). The paper
  doesn't explicitly exclude same-variable lags, but a causal DAG with
  self-loops isn't meaningful, so autoregressive terms are dropped from the
  discovered graph.
- Local-PC conditioning sets (Eq 12/13) are subsets of a target's own
  candidate-parent set, capped at size 3 -- otherwise the subset search
  over up to k=10 candidates explodes combinatorially.
- Orientation, normally the hard part of PC, is trivial here: every
  candidate parent is a strictly-lagged (l>=1) copy of some variable, so
  temporal precedence alone fixes edge direction. The paper's
  physical/control-logic orientation rules (needed for contemporaneous
  edges) aren't implemented since only strictly-lagged edges are used.
- Discovery runs on a contiguous row subsample (default 5000 rows) since
  CI-testing over full 100k+ row datasets is unnecessary for structure
  learning and would be slow in pure Python/sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

MAX_LAG = 10
MAX_CANDIDATE_PARENTS = 10
ALPHA = 0.05
MAX_CONDITIONING_SET_SIZE = 3
DISCOVERY_SUBSAMPLE_ROWS = 5000
PREFILTER_SHORTLIST_SIZE = 30


@dataclass(frozen=True)
class CandidateParent:
    var: str
    lag: int


def _subsample(df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, len(df) - n_rows + 1))
    return df.iloc[start : start + n_rows]


def _build_lagged_arrays(df: pd.DataFrame, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """present: (n_rows-tau, n_cols); lagged: (n_rows-tau, n_cols, tau) with
    lagged[:, :, l-1] holding each column shifted back by l steps."""
    values = df.to_numpy()
    n_rows = values.shape[0]
    present = values[tau:]
    lagged = np.stack([values[tau - l : n_rows - l] for l in range(1, tau + 1)], axis=-1)
    return present, lagged


def _tdmi_candidates(
    present: np.ndarray, lagged: np.ndarray, columns: list[str], target_idx: int, k: int, seed: int
) -> list[CandidateParent]:
    n_cols, tau = lagged.shape[1], lagged.shape[2]
    y = present[:, target_idx]

    cand_vars, cand_lags, cand_series = [], [], []
    for j in range(n_cols):
        if j == target_idx:
            continue
        for l in range(tau):
            cand_vars.append(j)
            cand_lags.append(l + 1)
            cand_series.append(lagged[:, j, l])
    if not cand_vars:
        return []
    X = np.stack(cand_series, axis=1)

    y_c = y - y.mean()
    X_c = X - X.mean(axis=0, keepdims=True)
    corr = np.abs((X_c.T @ y_c) / ((np.linalg.norm(X_c, axis=0) + 1e-12) * (np.linalg.norm(y_c) + 1e-12)))
    shortlist = np.argsort(-corr)[:PREFILTER_SHORTLIST_SIZE]

    mi = mutual_info_regression(X[:, shortlist], y, random_state=seed)
    top = shortlist[np.argsort(-mi)[:k]]

    return [CandidateParent(var=columns[cand_vars[i]], lag=cand_lags[i]) for i in top]


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    if z.shape[1] == 0:
        resid_x, resid_y = x - x.mean(), y - y.mean()
    else:
        Z = np.column_stack([np.ones(len(z)), z])
        beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid_x, resid_y = x - Z @ beta_x, y - Z @ beta_y
    # near-zero-variance residuals (e.g. a candidate constant within this
    # discovery subsample) are treated as uncorrelated rather than NaN.
    denom = np.std(resid_x) * np.std(resid_y)
    if denom < 1e-9:
        return 0.0
    return float(np.mean(resid_x * resid_y) / denom)


def _fisher_z_independent(rho: float, n: int, cond_size: int, alpha: float) -> bool:
    rho = float(np.clip(rho, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(max(n - cond_size - 3, 1))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value > alpha


def _local_pc_filter(
    present: np.ndarray,
    lagged: np.ndarray,
    columns: list[str],
    target_idx: int,
    candidates: list[CandidateParent],
    alpha: float,
) -> list[CandidateParent]:
    y = present[:, target_idx]
    col_index = {c: i for i, c in enumerate(columns)}
    series = {(c.var, c.lag): lagged[:, col_index[c.var], c.lag - 1] for c in candidates}
    n = len(y)

    surviving = []
    for cand in candidates:
        others = [c for c in candidates if c is not cand]
        x = series[(cand.var, cand.lag)]
        independent = False
        for size in range(0, min(MAX_CONDITIONING_SET_SIZE, len(others)) + 1):
            for subset in combinations(others, size):
                z = np.column_stack([series[(c.var, c.lag)] for c in subset]) if subset else np.empty((n, 0))
                rho = _partial_correlation(x, y, z)
                if _fisher_z_independent(rho, n, size, alpha):
                    independent = True
                    break
            if independent:
                break
        if not independent:
            surviving.append(cand)
    return surviving


def discover_graph(
    train: pd.DataFrame,
    tau: int = MAX_LAG,
    k: int = MAX_CANDIDATE_PARENTS,
    alpha: float = ALPHA,
    subsample_rows: int = DISCOVERY_SUBSAMPLE_ROWS,
    seed: int = 0,
) -> nx.DiGraph:
    """Run Phase I and return a summary DAG over the original (unlagged)
    variable names, with each edge's `lag` attribute set to the shortest
    surviving lag between that parent and child."""
    df = _subsample(train, subsample_rows, seed=seed)
    columns = list(df.columns)
    present, lagged = _build_lagged_arrays(df, tau)

    best_lag: dict[tuple[str, str], int] = {}
    for i, col in enumerate(columns):
        cands = _tdmi_candidates(present, lagged, columns, i, k, seed)
        for p in _local_pc_filter(present, lagged, columns, i, cands, alpha):
            key = (p.var, col)
            if key not in best_lag or p.lag < best_lag[key]:
                best_lag[key] = p.lag

    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for (parent_var, child_var), lag in best_lag.items():
        graph.add_edge(parent_var, child_var, lag=lag)

    # Collapsing per-lag edges into one summary edge per (parent, child) can
    # introduce cycles (e.g. A(t-1)->B(t) and B(t-1)->A(t) both survive).
    # Break cycles by dropping the longest-lag edge in each, since shorter
    # lags reflect tighter/more direct temporal coupling.
    while True:
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            break
        worst = max(cycle, key=lambda e: graph.edges[e[0], e[1]]["lag"])
        graph.remove_edge(*worst[:2])

    return graph
