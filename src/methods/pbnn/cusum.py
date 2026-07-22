"""CUSUM residual-based anomaly detector (PbNN paper Eqs 6,7,9,10).

The paper's Eqs 6-7 as transcribed (`P(t) = max(0, r(t)-target-b)`,
`N(t) = min(0, r(t)-target+b)`) omit the recursive P(t-1)/N(t-1) term that
every standard two-sided CUSUM control chart uses -- almost certainly lost
in the paper's PDF rendering, since the surrounding prose explicitly calls
this a "cumulative sum" technique and a purely point-wise max(0, ...) isn't
cumulative at all. We implement the standard recursive CUSUM here, which is
the well-established form the paper's own terminology and behavior point to.

`b` (slack) and UCL/LCL are not given numeric values -- the paper says only
"computed empirically using the values of P and N". Defaults here (b =
0.5 * train-residual-std; UCL/LCL = mean +/- k_sigma*std of P/N fit on
training residuals) are a standard, documented choice, not from the paper.
S_w/T_w (Eq 10's window size / violation-count threshold) are likewise
undocumented; defaults S_w=10, T_w=3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CUSUMParams:
    target: float
    b: float
    ucl: float
    lcl: float
    s_w: int = 10
    t_w: int = 3


def _cusum_stats(residuals: np.ndarray, target: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    p_arr = np.zeros(len(residuals))
    n_arr = np.zeros(len(residuals))
    p, n = 0.0, 0.0
    for i, r in enumerate(residuals):
        p = max(0.0, p + r - target - b)
        n = min(0.0, n + r - target + b)
        p_arr[i], n_arr[i] = p, n
    return p_arr, n_arr


def fit_cusum(train_residuals: np.ndarray, k_sigma: float = 3.0, s_w: int = 10, t_w: int = 3) -> CUSUMParams:
    target = float(np.mean(train_residuals))
    b = 0.5 * float(np.std(train_residuals) + 1e-8)
    p_arr, n_arr = _cusum_stats(train_residuals, target, b)
    ucl = float(np.mean(p_arr) + k_sigma * np.std(p_arr))
    lcl = float(np.mean(n_arr) - k_sigma * np.std(n_arr))
    return CUSUMParams(target=target, b=b, ucl=ucl, lcl=lcl, s_w=s_w, t_w=t_w)


def cusum_violations(residuals: np.ndarray, params: CUSUMParams) -> np.ndarray:
    """Eq 9: f(t) = 1 if P(t) > UCL or N(t) < LCL else 0."""
    p_arr, n_arr = _cusum_stats(residuals, params.target, params.b)
    return ((p_arr > params.ucl) | (n_arr < params.lcl)).astype(int)


def cusum_score(residuals: np.ndarray, params: CUSUMParams) -> np.ndarray:
    """Continuous version of Eq 10: fraction of the trailing S_w window that
    violated UCL/LCL. Thresholding this at t_w/s_w recovers Eq 10's binary
    alert rule exactly."""
    violations = cusum_violations(residuals, params)
    counts = pd.Series(violations).rolling(window=params.s_w, min_periods=1).sum().to_numpy()
    return counts / params.s_w


def windowed_alerts(residuals: np.ndarray, params: CUSUMParams) -> np.ndarray:
    """Eq 10: alert at t if violation count over the trailing S_w window > T_w."""
    return (cusum_score(residuals, params) * params.s_w > params.t_w).astype(int)
