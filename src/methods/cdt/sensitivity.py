"""Eq 26: Robins' bias-bound sensitivity analysis for hidden confounding.

A standalone diagnostic, not part of the main fit/score pipeline -- the
paper itself only applies it in one ad hoc case study with a
user-specified candidate confounder `U` (e.g. an unmeasured ambient
temperature), not as a generic step every causal-effect estimate runs
through.
"""
from __future__ import annotations

import numpy as np


def robins_bias_bound(x: np.ndarray, y: np.ndarray, u: np.ndarray) -> float:
    """|CE_hat - CE_true| <= alpha_U * beta_U for a candidate unmeasured
    confounder `u` (same length as `x`/`y`), where:
        alpha_U = max_x |P(Y|X=x,U=1) - P(Y|X=x,U=0)|
        beta_U  = |Cov(X,U)|
    `u` is treated as binary (0/1); `y` and `x` are treated as continuous.
    """
    u = u.astype(bool)
    alpha_u = 0.0
    for x_val in np.unique(x):
        mask = x == x_val
        y_u1 = y[mask & u]
        y_u0 = y[mask & ~u]
        if len(y_u1) == 0 or len(y_u0) == 0:
            continue
        alpha_u = max(alpha_u, abs(y_u1.mean() - y_u0.mean()))
    beta_u = abs(np.cov(x, u.astype(float))[0, 1])
    return float(alpha_u * beta_u)
