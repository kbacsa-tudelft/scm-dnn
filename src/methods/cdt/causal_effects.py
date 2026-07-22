"""Phase III of CDT: interventional / do-calculus quantities (Eqs 17-18).

`do_predict` is the workhorse used by scoring.py and attribution.py. Because
every edge in the discovered graph already points from a variable's
*parents* (its full backdoor-blocking adjustment set, by construction of
Phase I) to the child, `E[V_i | do(PA(V_i)=pa)]` collapses to simply
evaluating the fitted structural equation at `pa` -- conditioning on a
variable's full parent set *is* backdoor adjustment with Z = PA(X) itself
(Eq 17 with the sum over Z degenerating to the single fitted-mechanism
evaluation, since P(z) is absorbed into whatever parent values are passed
in). `backdoor_adjustment`/`frontdoor_adjustment` below are standalone,
general-purpose implementations of Eqs 17/18 (simple linear-regression
adjustment) provided for completeness/reproduction -- they are not on the
main fit/score path, which only needs `do_predict`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scm import FittedUnit


def do_predict(unit: FittedUnit, parent_values: np.ndarray) -> np.ndarray:
    """E[V_i | do(PA(V_i)=parent_values)], vectorized over rows."""
    return unit.predict(parent_values)


def backdoor_adjustment(
    train: pd.DataFrame, treatment: str, outcome: str, adjustment_set: list[str], x_value: float
) -> float:
    """Eq 17: P(y|do(x)) = sum_z P(y|x,z) P(z), via linear regression
    adjustment -- fit E[Y|X,Z] by OLS, then average its prediction at
    X=x_value over the empirical distribution of Z observed in training data."""
    Z = train[adjustment_set].to_numpy() if adjustment_set else np.empty((len(train), 0))
    X = train[[treatment]].to_numpy()
    y = train[outcome].to_numpy()
    design = np.column_stack([np.ones(len(train)), X, Z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)

    design_at_x = np.column_stack([np.ones(len(train)), np.full(len(train), x_value), Z])
    return float((design_at_x @ beta).mean())


def frontdoor_adjustment(train: pd.DataFrame, treatment: str, outcome: str, mediator: str, x_value: float) -> float:
    """Eq 18: P(y|do(x)) = sum_z P(z|x) sum_x' P(y|x',z) P(x'), single-mediator
    case, via two linear-regression stages (secondary/fallback path, used only
    when no valid backdoor set exists)."""
    x, m, y = train[treatment].to_numpy(), train[mediator].to_numpy(), train[outcome].to_numpy()

    design_m = np.column_stack([np.ones(len(train)), x])
    beta_m, *_ = np.linalg.lstsq(design_m, m, rcond=None)
    m_given_x = float(np.array([1.0, x_value]) @ beta_m)

    design_y = np.column_stack([np.ones(len(train)), x, m])
    beta_y, *_ = np.linalg.lstsq(design_y, y, rcond=None)
    # E_x'[E[Y|X=x',M=m_given_x]] -- average the fitted Y-model over the
    # training X distribution at the mediator value implied by x_value.
    design_y_at_m = np.column_stack([np.ones(len(train)), x, np.full(len(train), m_given_x)])
    return float((design_y_at_m @ beta_y).mean())
