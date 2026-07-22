"""Root-cause attribution (Eqs 21-22).

Both use a common "replacement game" over a target variable's fitted
structural equation: `f(S) = -|target_observed - unit.predict(x_S)|` where
`x_S` takes each parent's *observed* value for parents in `S` and its
*baseline* (train-mean) value otherwise. A parent that most improves the
reconstruction of the (anomalous) observed target when switched from
baseline to observed is implicated as a root cause -- its true behavior is
what the causal mechanism needed to explain the anomalous reading. This is
a SHAP-style anomaly-attribution construction, scoped to one target
variable's direct parents (the "candidate causes" for that variable, per
the discovered graph) rather than all variables in the dataset, since the
paper's Eq 22 game `f(S)` is otherwise unspecified.

`causal_effect_do` is the exact Eq 21 special case (coalition size 0 vs 1,
i.e. one parent swapped in isolation) computed directly. `shapley_values`
is the full Eq 22 game, approximated via Monte Carlo permutation sampling
since exact Shapley is exponential in the number of parents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .scm import FittedSCM, FittedUnit

SHAPLEY_SAMPLES = 300


def _game_value(unit: FittedUnit, observed: np.ndarray, baseline: np.ndarray, mask: np.ndarray, target_observed: float) -> float:
    x = np.where(mask, observed, baseline).reshape(1, -1)
    pred = unit.predict(x)[0]
    return -abs(target_observed - pred)


def causal_effect_do(unit: FittedUnit, observed: np.ndarray, baseline: np.ndarray, target_observed: float) -> np.ndarray:
    """Eq 21 per parent: E[.|do(parent=observed)] - E[.|do(parent=normal)],
    others held at baseline."""
    n_parents = len(observed)
    empty_mask = np.zeros(n_parents, dtype=bool)
    f_empty = _game_value(unit, observed, baseline, empty_mask, target_observed)
    effects = np.zeros(n_parents)
    for i in range(n_parents):
        mask = empty_mask.copy()
        mask[i] = True
        effects[i] = _game_value(unit, observed, baseline, mask, target_observed) - f_empty
    return effects


def shapley_values(
    unit: FittedUnit, observed: np.ndarray, baseline: np.ndarray, target_observed: float,
    n_samples: int = SHAPLEY_SAMPLES, seed: int = 0,
) -> np.ndarray:
    """Eq 22, Monte Carlo permutation approximation."""
    n_parents = len(observed)
    if n_parents == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    phi = np.zeros(n_parents)
    for _ in range(n_samples):
        perm = rng.permutation(n_parents)
        mask = np.zeros(n_parents, dtype=bool)
        prev = _game_value(unit, observed, baseline, mask, target_observed)
        for idx in perm:
            mask[idx] = True
            curr = _game_value(unit, observed, baseline, mask, target_observed)
            phi[idx] += curr - prev
            prev = curr
    return phi / n_samples


def _observed_parent_vector(df: pd.DataFrame, parents: list[tuple[str, int]], t: int) -> np.ndarray:
    return np.array([df[var].iloc[t - lag] for var, lag in parents])


def root_cause(
    scm: FittedSCM, weights: dict[str, float], test: pd.DataFrame, t: int, scores_row: dict[str, float], top_k: int = 5
) -> list[tuple[str, float]]:
    """Picks the highest weighted-score node at row `t` as the anomaly's
    proximate location, then ranks that node's parents by Shapley
    attribution as the candidate root causes."""
    if t < scm.global_lag:
        return []

    target = max(scores_row, key=lambda v: weights.get(v, 0.0) * scores_row[v])
    unit = scm.units[target]
    if not unit.parents:
        return [(target, float(scores_row[target]))]

    observed = _observed_parent_vector(test, unit.parents, t)
    baseline = unit.parent_mean
    target_observed = float(test[target].iloc[t])

    phi = shapley_values(unit, observed, baseline, target_observed)
    labels = [f"{var}(t-{lag})" for var, lag in unit.parents]
    ranked = sorted(zip(labels, phi), key=lambda kv: -kv[1])
    return ranked[:top_k]
