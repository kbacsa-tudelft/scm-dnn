"""PbNN (Raman & Mathur 2022): one DCNN + CUSUM detector per process
invariant, combined into a single per-timestep anomaly score.

Eq 10 is inherently a per-invariant binary alert; the paper never defines
how to combine invariants into one detector-level score/decision. We use
the mean of each invariant's `cusum_score` (trailing-window violation
fraction) across invariants as the combined continuous score -- a
documented choice, not from the paper.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd

from ..base import AnomalyDetectionMethod
from .cusum import CUSUMParams, cusum_score, fit_cusum
from .dcnn import DCNN, DCNNConfig, DEFAULT_CONFIG, predict_dcnn, train_dcnn
from .invariants import Invariant, build_invariants


def _guess_dataset(columns: list[str]) -> str:
    if any(c.startswith(("1_", "2_", "3_")) for c in columns):
        return "wadi"
    if any(c.startswith(("P1_", "P2_", "P3_", "P4_")) for c in columns):
        return "hai"
    return "swat"


class PbNN(AnomalyDetectionMethod):
    def __init__(
        self,
        config: DCNNConfig = DEFAULT_CONFIG,
        s_w: int = 10,
        t_w: int = 3,
        k_sigma: float = 3.0,
    ):
        self.config = config
        self.s_w, self.t_w, self.k_sigma = s_w, t_w, k_sigma
        self.invariants: list[Invariant] = []
        self.models: dict[str, DCNN] = {}
        self.dcnn_stats: dict[str, dict] = {}
        self.cusum_params: dict[str, CUSUMParams] = {}
        self.threshold_: float = 0.0

    def fit(self, train: pd.DataFrame) -> None:
        self.invariants = build_invariants(list(train.columns), dataset_name=_guess_dataset(list(train.columns)))
        if not self.invariants:
            raise ValueError("No PbNN invariants could be built from the given columns")

        for inv in self.invariants:
            predictors = train[inv.predictors].to_numpy()
            target = train[inv.target].to_numpy()
            model, stats = train_dcnn(predictors, target, self.config)
            self.models[inv.name] = model
            self.dcnn_stats[inv.name] = stats

            pred = predict_dcnn(model, predictors, self.config, stats)
            residuals = target[self.config.t_k :] - pred
            self.cusum_params[inv.name] = fit_cusum(residuals, self.k_sigma, self.s_w, self.t_w)

        train_scores = self.score(train)
        self.threshold_ = float(np.mean(train_scores) + 3 * np.std(train_scores))

    def score(self, test: pd.DataFrame) -> np.ndarray:
        n = len(test)
        per_invariant_scores = []
        for inv in self.invariants:
            predictors = test[inv.predictors].to_numpy()
            target = test[inv.target].to_numpy()
            model = self.models[inv.name]
            stats = self.dcnn_stats[inv.name]
            pred = predict_dcnn(model, predictors, self.config, stats)
            residuals = target[self.config.t_k :] - pred
            s = cusum_score(residuals, self.cusum_params[inv.name])
            full = np.concatenate([np.zeros(self.config.t_k), s])
            per_invariant_scores.append(full[:n])
        return np.mean(per_invariant_scores, axis=0)

    def causal_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for inv in self.invariants:
            for p in inv.predictors:
                g.add_edge(p, inv.target)
        return g

    def root_cause(self, test: pd.DataFrame, t: int, top_k: int = 5) -> list[tuple[str, float]]:
        contributions = []
        for inv in self.invariants:
            predictors = test[inv.predictors].to_numpy()
            target = test[inv.target].to_numpy()
            model = self.models[inv.name]
            stats = self.dcnn_stats[inv.name]
            pred = predict_dcnn(model, predictors, self.config, stats)
            idx = t - self.config.t_k
            if not (0 <= idx < len(pred)):
                continue
            residual = abs(target[self.config.t_k :][idx] - pred[idx])
            contributions.append((inv.target, float(residual)))
        contributions.sort(key=lambda x: -x[1])
        return contributions[:top_k]
