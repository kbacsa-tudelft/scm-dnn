"""CDT: ties discovery (Phase I) -> SCM fitting (Phase II) -> multi-scale
interventional scoring (Phase III) together behind the AnomalyDetectionMethod
interface."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ..base import AnomalyDetectionMethod
from . import attribution
from .discovery import ALPHA, MAX_CANDIDATE_PARENTS, MAX_LAG, discover_graph
from .scm import EPOCHS, LR, fit_scm
from .scoring import (
    centrality_weights,
    fit_window_norm_stats,
    mcai,
    multiscale_ensemble_score,
    per_variable_scores,
)

THRESHOLD_QUANTILE = 0.99


class CDT(AnomalyDetectionMethod):
    def __init__(
        self,
        tau: int = MAX_LAG,
        k: int = MAX_CANDIDATE_PARENTS,
        alpha: float = ALPHA,
        epochs: int = EPOCHS,
        lr: float = LR,
        threshold_quantile: float = THRESHOLD_QUANTILE,
        seed: int = 0,
    ):
        self.tau = tau
        self.k = k
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.threshold_quantile = threshold_quantile
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> None:
        self.graph_ = discover_graph(train, tau=self.tau, k=self.k, alpha=self.alpha, seed=self.seed)
        self.scm_ = fit_scm(train, self.graph_, epochs=self.epochs, lr=self.lr, seed=self.seed)
        self.weights_ = centrality_weights(self.graph_)

        train_scores = per_variable_scores(self.scm_, train)
        train_mcai = mcai(train_scores, self.weights_)
        self.norm_stats_ = fit_window_norm_stats(train_mcai)
        train_ensemble = multiscale_ensemble_score(train_mcai, self.norm_stats_)

        self.threshold_ = float(np.quantile(train_ensemble, self.threshold_quantile))

    def _ensemble_score(self, df: pd.DataFrame) -> np.ndarray:
        scores = per_variable_scores(self.scm_, df)
        return multiscale_ensemble_score(mcai(scores, self.weights_), self.norm_stats_)

    def score(self, test: pd.DataFrame) -> np.ndarray:
        ensemble = self._ensemble_score(test)
        # per_variable_scores drops the first `global_lag` rows (no history
        # to evaluate lagged parents); pad the front so score() returns one
        # value per row of `test`, per the AnomalyDetectionMethod contract.
        pad = np.full(self.scm_.global_lag, ensemble[0] if len(ensemble) else 0.0)
        return np.concatenate([pad, ensemble])

    def causal_graph(self) -> nx.DiGraph:
        return self.graph_

    def root_cause(self, test: pd.DataFrame, t: int, top_k: int = 5) -> list[tuple[str, float]] | None:
        if t < self.scm_.global_lag:
            return None
        row_scores = per_variable_scores(self.scm_, test).iloc[t - self.scm_.global_lag].to_dict()
        return attribution.root_cause(self.scm_, self.weights_, test, t, row_scores, top_k)
