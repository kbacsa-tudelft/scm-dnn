"""Shared interface every method (CDT, PbNN, ...) implements, so the
benchmark runner can treat them interchangeably."""
from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import pandas as pd


class AnomalyDetectionMethod(ABC):
    """Fit on attack-free training data, then score/predict on test data."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """`train` has one column per sensor/actuator tag, attack-free."""

    @abstractmethod
    def score(self, test: pd.DataFrame) -> np.ndarray:
        """Return one continuous anomaly score per row of `test` (higher = more anomalous)."""

    def predict(self, test: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        """Binary 0/1 per row. If `threshold` is None, uses a threshold fit in `fit()`."""
        scores = self.score(test)
        thresh = threshold if threshold is not None else self.threshold_
        return (scores > thresh).astype(int)

    def causal_graph(self) -> nx.DiGraph | None:
        """Discovered causal/dependency graph over training columns, if the method produces one."""
        return None

    def root_cause(self, test: pd.DataFrame, t: int, top_k: int = 5) -> list[tuple[str, float]] | None:
        """Ranked (variable, attribution score) pairs explaining the anomaly at row `t`, if supported."""
        return None
