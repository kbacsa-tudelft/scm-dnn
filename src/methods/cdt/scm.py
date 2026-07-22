"""Phase II of CDT: additive-noise structural causal model estimation
(Eqs 14-16), fit with PyTorch (Adam) per Eq 16's "MLE parameter estimation".

Each non-root variable's structural equation is a single linear layer over
its (standardized) parents plus one shared quadratic term
`sum_k PA_k^2` (Eq 14), or a logistic unit for low-cardinality
actuator-like variables (Eq 15). Parent features and (for continuous
variables) the target are standardized before fitting purely for
optimizer stability -- this reparameterization doesn't change the
functional form (linear + quadratic in the parents), only its scale.
Minimizing MSE for a fixed-variance Gaussian additive-noise model *is* the
MLE solution (Eq 16), so no separate variance parameter is learned; sigma
is computed empirically from residuals after fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DISCRETE_MAX_CARDINALITY = 5
EPOCHS = 300
LR = 0.05


class ContinuousSCMUnit(nn.Module):
    def __init__(self, n_parents: int):
        super().__init__()
        self.linear = nn.Linear(n_parents + 1, 1)  # +1 for the shared quadratic term

    def forward(self, parents: torch.Tensor) -> torch.Tensor:
        quad = (parents**2).sum(dim=1, keepdim=True)
        return self.linear(torch.cat([parents, quad], dim=1)).squeeze(-1)


class DiscreteSCMUnit(nn.Module):
    def __init__(self, n_parents: int):
        super().__init__()
        self.linear = nn.Linear(n_parents, 1)

    def forward(self, parents: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(parents).squeeze(-1))


@dataclass
class FittedUnit:
    var: str
    kind: str  # "root" | "continuous" | "discrete"
    parents: list[tuple[str, int]]  # (parent_var, lag) in the order model inputs expect
    model: nn.Module | None = None
    mean: float = 0.0
    std: float = 1.0
    parent_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    parent_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    target_mean: float = 0.0
    target_std: float = 1.0
    target_min: float = 0.0
    target_max: float = 1.0

    def predict(self, parent_matrix: np.ndarray) -> np.ndarray:
        """Vectorized prediction of E[V_i | parents] in original units."""
        n = parent_matrix.shape[0] if parent_matrix.ndim == 2 else len(parent_matrix)
        if self.kind == "root" or self.model is None:
            return np.full(n, self.mean)
        x = (parent_matrix - self.parent_mean) / self.parent_std
        with torch.no_grad():
            out = self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        if self.kind == "discrete":
            scale = self.target_max - self.target_min
            return out * scale + self.target_min
        return out * self.target_std + self.target_mean


@dataclass
class FittedSCM:
    units: dict[str, FittedUnit]
    graph: nx.DiGraph
    global_lag: int

    def order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))


def _train(model: nn.Module, X: np.ndarray, y: np.ndarray, kind: str, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    loss_fn = F.mse_loss if kind == "continuous" else F.binary_cross_entropy
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_t), y_t)
        loss.backward()
        opt.step()


def fit_scm(train: pd.DataFrame, graph: nx.DiGraph, epochs: int = EPOCHS, lr: float = LR, seed: int = 0) -> FittedSCM:
    torch.manual_seed(seed)
    lags = [d["lag"] for _, _, d in graph.edges(data=True)]
    global_lag = max(lags) if lags else 0
    n = len(train)

    units: dict[str, FittedUnit] = {}
    for var in nx.topological_sort(graph):
        parent_info = [(p, graph.edges[p, var]["lag"]) for p in graph.predecessors(var)]
        target = train[var].to_numpy()[global_lag:]

        if not parent_info:
            units[var] = FittedUnit(
                var=var, kind="root", parents=[], mean=float(target.mean()), std=float(target.std() + 1e-6)
            )
            continue

        parent_matrix = np.column_stack(
            [train[p].to_numpy()[global_lag - lag : n - lag] for p, lag in parent_info]
        )
        p_mean, p_std = parent_matrix.mean(axis=0), parent_matrix.std(axis=0) + 1e-6
        parent_matrix_std = (parent_matrix - p_mean) / p_std

        is_discrete = train[var].nunique() <= DISCRETE_MAX_CARDINALITY
        if is_discrete:
            t_min, t_max = float(target.min()), float(target.max())
            scale = (t_max - t_min) or 1.0
            target_scaled = (target - t_min) / scale
            model = DiscreteSCMUnit(len(parent_info))
            _train(model, parent_matrix_std, target_scaled, "discrete", epochs, lr)
            with torch.no_grad():
                pred_orig = model(torch.tensor(parent_matrix_std, dtype=torch.float32)).numpy() * scale + t_min
            units[var] = FittedUnit(
                var=var, kind="discrete", parents=parent_info, model=model,
                parent_mean=p_mean, parent_std=p_std,
                target_min=t_min, target_max=t_max,
                std=float(np.std(target - pred_orig) + 1e-6),
            )
        else:
            t_mean, t_std = float(target.mean()), float(target.std() + 1e-6)
            target_std_scaled = (target - t_mean) / t_std
            model = ContinuousSCMUnit(len(parent_info))
            _train(model, parent_matrix_std, target_std_scaled, "continuous", epochs, lr)
            with torch.no_grad():
                pred_orig = model(torch.tensor(parent_matrix_std, dtype=torch.float32)).numpy() * t_std + t_mean
            units[var] = FittedUnit(
                var=var, kind="continuous", parents=parent_info, model=model,
                parent_mean=p_mean, parent_std=p_std,
                target_mean=t_mean, target_std=t_std,
                std=float(np.std(target - pred_orig) + 1e-6),
            )

    return FittedSCM(units=units, graph=graph, global_lag=global_lag)
