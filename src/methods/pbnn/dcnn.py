"""DCNN per invariant (PbNN paper Section III-B / Table IV hyperparameters).

Eq 4 is the MSE training loss; Eq 5 is Theil's U1 used in the paper to pick
the best of DCNN1/DCNN2/DCNN3 (1/2/3 conv blocks). Kernel size, activation,
optimizer and pooling scheme are not specified in the paper at all -- ReLU,
Adam, kernel_size=3 are standard, documented defaults, not paper-derived.
Because the lag window `t_k` is short (2-10 steps), repeated intermediate
max-pooling would degenerate the sequence to length 0-1 after a couple of
blocks; instead each block is conv+ReLU only, with a single global average
pool before the FC head -- a documented deviation from a literal
"conv+pool block" reading, standard practice for short 1D sequences.

The full Table IV grid (~3 filter settings x ~20 hidden-unit values x ~9
t_k values x 3 epoch settings x 2 batch sizes x 3 architectures, per
invariant) is computationally infeasible for a CPU-only benchmark harness.
`grid_search_dcnn` implements the search machinery but is opt-in; the
default path (`DEFAULT_CONFIG`) fixes one reasonable point in the grid.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class DCNNConfig:
    n_blocks: int = 2  # "DCNN2" in the paper's naming
    filters: int = 64
    hidden_units: int = 32
    t_k: int = 5
    epochs: int = 20
    batch_size: int = 64
    kernel_size: int = 3  # not specified in paper
    lr: float = 1e-3


DEFAULT_CONFIG = DCNNConfig()

GRID: dict[str, list] = dict(
    n_blocks=[1, 2, 3],
    filters=[32, 64, 128],
    hidden_units=[8, 32, 64, 100],
    t_k=[2, 5, 10],
    epochs=[10, 20, 50],
    batch_size=[32, 64],
)


class DCNN(nn.Module):
    """1D-conv net over a short lag window of predictor values, predicting
    the invariant's target one step ahead."""

    def __init__(self, n_predictors: int, config: DCNNConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_predictors
        for _ in range(config.n_blocks):
            k = min(config.kernel_size, config.t_k)
            layers += [nn.Conv1d(in_ch, config.filters, kernel_size=k, padding=k // 2), nn.ReLU()]
            in_ch = config.filters
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)  # (batch, filters, t_k')
        h = self.pool(h).squeeze(-1)  # (batch, filters)
        return self.fc(h).squeeze(-1)


def make_windows(predictors: np.ndarray, target: np.ndarray, t_k: int) -> tuple[np.ndarray, np.ndarray]:
    """X[i] = predictors[i:i+t_k].T, predicting target[i+t_k] one step ahead."""
    n = len(target)
    x = np.stack([predictors[i : i + t_k].T for i in range(n - t_k)])
    y = target[t_k:]
    return x.astype(np.float32), y.astype(np.float32)


def theils_u1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.sqrt(np.mean(y_true**2)) + 1e-12
    return float(num / den)


def train_dcnn(predictors: np.ndarray, target: np.ndarray, config: DCNNConfig = DEFAULT_CONFIG) -> tuple[DCNN, dict]:
    x, y = make_windows(predictors, target, config.t_k)

    x_mean, x_std = x.mean(axis=(0, 2), keepdims=True), x.std(axis=(0, 2), keepdims=True) + 1e-8
    y_mean, y_std = float(y.mean()), float(y.std()) + 1e-8
    x_n = (x - x_mean) / x_std
    y_n = (y - y_mean) / y_std

    model = DCNN(predictors.shape[1], config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_t, y_t = torch.from_numpy(x_n), torch.from_numpy(y_n)
    n = len(y_n)

    model.train()
    for _ in range(config.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_final = model(x_t).numpy()
    stats = dict(
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        theils_u1=theils_u1(y_n, pred_final),
        final_mse=float(np.mean((pred_final - y_n) ** 2)),
    )
    return model, stats


def predict_dcnn(model: DCNN, predictors: np.ndarray, config: DCNNConfig, stats: dict) -> np.ndarray:
    """Returns predictions aligned to `predictors[config.t_k:]` (i.e. length
    len(predictors) - t_k), in the *original* (un-normalized) target scale."""
    x, _ = make_windows(predictors, np.zeros(len(predictors), dtype=np.float32), config.t_k)
    x_n = (x - stats["x_mean"]) / stats["x_std"]
    model.eval()
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_n.astype(np.float32))).numpy()
    return pred_n * stats["y_std"] + stats["y_mean"]


def grid_search_dcnn(
    predictors: np.ndarray, target: np.ndarray, max_configs: int | None = 20
) -> tuple[DCNN, DCNNConfig, dict]:
    """Exhaustive Table IV grid search, selecting the config with lowest
    Theil's U1 (Eq 5). Opt-in only (see module docstring); `max_configs`
    caps the search since the full grid is thousands of points."""
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    if max_configs is not None:
        combos = combos[:max_configs]

    best: tuple[DCNN, DCNNConfig, dict] | None = None
    for combo in combos:
        cfg = DCNNConfig(**dict(zip(keys, combo)))
        model, stats = train_dcnn(predictors, target, cfg)
        if best is None or stats["theils_u1"] < best[2]["theils_u1"]:
            best = (model, cfg, stats)
    assert best is not None
    return best
