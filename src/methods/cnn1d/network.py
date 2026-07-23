"""1D-CNN prediction network (Kravchik & Shabtai 2018, CPS-SPC'18, "Detecting
Cyber Attacks in Industrial Control Systems Using Convolutional Neural
Networks").

Sequence-to-vector prediction (Eq 8-10): predicts the next single timestep's
full feature vector from a window of `WINDOW_LEN` past timesteps -- the
paper explicitly found this outperforms sequence-to-sequence prediction.
Architecture is the paper's best/headline 8-layer configuration (Tables 1-3):
8 Conv1d+ReLU+MaxPool(2) blocks, filters doubling every even layer
(32,32,64,64,128,128,256,256), kernel_size=2. Pool size of 2 is not stated as
a bare number anywhere, but it's *forced* by the paper's own numbers (2**8 ==
256 == the stated window length), not a guess. Flatten -> Dropout -> one FC
-> linear output (plain regression, no output activation).

Genuinely underspecified in the paper (documented defaults, not paper-given):
dropout rate ("we used a dropout layer", no rate -- default 0.5, standard),
padding scheme (kernel_size=2 doesn't evenly halve a length without padding;
right-pad-by-1 then valid conv preserves length pre-pool, consistent with
"SAME"-style padding common in the paper's original TensorFlow
implementation -- not stated explicitly there either), batch size (never
given a number anywhere in the paper -- default 64, matching this repo's
other methods' convention), learning rate/decay (paper gives only a search
range 0.001-0.00001 with an unspecified decay schedule -- default fixed
lr=1e-3, no decay), early-stopping patience (paper says "early stopping on
validation loss" with no patience number -- default 10 epochs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

WINDOW_LEN = 256
FILTERS = [32, 32, 64, 64, 128, 128, 256, 256]
KERNEL_SIZE = 2
POOL_SIZE = 2


@dataclass
class CNN1DConfig:
    window_len: int = WINDOW_LEN
    dropout: float = 0.5
    batch_size: int = 64
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    val_frac: float = 0.2


DEFAULT_CONFIG = CNN1DConfig()


class PredictorNet(nn.Module):
    def __init__(self, n_features: int, config: CNN1DConfig = DEFAULT_CONFIG):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for filters in FILTERS:
            layers += [
                nn.ConstantPad1d((0, KERNEL_SIZE - 1), 0.0),  # right-pad: preserve length pre-pool
                nn.Conv1d(in_ch, filters, kernel_size=KERNEL_SIZE),
                nn.ReLU(),
                nn.MaxPool1d(POOL_SIZE),
            ]
            in_ch = filters
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_ch, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)            # (batch, 256, 1) after 8 halvings of window_len=256
        h = h.squeeze(-1)            # (batch, 256)
        h = self.dropout(h)
        return self.fc(h)            # (batch, n_features)


class _WindowDataset(Dataset):
    """On-demand sliding windows -- materializing every window of a
    million-plus-row, ~50-130-feature dataset at `window_len=256` upfront
    would be tens of GB; slicing per-`__getitem__` keeps memory bounded to
    one batch at a time."""

    def __init__(self, data: np.ndarray, window_len: int):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.window_len = window_len

    def __len__(self) -> int:
        return len(self.data) - self.window_len

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.data[i : i + self.window_len].T  # (n_features, window_len)
        y = self.data[i + self.window_len]         # (n_features,)
        return x, y


def train_predictor(
    train_data: np.ndarray, config: CNN1DConfig = DEFAULT_CONFIG, seed: int = 0
) -> PredictorNet:
    """Trains on `train_data` (n_samples, n_features), holding out
    `config.val_frac` for early stopping on validation MSE (Eq 11 loss, and
    "20% of training data withheld for validation" per the paper)."""
    torch.manual_seed(seed)
    n_features = train_data.shape[1]
    split = int(len(train_data) * (1 - config.val_frac))
    train_ds = _WindowDataset(train_data[:split], config.window_len)
    val_ds = _WindowDataset(train_data[split:], config.window_len)

    model = PredictorNet(n_features, config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for _epoch in range(config.max_epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                val_losses.append(nn.functional.mse_loss(model(x), y).item() * len(x))
        val_loss = sum(val_losses) / len(val_ds) if len(val_ds) else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: PredictorNet, data: np.ndarray, config: CNN1DConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Returns predictions aligned to `data[window_len:]` (length
    len(data) - window_len), one row per predicted next-timestep vector."""
    ds = _WindowDataset(data, config.window_len)
    loader = DataLoader(ds, batch_size=max(config.batch_size, 256), shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _y in loader:
            preds.append(model(x).numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0, data.shape[1]))
