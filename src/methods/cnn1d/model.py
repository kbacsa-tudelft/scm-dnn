"""CNN1D (Kravchik & Shabtai 2018): 1D-CNN next-timestep predictor(s),
anomaly score = per-feature z-scored prediction error.

Two configurations, both described in the paper (Table 3): a single
"combined" model over every feature (dataset-agnostic, no grouping needed),
or a "per_stage" ensemble of one model per process stage/subsystem, scores
combined by taking the max across stages (the continuous analog of the
paper's OR-at-the-alert-level combination rule for its 5-stage SWaT
ensemble -- its actual best-reported result, F1 0.886 vs. 0.767 combined).
Stage/subsystem grouping for datasets other than SWaT is this repo's own
extension (see `stages.py`), not from the paper.

`causal_graph()` is intentionally left as the base class default (None):
this method predicts every feature from every other feature jointly via a
shared network, so there's no discovered sparse dependency structure to
report the way CDT/PbNN have -- returning a trivial complete graph would
overstate what the method actually produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import AnomalyDetectionMethod
from .network import CNN1DConfig, DEFAULT_CONFIG, predict, train_predictor
from .scoring import DEFAULT_T, DEFAULT_W, ErrorStats, fit_error_stats, per_timestep_score
from .stages import stage_groups

DISCRETE_MAX_CARDINALITY = 5  # matches src/methods/cdt/scm.py's convention


class CNN1D(AnomalyDetectionMethod):
    def __init__(
        self,
        config: CNN1DConfig = DEFAULT_CONFIG,
        threshold: float | None = None,
        threshold_quantile: float = 0.99,
        window: int = DEFAULT_W,
        ensemble: str = "combined",
        warmup_trim: int = 0,
        seed: int = 0,
    ):
        if ensemble not in ("combined", "per_stage"):
            raise ValueError(f"ensemble must be 'combined' or 'per_stage', got {ensemble!r}")
        self.config = config
        # T has no paper-given fixed value, only a search range (1.8-3.0) --
        # and empirically, this implementation's z_e_t scale doesn't match
        # that range (real SWaT run: normal-row z-scores commonly land in
        # the 5-15 range, not under 3). Rather than hardcode a paper-range
        # constant that's miscalibrated for this implementation's actual
        # score distribution, the default calibrates a threshold from the
        # held-out validation scores (same `threshold_quantile` pattern
        # CDT/PbNN already use). Pass `threshold=` directly to force a
        # literal value (e.g. DEFAULT_T=2.4, the range's midpoint) instead.
        self.threshold = threshold
        self.threshold_quantile = threshold_quantile
        self.window = window
        self.ensemble = ensemble
        # Paper trims SWaT's first 16,000 training rows (system "unstable"
        # during warm-up) -- dataset-specific, so left as an opt-in kwarg
        # (0 = no trim) rather than auto-detected, keeping the method itself
        # dataset-agnostic. Pass warmup_trim=16000 when benchmarking on SWaT
        # to reproduce the paper's own preprocessing.
        self.warmup_trim = warmup_trim
        self.seed = seed

    def _groups(self, columns: list[str]) -> dict[str, list[str]]:
        return {"all": list(columns)} if self.ensemble == "combined" else stage_groups(list(columns))

    def fit(self, train: pd.DataFrame) -> None:
        if self.warmup_trim:
            train = train.iloc[self.warmup_trim :]

        self.col_min_ = train.min()
        self.col_max_ = (train.max() - self.col_min_).replace(0.0, 1.0)
        train_norm = (train - self.col_min_) / self.col_max_

        self.groups_ = self._groups(list(train.columns))
        self.models_: dict[str, object] = {}
        self.stats_: dict[str, ErrorStats] = {}
        self.score_masks_: dict[str, np.ndarray] = {}
        val_scores_by_group = {}
        for name, cols in self.groups_.items():
            data = train_norm[cols].to_numpy()
            model = train_predictor(data, self.config, seed=self.seed)

            # Error stats (mu_e/sigma_e, Eq 2) come from the held-out
            # validation split, not the data the model was fit on: in-sample
            # error is systematically smaller (the model was optimized
            # against exactly that loss), so calibrating against it makes
            # ordinary out-of-sample data -- including normal test rows --
            # look anomalous purely from the train/generalization gap, not
            # real deviation. Same split fraction `train_predictor` used
            # internally for early stopping.
            split = int(len(data) * (1 - self.config.val_frac))
            val_data = data[split:]
            pred = predict(model, val_data, self.config)
            actual = val_data[self.config.window_len :]
            self.models_[name] = model
            self.stats_[name] = fit_error_stats(actual, pred)
            # See scoring.py's module docstring: near-binary actuator
            # columns dominate max_features(z_e_t) trivially, so the
            # aggregate score is restricted to continuous columns.
            self.score_masks_[name] = np.array([train[c].nunique() > DISCRETE_MAX_CARDINALITY for c in cols])
            val_scores_by_group[name] = per_timestep_score(actual, pred, self.stats_[name], self.score_masks_[name])

        if self.threshold is not None:
            self.threshold_ = self.threshold
        else:
            combined_val_scores = np.max(list(val_scores_by_group.values()), axis=0)
            self.threshold_ = float(np.quantile(combined_val_scores, self.threshold_quantile))

    def _group_scores(self, test: pd.DataFrame) -> dict[str, np.ndarray]:
        test_norm = (test - self.col_min_) / self.col_max_
        n = len(test)
        scores = {}
        for name, cols in self.groups_.items():
            data = test_norm[cols].to_numpy()
            pred = predict(self.models_[name], data, self.config)
            actual = data[self.config.window_len :]
            s = per_timestep_score(actual, pred, self.stats_[name], self.score_masks_[name])
            pad = np.full(self.config.window_len, s[0] if len(s) else 0.0)
            scores[name] = np.concatenate([pad, s])[:n]
        return scores

    def score(self, test: pd.DataFrame) -> np.ndarray:
        group_scores = self._group_scores(test)
        return np.max(list(group_scores.values()), axis=0)

    def root_cause(self, test: pd.DataFrame, t: int, top_k: int = 5) -> list[tuple[str, float]] | None:
        if t < self.config.window_len:
            return None
        group_scores = self._group_scores(test)
        worst_group = max(group_scores, key=lambda g: group_scores[g][t])
        cols = self.groups_[worst_group]

        test_norm = (test - self.col_min_) / self.col_max_
        # window_len+1 rows so predict() has exactly one full window (the
        # first window_len rows) plus its target (row t) to score.
        window = test_norm[cols].to_numpy()[t - self.config.window_len : t + 1]
        pred = predict(self.models_[worst_group], window, self.config)
        if len(pred) == 0:
            return None
        actual = window[self.config.window_len :]
        z = np.abs(actual - self.stats_[worst_group].mean) / self.stats_[worst_group].std
        mask = self.score_masks_[worst_group]
        candidates = [(c, v) for c, v, keep in zip(cols, z[-1], mask) if keep]
        ranked = sorted(candidates, key=lambda kv: -kv[1])
        return [(c, float(v)) for c, v in ranked[:top_k]]
