# CNN1D — 1D-CNN Next-Step Predictor

Implementation of: Moshe Kravchik, Asaf Shabtai — *"Detecting Cyber Attacks
in Industrial Control Systems Using Convolutional Neural Networks"*,
CPS-SPC'18 (ACM Workshop on Cyber-Physical Systems Security & Privacy,
co-located with CCS'18), `papers/cnn1d.pdf`. This is the "CNN-1D" baseline
already cited by name in the CDT paper's own comparison tables (`docs/cdt.md`).

Code: `src/methods/cnn1d/`. Entry point: `CNN1D` in
`src/methods/cnn1d/model.py` (implements `AnomalyDetectionMethod`, see
`src/methods/base.py`). The paper's own code isn't public, so nothing here
is checked against a reference implementation.

## Method

Unlike CDT (causal graph + SCM) or PbNN (physics-informed invariants +
CUSUM), this is a much more generic deep-learning approach: a 1D-CNN
predicts the **next single timestep's full sensor vector** from a window of
256 past timesteps (sequence-to-vector; the paper explicitly found this
beats sequence-to-sequence prediction). The anomaly signal is how far off
that prediction is, in a per-feature-normalized sense.

### Architecture (`network.py`, paper's Tables 1-3, best/8-layer config)

8 stacked `Conv1d → ReLU → MaxPool(2)` blocks, kernel size 2, filters
doubling every even layer (32, 32, 64, 64, 128, 128, 256, 256), then
`Flatten → Dropout → one Linear layer` producing the predicted next-step
vector (plain regression, no output activation). Pool size 2 is not stated
as a bare number in the paper, but it's *forced* by its own numbers
(`2**8 == 256`, the stated window length) — not an assumption.

**Not specified in the paper at all** (documented defaults, not paper-derived):
- Dropout rate (paper: "we used a dropout layer", no rate) — default 0.5.
- Padding scheme (kernel=2 doesn't evenly halve a length without padding) —
  right-pad by 1 then valid conv, preserving length pre-pool, consistent
  with "SAME"-style padding common in the paper's original TensorFlow
  implementation.
- Batch size (never given a number anywhere) — default 64, matching PbNN's
  convention elsewhere in this repo.
- Learning rate / decay (paper gives only a search range, 0.001-0.00001,
  with an unnamed decay schedule) — default fixed `lr=1e-3`, no decay.
- Early-stopping patience (paper: "early stopping on validation loss", no
  patience number) — default 10 epochs without improvement.

### Input & preprocessing

- Window length 256 (paper's stated value; see architecture note above for
  why this also fixes pool size).
- Min-max normalization to [0, 1], fit on training data only, applied to
  test — the paper's own stated preprocessing (distinct from CDT/PbNN's
  z-score/per-unit standardization).
- Windows are generated on demand via a `torch.utils.data.Dataset`
  (`network.py::_WindowDataset`) rather than materialized upfront — at
  window length 256 over a full-scale dataset (millions of rows), a
  pre-stacked array of all windows would be tens of GB; slicing per
  `__getitem__` keeps memory bounded to one batch at a time.
- SWaT-specific: the paper trims the first 16,000 training rows (system
  "unstable" during warm-up). This is **not applied automatically** — kept
  as an opt-in `warmup_trim` constructor kwarg (0 by default) so the method
  itself stays dataset-agnostic; pass `warmup_trim=16000` (or
  `run_benchmark.py --cnn1d-warmup-trim 16000`) to reproduce the paper's
  SWaT preprocessing specifically.
- The paper also describes an optional lagged-first-difference feature
  augmentation (its Eq 7) as improving results, but doesn't clearly confirm
  it's part of the headline configuration — treated as opt-in and **not
  implemented**, since it's ambiguous whether it belongs in the config being
  reproduced.

### Training

Plain MSE loss (Eq 11), Adam optimizer, 20% of training data held out for
validation/early-stopping (paper's stated split), up to 100 epochs.

### Scoring (`scoring.py`, exact formulas, Eqs 1-4)

- Eq 1: `e_t = |y_t - yhat_t|` (per-feature absolute prediction error).
- Eq 2: `z_e_t = |e_t - mu_e| / sigma_e`. The paper says mu_e/sigma_e are
  "calculated over all of the data" — ambiguous train-only vs. train+test;
  train-only is used (the standard non-leaking choice). More specifically,
  they're computed from the model's **held-out validation split**, not the
  data it was directly fit on: in-sample error is systematically smaller
  than out-of-sample error for a network this size relative to the dataset
  sizes here, so calibrating against in-sample error made ordinary held-out
  data — including normal test rows — look anomalous purely from the
  generalization gap, not genuine deviation. This was caught during
  smoke-testing (every row was flagged) before switching to validation-split
  statistics.
- **Deviation, also found during smoke-testing**: `max_features` is taken
  over *continuous* columns only (same `nunique() <= 5` heuristic CDT's SCM
  uses to tell sensors from actuators), not literally every column as the
  paper describes ("no special treatment" for discrete columns). Near-binary
  actuator columns are trivially easy to predict, so their calibration-period
  error std is tiny, and any single state change during test then produces
  an enormous z-score that dominates the max regardless of the continuous
  sensors — confirmed empirically (one such column alone drove the max on
  ~49% of test rows in a real run before this fix).
- Eq 3: `max_features(z_e_t) > T` — per-timestep breach flag.
- Eq 4: alert at `t` only if **every** step in the trailing window `[t-W, t]`
  breached `T` (`scoring.py::windowed_and_alerts`) — an AND rule, stricter
  in kind than PbNN's CUSUM count-based partial-violation rule or CDT's
  majority-vote rule. `CNN1D.score()` returns the continuous
  `max_features(z_e_t)` value (not the windowed AND result) for the
  harness's threshold-free comparison.
- **`T` and `W` are never given final numeric values, only grid-search
  ranges** (`T`: 1.8-3.0, `W`: 50-300 "seconds"). Unlike CDT/PbNN's other
  paper-range gaps, a fixed literal default for `T` turned out not to be
  viable here: this implementation's actual `z_e_t` scale (a function of the
  exact network/training details, several of which are themselves
  undocumented defaults) doesn't match the paper's assumed 1.8-3.0 range —
  real SWaT runs show normal-row z-scores commonly in the 5-15 range. So
  **`self.threshold_` defaults to a calibrated quantile of held-out
  validation scores** (`threshold_quantile=0.99`, the same pattern CDT/PbNN
  already use), not a literal `T`. Pass `threshold=2.4` (`DEFAULT_T`, the
  range's midpoint) to force the paper's literal framing instead. `W=100`
  (`DEFAULT_W`) remains a documented midpoint default for
  `windowed_and_alerts`, which isn't on the main `score()`/`predict()` path.

## Two configurations (both implemented, per the paper's own Table 3)

- **`ensemble="combined"`** (default): one model over every sensor jointly.
  Dataset-agnostic by construction — no hardcoded tag names or subsystem
  groupings needed, unlike PbNN. Paper's reported F1 on SWaT: 0.767
  (attack-based) / 0.871 (record-based).
- **`ensemble="per_stage"`**: one model per process stage/subsystem
  (`stages.py`), scores combined via `max` across stages — the continuous
  analog of the paper's OR-at-the-alert-level rule for its 5-stage SWaT
  ensemble, which is the paper's actual **best-reported result** (F1 0.886
  attack-based / 0.860 record-based) — better than the combined model.
  SWaT's stages are its native P1-P6 process stages (reusing the same
  regex as `scripts/eda_swat.py`'s stage grouping); WADI/HAI/Z24 have no
  equivalent table in this paper at all, so they fall back to the same
  subsystem-prefix heuristic already used for PbNN's invariants
  (`src/methods/subsystem_grouping.py`, shared between both methods to
  avoid duplicating this a second time) — this repo's own extension, not
  from the paper.

`causal_graph()` is intentionally left as the base-class default (`None`):
this method predicts every feature from every other feature jointly via a
shared network, so there's no discovered sparse dependency structure to
report the way CDT/PbNN have — returning a trivial complete graph would
overstate what the method actually produces.

## Datasets the paper itself evaluates

**SWaT only** (like PbNN) — no WADI/HAI. Its own SWaT record counts
(496,800 normal / 449,919 attack, 36 attacks) don't match this repo's
`datasets/raw/swat` (1,387,098 normal / 54,621 attack, see `docs/README`
caveats on the SWaT repackaging) — likely a different SWaT release, so
direct numeric comparison to the paper's own Table 3 numbers should be
treated as approximate, same caveat as CDT/PbNN's paper-vs-repo comparisons
already documented in the root `README.md`.

## Reported results (paper's Table 3, for later comparison)

| Configuration | Precision | Recall | F1 |
|---|---|---|---|
| Combined model, record-based | 0.968 | 0.791 | 0.871 |
| Combined model, attack-based | 0.958 | 0.639 | 0.767 |
| Per-stage ensemble, record-based | 0.867 | 0.854 | 0.860 |
| **Per-stage ensemble, attack-based (headline)** | **0.912** | **0.861** | **0.886** |

"Attack-based" credits a whole labeled attack as detected if the alert
intersects it (plus an unspecified extension period) — conceptually close
to this repo's point-adjusted (`_pa`) metrics, but not identical (the paper
counts discrete attacks, not rows).

## Usage

```python
from data.swat import load_swat
from methods.cnn1d.model import CNN1D

dataset = load_swat(nrows=20000)
method = CNN1D()                                   # combined model, no warm-up trim
# method = CNN1D(ensemble="per_stage", warmup_trim=16000)  # paper's SWaT headline config
method.fit(dataset.train)

scores = method.score(dataset.test)          # continuous max_features(z_e_t) per row
preds = method.predict(dataset.test)         # 0/1, via method.threshold_ (calibrated, or literal T if passed)

attack_row = int(dataset.test_labels.nonzero()[0][0])
method.root_cause(dataset.test, attack_row)  # [(feature, z_score), ...] from the worst-scoring stage
```

## Known limitations

- Window length 256 means the first 256 rows of any `score()`/`predict()`
  call have no real prediction to compare against (padded with the first
  real score, matching the convention `CDT`/`PbNN` already use for their
  own lag warm-up).
- The optional lag-difference feature augmentation (Eq 7) isn't implemented
  (see "Input & preprocessing" above).
- `per_stage` ensemble's WADI/HAI/Z24 grouping is a heuristic extension, not
  the paper's own domain knowledge (same caveat as PbNN's invariants) —
  expect it to be less faithful there than on SWaT.
- **WADI specifically shows poor normal/attack separation** even after the
  fixes above (real run: normal-row score median 16.7 vs. attack-row median
  19.7 — barely separated, vs. SWaT's ~46x gap). Diagnosed to a few
  "continuous" WADI sensors (`2_DPIT_001_PV`, `2_FIC_401_CO`,
  analyzer/control-output tags) that are technically above the discrete-
  column cardinality cutoff but still low-variance/noisy enough to dominate
  the max score without being genuinely predictive of attacks — one pair of
  columns alone accounted for the argmax on over 80% of test rows in a real
  run. This mirrors the same "SWaT works, WADI/HAI are much harder"
  pattern already documented for CDT and PbNN, and is treated the same way
  here: a real, dataset-driven limitation rather than something patched
  further with more masking heuristics.
- Training an 8-layer CNN per stage (up to 6+ separate networks in
  `per_stage` mode) is more compute than CDT's discovery or a single PbNN
  invariant; budget accordingly on WADI/HAI's wider column counts.
