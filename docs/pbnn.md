# PbNN — Physics-based Neural Network

Implementation of: M.R. Gauthama Raman & Aditya P. Mathur — *"A Hybrid
Physics-Based Data-Driven Framework for Anomaly Detection in Industrial
Control Systems"*, IEEE Trans. Systems, Man, and Cybernetics: Systems, vol.
52, no. 9, Sept 2022,
`papers/A_Hybrid_Physics-Based_Data-Driven_Framework_for_Anomaly_Detection_in_Industrial_Control_Systems.pdf`.

Code: `src/methods/pbnn/`. Entry point: `PbNN` in
`src/methods/pbnn/model.py` (implements `AnomalyDetectionMethod`, see
`src/methods/base.py`).

The paper only evaluates SWaT. It combines P&ID-derived "process
invariants" (small groups of physically related sensors/actuators) with one
DCNN per invariant, and flags anomalies via CUSUM on each DCNN's residuals.

## Pipeline

`PbNN.fit()`:

1. **Build invariants** (`invariants.py`) for the training columns.
2. **Train one DCNN per invariant** (`dcnn.py`) to predict its target
   sensor from its predictor sensors.
3. **Fit CUSUM parameters** (`cusum.py`) from each invariant's
   training-period residuals.

`PbNN.score()` re-runs each invariant's DCNN + CUSUM over the test set and
averages the per-invariant CUSUM violation-fraction into one continuous
score per timestep (the paper never defines how to combine per-invariant
alerts into a single detector-level output — this is a documented choice,
not from the paper).

### Process invariants (Table III)

The paper only tabulates invariants for **SWaT**; `invariants.py` hardcodes
that table (used automatically whenever enough SWaT tag names are present
in the training columns):

| Invariant | Target | Predictors |
|---|---|---|
| I1 | `FIT101` | `FIT101`, `MV101` |
| I2 | `LIT101` | `FIT101`, `MV101`, `FIT201` |
| I3 | `FIT201` | `LIT101`, `MV101`, `P601`, `P101` |
| I4 | `LIT301` | `FIT201`, `MV201`, `P101` |
| I5 | `FIT301`* | `LIT301`, `FIT201` |
| I6 | `LIT401`* | `LIT301`, `MV201`, `P301` |

`*` Table III's target cell is blank/merged in the source PDF for I5/I6 —
these two targets are a best-effort, physically-motivated completion (I5:
`FIT301` is the only sensor among its listed predictors not already a
predictor elsewhere upstream; I6: `LIT401` is the natural downstream level
sensor fed by `MV201`/`P301`), not a verbatim transcription. A 7th
invariant (I7) is plotted in the paper's Fig. 4 but never defined in Table
III at all — it is dropped rather than guessed. Section IV-A's own worked
example (`x1(t+1) = f(x1(t), x2(t-t_k), x3(t-t_k))` with LIT101 as the
dependent variable) matches I2 above and was used to sanity-check the
transcription.

**WADI and HAI** have no equivalent table in this paper — `infer_invariants()`
is this codebase's own extension of the *method* (not from the paper):
group columns by their subsystem-prefix tag convention (WADI: `1_`, `2_`,
`3_`; HAI: `P1_`, `P2_`, `P3_`, `P4_`), then within each subsystem group
treat sensor-like tags (matched by substring, e.g. `FIT`/`LIT`/`AIT`/`PV`)
as targets and other same-subsystem tags as predictors.
`PbNN.fit()`/`build_invariants()` picks SWaT's hardcoded table when at
least 3 of its tags are present, otherwise falls back to
`infer_invariants()`.

### DCNN (Section III-B, Eqs 4–5, Table IV)

One 1D-conv network per invariant, predicting its target one step ahead
from a `t_k`-step window of its predictors:

- Architecture: `n_blocks` conv+ReLU blocks (the paper's "DCNN1/2/3" =
  1/2/3 blocks), then one global-average-pool, then a 2-layer FC head.
  Because `t_k` is short (2–10 steps), repeated intermediate max-pooling
  would collapse the sequence to length 0–1 after a couple of blocks — a
  single pool before the FC head is used instead of a literal "conv+pool"
  block per block, standard practice for short 1D sequences.
- Loss: plain MSE (Eq 4).
- Model selection: Theil's U1 (Eq 5), `sqrt(mean((y-yhat)^2)) /
  sqrt(mean(y^2))`, used by the optional grid search below.
- **Not specified in the paper at all**: kernel size, activation function,
  optimizer, pooling type. Defaults used: kernel size 3, ReLU, Adam —
  standard choices, not paper-derived.

**Table IV hyperparameter grid** (filters ∈ {32,64,128}, hidden units ∈
[5,100], `t_k` ∈ [2,10], epochs ∈ {10,20,50}, batch size ∈ {32,64}, ×3
architectures) is implemented in full in `grid_search_dcnn()`, but is
**opt-in**, not the default — an exhaustive search per invariant is
computationally infeasible on a CPU-only benchmark run. The default path
(`DEFAULT_CONFIG`) fixes one reasonable point in the grid:

| Hyperparameter | Default |
|---|---|
| `n_blocks` | 2 (DCNN2) |
| `filters` | 64 |
| `hidden_units` | 32 |
| `t_k` | 5 |
| `epochs` | 20 |
| `batch_size` | 64 |

### CUSUM (Eqs 6, 7, 9, 10)

- Residual: `r(t) = x(t) - xhat(t)` from the invariant's DCNN.
- Two-sided recursive CUSUM: `P(t) = max(0, P(t-1) + r(t) - target - b)`,
  `N(t) = min(0, N(t-1) + r(t) - target + b)`.

  **Note on the paper's transcription:** Eqs 6–7 as printed omit the
  recursive `P(t-1)`/`N(t-1)` term (`P(t) = max(0, r(t)-target-b)`, i.e.
  purely point-wise). This is almost certainly a rendering artifact of the
  PDF, not the paper's intended algorithm — the surrounding prose
  explicitly calls this a "cumulative sum" technique, and a point-wise
  `max(0, ...)` isn't cumulative. The standard recursive CUSUM (implemented
  here) is the well-established form the paper's own terminology points to.
- Violation flag (Eq 9): `f(t) = 1 if P(t)>UCL or N(t)<LCL else 0`.
- Windowed alert (Eq 10): alert if the violation count in the trailing
  `S_w` window exceeds `T_w`.
- **Not given numeric values anywhere** ("computed empirically using the
  values of P and N"): slack `b`, `UCL`/`LCL`, `S_w`, `T_w`. Documented
  defaults, all fit from training-period residuals:

| Parameter | Default | Basis |
|---|---|---|
| `b` | `0.5 * std(train residuals)` | slack |
| `UCL` / `LCL` | `mean(P/N) +/- 3*std(P/N)` | control limits |
| `S_w` | 10 | trailing window size |
| `T_w` | 3 | violation-count threshold |

### Metrics used in the paper (Eqs 11–13)

`Dr = Tp/(Tp+Fn)`, `Fr = Fp/(Fp+Tn)`, `CiF = c1*Dr - c2*Fr` with the paper's
exact weights `c1=0.4, c2=0.6` — implemented in
`src/metrics/detection.py` (shared across methods, not duplicated here).

## Usage

```python
from data.swat import load_swat
from methods.pbnn.model import PbNN

dataset = load_swat(nrows=20000)
method = PbNN()                      # or PbNN(config=DCNNConfig(...), s_w=10, t_w=3)
method.fit(dataset.train)

scores = method.score(dataset.test)
preds = method.predict(dataset.test)
graph = method.causal_graph()        # sparse predictor->target edges from the invariants
method.root_cause(dataset.test, t)   # invariants ranked by residual magnitude at row t
```

## Known limitations

- WADI/HAI/BATADAL invariants are a heuristic extension, not the paper's own
  domain knowledge — expect noticeably weaker performance there than on
  SWaT (see the benchmark results in the root `README.md`). BATADAL's tag
  naming (`L_T1`, `F_PU1`, `P_J280`, ...) doesn't match any of
  `subsystem_grouping.py`'s dataset-specific prefix patterns either, so it
  falls all the way through to `infer_invariants`'s SWaT-shaped fallback,
  which only picks up a handful of `P_J*` junction-pressure tags via an
  incidental regex match — a real run collapsed to the same "flag
  everything" pattern already seen on WADI/HAI (F1 ~0.10, false-alarm-rate
  ~1.0). Unlike TEP (dataset-agnostic by construction, no grouping needed
  at all -- see `docs/cnn1d.md`'s note on CNN1D's combined mode), PbNN's
  invariant-based design means every new SCADA-tag naming convention needs
  its own grouping heuristic to work well; BATADAL's hasn't been added.
- Full Table IV grid search (`grid_search_dcnn`) is available but not run
  by default; `PbNN`'s constructor always uses `DEFAULT_CONFIG` unless you
  build and pass a custom `DCNNConfig` yourself (grid search isn't wired
  into `PbNN.fit()` — call `dcnn.grid_search_dcnn()` directly if needed).
- Attack-table replication (the paper's specific SWaT attacks, Tables I/II)
  is out of scope — this is a method reproduction, not an evaluation-
  protocol reproduction.
