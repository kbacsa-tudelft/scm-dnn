# CDT — Causal Digital Twin

Implementation of: Homaei, Tarif, García Rodríguez, Caro, Ávila — *"Causal
Digital Twins for cyber–physical security in water systems: A framework for
robust anomaly detection"*, Machine Learning with Applications, vol. 23
(2026), `papers/1-s2.0-S2666827025002075-main.pdf`.

Code: `src/methods/cdt/`. Entry point: `CDT` in `src/methods/cdt/model.py`
(implements `AnomalyDetectionMethod`, see `src/methods/base.py`).

The paper's own code/data are not public (its Data Availability statement
says so explicitly), so nothing here is checked against a reference
implementation — this is a from-scratch reproduction from the paper text.

## Pipeline

CDT has three phases, run in `CDT.fit()`:

1. **Discovery** (`discovery.py`) — learn a causal DAG over sensor/actuator
   tags from lagged time series.
2. **SCM estimation** (`scm.py`) — fit an additive-noise structural equation
   per variable, given its parents from the discovered DAG.
3. **Scoring** (`scoring.py`) — turn per-variable structural-equation
   residuals into a single anomaly score per timestep.

Two further modules support root-cause analysis and are *not* on the
fit/score path:

- **Causal effects** (`causal_effects.py`) — do-calculus utilities (backdoor
  and frontdoor adjustment).
- **Attribution** (`attribution.py`) — Shapley-value root-cause ranking,
  used by `CDT.root_cause()`.
- **Sensitivity** (`sensitivity.py`) — Robins' hidden-confounder bias bound,
  a standalone diagnostic function, not called anywhere in `fit`/`score`.

### 1. Discovery (Algorithm 1, Eqs 10–13)

For every variable `V_i`, build lagged copies `V_i(t-1), ..., V_i(t-tau)`
(`tau=10` by default, Eq 10), then:

- **TDMI candidate filtering** (Eq 11): score every other variable's lagged
  copies by mutual information with the target, keep the top `k=10`. The
  paper cites Fraser & Swinney (1986) for TDMI but gives no closed form and
  never gives a numeric value for its "> delta" threshold — since `k`
  already caps the candidate set, no separate delta is applied. For
  tractability, a cheap vectorized correlation prefilter narrows candidates
  to a shortlist of 30 before running sklearn's k-NN mutual-information
  estimator on that shortlist.
- **Local PC skeleton test** (Eqs 12–13): for each candidate parent, test
  conditional independence via lagged partial correlation (Eq 13, Fisher
  z-test, `alpha=0.05`) against conditioning sets drawn from the target's
  *other* candidate parents, capped at conditioning-set size 3 for
  tractability.
- **Orientation**: trivial by construction — every candidate parent is
  strictly earlier in time (`lag >= 1`) than its child, so temporal
  precedence alone fixes edge direction. The paper's physical/control-logic
  orientation rules (needed only for contemporaneous edges) aren't
  implemented since only strictly-lagged edges are used.
- Per-lag edges are collapsed into one summary `nx.DiGraph` over the
  original (unlagged) variable names, keeping the shortest surviving lag as
  an edge attribute. If collapsing introduces a cycle, the longest-lag edge
  in the cycle is dropped.
- Runs on a contiguous 5,000-row subsample of `train` by default (CI-testing
  over 100k+ rows would be slow in pure Python/sklearn, and discovery time
  scales worse than linearly with row count). Configurable via
  `CDT(discovery_subsample_rows=...)` or `run_benchmark.py`'s
  `--cdt-discovery-rows` flag for a more thorough (paper-scale) pass.
  Measured on SWaT: 5,000 rows -> 37s / 17 edges, 20,000 -> 192s / 27 edges,
  50,000 -> 608s (~10 min) / 30 edges -- more rows reliably finds more
  structure (more statistical power for the CI tests), at a steep runtime
  cost. 50,000 is a reasonable "thorough" setting; expect it to take at
  least as long on WADI/HAI, which have more target variables per discovery
  pass. The default stays at 5,000 so quick/smoke-test runs remain fast --
  see "Known limitations" below.

### 2. SCM estimation (Eqs 14–16)

Each variable's structural equation is fit independently, in topological
order of the discovered DAG, using PyTorch (`nn.Linear` + Adam):

- **Continuous** variables (Eq 14): `V_i := alpha_i + sum_j beta_ij*PA_j +
  gamma_i*sum_k PA_k^2 + U_i`, `U_i ~ N(0, sigma_i^2)` — a linear layer over
  `[parents, sum(parents^2)]`, fit by MSE. Minimizing MSE for a
  fixed-variance Gaussian additive-noise model *is* the MLE solution
  (Eq 16), so no separate variance parameter is learned; `sigma_i` is
  computed from residuals after fitting.
- **Discrete/actuator** variables (Eq 15, detected as `nunique() <= 5`):
  `P(A_i=1|PA) = sigmoid(alpha_i + sum beta_ij*V_j)`, fit by BCE loss.
- Root variables (no parents) are stored as their train-period mean/std.

### 3. Scoring (Eqs 19–20, 30–31)

- **Interventional score** (Eq 19): `Score_causal(V_i,t) = |V_i(t) -
  E[V_i|do(PA(V_i)=pa_i(t))]| / sigma_{V_i|do(PA(V_i))}`. Because every DAG
  edge already points from a variable's full parent set (its own
  backdoor-blocking adjustment set by construction), `E[V_i|do(PA=pa)]`
  collapses to evaluating the fitted structural equation at the observed
  parent values — no separate adjustment-set search is needed.
- **MCAI** (Eq 20): `alpha_i = (|Descendants(V_i)| + |Ancestors(V_i)|) /
  (2n)` (exact centrality-weight formula from the paper), `MCAI(t) =
  sum_i alpha_i * Score_causal(V_i,t)`.
- **Multi-scale ensemble** (Eqs 30–31): windowed cumulative sums of MCAI at
  three scales, `tau in {5, 10, 20}` (distinct from the `tau=10` discovery
  lag — an unfortunate notation collision in the paper), z-normalized
  per-window using train statistics, combined with the paper's exact
  weights `w5=0.5, w10=0.3, w20=0.2`. This weighted combination is what
  `CDT.score()` returns.

  **Deviation:** the paper's literal Eq 31 alert rule (`sum w_r *
  1[MCAI_r(t)>theta_r] >= 2`) is dimensionally inconsistent as printed —
  the weights sum to 1.0, so a weighted sum of 0/1 indicators can never
  reach 2 — and no numeric `theta_r` is ever given. `scoring.py`'s
  `multiscale_votes()` implements an unweighted majority-vote alternative
  instead (alert if at least 2 of the 3 scales independently exceed their
  own train-calibrated threshold); the continuous weighted score above is
  used for the harness's threshold-free comparison.

### Root-cause attribution (Eqs 21–22)

`CDT.root_cause(test, t)` picks the highest weighted-score variable at row
`t` as the anomaly's proximate location, then ranks *that variable's
parents* by a Shapley-value game:

- The game `f(S)` is not defined anywhere in the paper for Eq 22 — it's
  implemented here as a SHAP-style reconstruction game: `f(S) =
  -|observed_target - fitted_equation(x_S)|`, where `x_S` uses each
  parent's *observed* value for parents in `S` and its *baseline*
  (train-mean) value otherwise. This scopes the game to one variable's
  direct parents (its candidate causes per the discovered graph) rather
  than every variable in the dataset.
- `causal_effect_do()` is the exact Eq 21 special case (one parent swapped
  in isolation, coalition size 0 vs. 1).
- `shapley_values()` is the full Eq 22 game (`phi_i = sum_S
  [|S|!(n-|S|-1)!/n!]*[f(S∪{i})-f(S)]`), approximated via Monte Carlo
  permutation sampling (300 samples) since exact Shapley is exponential in
  the number of parents.

### Sensitivity analysis (Eq 26, standalone utility)

`sensitivity.robins_bias_bound(x, y, u)` implements Robins' bound
`|CE_hat - CE_true| <= alpha_U * beta_U` for a user-supplied candidate
confounder `u`. The paper only applies this in one ad hoc case study with a
synthetic confounder — it's provided as a diagnostic function, not called
during `fit()`/`score()`.

## Default parameters (given numerically in the paper, Algorithm 1)

| Parameter | Value | Meaning |
|---|---|---|
| `alpha` | 0.05 | PC conditional-independence significance level |
| `k` | 10 | max candidate parents per variable (TDMI top-k) |
| `tau` | 10 | temporal augmentation max lag (discovery) |
| multi-scale windows | {5, 10, 20} | ensemble scoring window sizes |
| multi-scale weights | {0.5, 0.3, 0.2} | ensemble combination weights |

All are constructor kwargs on `CDT(...)`.

`discovery_subsample_rows` (default 5,000) is not a paper-given parameter --
it's this implementation's tractability knob for Phase I's row cap (see
"Discovery" above for the measured runtime/edge-count tradeoff at larger
values).

## Usage

```python
from data.swat import load_swat
from methods.cdt.model import CDT

dataset = load_swat(nrows=20000)
method = CDT()                       # or CDT(tau=10, k=10, alpha=0.05, ...)
# method = CDT(discovery_subsample_rows=50000)  # more thorough discovery, ~10min/dataset
method.fit(dataset.train)

scores = method.score(dataset.test)          # continuous anomaly score per row
preds = method.predict(dataset.test)         # 0/1, via method.threshold_
graph = method.causal_graph()                # nx.DiGraph over dataset.columns

attack_row = int(dataset.test_labels.nonzero()[0][0])
method.root_cause(dataset.test, attack_row)  # [(parent_label, shapley_value), ...]
```

## Known limitations

- Discovery subsamples training data to 5,000 rows by default; very rare
  causal edges that only manifest outside that window may be missed.
  Raise `discovery_subsample_rows` (or `run_benchmark.py --cdt-discovery-rows`)
  for a more thorough pass, at a steep runtime cost -- see "Discovery" above.
- Graph-accuracy metrics (SHD, edge precision/recall — see
  `src/metrics/graph.py`) are only meaningful when a ground-truth graph
  shares node names with the discovered graph. Only HAI ships a
  ground-truth graph (`datasets/raw/hai/graph/boiler/*.json`), and it uses
  physical-component ids (`TK01`, `PP01A`, ...) rather than HAI's dataset
  tag names (`P1_FT01`, ...), so overlap — and therefore these metrics —
  will be near-zero in practice. See `src/data/hai.py`.
- No general do-calculus identification algorithm (arbitrary backdoor-set
  search) is implemented — only the SCM-propagation shortcut described
  above, plus standalone single-adjustment-set backdoor/frontdoor
  functions in `causal_effects.py` for reference/completeness.
