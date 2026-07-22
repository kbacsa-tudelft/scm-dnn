# Causal Learning Benchmark Harness

Compares two causal/physics-informed anomaly detection methods for
industrial control systems (ICS), reproduced from the papers in
`papers/`, on the SWaT, WADI and HAI datasets in `datasets/`.

| Method | Paper | Docs |
|---|---|---|
| **CDT** (Causal Digital Twin) | Homaei et al., *Machine Learning with Applications* (2026) | [`docs/cdt.md`](docs/cdt.md) |
| **PbNN** (Physics-based Neural Network) | Raman & Mathur, *IEEE T-SMC* (2022) | [`docs/pbnn.md`](docs/pbnn.md) |

Exploratory data analysis for each dataset (schema, data-quality issues,
distributions, correlations, class balance, PCA separability) lives in
[`reports/eda_swat.md`](reports/eda_swat.md),
[`reports/eda_wadi.md`](reports/eda_wadi.md) and
[`reports/eda_hai.md`](reports/eda_hai.md) — read these before trusting any
benchmark number, since they surface real data-quality issues (e.g. WADI has
a genuine ~3-day mid-collection outage and an inconsistent date format in
its raw training file) that the modeling pipeline works around rather than fixes.

Both are from-scratch reproductions (neither paper's own code is public).
Where a paper is genuinely ambiguous or missing a numeric value, the code
uses a documented default — see the method docs above for the full list of
those judgment calls before trusting any specific number.

## Repository layout

```
papers/                     the two source papers (PDF)
datasets/                   raw dataset archives (checked in)
  swat.zip / wadi.zip / hal.zip
datasets/raw/                <- extracted CSVs/JSON live here (gitignored, see Setup)

src/
  data/                     one loader per dataset -> common ICSDataset shape
    base.py                   ICSDataset dataclass + shared cleaning helpers
    swat.py / wadi.py / hai.py

  methods/
    base.py                  AnomalyDetectionMethod interface every method implements
    cdt/                     Causal Digital Twin -- see docs/cdt.md
      discovery.py             Phase I: temporal-lag causal graph discovery
      scm.py                   Phase II: additive-noise SCM fit (PyTorch)
      scoring.py                Phase III: interventional score, MCAI, multi-scale ensemble
      causal_effects.py        backdoor/frontdoor do-calculus utilities
      attribution.py           Shapley-based root-cause ranking
      sensitivity.py           Robins' hidden-confounder bias bound (standalone)
      model.py                  CDT class tying the phases together
    pbnn/                    Physics-based Neural Network -- see docs/pbnn.md
      invariants.py             SWaT process-invariant table + WADI/HAI auto-inference
      dcnn.py                   per-invariant 1D-CNN (PyTorch) + Theil's U1 + grid search
      cusum.py                  CUSUM residual detector
      model.py                  PbNN class tying it together

  metrics/
    detection.py             precision/recall/F1, point-adjustment, detection rate,
                              false alarm rate, conflict-index factor
    graph.py                 structural Hamming distance, edge precision/recall/F1

  benchmark/
    runner.py                 runs every (method, dataset) pair, catches per-run errors
    report.py                  formats/saves results as a table (CSV + JSON)

  eda/                      shared EDA library used by scripts/eda_*.py
    raw_loaders.py            per-dataset raw CSV parsing, no cleaning (see docs below)
    stats.py                  missingness, class balance, correlation, PCA-ready helpers
    plots.py                  every figure generator (histograms, heatmaps, PCA, graph, ...)
    style.py                  shared color palette / matplotlib rcParams
    report.py                  minimal markdown builder

scripts/
  eda_swat.py / eda_wadi.py / eda_hai.py   generate reports/eda_<name>.md + figures/<name>/*.png

run_benchmark.py            CLI entry point
requirements.txt
tests/                      pytest suite (data loaders, metrics, end-to-end integration)
results/                    benchmark output lands here (gitignored)
reports/                    EDA reports (eda_swat.md, eda_wadi.md, eda_hai.md) + figures/
```

## Setup

```bash
pip install -r requirements.txt

# Extract the datasets once (datasets/raw/ is gitignored -- regenerate from the zips)
cd datasets
mkdir -p raw/swat raw/wadi raw/hai
unzip -o -q swat.zip -d raw/swat
unzip -o -q wadi.zip -d raw/wadi
unzip -o -q hal.zip -x "haiend-23.05/*" -d raw/hai   # haiend-23.05 excluded: ~800MB, unused by either method
cd ..
```

(`haiend-23.05` is a separate, much larger HAI variant not referenced by
either paper; skip it unless you specifically need it, then unzip without
the `-x` filter.)

## Running the benchmark

```bash
# Quick smoke test (small row cap, a couple minutes)
python run_benchmark.py --datasets swat --methods cdt,pbnn --nrows 20000

# Everything, full data (slow -- see "Performance" below)
python run_benchmark.py

# A specific subset
python run_benchmark.py --datasets wadi,hai --methods cdt --out results/cdt_only
```

Flags: `--datasets` and `--methods` are comma-separated subsets of
`swat,wadi,hai` / `cdt,pbnn`; `--nrows` caps rows read per CSV (omit for
full data); `--out` sets the output directory (default `results/`).

Output: `results.csv` and `results.json`, one row per (method, dataset)
pair, with precision/recall/F1 (raw and point-adjusted), detection rate,
false alarm rate, the paper's conflict-index factor, graph metrics (SHD,
edge P/R/F1 — only non-null when the dataset ships a ground-truth graph,
currently HAI only) and fit/score timing. A run that errors is still
reported (with the error message in the `error` column) rather than
aborting the whole comparison.

### Using a method directly (bypassing the CLI)

```python
import sys; sys.path.insert(0, "src")
from data.swat import load_swat
from methods.cdt.model import CDT
from methods.pbnn.model import PbNN

dataset = load_swat(nrows=20000)
method = CDT()               # or PbNN()
method.fit(dataset.train)
scores = method.score(dataset.test)          # continuous anomaly score per row
preds = method.predict(dataset.test)         # 0/1, thresholded via method.threshold_
```

See `docs/cdt.md` / `docs/pbnn.md` for each method's full interface
(`causal_graph()`, `root_cause()`, hyperparameters).

## Exploratory data analysis

```bash
python scripts/eda_swat.py
python scripts/eda_wadi.py
python scripts/eda_hai.py
```

Each script reads the full dataset (no row cap; ~15-80s per dataset) and
(re)writes `reports/eda_<name>.md` plus its figures under
`reports/figures/<name>/`. Data-quality findings (missingness, constant/empty
columns, label conventions, timestamp gaps) are computed from the raw CSVs
directly via `src/eda/raw_loaders.py`, deliberately bypassing `src/data/*.py`'s
cleaning so the reports show what's really in the source files; everything
else (distributions, correlation, time series, class balance, PCA) uses the
same cleaned train/test split the benchmark harness consumes.

## Tests

```bash
python -m pytest tests/ -q
```

- `test_data_loaders.py` — each dataset loads into the common shape, no NaNs leak through.
- `test_metrics.py` — detection/graph metric formulas against hand-computed cases.
- `test_integration.py` — both methods run end-to-end through the actual `benchmark.runner`, not just in isolation.

Full suite takes a few minutes (it fits real models on real data, not mocks).

## Performance

Everything runs on CPU (no GPU in this environment). Fit times scale with
row count; PbNN in particular trains one DCNN per invariant with multiple
epochs over mini-batches, so it's the slower of the two on large inputs
(HAI's full training files, uncapped, took several minutes in testing).
Use `--nrows` while iterating; drop it only when you want real numbers.

## Known caveats (see the method docs for the full list)

- **Ground-truth causal graphs** exist only for HAI's boiler subsystem
  (`datasets/raw/hai/graph/boiler/*.json`), and use physical-component ids
  that don't overlap with HAI's dataset tag columns — so SHD/edge-P/R for
  HAI will be near-zero regardless of method quality. SWaT and WADI have no
  ground-truth graph at all, so `causal_graph()` output for those can only
  be inspected qualitatively, not scored.
- **WADI/HAI process invariants for PbNN** are an auto-inferred extension
  (subsystem-prefix grouping), not the paper's own P&ID knowledge — expect
  weaker PbNN results there than on SWaT.
- **SWaT's `attack.csv`** in this dataset repackaging is 100% attack-labeled
  (a paper-specific extraction of attack windows, not the canonical mixed
  test period); `data/swat.py` builds a leak-free test set by holding out a
  tail slice of `normal.csv` and concatenating it with `attack.csv` rather
  than using the shipped `merged.csv` directly (see that file's docstring).
