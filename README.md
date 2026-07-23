# Causal Learning Benchmark Harness

Compares causal/physics-informed/deep-learning anomaly detection methods for
industrial control systems (ICS), reproduced from the papers in
`papers/`, on the SWaT, WADI and HAI datasets in `datasets/`.

| Method | Paper | Docs |
|---|---|---|
| **CDT** (Causal Digital Twin) | Homaei et al., *Machine Learning with Applications* (2026) | [`docs/cdt.md`](docs/cdt.md) |
| **PbNN** (Physics-based Neural Network) | Raman & Mathur, *IEEE T-SMC* (2022) | [`docs/pbnn.md`](docs/pbnn.md) |
| **CNN1D** (1D-CNN next-step predictor) | Kravchik & Shabtai, *CPS-SPC'18* | [`docs/cnn1d.md`](docs/cnn1d.md) |

Exploratory data analysis for each dataset (schema, data-quality issues,
distributions, correlations, class balance, PCA separability) lives in
[`reports/eda_swat.md`](reports/eda_swat.md),
[`reports/eda_wadi.md`](reports/eda_wadi.md) and
[`reports/eda_hai.md`](reports/eda_hai.md) — read these before trusting any
benchmark number, since they surface real data-quality issues (e.g. WADI has
a genuine ~3-day mid-collection outage and an inconsistent date format in
its raw training file) that the modeling pipeline works around rather than fixes.

A fourth dataset, the **Z24 Bridge Progressive Damage Test (PDT)** campaign
(`datasets/data-z24/`), has been processed and profiled in
[`reports/eda_z24.md`](reports/eda_z24.md) but is **not yet wired into the
benchmark harness** — it's structurally different (17 discrete, mostly-
irreversible damage scenarios rather than a continuous normal/attack time
series) and integration is deferred. See `src/data/z24.py` for the loader.

All three are from-scratch reproductions (none of the papers' own code is
public). Where a paper is genuinely ambiguous or missing a numeric value, the
code uses a documented default — see the method docs above for the full list
of those judgment calls before trusting any specific number.

## Repository layout

```
papers/                     the three source papers (PDF)
datasets/                   raw dataset archives (checked in)
  swat.zip / wadi.zip / hal.zip / batadal.zip / tep.zip
  data-z24/                  Z24 bridge PDT archives + PDF documentation (see Setup)
datasets/raw/                <- extracted/processed data lives here (gitignored, see Setup)

src/
  data/                     one loader per dataset
    base.py                   ICSDataset dataclass + shared cleaning helpers (swat/wadi/hai/batadal/tep)
    swat.py / wadi.py / hai.py / batadal.py / tep.py  -> common ICSDataset shape
    z24.py                    Z24 PDT loader -- different shape, see "Z24" section below

  methods/
    base.py                  AnomalyDetectionMethod interface every method implements
    subsystem_grouping.py    shared column-grouping-by-subsystem heuristics (PbNN + CNN1D)
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
    cnn1d/                   1D-CNN next-step predictor -- see docs/cnn1d.md
      network.py                8-layer Conv1d predictor (PyTorch), on-demand windowing
      scoring.py                z-scored prediction error, windowed AND-rule alert
      stages.py                 per-stage/subsystem ensemble grouping
      model.py                  CNN1D class tying it together (combined or per_stage)

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
  eda_swat.py / eda_wadi.py / eda_hai.py / eda_batadal.py
                              generate reports/eda_<name>.md + figures/<name>/*.png
  build_z24_manifest.py     one-time Z24 processing step (combines setups -> per-scenario parquet)
                              --test-type fvt (default) or avt
  eda_z24.py                generates reports/eda_z24.md + figures/z24/* (forced/FVT)
  eda_z24_avt.py             generates reports/eda_z24_avt.md + figures/z24_avt/* (ambient/AVT)
  eda_z24_comparison.py      generates reports/eda_z24_comparison.md + figures/z24_comparison/*
  build_tep_parquet.py      one-time TEP processing step (RData -> parquet via pyreadr)
  eda_tep.py                 generates reports/eda_tep.md + figures/tep/* -- defaults to a small
                              --nrows-capped smoke test (uncapped OOM'd this environment)

run_benchmark.py            CLI entry point
requirements.txt
tests/                      pytest suite (data loaders, metrics, end-to-end integration)
results/                    benchmark output lands here (gitignored)
reports/                    EDA reports (eda_swat.md, eda_wadi.md, eda_hai.md, eda_batadal.md,
                              eda_tep.md, eda_z24.md, eda_z24_avt.md, eda_z24_comparison.md) + figures/
```

## Setup

```bash
pip install -r requirements.txt

# Extract the datasets once (datasets/raw/ is gitignored -- regenerate from the zips)
cd datasets
mkdir -p raw/swat raw/wadi raw/hai raw/batadal
unzip -o -q swat.zip -d raw/swat
unzip -o -q wadi.zip -d raw/wadi
unzip -o -q hal.zip -x "haiend-23.05/*" -d raw/hai   # haiend-23.05 excluded: ~800MB, unused by either method
unzip -o -q batadal.zip -d raw/batadal
cd ..
```

(`haiend-23.05` is a separate, much larger HAI variant not referenced by
either paper; skip it unless you specifically need it, then unzip without
the `-x` filter.)

**TEP** needs an extra conversion step (RData -> parquet) since its source
format isn't directly readable by pandas:

```bash
pip install pyreadr   # pure-Python RData reader, no R installation needed to read
mkdir -p datasets/raw/tep
cd datasets
unzip -q tep.zip "tep_files/TEP_FaultFree_Training.RData" "tep_files/TEP_FaultFree_Testing.RData" \
  "tep_files/TEP_Faulty_Testing.RData" "tep_files/description.txt" -d raw/tep
cd ..
python scripts/build_tep_parquet.py   # one-time; faulty_testing alone is ~800MB, several GB decompressed
```

(`TEP_Faulty_Training.RData`, ~500MB, is deliberately not extracted at all —
every method here trains on fault-free data only, so it's never needed; see
`src/data/tep.py`'s module docstring.)

## Z24 bridge dataset (processed + profiled, not yet in the benchmark)

```bash
mkdir -p datasets/raw/z24
unzip -q datasets/data-z24/pdt_01-08.zip "*/fvt/*" "*/FVT/*" -d datasets/raw/z24
unzip -q datasets/data-z24/pdt_09_17.zip "*/fvt/*" "*/FVT/*" -d datasets/raw/z24
unzip -q datasets/data-z24/pdt_01-08.zip "*/avt/*" "*/AVT/*" -d datasets/raw/z24
unzip -q datasets/data-z24/pdt_09_17.zip "*/avt/*" "*/AVT/*" -d datasets/raw/z24

python scripts/build_z24_manifest.py                    # forced (FVT) -> datasets/raw/z24/combined/<NN>.parquet
python scripts/build_z24_manifest.py --test-type avt    # ambient (AVT) -> datasets/raw/z24/combined_avt/<NN>.parquet

python scripts/eda_z24.py              # -> reports/eda_z24.md + reports/figures/z24/
python scripts/eda_z24_avt.py          # -> reports/eda_z24_avt.md + reports/figures/z24_avt/
python scripts/eda_z24_comparison.py   # -> reports/eda_z24_comparison.md + reports/figures/z24_comparison/
```

Both the forced-vibration (`fvt`/`FVT`) and ambient-vibration (`avt`/`AVT`)
campaigns are now extracted and profiled. They use **different roving-array
sensor grids with zero location-channel overlap** — confirmed directly from
the `.mat` files, not assumed — so only the 5 reference channels
(`R1V`,`R2L`,`R2T`,`R2V`,`R3V`) present in both allow a real comparison;
[`reports/eda_z24_comparison.md`](reports/eda_z24_comparison.md) covers that
(forced excitation runs ~11-44x higher RMS amplitude than ambient, but PSD
shapes correlate 0.39-0.73 across campaigns — the same underlying structural
resonances show up under both excitation types, just at very different
energy levels). The pre-existing `datasets/data-z24/accelerations_*.csv`
files (5.6GB) were investigated and found to be unrelated to the 17-scenario
PDT structure (no headers, mismatched row-count pattern, much newer
timestamps than the source zips) — they're intentionally untouched and
unused. See `src/data/z24.py`'s module docstring and
[`reports/eda_z24.md`](reports/eda_z24.md) /
[`reports/eda_z24_avt.md`](reports/eda_z24_avt.md) for the full picture,
including two scenarios (03, 17) whose damage-scenario labels are
inferred/best-effort rather than confirmed from source documentation.

## Running the benchmark

```bash
# Quick smoke test (small row cap, a couple minutes)
python run_benchmark.py --datasets swat --methods cdt,pbnn --nrows 20000

# Everything, full data (slow -- see "Performance" below)
python run_benchmark.py

# A specific subset
python run_benchmark.py --datasets wadi,hai --methods cdt --out results/cdt_only

# More thorough CDT discovery (paper-scale-ish; ~10min/dataset, see docs/cdt.md)
python run_benchmark.py --methods cdt --cdt-discovery-rows 50000

# CNN1D's per-stage ensemble (the paper's own best-reported config, see docs/cnn1d.md)
python run_benchmark.py --methods cnn1d --cnn1d-ensemble per_stage

# TEP (excluded from the default --datasets set -- see "Performance" below)
python run_benchmark.py --datasets tep --nrows 20000
```

Flags: `--datasets` and `--methods` are comma-separated subsets of
`swat,wadi,hai,batadal,tep` (tep needs `datasets/raw/tep/tep_files/*.parquet`
built first, see Setup) / `cdt,pbnn,cnn1d`; `--nrows` caps rows read per CSV
(omit for full data); `--out` sets the output directory (default `results/`);
`--cdt-discovery-rows` raises CDT's discovery-phase row cap above its
default 5,000 for a more thorough (slower) structure-learning pass — see
`docs/cdt.md`'s "Discovery" section for the measured runtime/edge-count
tradeoff.

Output: `results.csv` and `results.json`, one row per (method, dataset)
pair, with precision/recall/F1 (raw and point-adjusted), detection rate,
false alarm rate, the paper's conflict-index factor, graph metrics (SHD,
edge P/R/F1 — only non-null when the dataset ships a ground-truth graph,
currently HAI only) and fit/score timing. A run that errors is still
reported (with the error message in the `error` column) rather than
aborting the whole comparison.

Each row also carries a `config` field — a snapshot of the method's
constructor-set hyperparameters (e.g. CDT's `discovery_subsample_rows`,
PbNN's `DCNNConfig`) taken before `fit()` runs (see
`benchmark/runner.py::_extract_config`). In `results.json` it's a nested
object; in `results.csv` it's a JSON string in the `config` column
(`json.loads(row["config"])` to get it back). This exists so a saved
results file records exactly which settings produced it, rather than
requiring that to be reconstructed from memory or a paper trail elsewhere.

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

(For Z24, see the "Z24 bridge dataset" section above — it has its own EDA script and a materially different report structure.)

```bash
python scripts/eda_swat.py
python scripts/eda_wadi.py
python scripts/eda_hai.py
python scripts/eda_batadal.py

# TEP defaults to a small --nrows-capped smoke test (see below), not the full report
python scripts/eda_tep.py
```

Each script reads the full dataset (no row cap; ~15-80s per dataset) and
(re)writes `reports/eda_<name>.md` plus its figures under
`reports/figures/<name>/`. Data-quality findings (missingness, constant/empty
columns, label conventions, timestamp gaps) are computed from the raw CSVs
directly via `src/eda/raw_loaders.py`, deliberately bypassing `src/data/*.py`'s
cleaning so the reports show what's really in the source files; everything
else (distributions, correlation, time series, class balance, PCA) uses the
same cleaned train/test split the benchmark harness consumes.

**`scripts/eda_tep.py` is the one exception** — it defaults to
`--nrows 5000` (a smoke-test size), not full data. An uncapped run OOM-killed
this development environment in practice: TEP's full `faulty_testing.parquet`
is ~800MB on disk (several GB decompressed), and cleaning/concatenation code
makes several full in-memory copies along the way. The generated report
says clearly at the top whether it ran capped or full. Pass `--nrows 0` for
the real, full-scale report — only on a machine with plenty of spare RAM
(double-digit GB free), which this development environment did not
reliably have once TEP entered the picture.

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
(HAI's full training files, uncapped, took ~49 minutes to fit in a full-scale
run). Use `--nrows` while iterating; drop it only when you want real numbers.

CDT's own fit time is largely decoupled from row count (its discovery phase
subsamples to a fixed 5,000 rows by default) -- but that row cap is also a
real fidelity ceiling: `--cdt-discovery-rows 50000` finds meaningfully more
graph structure at the cost of ~10x longer discovery (measured on SWaT: 5k
rows/37s/17 edges vs 50k rows/608s/30 edges — see `docs/cdt.md`). CNN1D is
heavier still per row than either (an 8-layer conv net over 256-step
windows, vs. PbNN's 1-3-layer nets over 5-10-step windows) -- expect it to
be the slowest of the three on anything beyond a quick `--nrows`-capped run,
and its `per_stage` ensemble multiplies that by however many stages/
subsystems a dataset groups into (see `docs/cnn1d.md`).

**TEP is a different scale problem entirely**: its full faulty-testing file
alone is ~9.6M rows, ~10x SWaT's full dataset size. It's deliberately left
out of `run_benchmark.py`'s default `--datasets` set for this reason --
pass `--datasets tep` explicitly, and use `--nrows` unless you're prepared
for a very long run.

## Known caveats (see the method docs for the full list)

- **Ground-truth causal graphs** exist only for HAI's boiler subsystem
  (`datasets/raw/hai/graph/boiler/*.json`), and use physical-component ids
  that don't overlap with HAI's dataset tag columns — so SHD/edge-P/R for
  HAI will be near-zero regardless of method quality. SWaT and WADI have no
  ground-truth graph at all, so `causal_graph()` output for those can only
  be inspected qualitatively, not scored.
- **WADI/HAI/BATADAL process invariants for PbNN** are an auto-inferred
  extension (subsystem-prefix grouping), not the paper's own P&ID knowledge
  — expect weaker PbNN results there than on SWaT. BATADAL's tag naming
  doesn't match any of the existing grouping patterns at all and falls back
  to a weak default, collapsing to the same "flag everything" pattern
  already seen on WADI/HAI (see `docs/pbnn.md`).
- **SWaT's `attack.csv`** in this dataset repackaging is 100% attack-labeled
  (a paper-specific extraction of attack windows, not the canonical mixed
  test period); `data/swat.py` builds a leak-free test set by holding out a
  tail slice of `normal.csv` and concatenating it with `attack.csv` rather
  than using the shipped `merged.csv` directly (see that file's docstring).
- **BATADAL's `-999` label** in `BATADAL_dataset04.csv` is treated as normal
  (0) — the standard convention for this dataset (only confirmed attacks
  get an explicit `1`; everything else is unlabeled-by-omission, not
  unlabeled-as-unknown). See `src/data/batadal.py`.
- **TEP's per-run structure** means the concatenated train/test tables have
  artificial "seams" between independent simulation runs (500/960 samples
  each) — not a real continuous process, just a flat table built from many
  short ones. Fault labels within `faulty_testing` correctly account for
  the documented 8-hour/160-sample fault-introduction delay (rows before
  that point in a "faulty" run are labeled normal). All 20 TEP fault types
  are merged into one binary normal/anomaly problem; see `src/data/tep.py`
  for the per-fault alternative this could be extended into later.
