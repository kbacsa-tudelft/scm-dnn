#!/usr/bin/env python3
"""Generates reports/eda_hai.md and reports/figures/hai/*.png.

Uses the hai-22.04 version (same default as src/data/hai.py and the
benchmark harness) so this report describes exactly what the harness
consumes. Data-quality section uses the raw per-file CSVs directly
(src/eda/raw_loaders.py); everything else uses the cleaned train/test split
(src/data/hai.py::load_hai) on the full dataset (no row cap).
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from sklearn.decomposition import PCA

from data.hai import load_hai
from eda import plots, stats
from eda.raw_loaders import load_raw_hai
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/hai"
REPORT_PATH = "reports/eda_hai.md"
PCA_SAMPLE_SIZE = 50_000
VERSION = "hai-22.04"


def _subsystem(col: str) -> str | None:
    m = re.match(r"^(P\d+)_", col)
    return m.group(1) if m else None


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw_hai(version=VERSION)
    train_raw = pd.concat(raw["train"].values(), ignore_index=True)
    test_raw = pd.concat(raw["test"].values(), ignore_index=True)
    ds = load_hai(version=VERSION)  # full data

    report = MarkdownReport("HAI — Exploratory Data Analysis")
    report.para(
        f"HIL-based Augmented ICS (HAI) testbed data, version `{VERSION}`: a "
        f"combined boiler/turbine/water-treatment process, ~1 reading/second. "
        f"Source files: `datasets/raw/hai/{VERSION}/train*.csv` "
        f"({len(raw['train'])} files) and `test*.csv` ({len(raw['test'])} files), "
        f"each already carrying an inline `Attack` label (unlike HAI 23.05, which "
        f"needs a separate label file joined on timestamp -- see `src/data/hai.py`). "
        f"See [`docs/cdt.md`](cdt.md) / [`docs/pbnn.md`](pbnn.md) for how this feeds "
        f"the two methods."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    meta_cols = ("timestamp", "Attack")
    raw_cols = [c for c in train_raw.columns if c not in meta_cols]
    continuous, discrete = stats.classify_columns(ds.train, ds.columns)
    report.bullets([
        f"Train files ({', '.join(raw['train'].keys())}): {train_raw.shape[0]:,} rows total, "
        f"all labeled `Attack=0` in the source (`{int(train_raw['Attack'].sum())}` non-zero found).",
        f"Test files ({', '.join(raw['test'].keys())}): {test_raw.shape[0]:,} rows total, "
        f"`Attack=1` for {int(test_raw['Attack'].sum()):,} rows.",
        f"Raw tag count: {len(raw_cols)} (before dropping constant columns).",
        f"After cleaning: train {ds.train.shape[0]:,} rows, test {ds.test.shape[0]:,} rows, "
        f"{len(ds.columns)} non-constant tags ({len(continuous)} continuous, {len(discrete)} discrete).",
        f"Test-set attack rate: {100*ds.test_labels.mean():.2f}% "
        f"({int(ds.test_labels.sum()):,} / {len(ds.test_labels):,} rows).",
        f"Other HAI versions available but not used here: 20.07, 21.03, 23.05 "
        f"(23.05 needs the separate-label-file join mentioned above).",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality (raw files)")
    miss = stats.missingness(train_raw, raw_cols)
    n_missing_cols = int((miss > 0).sum())
    report.para(
        f"Columns with any missing values across all {len(raw['train'])} train files: "
        f"{n_missing_cols} / {len(raw_cols)} -- HAI's raw data is complete, unlike WADI's."
    )
    ts = pd.to_datetime(train_raw["timestamp"])
    gap_seconds = ts.diff().dropna().dt.total_seconds()
    non_unit_gaps = int((gap_seconds != 1).sum())
    report.para(
        f"{non_unit_gaps} non-1-second timestamp jump(s) in the concatenated train "
        f"files -- exactly the {len(raw['train']) - 1} boundaries between the "
        f"{len(raw['train'])} separately-recorded train files stitched together here, "
        f"not a real data gap (contrast with WADI's genuine mid-collection outage)."
    )
    const_cols = sorted(set(stats.constant_columns(train_raw, raw_cols)))
    report.para(f"Constant columns dropped by the loader: {len(const_cols)} (e.g. {', '.join(const_cols[:8])}...).")

    # --- Univariate distributions ---------------------------------------
    report.h2("Univariate distributions")
    report.para(f"All {len(continuous)} continuous sensors, training period:")
    plots.plot_histograms(ds.train, continuous, f"{FIG_DIR}/histograms_continuous.png", ncols=6)
    report.image("figures/hai/histograms_continuous.png", "Continuous sensor histograms")

    if discrete:
        report.para(f"All {len(discrete)} discrete/binary columns, training period:")
        plots.plot_actuator_bars(ds.train, discrete, f"{FIG_DIR}/bars_discrete.png", ncols=6)
        report.image("figures/hai/bars_discrete.png", "Discrete column state frequencies")

    # --- Temporal structure ----------------------------------------------
    report.h2("Temporal structure")
    subsystem_reps = []
    seen = set()
    for c in continuous:
        s = _subsystem(c)
        if s and s not in seen:
            subsystem_reps.append(c)
            seen.add(s)
    report.para(
        f"One representative continuous tag per process subsystem "
        f"({', '.join(sorted(seen))}), across the full test period, downsampled "
        f"for plotting; shaded bands are attack windows:"
    )
    test_ds = stats.downsample_for_plot(ds.test.assign(_label=ds.test_labels), max_points=8000)
    plots.plot_timeseries(
        test_ds, subsystem_reps, f"{FIG_DIR}/timeseries_by_subsystem.png",
        attack_mask=test_ds["_label"].to_numpy(),
    )
    report.image("figures/hai/timeseries_by_subsystem.png", "Representative sensor per subsystem over time")

    # --- Correlation structure ---------------------------------------------
    report.h2("Correlation structure")
    corr = stats.correlation_matrix(ds.train, continuous)
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/correlation_heatmap.png")
    report.image("figures/hai/correlation_heatmap.png", "Pairwise correlation, continuous sensors")
    report.para("Top 10 most correlated sensor pairs (training period):")
    report.table(stats.top_correlated_pairs(corr, n=10))

    # --- Class balance & attack segments -----------------------------------
    report.h2("Class balance & attack segments")
    balance = stats.class_balance(ds.test_labels)
    plots.plot_class_balance(balance, f"{FIG_DIR}/class_balance.png")
    report.image("figures/hai/class_balance.png", "Normal vs. attack row counts (test set)")
    report.bullets([
        f"{balance['n_segments']} contiguous attack segments.",
        f"Segment length -- mean {balance['mean_segment_length']:.0f}, "
        f"median {balance['median_segment_length']:.0f}, max {balance['max_segment_length']} rows.",
    ])
    seg_path = plots.plot_segment_length_hist(balance["segment_lengths"], f"{FIG_DIR}/segment_lengths.png")
    if seg_path:
        report.image("figures/hai/segment_lengths.png", "Attack segment length distribution")

    # --- Separability projection ------------------------------------------
    report.h2("Separability projection (PCA)")
    sample = ds.test.sample(n=min(PCA_SAMPLE_SIZE, len(ds.test)), random_state=0)
    labels_sample = ds.test_labels[sample.index.to_numpy()]
    X = (sample[continuous] - ds.train[continuous].mean()) / (ds.train[continuous].std() + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.to_numpy())
    plots.plot_pca_projection(coords, labels_sample, tuple(pca.explained_variance_ratio_), f"{FIG_DIR}/pca.png")
    report.image("figures/hai/pca.png", "2D PCA projection, normal vs. attack",
                 caption=f"{min(PCA_SAMPLE_SIZE, len(ds.test)):,}-row sample; "
                         f"standardized using training-period mean/std.")

    # --- Ground-truth graph -------------------------------------------------
    report.h2("Ground-truth boiler causal graph")
    graph = ds.ground_truth_graph
    overlap = set(graph.nodes()) & set(ds.columns)
    plots.plot_graph_topology(graph, f"{FIG_DIR}/boiler_graph.png")
    report.image("figures/hai/boiler_graph.png", "HAI boiler subsystem ground-truth causal graph")
    report.para(
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} directed edges "
        f"(`datasets/raw/hai/graph/boiler/phy_boiler.json`). **Node-id overlap with "
        f"this dataset's {len(ds.columns)} tag columns: {len(overlap)}** -- the graph "
        f"uses physical-component ids (`TK01`, `PP01A`, ...) while the CSV columns "
        f"use DCS tag names (`P1_LIT01`, ...), so this graph cannot be used directly "
        f"for structural-accuracy scoring against a discovered graph over these "
        f"columns (see `src/data/hai.py` and `src/metrics/graph.py`); it's included "
        f"here for topology reference only."
    )

    # --- Dataset-specific notes ---------------------------------------------
    report.h2("HAI-specific notes")
    report.bullets([
        "Tags are prefixed by process subsystem (`P1`-`P4`).",
        f"{len(const_cols)} constant column(s) in the raw data were dropped before modeling.",
        f"Using version `{VERSION}` for consistency with the benchmark harness default; "
        f"other versions differ in label convention and would need separate EDA if used.",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
