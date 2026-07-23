#!/usr/bin/env python3
"""Generates reports/eda_tep.md and reports/figures/tep/*.png.

TEP's shape differs from every other dataset here: many independent short
simulation runs (500/960 samples each) rather than one continuous
recording, and all 52 variables are continuous (no discrete actuators).
The concatenated ICSDataset view (src/data/tep.py) drops the
faultNumber/simulationRun/sample metadata that would let you tell where one
run ends and the next begins -- so the "Temporal structure" section below
uses the raw parquet files (via eda/raw_loaders.py) instead, to plot a
single real run rather than a confusing multi-run flattening.

Defaults to a small `--nrows` cap, not the full ~10M-row test set: an
uncapped run OOM-killed this environment in practice (faulty_testing.parquet
alone is ~800MB on disk, several GB decompressed, and cleaning/concatenation
code makes several full copies along the way). Pass `--nrows 0` (or a large
number) for the full, real report -- only on a machine with plenty of spare
RAM (double-digit GB free), not verified safe here.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.decomposition import PCA

from data.tep import FAULT_INTRODUCED_SAMPLE, load_tep
from eda import plots, stats
from eda.raw_loaders import load_raw_tep
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/tep"
REPORT_PATH = "reports/eda_tep.md"
PCA_SAMPLE_SIZE = 50_000
DEFAULT_NROWS = 5_000  # safe smoke-test size; see module docstring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nrows", type=int, default=DEFAULT_NROWS,
                         help=f"Cap rows read per source file (default {DEFAULT_NROWS}, a smoke-test size). "
                              f"Pass 0 for the full, uncapped dataset -- OOM'd this environment; only "
                              f"attempt on a machine with plenty of spare RAM.")
    args = parser.parse_args()
    nrows = None if args.nrows == 0 else args.nrows

    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw_tep(nrows=nrows)
    train_raw, fault_free_test_raw, faulty_test_raw = (
        raw["fault_free_training"], raw["fault_free_testing"], raw["faulty_testing"]
    )
    ds = load_tep(nrows=nrows)

    report = MarkdownReport("TEP — Exploratory Data Analysis")
    report.para(
        "Tennessee Eastman Process (TEP), Rieth et al. 2017 simulation "
        "release: a chemical process simulation, sampled every 3 minutes. "
        "Source: `datasets/raw/tep/tep_files/*.parquet` (converted from the "
        "original RData via `scripts/build_tep_parquet.py`). Unlike every "
        "other dataset here, TEP is organized as **many independent short "
        "simulation runs** (500 samples/25h training, 960 samples/48h "
        "testing) across 21 conditions (0=normal, 1-20=distinct fault "
        "types) rather than one continuous recording -- see "
        "`src/data/tep.py`'s module docstring for how this becomes one "
        "binary normal/anomaly `ICSDataset` (all 20 faults merged, per the "
        "user's choice)."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    if nrows is not None:
        report.para(
            f"**This report ran with `--nrows {nrows}`** (a per-file row cap, not the full "
            f"dataset) -- an uncapped run OOM-killed this environment in practice, so the "
            f"numbers below reflect a small slice, not TEP's actual full scale (nominally "
            f"500 runs x 500/960 samples x 21 conditions -- see the source-file row counts "
            f"quoted below for what was *actually* read here). Re-run with `--nrows 0` on a "
            f"machine with plenty of spare RAM for the real report."
        )
    continuous, discrete = stats.classify_columns(ds.train, ds.columns)
    report.bullets([
        f"Fault-free training: {train_raw.shape[0]:,} rows read"
        + ("" if nrows is None else f" (capped at --nrows {nrows})") + ".",
        f"Fault-free testing: {fault_free_test_raw.shape[0]:,} rows read"
        + ("" if nrows is None else f" (capped at --nrows {nrows})") + ".",
        f"Faulty testing: {faulty_test_raw.shape[0]:,} rows read"
        + ("" if nrows is None else f" (capped at --nrows {nrows})")
        + " -- **`faulty_training` (~5M rows) is never used**, since every method "
        f"here trains on fault-free data only (not even extracted, see `scripts/build_tep_parquet.py`).",
        f"52 process variables (41 `xmeas_*` measured + 11 `xmv_*` manipulated), all continuous "
        f"-- {len(continuous)}/{len(ds.columns)} after cleaning, {len(discrete)} discrete.",
        f"After cleaning: train {ds.train.shape[0]:,} rows, test {ds.test.shape[0]:,} rows.",
        f"Test-set attack (fault) rate: {100*ds.test_labels.mean():.2f}% "
        f"({int(ds.test_labels.sum()):,} / {len(ds.test_labels):,} rows).",
        "Sample rate: one reading every 3 minutes (much coarser than SWaT/WADI/HAI's ~1Hz, "
        "coarser even than BATADAL's hourly rate is dense-per-day, but each run only spans "
        "25-48 simulated hours).",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality (raw files)")
    process_cols = [c for c in train_raw.columns if c not in ("faultNumber", "simulationRun", "sample")]
    miss = stats.missingness(train_raw, process_cols)
    report.para(
        f"Columns with any missing values in fault-free training: {int((miss > 0).sum())} / "
        f"{len(process_cols)} -- a clean simulated dataset, unlike the real ICS datasets."
    )
    const_cols = sorted(set(stats.constant_columns(train_raw, process_cols)))
    report.para(f"Constant columns dropped by the loader: {', '.join(const_cols) if const_cols else 'none'}.")
    report.para(
        f"Fault labeling within `faulty_testing`: per the dataset's own documentation, faults "
        f"are introduced {FAULT_INTRODUCED_SAMPLE} samples (8 hours) into each faulty run -- rows "
        f"before that point are labeled 0 (still genuinely fault-free) even though the run's "
        f"`faultNumber` is already nonzero. Confirmed directly: first faulty run's samples "
        f"1-{FAULT_INTRODUCED_SAMPLE-1} are labeled "
        f"{'0' if ds.test_labels[len(fault_free_test_raw):len(fault_free_test_raw)+FAULT_INTRODUCED_SAMPLE-1].sum() == 0 else 'MISMATCH'}, "
        f"sample {FAULT_INTRODUCED_SAMPLE} onward is labeled 1."
    )
    report.para(
        "Concatenating 500+ independent runs end-to-end creates artificial \"seams\" at every "
        "run boundary (a discontinuous jump in process state) -- inherent to this data's shape, "
        "not a cleaning artifact; see \"Temporal structure\" below for what a single real run "
        "actually looks like."
    )

    # --- Univariate distributions ---------------------------------------
    report.h2("Univariate distributions")
    report.para(f"All {len(continuous)} continuous process variables, training period (fault-free):")
    plots.plot_histograms(ds.train, continuous, f"{FIG_DIR}/histograms_continuous.png", ncols=7)
    report.image("figures/tep/histograms_continuous.png", "Continuous process-variable histograms")

    # --- Temporal structure ----------------------------------------------
    report.h2("Temporal structure")
    report.para(
        "One real run, not the flattened multi-run table: `xmeas_1` (a measured variable) for "
        "fault-free run 1 (training) vs. faulty run 1 (fault 1, testing) -- shaded band marks "
        f"the post-fault-introduction period (sample >= {FAULT_INTRODUCED_SAMPLE}):"
    )
    run1_free = train_raw[(train_raw.simulationRun == 1)].sort_values("sample")
    run1_faulty = faulty_test_raw[
        (faulty_test_raw.simulationRun == 1) & (faulty_test_raw.faultNumber == 1)
    ].sort_values("sample")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharey=True)
    axes[0].plot(run1_free["sample"].to_numpy(), run1_free["xmeas_1"].to_numpy(), color="#2a78d6")
    axes[0].set_title("fault-free training, run 1", fontsize=9, loc="left")
    axes[1].plot(run1_faulty["sample"].to_numpy(), run1_faulty["xmeas_1"].to_numpy(), color="#2a78d6")
    axes[1].axvspan(FAULT_INTRODUCED_SAMPLE, run1_faulty["sample"].max(), color="#e34948", alpha=0.15)
    axes[1].set_title("faulty testing, run 1, fault 1", fontsize=9, loc="left")
    axes[1].set_xlabel("sample (within this run)")
    for ax in axes:
        ax.set_ylabel("xmeas_1")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/timeseries_single_run.png")
    plt.close(fig)
    report.image("figures/tep/timeseries_single_run.png", "xmeas_1, one fault-free run vs. one faulty run")

    # --- Correlation structure ---------------------------------------------
    report.h2("Correlation structure")
    corr = stats.correlation_matrix(ds.train, continuous)
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/correlation_heatmap.png")
    report.image("figures/tep/correlation_heatmap.png", "Pairwise correlation, continuous process variables")
    report.para("Top 10 most correlated variable pairs (training period):")
    report.table(stats.top_correlated_pairs(corr, n=10))

    # --- Class balance & attack segments -----------------------------------
    report.h2("Class balance & fault segments")
    balance = stats.class_balance(ds.test_labels)
    plots.plot_class_balance(balance, f"{FIG_DIR}/class_balance.png")
    report.image("figures/tep/class_balance.png", "Normal vs. faulty row counts (test set)")
    report.bullets([
        f"{balance['n_segments']} contiguous \"fault\" segments -- with 500 runs x 20 fault "
        f"types, expect this to be large and fairly uniform (each faulty run contributes "
        f"one segment of ~{960-FAULT_INTRODUCED_SAMPLE} samples, by construction).",
        f"Segment length -- mean {balance['mean_segment_length']:.0f}, "
        f"median {balance['median_segment_length']:.0f}, max {balance['max_segment_length']} samples.",
    ])

    # --- Separability projection ------------------------------------------
    report.h2("Separability projection (PCA)")
    sample = ds.test.sample(n=min(PCA_SAMPLE_SIZE, len(ds.test)), random_state=0)
    labels_sample = ds.test_labels[sample.index.to_numpy()]
    X = (sample[continuous] - ds.train[continuous].mean()) / (ds.train[continuous].std() + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.to_numpy())
    plots.plot_pca_projection(coords, labels_sample, tuple(pca.explained_variance_ratio_), f"{FIG_DIR}/pca.png")
    n_faults_present = faulty_test_raw["faultNumber"].nunique()
    report.image("figures/tep/pca.png", "2D PCA projection, normal vs. faulty",
                 caption=f"{min(PCA_SAMPLE_SIZE, len(ds.test)):,}-row sample covering {n_faults_present} of 20 "
                         f"fault types present in this run's data (dominant faults may visually crowd out "
                         f"subtler ones; a small --nrows cap covers fewer fault types, since faulty_testing "
                         f"is ordered fault-ascending); standardized using training-period mean/std.")

    # --- Dataset-specific notes ---------------------------------------------
    report.h2("TEP-specific notes")
    report.bullets([
        "All 20 fault types are merged into one binary label here -- some are far easier to "
        "detect than others in the TEP literature (e.g. fault 1/step changes vs. fault 3/19, "
        "notoriously hard); a per-fault breakdown would need the 20-separate-datasets "
        "alternative noted in `src/data/tep.py`, not built here.",
        "Every method in this harness trains on fault-free data only, so `faulty_training` "
        "is never extracted or used -- only `fault_free_training`, `fault_free_testing` and "
        "`faulty_testing` are.",
        f"{len(const_cols)} constant column(s) in the raw data were dropped before modeling.",
        "Full-scale `test` is ~10.1M rows -- about 7x SWaT's full dataset -- and an uncapped "
        "EDA run OOM-killed this environment in practice; see the capped-run notice at the "
        "top of this report for what was actually analyzed here. Benchmark runs should use "
        "`--nrows` for the same reason (see root README's Performance section).",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
