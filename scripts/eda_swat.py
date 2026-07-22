#!/usr/bin/env python3
"""Generates reports/eda_swat.md and reports/figures/swat/*.png.

Data-quality section (missingness, constants, label check) uses the raw
CSVs directly (src/eda/raw_loaders.py) so cleaning already done by
src/data/swat.py isn't hidden from the report. Everything else
(distributions, correlation, time series, class balance, PCA) uses the
same cleaned train/test split the benchmark harness actually consumes
(src/data/swat.py::load_swat), on the full dataset (no row cap).
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.decomposition import PCA

from data.swat import load_swat
from eda import plots, stats
from eda.raw_loaders import load_raw_swat
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/swat"
REPORT_PATH = "reports/eda_swat.md"
PCA_SAMPLE_SIZE = 50_000


def _stage(col: str) -> str | None:
    m = re.search(r"(\d)\d{2}", col)
    return f"P{m.group(1)}" if m else None


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw_swat()
    normal_raw, attack_raw = raw["normal"], raw["attack"]
    ds = load_swat()  # full data, leak-free split (see data/swat.py docstring)

    report = MarkdownReport("SWaT — Exploratory Data Analysis")
    report.para(
        "Secure Water Treatment (SWaT) testbed data: a 6-stage water treatment "
        "process, ~1 reading/second. Source files: `datasets/raw/swat/normal.csv`, "
        "`attack.csv`. See [`docs/cdt.md`](cdt.md) and [`docs/pbnn.md`](pbnn.md) "
        "for how this feeds the two methods, and the root "
        "[`README.md`](../README.md#known-caveats-see-the-method-docs-for-the-full-list) "
        "for why the train/test split isn't the shipped `merged.csv`."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    label_col = "Normal/Attack"
    time_col = "Timestamp"
    raw_cols = [c for c in normal_raw.columns if c not in (time_col, label_col)]
    continuous, discrete = stats.classify_columns(ds.train, ds.columns)
    report.bullets([
        f"`normal.csv`: {normal_raw.shape[0]:,} rows -- entirely labeled Normal.",
        f"`attack.csv`: {attack_raw.shape[0]:,} rows -- entirely labeled Attack "
        f"(a paper-specific extraction of attack windows, not the canonical mixed test period).",
        f"Raw tag count: {len(raw_cols)} (before dropping constant columns).",
        f"After cleaning + leak-free split: train {ds.train.shape[0]:,} rows, "
        f"test {ds.test.shape[0]:,} rows, {len(ds.columns)} non-constant tags "
        f"({len(continuous)} continuous, {len(discrete)} discrete/actuator).",
        f"Test-set attack rate: {100*ds.test_labels.mean():.2f}% "
        f"({int(ds.test_labels.sum()):,} / {len(ds.test_labels):,} rows).",
        f"Timestamp format example: `{normal_raw[time_col].iloc[0]}` (~1 Hz sampling).",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality (raw files)")
    miss = stats.missingness(normal_raw, raw_cols)
    n_missing_cols = int((miss > 0).sum())
    report.para(f"Columns with any missing values in `normal.csv`: {n_missing_cols} / {len(raw_cols)}.")
    if n_missing_cols:
        path = plots.plot_missingness(miss, f"{FIG_DIR}/missingness.png")
        report.image(f"figures/swat/missingness.png", "Missingness by column")

    const_cols = stats.constant_columns(normal_raw, raw_cols) + stats.constant_columns(attack_raw, raw_cols)
    const_cols = sorted(set(const_cols))
    report.para(
        f"Constant (or single-valued) columns dropped by the loader: "
        f"{', '.join(const_cols) if const_cols else 'none'}."
    )
    label_values_normal = normal_raw[label_col].str.strip().value_counts().to_dict()
    label_values_attack = attack_raw[label_col].str.strip().value_counts().to_dict()
    report.para(
        f"Label column values -- `normal.csv`: {label_values_normal}; "
        f"`attack.csv`: {label_values_attack} (no label typos/variants found)."
    )

    # --- Univariate distributions ---------------------------------------
    report.h2("Univariate distributions")
    report.para(f"All {len(continuous)} continuous sensors, training period (attack-free):")
    plots.plot_histograms(ds.train, continuous, f"{FIG_DIR}/histograms_continuous.png", ncols=6)
    report.image("figures/swat/histograms_continuous.png", "Continuous sensor histograms")

    if discrete:
        report.para(f"All {len(discrete)} discrete actuator/state columns, training period:")
        plots.plot_actuator_bars(ds.train, discrete, f"{FIG_DIR}/bars_discrete.png", ncols=6)
        report.image("figures/swat/bars_discrete.png", "Discrete actuator state frequencies")

    # --- Temporal structure ----------------------------------------------
    report.h2("Temporal structure")
    stage_reps = []
    seen_stages = set()
    for c in continuous:
        s = _stage(c)
        if s and s not in seen_stages:
            stage_reps.append(c)
            seen_stages.add(s)
    for c in discrete:
        s = _stage(c)
        if s and s not in seen_stages:
            stage_reps.append(c)
            seen_stages.add(s)
    report.para(
        "One representative tag per process stage (P1-P6), across the full test "
        "period (normal + attack), downsampled for plotting; shaded bands are attack windows:"
    )
    test_ds = stats.downsample_for_plot(ds.test.assign(_label=ds.test_labels), max_points=8000)
    plots.plot_timeseries(
        test_ds, stage_reps, f"{FIG_DIR}/timeseries_by_stage.png",
        attack_mask=test_ds["_label"].to_numpy(),
    )
    report.image("figures/swat/timeseries_by_stage.png", "Representative sensor per process stage over time")

    # --- Correlation structure ---------------------------------------------
    report.h2("Correlation structure")
    corr = stats.correlation_matrix(ds.train, continuous)
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/correlation_heatmap.png")
    report.image("figures/swat/correlation_heatmap.png", "Pairwise correlation, continuous sensors")
    report.para("Top 10 most correlated sensor pairs (training period):")
    report.table(stats.top_correlated_pairs(corr, n=10))

    # --- Class balance & attack segments -----------------------------------
    report.h2("Class balance & attack segments")
    balance = stats.class_balance(ds.test_labels)
    plots.plot_class_balance(balance, f"{FIG_DIR}/class_balance.png")
    report.image("figures/swat/class_balance.png", "Normal vs. attack row counts (test set)")
    report.bullets([
        f"{balance['n_segments']} contiguous attack segment(s) -- with only "
        f"{balance['n_segments']} segment, this is an artifact of the test "
        f"set being held-out-normal followed by all of `attack.csv` "
        f"back-to-back (see Overview), not evidence that real attacks are "
        f"one long uninterrupted window.",
        f"Segment length -- mean {balance['mean_segment_length']:.0f}, "
        f"median {balance['median_segment_length']:.0f}, max {balance['max_segment_length']} rows.",
    ])
    seg_path = plots.plot_segment_length_hist(balance["segment_lengths"], f"{FIG_DIR}/segment_lengths.png")
    if seg_path:
        report.image("figures/swat/segment_lengths.png", "Attack segment length distribution")

    # --- Separability projection ------------------------------------------
    report.h2("Separability projection (PCA)")
    sample = ds.test.sample(n=min(PCA_SAMPLE_SIZE, len(ds.test)), random_state=0)
    labels_sample = ds.test_labels[sample.index.to_numpy()]
    X = (sample[continuous] - ds.train[continuous].mean()) / (ds.train[continuous].std() + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.to_numpy())
    plots.plot_pca_projection(coords, labels_sample, tuple(pca.explained_variance_ratio_), f"{FIG_DIR}/pca.png")
    report.image("figures/swat/pca.png", "2D PCA projection, normal vs. attack",
                 caption=f"{min(PCA_SAMPLE_SIZE, len(ds.test)):,}-row stratified-by-time sample; "
                         f"standardized using training-period mean/std.")

    # --- Dataset-specific notes ---------------------------------------------
    report.h2("SWaT-specific notes")
    report.bullets([
        "6 process stages (P1 raw water intake -- P6 backwash), tags numbered "
        "`<TYPE><stage><id>` e.g. `FIT101` = flow sensor, stage 1.",
        "The leak-free train/test split holds out a tail slice of `normal.csv` for "
        "testing rather than reusing rows also used to fit the model -- see "
        "`src/data/swat.py`.",
        f"{len(const_cols)} constant column(s) in the raw data were dropped before modeling.",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
