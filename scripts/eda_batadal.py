#!/usr/bin/env python3
"""Generates reports/eda_batadal.md and reports/figures/batadal/*.png.

Mirrors scripts/eda_swat.py's structure. BATADAL (C-Town water distribution
network, iTrust/University of Cyprus) is much closer in shape to SWaT/WADI
than to Z24 -- one continuous hourly series with a train/test split -- so
the same EDA pattern applies directly. See src/data/batadal.py for the
dataset's own quirks (the -999 label convention, the unlabeled competition
test set that isn't used here).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.decomposition import PCA

from data.batadal import load_batadal
from eda import plots, stats
from eda.raw_loaders import load_raw_batadal
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/batadal"
REPORT_PATH = "reports/eda_batadal.md"
PCA_SAMPLE_SIZE = 50_000


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw_batadal()
    train_raw, test_raw = raw["dataset03"], raw["dataset04"]
    ds = load_batadal()

    report = MarkdownReport("BATADAL — Exploratory Data Analysis")
    report.para(
        "BATADAL (BATtle of the Attack Detection ALgorithms): SCADA data "
        "from the C-Town water distribution network testbed (EPANET "
        "simulation), hourly readings. Source: `datasets/raw/batadal/batadal/"
        "BATADAL_dataset0{3,4}.csv`. See [`docs/cdt.md`](cdt.md) / "
        "[`docs/pbnn.md`](pbnn.md) / [`docs/cnn1d.md`](cnn1d.md) for how "
        "this feeds the three methods."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    label_col = "ATT_FLAG"
    time_col = "DATETIME"
    raw_cols = [c for c in train_raw.columns if c not in (time_col, label_col)]
    continuous, discrete = stats.classify_columns(ds.train, ds.columns)
    report.bullets([
        f"`BATADAL_dataset03.csv`: {train_raw.shape[0]:,} rows, ~1 year hourly, "
        f"entirely normal operation (train).",
        f"`BATADAL_dataset04.csv`: {test_raw.shape[0]:,} rows, ~6 months hourly, "
        f"contains 7 documented attacks (test).",
        f"Raw tag count: {len(raw_cols)} (before dropping constant columns).",
        f"After cleaning: train {ds.train.shape[0]:,} rows, test {ds.test.shape[0]:,} rows, "
        f"{len(ds.columns)} non-constant tags ({len(continuous)} continuous, {len(discrete)} discrete/actuator).",
        f"Test-set attack rate: {100*ds.test_labels.mean():.2f}% "
        f"({int(ds.test_labels.sum()):,} / {len(ds.test_labels):,} rows).",
        f"Timestamp format example: `{train_raw[time_col].iloc[0]}` (hourly sampling, "
        f"much coarser than SWaT/WADI/HAI's ~1Hz).",
        "The original competition's held-out test set (`BATADAL_test_dataset.zip`) "
        "ships with no label column at all and isn't used here -- see `src/data/batadal.py`.",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality (raw files)")
    miss = stats.missingness(train_raw, raw_cols)
    n_missing_cols = int((miss > 0).sum())
    report.para(f"Columns with any missing values in `dataset03.csv`: {n_missing_cols} / {len(raw_cols)}.")
    if n_missing_cols:
        plots.plot_missingness(miss, f"{FIG_DIR}/missingness.png")
        report.image("figures/batadal/missingness.png", "Missingness by column")

    const_cols = sorted(set(stats.constant_columns(train_raw, raw_cols) + stats.constant_columns(test_raw, raw_cols)))
    report.para(
        f"Constant (or single-valued) columns dropped by the loader: "
        f"{', '.join(const_cols) if const_cols else 'none'}."
    )
    label_values_train = train_raw[label_col].value_counts().to_dict()
    label_values_test = test_raw[label_col].value_counts().to_dict()
    report.para(
        f"Label column (`ATT_FLAG`) values -- `dataset03.csv`: {label_values_train}; "
        f"`dataset04.csv`: {label_values_test}. There is no explicit `0` in `dataset04.csv` "
        f"-- only confirmed attacks are marked `1`, everything else is `-999` "
        f"(\"not confirmed either way\"), treated here as normal (0), the standard "
        f"convention for this dataset."
    )

    # --- Univariate distributions ---------------------------------------
    report.h2("Univariate distributions")
    report.para(f"All {len(continuous)} continuous sensors, training period (attack-free):")
    plots.plot_histograms(ds.train, continuous, f"{FIG_DIR}/histograms_continuous.png", ncols=6)
    report.image("figures/batadal/histograms_continuous.png", "Continuous sensor histograms")

    if discrete:
        report.para(f"All {len(discrete)} discrete actuator/state columns, training period:")
        plots.plot_actuator_bars(ds.train, discrete, f"{FIG_DIR}/bars_discrete.png", ncols=6)
        report.image("figures/batadal/bars_discrete.png", "Discrete actuator state frequencies")

    # --- Temporal structure ----------------------------------------------
    report.h2("Temporal structure")
    # Representative tags: tank levels (L_T*), pump flows (F_PU*), one valve, one junction pressure.
    reps = [c for c in continuous if c.startswith("L_T")][:3]
    reps += [c for c in continuous if c.startswith("F_PU")][:3]
    reps += [c for c in continuous if c.startswith(("F_V", "P_J"))][:2]
    report.para(
        "A handful of representative tags (tank levels, pump flows, valve flow, junction "
        "pressure) across the full test period, downsampled for plotting; shaded bands are attack windows:"
    )
    test_ds = stats.downsample_for_plot(ds.test.assign(_label=ds.test_labels), max_points=4000)
    plots.plot_timeseries(
        test_ds, reps, f"{FIG_DIR}/timeseries.png",
        attack_mask=test_ds["_label"].to_numpy(),
    )
    report.image("figures/batadal/timeseries.png", "Representative tags over time")

    # --- Correlation structure ---------------------------------------------
    report.h2("Correlation structure")
    corr = stats.correlation_matrix(ds.train, continuous)
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/correlation_heatmap.png")
    report.image("figures/batadal/correlation_heatmap.png", "Pairwise correlation, continuous sensors")
    report.para("Top 10 most correlated sensor pairs (training period):")
    report.table(stats.top_correlated_pairs(corr, n=10))

    # --- Class balance & attack segments -----------------------------------
    report.h2("Class balance & attack segments")
    balance = stats.class_balance(ds.test_labels)
    plots.plot_class_balance(balance, f"{FIG_DIR}/class_balance.png")
    report.image("figures/batadal/class_balance.png", "Normal vs. attack row counts (test set)")
    report.bullets([
        f"{balance['n_segments']} contiguous attack segments (BATADAL's own documentation "
        f"describes 7 attacks in this file; segment count may differ slightly if attacks "
        f"are adjacent/back-to-back in hourly resolution).",
        f"Segment length -- mean {balance['mean_segment_length']:.0f}, "
        f"median {balance['median_segment_length']:.0f}, max {balance['max_segment_length']} rows "
        f"(hours, given hourly sampling).",
    ])
    seg_path = plots.plot_segment_length_hist(balance["segment_lengths"], f"{FIG_DIR}/segment_lengths.png")
    if seg_path:
        report.image("figures/batadal/segment_lengths.png", "Attack segment length distribution")

    # --- Separability projection ------------------------------------------
    report.h2("Separability projection (PCA)")
    sample = ds.test.sample(n=min(PCA_SAMPLE_SIZE, len(ds.test)), random_state=0)
    labels_sample = ds.test_labels[sample.index.to_numpy()]
    X = (sample[continuous] - ds.train[continuous].mean()) / (ds.train[continuous].std() + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.to_numpy())
    plots.plot_pca_projection(coords, labels_sample, tuple(pca.explained_variance_ratio_), f"{FIG_DIR}/pca.png")
    report.image("figures/batadal/pca.png", "2D PCA projection, normal vs. attack",
                 caption=f"{min(PCA_SAMPLE_SIZE, len(ds.test)):,}-row sample (all of dataset04, given its "
                         f"size); standardized using training-period mean/std.")

    # --- Dataset-specific notes ---------------------------------------------
    report.h2("BATADAL-specific notes")
    report.bullets([
        "Tags: `L_T*` tank levels, `F_PU*`/`S_PU*` pump flow/status pairs, `F_V2`/`S_V2` "
        "valve flow/status, `P_J*` junction pressures.",
        "Hourly sampling is far coarser than SWaT/WADI/HAI's ~1Hz -- a year of data is "
        "only ~8,760 rows, much less than the other ICS datasets' millions.",
        f"{len(const_cols)} constant column(s) in the raw data were dropped before modeling.",
        "`CTOWN.INP` (the EPANET network topology model) ships alongside the CSVs but "
        "isn't parsed as a ground-truth graph here -- same category of gap as HAI's "
        "boiler graph (node ids would need a translation layer to the SCADA tags above).",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
