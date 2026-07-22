#!/usr/bin/env python3
"""Generates reports/eda_wadi.md and reports/figures/wadi/*.png.

Data-quality section uses the raw CSVs directly (src/eda/raw_loaders.py) --
WADI's raw files have a large mid-collection outage and inconsistent date
formatting that the cleaned loader doesn't need to touch (it never uses
Row/Date/Time), so this is the only place those issues get surfaced.
Everything else uses the cleaned train/test split (src/data/wadi.py::load_wadi),
on the full dataset (no row cap).
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.decomposition import PCA

from data.wadi import load_wadi
from eda import plots, stats
from eda.raw_loaders import load_raw_wadi
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/wadi"
REPORT_PATH = "reports/eda_wadi.md"
PCA_SAMPLE_SIZE = 50_000


def _subsystem(col: str) -> str | None:
    m = re.match(r"^(\d+[A-Za-z]?)_", col)
    return m.group(1) if m else None


def _row_gaps(row_col: np.ndarray) -> list[tuple[int, int, int]]:
    """(row_before, row_after, gap_size) for every discontinuity in a Row index."""
    diffs = np.diff(row_col)
    gap_positions = np.where(diffs > 1)[0]
    return [(int(row_col[i]), int(row_col[i + 1]), int(diffs[i] - 1)) for i in gap_positions]


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    raw = load_raw_wadi()
    days14_raw, attack_raw = raw["days14"], raw["attack"]
    ds = load_wadi()  # full data

    report = MarkdownReport("WADI — Exploratory Data Analysis")
    report.para(
        "Water Distribution (WADI) testbed data: a water distribution network "
        "downstream of SWaT, ~1 reading/second. Source files: "
        "`datasets/raw/wadi/WADI_14days_new.csv` (normal baseline), "
        "`WADI_attackdataLABLE.csv` (labeled attack period). See "
        "[`docs/cdt.md`](cdt.md) / [`docs/pbnn.md`](pbnn.md) for how this feeds "
        "the two methods."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    meta_cols = ("Row", "Date", "Time")
    raw_cols = [c for c in days14_raw.columns if c not in meta_cols]
    continuous, discrete = stats.classify_columns(ds.train, ds.columns)
    label_col = attack_raw.columns[-1]
    report.bullets([
        f"`WADI_14days_new.csv`: {days14_raw.shape[0]:,} rows, normal operation, no label column.",
        f"`WADI_attackdataLABLE.csv`: {attack_raw.shape[0]:,} rows, label column "
        f"`{label_col.strip()}` (1 = no attack, -1 = attack).",
        f"Raw tag count: {len(raw_cols)} (before dropping constant/empty columns).",
        f"After cleaning: train {ds.train.shape[0]:,} rows, test {ds.test.shape[0]:,} rows, "
        f"{len(ds.columns)} non-constant tags ({len(continuous)} continuous, {len(discrete)} discrete/status).",
        f"Test-set attack rate: {100*ds.test_labels.mean():.2f}% "
        f"({int(ds.test_labels.sum()):,} / {len(ds.test_labels):,} rows).",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality (raw files)")

    date_formats = sorted(set(
        "4-digit year" if re.search(r"/\d{4}$", d) else "2-digit year"
        for d in days14_raw["Date"].dropna().unique()
    ))
    report.para(
        f"**Date format is inconsistent within `WADI_14days_new.csv`**: "
        f"{', '.join(date_formats)} both appear (e.g. `9/25/2017` vs `10/7/17`) "
        f"-- naive date parsing without a flexible parser will silently "
        f"misinterpret or fail on part of the file."
    )

    gaps = _row_gaps(days14_raw["Row"].to_numpy())
    if gaps:
        total_missing = sum(g[2] for g in gaps)
        worst = max(gaps, key=lambda g: g[2])
        report.para(
            f"**{len(gaps)} discontinuity(ies) in the `Row` index totaling "
            f"{total_missing:,} missing rows** ({100*total_missing/(len(days14_raw)+total_missing):.1f}% "
            f"of the nominal 1 Hz timeline) -- the largest is a gap of "
            f"{worst[2]:,} rows (~{worst[2]/3600:.1f} hours) between Row "
            f"{worst[0]:,} and {worst[1]:,}. This is a real mid-collection outage "
            f"in the normal/training period, not a plotting artifact: any model "
            f"treating this file as one continuous time series should account "
            f"for the discontinuity rather than bridging across it."
        )
    else:
        report.para("No discontinuities found in the `Row` index.")

    miss = stats.missingness(days14_raw, raw_cols)
    fully_empty = [c for c in raw_cols if days14_raw[c].isna().all()]
    report.para(
        f"Columns with any missing values: {int((miss > 0).sum())} / {len(raw_cols)}, "
        f"of which {len(fully_empty)} are **entirely empty** in this file: "
        f"{', '.join(fully_empty) if fully_empty else 'none'}."
    )
    miss_path = plots.plot_missingness(miss, f"{FIG_DIR}/missingness.png")
    if miss_path:
        report.image("figures/wadi/missingness.png", "Missingness by column (WADI_14days_new.csv)")

    const_cols = sorted(set(stats.constant_columns(days14_raw, raw_cols)))
    report.para(
        f"Constant columns dropped by the loader (includes the fully-empty ones "
        f"above, since all-NaN counts as a single unique value): {len(const_cols)}."
    )

    # --- Univariate distributions ---------------------------------------
    report.h2("Univariate distributions")
    report.para(f"All {len(continuous)} continuous sensors, training period:")
    plots.plot_histograms(ds.train, continuous, f"{FIG_DIR}/histograms_continuous.png", ncols=6)
    report.image("figures/wadi/histograms_continuous.png", "Continuous sensor histograms")

    if discrete:
        report.para(f"All {len(discrete)} discrete status/alarm columns, training period:")
        plots.plot_actuator_bars(ds.train, discrete, f"{FIG_DIR}/bars_discrete.png", ncols=6)
        report.image("figures/wadi/bars_discrete.png", "Discrete status/alarm frequencies")

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
        f"One representative continuous tag per subsystem ({', '.join(sorted(seen))}), "
        f"across the full test period, downsampled for plotting; shaded bands are attack windows:"
    )
    test_ds = stats.downsample_for_plot(ds.test.assign(_label=ds.test_labels), max_points=8000)
    plots.plot_timeseries(
        test_ds, subsystem_reps, f"{FIG_DIR}/timeseries_by_subsystem.png",
        attack_mask=test_ds["_label"].to_numpy(),
    )
    report.image("figures/wadi/timeseries_by_subsystem.png", "Representative sensor per subsystem over time")

    # --- Correlation structure ---------------------------------------------
    report.h2("Correlation structure")
    corr = stats.correlation_matrix(ds.train, continuous)
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/correlation_heatmap.png")
    report.image("figures/wadi/correlation_heatmap.png", "Pairwise correlation, continuous sensors")
    report.para("Top 10 most correlated sensor pairs (training period):")
    report.table(stats.top_correlated_pairs(corr, n=10))

    # --- Class balance & attack segments -----------------------------------
    report.h2("Class balance & attack segments")
    balance = stats.class_balance(ds.test_labels)
    plots.plot_class_balance(balance, f"{FIG_DIR}/class_balance.png")
    report.image("figures/wadi/class_balance.png", "Normal vs. attack row counts (test set)")
    report.bullets([
        f"{balance['n_segments']} contiguous attack segments.",
        f"Segment length -- mean {balance['mean_segment_length']:.0f}, "
        f"median {balance['median_segment_length']:.0f}, max {balance['max_segment_length']} rows.",
    ])
    seg_path = plots.plot_segment_length_hist(balance["segment_lengths"], f"{FIG_DIR}/segment_lengths.png")
    if seg_path:
        report.image("figures/wadi/segment_lengths.png", "Attack segment length distribution")

    # --- Separability projection ------------------------------------------
    report.h2("Separability projection (PCA)")
    sample = ds.test.sample(n=min(PCA_SAMPLE_SIZE, len(ds.test)), random_state=0)
    labels_sample = ds.test_labels[sample.index.to_numpy()]
    X = (sample[continuous] - ds.train[continuous].mean()) / (ds.train[continuous].std() + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.to_numpy())
    plots.plot_pca_projection(coords, labels_sample, tuple(pca.explained_variance_ratio_), f"{FIG_DIR}/pca.png")
    report.image("figures/wadi/pca.png", "2D PCA projection, normal vs. attack",
                 caption=f"{min(PCA_SAMPLE_SIZE, len(ds.test)):,}-row sample; "
                         f"standardized using training-period mean/std.")

    # --- Dataset-specific notes ---------------------------------------------
    report.h2("WADI-specific notes")
    report.bullets([
        "Tags are prefixed by subsystem number (`1_`, `2_`, `2A_`, `2B_`, `3_`) "
        "and suffixed by signal type (`_PV` process value, `_STATUS`, `_AL` alarm, "
        "`_CO` control output, `_SP` setpoint).",
        f"{len(const_cols)} constant/empty column(s) in the raw data were dropped before modeling.",
        "The attack file's label column is `\"Attack LABLE (1:No Attack, -1:Attack)\"` "
        "-- note the embedded newline in the source header and the sic'd spelling; "
        "`src/data/wadi.py` selects it positionally rather than by exact name.",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
