#!/usr/bin/env python3
"""Generates reports/eda_z24.md and reports/figures/z24/*.png.

Unlike the other 3 EDA scripts, there's no train/test split or binary
label here -- each scenario is a distinct, mostly-irreversible structural
damage state, and each scenario's file already combines its 9 forced-
vibration setups (see src/data/z24.py). Analysis therefore centers on
cross-scenario comparison of the channels shared across (nearly) every
setup, rather than per-column distributions/correlations over one flat
table.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from data.z24 import SCENARIOS, load_z24_pdt
from eda import plots
from eda.report import MarkdownReport

FIG_DIR = "reports/figures/z24"
REPORT_PATH = "reports/eda_z24.md"
FS = 100.0  # Hz
SHARED_CHANNELS = ["R1V", "R2L", "R2T", "R2V", "R3V", "DP1V", "DP2V"]
SCENARIOS_SORTED = sorted(SCENARIOS)


def _setup_cols(df: pd.DataFrame, channel: str) -> list[str]:
    return sorted(c for c in df.columns if c.endswith("__" + channel))


def _psd_log_corr(cols_data: np.ndarray) -> float:
    """Mean pairwise correlation of log-PSD across setups (>=2 columns)."""
    logp = []
    for i in range(cols_data.shape[1]):
        _, p = welch(cols_data[:, i], fs=FS, nperseg=4096)
        logp.append(np.log10(p + 1e-12))
    corr = np.corrcoef(np.array(logp))
    n = corr.shape[0]
    return float(corr[np.triu_indices(n, k=1)].mean()) if n > 1 else float("nan")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    ds = load_z24_pdt()
    manifest = ds.manifest

    report = MarkdownReport("Z24 Bridge (PDT) — Exploratory Data Analysis")
    report.para(
        "Z24 Bridge Progressive Damage Test (PDT) campaign: 17 discrete, "
        "mostly-irreversible structural damage scenarios applied to a real "
        "bridge in 1998 before demolition, each measured with a forced-"
        "vibration test (shaker excitation). Source: `datasets/data-z24/pdt_*.zip`. "
        "Unlike SWaT/WADI/HAI, there is **no continuous time series or binary "
        "attack label** here -- each scenario is its own labeled structural "
        "state. Benchmark/method integration is deferred; see `src/data/z24.py` "
        "for how each scenario's 9 forced-vibration setups were combined into "
        "one file (setup-prefixed columns, e.g. `setup01__R2V`)."
    )

    # --- Overview -----------------------------------------------------
    report.h2("Overview")
    report.bullets([
        f"{len(manifest)} scenarios, 9 forced-vibration setups combined into each "
        f"(153 source `.mat` files total).",
        f"Sample rate: {FS:.0f} Hz. Samples per scenario: "
        f"{manifest.n_samples.min():,}-{manifest.n_samples.max():,} "
        f"(~{manifest.n_samples.min()/FS:.0f}-{manifest.n_samples.max()/FS:.0f}s).",
        f"Total channels per scenario (9 setups x 28-35 channels each, not "
        f"deduplicated): {manifest.n_channels_total.min()}-{manifest.n_channels_total.max()}.",
        f"{(manifest.label_confidence == 'inferred').sum()} scenario(s) have an "
        f"**inferred rather than confirmed** label (see Scenario notes below): "
        f"{', '.join(manifest.loc[manifest.label_confidence == 'inferred', 'scenario'])}.",
    ])

    # --- Data quality ---------------------------------------------------
    report.h2("Data quality")
    short = manifest[manifest.n_samples < manifest.n_samples.max()]
    report.para(
        f"Documented sample count is 65536 (Appendix J); actual per-scenario "
        f"sample count (after truncating each scenario's 9 setups to their "
        f"common length) is 65536 for {int((manifest.n_samples == 65536).sum())} "
        f"scenarios, but only {int(manifest.n_samples.min())} for scenario(s) "
        f"{', '.join(short.scenario)} -- a small, consistent trim, cause not "
        f"documented in the source material."
    )
    channel_counts = manifest.set_index("scenario").n_channels_total
    report.para(
        f"Channel count varies by scenario ({channel_counts.min()}-{channel_counts.max()}) "
        f"because not every setup carries every shared channel (e.g. `DP1V` is "
        f"documented as lost in scenario 01's setups 01 & 05 -- confirmed directly "
        f"in the data: {'present' if 'setup01__DP1V' in ds.load_scenario('01').columns else 'absent'} "
        f"in scenario 01 setup 01)."
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(channel_counts.index, channel_counts.values, color="#2a78d6")
    ax.set_xlabel("scenario")
    ax.set_ylabel("total channels (9 setups, undeduplicated)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/channel_counts.png")
    plt.close(fig)
    report.image("figures/z24/channel_counts.png", "Total channel count per scenario")

    # --- Same-forcing consistency check -----------------------------------
    report.h2("Same-forcing consistency check")
    report.para(
        "Each scenario's 9 setups share the same forcing, so the reference/"
        "driving-point channels that recur in nearly every setup should read "
        "consistently across setups *if* that assumption holds exactly. Raw "
        "time-domain correlation between setups turns out to be ~0 for these "
        "channels (confirmed: mean pairwise correlation for `R2V`/`DP2V` in "
        "scenario 01 was ~0.00) -- consistent with **stochastic (random-noise) "
        "shaker excitation**, where each setup gets a different random-noise "
        "realization rather than a repeated deterministic waveform, so sample-"
        "by-sample correlation isn't the right test. **Spectral** (PSD) "
        "correlation is the right test instead, and it's high (0.96-0.996 in "
        "spot checks) -- the excitation's statistical character, not its exact "
        "time trace, repeats across setups."
    )
    rows = []
    for scenario in SCENARIOS_SORTED:
        df = ds.load_scenario(scenario)
        for ch in SHARED_CHANNELS:
            cols = _setup_cols(df, ch)
            if len(cols) < 2:
                continue
            data = df[cols].to_numpy()
            rms = df[cols].std()
            rows.append(dict(
                scenario=scenario, channel=ch, n_setups=len(cols),
                psd_log_corr=_psd_log_corr(data),
                rms_ratio=float(rms.max() / rms.min()),
            ))
    consistency = pd.DataFrame(rows)
    per_scenario = consistency.groupby("scenario").agg(
        mean_psd_log_corr=("psd_log_corr", "mean"),
        max_rms_ratio=("rms_ratio", "max"),
    ).reindex(SCENARIOS_SORTED)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].bar(per_scenario.index, per_scenario.mean_psd_log_corr, color="#2a78d6")
    axes[0].set_ylabel("mean PSD log-corr\nacross setups")
    axes[1].bar(per_scenario.index, per_scenario.max_rms_ratio, color="#eb6834")
    axes[1].set_ylabel("max RMS ratio\nacross setups")
    axes[1].set_xlabel("scenario")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/consistency_check.png")
    plt.close(fig)
    report.image("figures/z24/consistency_check.png",
                 "Cross-setup spectral consistency of shared channels, per scenario")
    worst = per_scenario.mean_psd_log_corr.idxmin()
    report.para(
        f"Lowest mean spectral consistency: scenario {worst} "
        f"(mean PSD log-correlation {per_scenario.mean_psd_log_corr.loc[worst]:.3f}). "
        f"Full per-scenario numbers:"
    )
    report.table(per_scenario.reset_index())

    # --- Vibration amplitude across scenarios ------------------------------
    report.h2("Vibration amplitude across scenarios")
    report.para(
        "RMS amplitude of a representative shared channel (`R2V`, setup01) "
        "across all 17 scenarios -- a first-order look at whether response "
        "magnitude tracks damage progression:"
    )
    amps = []
    for scenario in SCENARIOS_SORTED:
        df = ds.load_scenario(scenario)
        col = "setup01__R2V"
        amps.append(df[col].std() if col in df.columns else np.nan)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(SCENARIOS_SORTED, amps, color="#1baf7a")
    ax.set_xlabel("scenario")
    ax.set_ylabel("R2V RMS (setup01)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/amplitude_by_scenario.png")
    plt.close(fig)
    report.image("figures/z24/amplitude_by_scenario.png", "Reference-channel RMS amplitude per scenario")

    # --- Time series --------------------------------------------------------
    report.h2("Time series")
    report.para("`R2V` (setup01), first 20s, for three contrasting scenarios:")
    example = {}
    for scenario in ["01", "06", "17"]:
        df = ds.load_scenario(scenario)
        col = "setup01__R2V"
        if col in df.columns:
            example[f"scenario {scenario} ({SCENARIOS[scenario]['label']})"] = df[col].iloc[:2000].to_numpy()
    fig, axes = plt.subplots(len(example), 1, figsize=(9, 2.0 * len(example)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, sig) in zip(axes, example.items()):
        ax.plot(np.arange(len(sig)) / FS, sig, color="#2a78d6", linewidth=1.0)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_ylabel("R2V", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/timeseries_examples.png")
    plt.close(fig)
    report.image("figures/z24/timeseries_examples.png", "R2V trace, reference vs. damaged scenarios")

    # --- Frequency-domain analysis -------------------------------------------
    report.h2("Frequency-domain analysis (PSD across all 17 scenarios)")
    report.para(
        "`R2V` (setup01) power spectral density for every scenario -- natural-"
        "frequency shift with progressive damage is the standard SHM signal "
        "for this exact dataset:"
    )
    signals = {}
    for scenario in SCENARIOS_SORTED:
        df = ds.load_scenario(scenario)
        col = "setup01__R2V"
        if col in df.columns:
            signals[f"{scenario}"] = df[col].to_numpy()
    plots.plot_psd_grid(signals, fs=FS, out_path=f"{FIG_DIR}/psd_grid.png", ncols=5)
    report.image("figures/z24/psd_grid.png", "PSD of R2V (setup01) per scenario")

    # --- Correlation among shared channels ----------------------------------
    report.h2("Correlation among shared channels")
    df01 = ds.load_scenario("01")
    shared_cols = [f"setup01__{ch}" for ch in SHARED_CHANNELS if f"setup01__{ch}" in df01.columns]
    missing = [ch for ch in SHARED_CHANNELS if f"setup01__{ch}" not in df01.columns]
    report.para(
        f"Within scenario 01, setup01: correlation among the {len(shared_cols)} of "
        f"{len(SHARED_CHANNELS)} candidate shared channels present here "
        f"(`{'`, `'.join(missing)}` absent in this particular setup, per the "
        f"documented channel loss noted above):"
    )
    corr = df01[shared_cols].corr()
    plots.plot_correlation_heatmap(corr, f"{FIG_DIR}/shared_channel_correlation.png")
    report.image("figures/z24/shared_channel_correlation.png", "Correlation among shared channels, scenario 01 setup01")

    # --- Scenario notes -------------------------------------------------------
    report.h2("Scenario notes")
    report.table(manifest[["scenario", "label", "label_confidence", "test_date", "notes"]])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
