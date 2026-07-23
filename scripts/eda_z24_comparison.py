#!/usr/bin/env python3
"""Generates reports/eda_z24_comparison.md and reports/figures/z24_comparison/*.png:
Ambient (AVT) vs. Forced (FVT) vibration, Z24 Bridge PDT campaign.

Scoped deliberately to what's actually comparable: the 5 reference channels
(R1V, R2L, R2T, R2V, R3V) present in both campaigns' setups. The other
~285-300 columns in each campaign's combined files are campaign-specific
roving-array channels covering non-overlapping bridge segments (confirmed
in src/data/z24.py's module docstring) and are not compared here.
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
from eda.report import MarkdownReport
from eda.style import categorical_color

FIG_DIR = "reports/figures/z24_comparison"
REPORT_PATH = "reports/eda_z24_comparison.md"
FS = 100.0
SHARED_CHANNELS = ["R1V", "R2L", "R2T", "R2V", "R3V"]
SCENARIOS_SORTED = sorted(SCENARIOS)
REP_CHANNEL = "R2V"  # representative channel, matches both per-campaign reports


def _psd(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return welch(sig, fs=FS, nperseg=4096)


def _psd_log_corr(a: np.ndarray, b: np.ndarray) -> float:
    _, pa = _psd(a)
    _, pb = _psd(b)
    return float(np.corrcoef(np.log10(pa + 1e-12), np.log10(pb + 1e-12))[0, 1])


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fvt = load_z24_pdt(test_type="fvt")
    avt = load_z24_pdt(test_type="avt")

    report = MarkdownReport("Z24 Bridge (PDT) — Ambient (AVT) vs. Forced (FVT) Vibration Comparison")
    report.para(
        "Compares the two Z24 PDT test campaigns per scenario: "
        "[`reports/eda_z24.md`](eda_z24.md) (Forced Vibration Test, shaker-driven) "
        "and [`reports/eda_z24_avt.md`](eda_z24_avt.md) (Ambient Vibration Test, "
        "traffic/wind-driven). **Scope note**: AVT and FVT use different roving-"
        "array sensor grids with zero location-channel overlap (confirmed directly "
        "from the `.mat` files -- see `src/data/z24.py`'s module docstring) -- "
        "only the 5 reference channels (`R1V`,`R2L`,`R2T`,`R2V`,`R3V`), present in "
        "nearly every setup of both campaigns, allow an apples-to-apples "
        "comparison. Everything below uses setup01's copy of these channels, "
        "consistent with the representative-channel choice in both per-campaign "
        "reports. This is not a comparison of the full ~300-column datasets."
    )

    # --- Amplitude comparison ------------------------------------------------
    report.h2("Amplitude: ambient vs. forced excitation")
    report.para(
        f"RMS amplitude of `{REP_CHANNEL}` (setup01), FVT vs. AVT, per scenario -- "
        f"ambient (traffic/wind) excitation is expected to be much lower-energy "
        f"than shaker-driven forced excitation:"
    )
    rows = []
    for scenario in SCENARIOS_SORTED:
        fvt_df = fvt.load_scenario(scenario)
        avt_df = avt.load_scenario(scenario)
        col = f"setup01__{REP_CHANNEL}"
        fvt_rms = fvt_df[col].std() if col in fvt_df.columns else np.nan
        avt_rms = avt_df[col].std() if col in avt_df.columns else np.nan
        rows.append(dict(scenario=scenario, fvt_rms=fvt_rms, avt_rms=avt_rms,
                          ratio_fvt_over_avt=fvt_rms / avt_rms if avt_rms else np.nan))
    amp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(amp))
    width = 0.38
    ax.bar(x - width / 2, amp.fvt_rms, width, label="FVT (forced)", color=categorical_color(0))
    ax.bar(x + width / 2, amp.avt_rms, width, label="AVT (ambient)", color=categorical_color(5))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(amp.scenario)
    ax.set_xlabel("scenario")
    ax.set_ylabel(f"{REP_CHANNEL} RMS (log scale)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/amplitude_comparison.png")
    plt.close(fig)
    report.image("figures/z24_comparison/amplitude_comparison.png",
                 "RMS amplitude, FVT vs AVT, per scenario (log scale)")
    report.para(
        f"FVT/AVT RMS ratio ranges {amp.ratio_fvt_over_avt.min():.1f}x-"
        f"{amp.ratio_fvt_over_avt.max():.1f}x across scenarios (median "
        f"{amp.ratio_fvt_over_avt.median():.1f}x) -- forced excitation is "
        f"consistently and substantially higher-energy, as expected for a "
        f"shaker vs. ambient traffic/wind."
    )

    # --- PSD shape comparison -------------------------------------------------
    report.h2("PSD shape: same structural resonances, different excitation")
    report.para(
        f"`{REP_CHANNEL}` (setup01) power spectral density, FVT and AVT overlaid, "
        f"per scenario -- if the same structural resonances show up under both "
        f"excitation types (as expected, same bridge), peaks should broadly align "
        f"even though absolute power differs (per the amplitude comparison above):"
    )
    n = len(SCENARIOS_SORTED)
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), sharex=True)
    axes = axes.flatten()
    for ax, scenario in zip(axes, SCENARIOS_SORTED):
        col = f"setup01__{REP_CHANNEL}"
        fvt_df, avt_df = fvt.load_scenario(scenario), avt.load_scenario(scenario)
        if col in fvt_df.columns:
            f, p = _psd(fvt_df[col].to_numpy())
            ax.semilogy(f, p, color=categorical_color(0), linewidth=0.9, label="FVT")
        if col in avt_df.columns:
            f, p = _psd(avt_df[col].to_numpy())
            ax.semilogy(f, p, color=categorical_color(5), linewidth=0.9, label="AVT")
        ax.set_title(scenario, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/psd_comparison_grid.png")
    plt.close(fig)
    report.image("figures/z24_comparison/psd_comparison_grid.png",
                 "PSD of R2V, FVT vs AVT overlaid, per scenario")

    # --- Cross-campaign spectral correlation ----------------------------------
    report.h2("Cross-campaign spectral correlation")
    report.para(
        f"Log-PSD correlation between FVT's and AVT's `{REP_CHANNEL}` (setup01) "
        f"per scenario -- a single number summarizing how much the resonance "
        f"structure agrees despite the different excitation source (this is "
        f"necessarily lower than either campaign's own cross-setup consistency, "
        f"reported in the two per-campaign reports, since it's comparing across "
        f"excitation types, not just across setups of the same type):"
    )
    corr_rows = []
    for scenario in SCENARIOS_SORTED:
        col = f"setup01__{REP_CHANNEL}"
        fvt_df, avt_df = fvt.load_scenario(scenario), avt.load_scenario(scenario)
        if col in fvt_df.columns and col in avt_df.columns:
            corr_rows.append(dict(scenario=scenario,
                                   cross_campaign_psd_log_corr=_psd_log_corr(
                                       fvt_df[col].to_numpy(), avt_df[col].to_numpy())))
    cross_corr = pd.DataFrame(corr_rows)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(cross_corr.scenario, cross_corr.cross_campaign_psd_log_corr, color=categorical_color(4))
    ax.set_xlabel("scenario")
    ax.set_ylabel("FVT-AVT PSD\nlog-correlation")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/cross_campaign_correlation.png")
    plt.close(fig)
    report.image("figures/z24_comparison/cross_campaign_correlation.png",
                 "Cross-campaign (FVT vs AVT) spectral correlation, per scenario")
    report.table(cross_corr)

    # --- What isn't comparable -------------------------------------------------
    report.h2("What this comparison does not cover")
    report.bullets([
        "The ~285-304 campaign-specific (non-reference) channels in each "
        "combined file cover non-overlapping bridge segments between AVT and "
        "FVT -- there is no shared ground truth to compare them against directly.",
        "AVT has no driving-point channels (`DP1V`/`DP2V`) at all -- no shaker, "
        "nothing to co-locate a sensor with.",
        "Sample-count and channel-count irregularities differ between the two "
        "campaigns (see each per-campaign report's Data quality section) -- "
        "this report uses setup01 specifically, which is unaffected in both.",
    ])

    report.write(REPORT_PATH)
    print(f"Wrote {REPORT_PATH} and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
