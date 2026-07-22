"""Loader for the Z24 Bridge Progressive Damage Test (PDT) dataset.

Unlike SWaT/WADI/HAI, this isn't a single continuous time series with a
binary normal/attack label. It's 17 discrete, mostly-irreversible structural
damage scenarios (folders `01`-`17`) applied in sequence to a real bridge in
1998, each with a forced-vibration test (`fvt`/`FVT`) consisting of 9 "setup"
recordings -- the accelerometer array physically moved between setups, so
each setup has a different channel set. Ambient-vibration test data (`avt`)
is out of scope here (not extracted; see the unzip commands below).

Since all 9 setups within a scenario used the same forcing (per the user),
sample index is a valid common time axis across them, and `combine_scenario`
collapses each scenario's 9 setups into one wide DataFrame rather than
keeping 153 separate per-setup files. Every column is prefixed by its setup
(`setup01__139V`, ..., `setup09__DP2V`) -- including the reference/
driving-point channels that recur in (almost) every setup -- rather than
deduplicating them, since averaging or keeping only one copy would hide it
if the "same forcing" assumption doesn't hold exactly for a given scenario
(the source documentation already flags "strange behaviour" in a few
scenarios' repeated channels). See `scripts/eda_z24.py` for the consistency
check this makes possible.

Extraction (raw `.mat` files, not checked into git):
    mkdir -p datasets/raw/z24
    unzip -q datasets/data-z24/pdt_01-08.zip "*/fvt/*" "*/FVT/*" -d datasets/raw/z24
    unzip -q datasets/data-z24/pdt_09_17.zip "*/fvt/*" "*/FVT/*" -d datasets/raw/z24

`readme.txt` in datasets/data-z24/ confirms: each `.mat` file has a `data`
matrix (rows=samples, cols=channels) and a `labelshulp` array of channel
names, sampled at 100 Hz.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.io as sio

SCENARIOS: dict[str, dict] = {
    "01": dict(label="First reference measurement", confidence="confirmed",
               test_date="1998-08-04/1998-08-05",
               notes="Prior to Koppigen Pier installation (undamaged baseline, original bearings). "
                     "Cabling error lost signals 223,228,233,238,243 (renamed to Utzenstorf pier); "
                     "DP1V lost in setups 01 & 05; force from driving point 2 (DP2) must be "
                     "multiplied by a factor of 6.25."),
    "02": dict(label="Second reference measurement", confidence="confirmed",
               test_date="1998-08-09/1998-08-10",
               notes="After Koppigen Pier installation (new temporary support hardware installed "
                     "at Koppigen pier, still undamaged)."),
    "03": dict(label="Pier settlement, 20 mm", confidence="inferred",
               test_date="1998-08-11/1998-08-12",
               notes="No test report (.DOC) present in either zip for this scenario -- label and "
                     "date inferred from Appendix F's deflection progression (Level 1 = -20mm) and "
                     "the chronological gap between scenario 02 (09-10 Aug) and scenario 04 "
                     "(13-14 Aug, 40mm). Not a document quote."),
    "04": dict(label="Pier settlement, 40 mm", confidence="confirmed",
               test_date="1998-08-13/1998-08-14", notes="None."),
    "05": dict(label="Pier settlement, 80 mm", confidence="confirmed",
               test_date="1998-08-17/1998-08-18",
               notes="AVT report only (no FVT .DOC, though fvt/*.mat data exists). \"Some strange "
                     "behaviour of signals 100V,105V,110V... and so forth, 200V,205V,210V... "
                     "No explanation found\" (quoted from the source report)."),
    "06": dict(label="Pier settlement, 95 mm", confidence="confirmed",
               test_date="1998-08-18/1998-08-19",
               notes="Extra raw per-channel .aaa exports exist under avt/ for this scenario "
                     "(out of scope here since only fvt/ was extracted)."),
    "07": dict(label="Tilt of foundation", confidence="confirmed",
               test_date="1998-08-19/1998-08-20",
               notes="Relative difference of 6mm at Koppigen Pier."),
    "08": dict(label="Third reference measurement", confidence="confirmed",
               test_date="1998-08-20/1998-08-21",
               notes="Reference after settlement/tilt scenarios undone (back to nominal support "
                     "condition). Redundant nested avt.zip/fvt.zip duplicates of the same .mat "
                     "files also exist in the archive; not extracted (see module docstring)."),
    "09": dict(label="Spalling of concrete, 12 sq m", confidence="confirmed",
               test_date="1998-08-25/1998-08-26", notes="No measurements with sensor (KW)."),
    "10": dict(label="Spalling of concrete, 24 sq m", confidence="confirmed",
               test_date="1998-08-26/1998-08-27",
               notes="\"A strange behaviour in some points for an unknown reason\" (source report)."),
    "11": dict(label="Landslide", confidence="confirmed",
               test_date="1998-08-27/1998-08-28", notes="None."),
    "12": dict(label="Failure of concrete hinges at abutment pier(s)", confidence="confirmed",
               test_date="1998-08-31/1998-09-01", notes="None."),
    "13": dict(label="Failure of anchor heads of post-tensioning cables (1 head)", confidence="confirmed",
               test_date="1998-09-02/1998-09-03",
               notes="During setup 08 a big drift occurred in the signal from point 107V."),
    "14": dict(label="Failure of anchor heads of post-tensioning cables (4 heads)", confidence="confirmed",
               test_date="1998-09-03/1998-09-04", notes="None."),
    "15": dict(label="Rupture of tendons #1", confidence="confirmed",
               test_date="1998-09-03/1998-09-04",
               notes="The source AVT report's own date field is a likely copy-paste error "
                     "(identical to scenario 01's date, but the doc's creation timestamp is "
                     "08.09.98). Date here is inferred from chronological position between "
                     "scenario 14 (03-04 Sept) and scenario 16 (08-09 Sept), not a literal quote. "
                     "No FVT .DOC present, though fvt/*.mat data exists. \"Some strange behaviour "
                     "of signals 100V,105V,110V and so forth... No explanation found\" (source report)."),
    "16": dict(label="Rupture of tendons #2 (4 tendons cut)", confidence="confirmed",
               test_date="1998-09-08/1998-09-09", notes="None."),
    "17": dict(label="Rupture of tendons #3 (6 tendons cut)", confidence="inferred",
               test_date="1998-09-09/1998-09-10",
               notes="The AVT and FVT source reports for this scenario disagree: AVT calls it "
                     "\"Rupture of tendons #2\" (6 tendons), FVT calls it \"Rupture of tendons #3\". "
                     "Since scenario 16 already used \"#2\" for a 4-tendon cut, \"#3\" (this being "
                     "the 3rd cutting event) is used here as the best-effort resolution -- treat "
                     "as inferred, not a clean document quote."),
}


@dataclass
class Z24Dataset:
    manifest: pd.DataFrame
    root: str

    def load_scenario(self, scenario: str) -> pd.DataFrame:
        path = self.manifest.loc[self.manifest.scenario == scenario, "path"].iloc[0]
        return pd.read_parquet(path)


def _fvt_dir(root: str, scenario: str) -> str:
    for name in ("fvt", "FVT"):
        candidate = f"{root}/{scenario}/{name}"
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"No fvt/FVT folder found for scenario {scenario!r} under {root}/{scenario} -- "
        f"have you extracted the PDT zips yet? See the module docstring in src/data/z24.py."
    )


def _setup_files(root: str, scenario: str) -> list[str]:
    fvt_dir = _fvt_dir(root, scenario)
    paths = sorted(glob.glob(f"{fvt_dir}/{scenario}setup*.mat"))
    if not paths:
        raise FileNotFoundError(f"No {scenario}setup*.mat files found in {fvt_dir}")
    return paths


def load_setup_mat(path: str) -> pd.DataFrame:
    mat = sio.loadmat(path)
    labels = [str(l).strip() for l in mat["labelshulp"]]
    return pd.DataFrame(mat["data"], columns=labels)


def combine_scenario(root: str, scenario: str) -> pd.DataFrame:
    """Load all 9 fvt setups for `scenario` and concatenate column-wise,
    prefixing every column by its setup number (see module docstring for why
    shared channels aren't deduplicated)."""
    paths = _setup_files(root, scenario)
    frames = []
    for path in paths:
        setup_id = os.path.basename(path).split("setup")[1].split(".")[0]
        df = load_setup_mat(path)
        df.columns = [f"setup{setup_id}__{c}" for c in df.columns]
        frames.append(df)
    n_samples = min(len(f) for f in frames)
    return pd.concat([f.iloc[:n_samples].reset_index(drop=True) for f in frames], axis=1)


def build_manifest(root: str = "datasets/raw/z24", out_dir: str = "datasets/raw/z24/combined") -> pd.DataFrame:
    """Combines every scenario's 9 setups into one parquet file each and
    writes/returns the manifest (one row per scenario)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for scenario in sorted(SCENARIOS):
        combined = combine_scenario(root, scenario)
        out_path = f"{out_dir}/{scenario}.parquet"
        combined.to_parquet(out_path)
        meta = SCENARIOS[scenario]
        rows.append(dict(
            scenario=scenario,
            label=meta["label"],
            label_confidence=meta["confidence"],
            test_date=meta["test_date"],
            notes=meta["notes"],
            n_samples=len(combined),
            n_channels_total=combined.shape[1],
            path=out_path,
        ))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(f"{root}/manifest.csv", index=False)
    return manifest


def load_z24_pdt(root: str = "datasets/raw/z24") -> Z24Dataset:
    manifest_path = f"{root}/manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found -- run scripts/build_z24_manifest.py first."
        )
    return Z24Dataset(manifest=pd.read_csv(manifest_path, dtype={"scenario": str}), root=root)
