"""Raw (uncleaned) dataset loading for EDA purposes.

Deliberately separate from `src/data/{swat,wadi,hai}.py`: those loaders
coerce dtypes, forward/back-fill NaNs, drop constant columns and z-score --
all things an EDA report needs to characterize as *findings*, not have
silently fixed before it sees the data. Parsing mechanics (which rows/columns
to skip, how each file's label convention works) intentionally mirror those
loaders since that part isn't "cleaning", just correct file reading.
"""
from __future__ import annotations

import glob

import pandas as pd


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw_swat(root: str = "datasets/raw/swat") -> dict[str, pd.DataFrame]:
    return {
        "normal": _strip_cols(pd.read_csv(f"{root}/normal.csv")),
        "attack": _strip_cols(pd.read_csv(f"{root}/attack.csv")),
    }


def load_raw_wadi(root: str = "datasets/raw/wadi") -> dict[str, pd.DataFrame]:
    return {
        "days14": _strip_cols(pd.read_csv(f"{root}/WADI_14days_new.csv")),
        "attack": _strip_cols(pd.read_csv(f"{root}/WADI_attackdataLABLE.csv", skiprows=1)),
    }


def load_raw_hai(root: str = "datasets/raw/hai", version: str = "hai-22.04") -> dict[str, dict[str, pd.DataFrame]]:
    version_dir = f"{root}/{version}"
    train = {p.split("/")[-1]: pd.read_csv(p) for p in sorted(glob.glob(f"{version_dir}/train*.csv"))}
    test = {p.split("/")[-1]: pd.read_csv(p) for p in sorted(glob.glob(f"{version_dir}/test*.csv"))}
    return {"train": train, "test": test}
