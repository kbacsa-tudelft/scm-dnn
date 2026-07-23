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
import pyarrow.parquet as pq


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw_swat(root: str = "datasets/raw/swat") -> dict[str, pd.DataFrame]:
    return {
        "normal": _strip_cols(pd.read_csv(f"{root}/normal.csv")),
        "attack": _strip_cols(pd.read_csv(f"{root}/attack.csv")),
    }


def load_raw_wadi(root: str = "datasets/raw/wadi") -> dict[str, pd.DataFrame]:
    # encoding="latin-1": WADI's CSVs carry stray non-UTF-8 bytes on some
    # machines/pandas versions (see src/data/wadi.py's module docstring).
    return {
        "days14": _strip_cols(pd.read_csv(f"{root}/WADI_14days_new.csv", encoding="latin-1")),
        "attack": _strip_cols(pd.read_csv(f"{root}/WADI_attackdataLABLE.csv", skiprows=1, encoding="latin-1")),
    }


def load_raw_hai(root: str = "datasets/raw/hai", version: str = "hai-22.04") -> dict[str, dict[str, pd.DataFrame]]:
    version_dir = f"{root}/{version}"
    train = {p.split("/")[-1]: pd.read_csv(p) for p in sorted(glob.glob(f"{version_dir}/train*.csv"))}
    test = {p.split("/")[-1]: pd.read_csv(p) for p in sorted(glob.glob(f"{version_dir}/test*.csv"))}
    return {"train": train, "test": test}


def load_raw_batadal(root: str = "datasets/raw/batadal/batadal") -> dict[str, pd.DataFrame]:
    return {
        "dataset03": _strip_cols(pd.read_csv(f"{root}/BATADAL_dataset03.csv")),
        "dataset04": _strip_cols(pd.read_csv(f"{root}/BATADAL_dataset04.csv")),
    }


def _read_parquet_capped(path: str, nrows: int | None) -> pd.DataFrame:
    """Full read when nrows is None; otherwise reads only the first row
    group(s) covering `nrows` via pyarrow directly, never materializing the
    whole file. `faulty_testing.parquet` alone is ~9.6M rows/~800MB on disk
    (several GB decompressed) -- an uncapped read of it plus the other two
    TEP files, plus the several DataFrame copies EDA/cleaning code makes
    along the way, was enough to OOM this environment in practice. Same
    approach as src/data/tep.py's loader."""
    if nrows is None:
        return pd.read_parquet(path)
    batches = pq.ParquetFile(path).iter_batches(batch_size=nrows)
    return next(batches).to_pandas()


def load_raw_tep(root: str = "datasets/raw/tep/tep_files", nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """"Raw" here means the parquet files as converted from RData (see
    scripts/build_tep_parquet.py) -- the conversion doesn't clean/alter any
    values, only the file format, so these are a faithful uncleaned view.
    `nrows` caps rows read per file -- see `_read_parquet_capped`; leave it
    unset only on a machine with plenty of spare RAM (measured OOM on this
    dataset's full scale during EDA -- see scripts/eda_tep.py)."""
    return {
        "fault_free_training": _read_parquet_capped(f"{root}/fault_free_training.parquet", nrows),
        "fault_free_testing": _read_parquet_capped(f"{root}/fault_free_testing.parquet", nrows),
        "faulty_testing": _read_parquet_capped(f"{root}/faulty_testing.parquet", nrows),
    }
