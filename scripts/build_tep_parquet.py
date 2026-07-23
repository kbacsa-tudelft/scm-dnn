#!/usr/bin/env python3
"""One-time processing step: converts the TEP RData files to parquet
(TEP_Faulty_Testing.RData alone is ~800MB compressed / several GB as an
in-memory R dataframe, and reading RData is slow -- convert once here,
load fast parquet thereafter). Requires `pyreadr` (pure-Python RData
reader, no R installation needed to *read* -- R itself was used upstream
by the dataset's original authors to produce the RData files).

Only fault_free_training, fault_free_testing and faulty_testing are used:
faulty_training is never needed since every method in this harness trains
on fault-free data only (see src/data/tep.py's module docstring) -- it's
not even extracted from the zip, saving ~500MB/time.
"""
from __future__ import annotations

import os

import pyreadr

ROOT = "datasets/raw/tep/tep_files"
FILES = {
    "fault_free_training": "TEP_FaultFree_Training.RData",
    "fault_free_testing": "TEP_FaultFree_Testing.RData",
    "faulty_testing": "TEP_Faulty_Testing.RData",
}


def main() -> None:
    for name, filename in FILES.items():
        out_path = f"{ROOT}/{name}.parquet"
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping")
            continue
        print(f"Reading {filename}...")
        result = pyreadr.read_r(f"{ROOT}/{filename}")
        df = result[name]
        print(f"  shape={df.shape}, writing {out_path}")
        df.to_parquet(out_path)
        del df, result


if __name__ == "__main__":
    main()
