#!/usr/bin/env python3
"""One-time processing step: combines each Z24 PDT scenario's 9 forced-
vibration-test setups into datasets/raw/z24/combined/<scenario>.parquet and
writes datasets/raw/z24/manifest.csv. Run after extracting the raw .mat
files (see src/data/z24.py's module docstring for the unzip commands).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.z24 import build_manifest


def main() -> None:
    manifest = build_manifest()
    print(manifest.to_string(index=False))
    print(f"\nWrote datasets/raw/z24/manifest.csv and {len(manifest)} combined parquet files.")


if __name__ == "__main__":
    main()
