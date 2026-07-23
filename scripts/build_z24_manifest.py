#!/usr/bin/env python3
"""One-time processing step: combines each Z24 PDT scenario's 9 setups
(forced-vibration by default, or ambient via --test-type avt) into
datasets/raw/z24/combined[_avt]/<scenario>.parquet and writes the matching
manifest CSV. Run after extracting the raw .mat files (see src/data/z24.py's
module docstring for the unzip commands, both fvt/FVT and avt/AVT).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.z24 import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-type", default="fvt", choices=["fvt", "avt"],
                         help="Which campaign to combine (default: fvt)")
    args = parser.parse_args()

    manifest = build_manifest(test_type=args.test_type)
    print(manifest.to_string(index=False))
    manifest_name = "manifest.csv" if args.test_type == "fvt" else f"manifest_{args.test_type}.csv"
    print(f"\nWrote datasets/raw/z24/{manifest_name} and {len(manifest)} combined parquet files.")


if __name__ == "__main__":
    main()
