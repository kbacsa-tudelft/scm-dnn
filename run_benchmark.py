#!/usr/bin/env python3
"""CLI entry point: compare CDT and PbNN across SWaT/WADI/HAI.

Examples:
    python run_benchmark.py                              # everything, full data
    python run_benchmark.py --datasets swat --nrows 20000 # quick smoke test
    python run_benchmark.py --methods cdt --datasets wadi
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.swat import load_swat
from data.wadi import load_wadi
from data.hai import load_hai
from methods.cdt.model import CDT
from methods.pbnn.model import PbNN
from benchmark.runner import run_benchmark
from benchmark.report import print_report, save_report

_DATASET_LOADERS = {
    "swat": lambda nrows: load_swat(nrows=nrows),
    "wadi": lambda nrows: load_wadi(nrows=nrows),
    "hai": lambda nrows: load_hai(nrows=nrows),
}

_METHOD_FACTORIES = {
    "cdt": lambda: CDT(),
    "pbnn": lambda: PbNN(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="swat,wadi,hai", help="Comma-separated subset of: swat,wadi,hai")
    parser.add_argument("--methods", default="cdt,pbnn", help="Comma-separated subset of: cdt,pbnn")
    parser.add_argument("--nrows", type=int, default=None, help="Cap rows read per CSV, for fast dev runs")
    parser.add_argument("--out", default="results", help="Directory to write results.csv/results.json")
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    method_names = args.methods.split(",")

    datasets = {}
    for name in dataset_names:
        print(f"Loading {name}...", file=sys.stderr)
        datasets[name] = _DATASET_LOADERS[name](args.nrows)
        print(f"  train={datasets[name].train.shape} test={datasets[name].test.shape} "
              f"attacks={int(datasets[name].test_labels.sum())}/{len(datasets[name].test_labels)}",
              file=sys.stderr)

    methods = {name: _METHOD_FACTORIES[name] for name in method_names}

    results = run_benchmark(methods, datasets)

    os.makedirs(args.out, exist_ok=True)
    save_report(results, args.out)
    print_report(results)
    print(f"\nSaved to {args.out}/results.csv and {args.out}/results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
