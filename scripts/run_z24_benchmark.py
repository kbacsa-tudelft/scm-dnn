#!/usr/bin/env python3
"""Runs CDT and PbNN across all 14 Z24 healthy-vs-damaged pairings (see
src/data/z24_anomaly.py) and reports results the same way as run_benchmark.py.

This reuses src/benchmark/{runner,report}.py rather than duplicating them --
Z24 doesn't fit run_benchmark.py's fixed --datasets swat,wadi,hai design
since each "dataset" here is parameterized by which damaged scenario it's
paired with, so it gets its own entry point instead of a CLI flag.

Note: Z24's combined per-scenario files have ~300 columns (9 setups x
28-35 channels each, not deduplicated) -- far wider than SWaT/WADI/HAI
(37-127 columns) -- and CDT's discovery phase loops once per column, so
per-pairing fit time is substantially longer here even at the default
5,000-row discovery subsample. Budget accordingly; see the module docstring
in src/methods/cdt/discovery.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.z24_anomaly import DAMAGED_SCENARIOS, build_z24_binary_dataset
from methods.cdt.model import CDT
from methods.pbnn.model import PbNN
from benchmark.runner import run_benchmark
from benchmark.report import print_report, save_report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=",".join(DAMAGED_SCENARIOS),
                         help=f"Comma-separated subset of damaged scenarios: {','.join(DAMAGED_SCENARIOS)}")
    parser.add_argument("--methods", default="cdt,pbnn", help="Comma-separated subset of: cdt,pbnn")
    parser.add_argument("--out", default="results/z24", help="Directory to write results.csv/results.json")
    args = parser.parse_args()

    scenarios = args.scenarios.split(",")
    method_names = args.methods.split(",")

    print(f"Building {len(scenarios)} healthy-vs-damaged pairings...", file=sys.stderr)
    datasets = {}
    for scenario in scenarios:
        ds = build_z24_binary_dataset(damaged=scenario)
        datasets[ds.name] = ds
        print(f"  {ds.name}: train={ds.train.shape} test={ds.test.shape} "
              f"attack_rate={ds.test_labels.mean():.3f}", file=sys.stderr)

    method_factories = {"cdt": lambda: CDT(), "pbnn": lambda: PbNN()}
    methods = {name: method_factories[name] for name in method_names}

    results = run_benchmark(methods, datasets)

    os.makedirs(args.out, exist_ok=True)
    save_report(results, args.out)
    print_report(results)
    print(f"\nSaved to {args.out}/results.csv and {args.out}/results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
