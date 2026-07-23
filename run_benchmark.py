#!/usr/bin/env python3
"""CLI entry point: compare CDT, PbNN and CNN1D across SWaT/WADI/HAI.

Examples:
    python run_benchmark.py                              # everything, full data
    python run_benchmark.py --datasets swat --nrows 20000 # quick smoke test
    python run_benchmark.py --methods cdt --datasets wadi
    python run_benchmark.py --methods cnn1d --cnn1d-ensemble per_stage
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.swat import load_swat
from data.wadi import load_wadi
from data.hai import load_hai
from data.batadal import load_batadal
from data.tep import load_tep
from methods.cdt.model import CDT
from methods.pbnn.model import PbNN
from methods.cnn1d.model import CNN1D
from benchmark.runner import run_benchmark
from benchmark.report import print_report, save_report

_DATASET_LOADERS = {
    "swat": lambda nrows: load_swat(nrows=nrows),
    "wadi": lambda nrows: load_wadi(nrows=nrows),
    "hai": lambda nrows: load_hai(nrows=nrows),
    "batadal": lambda nrows: load_batadal(nrows=nrows),
    "tep": lambda nrows: load_tep(nrows=nrows),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="swat,wadi,hai,batadal",
                         help="Comma-separated subset of: swat,wadi,hai,batadal,tep "
                              "(tep excluded from the default set -- its full-scale test set is "
                              "~10x SWaT's; pass --datasets tep explicitly, ideally with --nrows)")
    parser.add_argument("--methods", default="cdt,pbnn,cnn1d", help="Comma-separated subset of: cdt,pbnn,cnn1d")
    parser.add_argument("--nrows", type=int, default=None, help="Cap rows read per CSV, for fast dev runs")
    parser.add_argument("--out", default="results", help="Directory to write results.csv/results.json")
    parser.add_argument(
        "--cdt-discovery-rows", type=int, default=None,
        help="Rows CDT's discovery phase subsamples for CI testing (default: CDT's own, 5000). "
             "Discovery time scales worse than linearly with this (measured on SWaT: "
             "5k rows/37s/17 edges, 20k/192s/27 edges, 50k/608s/30 edges) -- 50000 is a "
             "reasonable 'thorough' setting if you want more paper-scale discovery and can "
             "spare the time (expect several minutes per dataset, more on WADI/HAI's wider "
             "column sets).",
    )
    parser.add_argument(
        "--cnn1d-ensemble", choices=["combined", "per_stage"], default="combined",
        help="CNN1D: one model over all sensors (default, dataset-agnostic) or the paper's "
             "per-stage/subsystem ensemble (its own best-reported result on SWaT, generalized "
             "to WADI/HAI/Z24 via the same subsystem-grouping heuristic PbNN's invariants use).",
    )
    parser.add_argument(
        "--cnn1d-warmup-trim", type=int, default=0,
        help="CNN1D: rows to trim from the start of training (paper trims SWaT's first 16000 "
             "rows as an unstable warm-up period; 0 = no trim, since this is SWaT-specific and "
             "not auto-applied).",
    )
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

    def make_cdt():
        kwargs = {}
        if args.cdt_discovery_rows is not None:
            kwargs["discovery_subsample_rows"] = args.cdt_discovery_rows
        return CDT(**kwargs)

    def make_cnn1d():
        return CNN1D(ensemble=args.cnn1d_ensemble, warmup_trim=args.cnn1d_warmup_trim)

    method_factories = {"cdt": make_cdt, "pbnn": lambda: PbNN(), "cnn1d": make_cnn1d}
    methods = {name: method_factories[name] for name in method_names}

    results = run_benchmark(methods, datasets)

    os.makedirs(args.out, exist_ok=True)
    save_report(results, args.out)
    print_report(results)
    print(f"\nSaved to {args.out}/results.csv and {args.out}/results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
