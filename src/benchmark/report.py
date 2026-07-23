"""Formats RunResult lists as a printable table and saves them to disk."""
from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .runner import RunResult

_COLUMNS = [
    "method", "dataset", "precision", "recall", "f1",
    "precision_pa", "recall_pa", "f1_pa",
    "detection_rate", "false_alarm_rate", "conflict_index_factor",
    "shd", "edge_precision", "edge_recall", "edge_f1",
    "fit_seconds", "score_seconds", "error", "config",
]


def to_dataframe(results: list[RunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"method": r.method, "dataset": r.dataset, "fit_seconds": round(r.fit_seconds, 2),
               "score_seconds": round(r.score_seconds, 2), "error": r.error,
               "config": json.dumps(r.config) if r.config is not None else None}
        row.update(r.metrics)
        if r.graph_metrics:
            row.update(r.graph_metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[_COLUMNS]


def print_report(results: list[RunResult]) -> None:
    df = to_dataframe(results)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


def save_report(results: list[RunResult], out_dir: str) -> None:
    df = to_dataframe(results)
    df.to_csv(f"{out_dir}/results.csv", index=False)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
