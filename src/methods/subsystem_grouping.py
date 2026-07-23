"""Shared "group these columns by dataset-specific subsystem naming
convention" heuristics, used by both PbNN's invariant inference
(`pbnn/invariants.py`) and CNN1D's per-stage ensemble (`cnn1d/stages.py`) --
factored out here rather than duplicated a second time, since it's the same
generic idea (infer which sensors belong to the same physical subsystem from
tag-naming convention alone) applied by two different methods.

None of this is from either paper -- it's this repo's own extension of both
methods' ideas to datasets/groupings they don't natively define.
"""
from __future__ import annotations

import re


def guess_dataset(columns: list[str]) -> str:
    """Best-effort dataset identification from column-naming convention alone."""
    if any(c.startswith("setup") and "__" in c for c in columns):
        return "z24"
    if any(c.startswith(("1_", "2_", "3_")) for c in columns):
        return "wadi"
    if any(c.startswith(("P1_", "P2_", "P3_", "P4_")) for c in columns):
        return "hai"
    return "swat"


def subsystem_of(col: str, dataset_name: str) -> str | None:
    """The subsystem/process-stage tag embedded in a single column name, or
    None if the naming convention for `dataset_name` doesn't yield one."""
    if dataset_name == "swat":
        # Tags are "<TYPE><stage><id>" e.g. FIT101 -> stage 1, LIT301 -> stage 3
        # (see scripts/eda_swat.py's `_stage` helper, same regex).
        m = re.search(r"(\d)\d{2}", col)
        return f"P{m.group(1)}" if m else None
    if dataset_name == "wadi":
        m = re.match(r"^(\d+[A-Za-z]?)_", col)
        return m.group(1) if m else None
    if dataset_name.startswith("hai"):
        m = re.match(r"^(P\d+)_", col)
        return m.group(1) if m else None
    if dataset_name.startswith("z24"):
        # Z24 columns are "setupNN__channel" (see data/z24.py) -- each setup
        # is its own physically-moved sensor array, the natural "subsystem"
        # grouping here.
        m = re.match(r"^(setup\d+)__", col)
        return m.group(1) if m else None
    return None


def group_columns(columns: list[str], dataset_name: str) -> dict[str, list[str]]:
    """Maps subsystem tag -> the columns belonging to it, in first-seen order."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        prefix = subsystem_of(col, dataset_name)
        if prefix is not None:
            groups.setdefault(prefix, []).append(col)
    return groups
