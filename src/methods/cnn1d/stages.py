"""Per-stage ensemble (Kravchik & Shabtai's headline result: 5 separate
per-SWaT-stage models OR'd together, F1 0.886 vs. the combined single-model's
0.767). The paper's stages are SWaT's native P1-P5 process stages (P6
excluded -- "some of the sensors and actuators... were not used for data
collection" for P6, per the paper).

WADI/HAI/Z24 have no equivalent stage table in this paper at all -- it only
evaluates SWaT. Generalizing to them via the same subsystem-prefix grouping
already used for PbNN's invariants (`subsystem_grouping.py`) is this
codebase's own extension, not from the paper.
"""
from __future__ import annotations

from ..subsystem_grouping import group_columns, guess_dataset


def stage_groups(columns: list[str]) -> dict[str, list[str]]:
    """Maps stage/subsystem name -> its columns. For SWaT this reproduces
    the paper's P1-P6 grouping exactly (P6 columns, if present, form their
    own group here rather than being dropped -- the paper excluded P6 only
    because its own dataset lacked usable P6 sensors, not as a methodological
    choice); for WADI/HAI/Z24 it's the same subsystem-prefix heuristic PbNN
    uses."""
    dataset_name = guess_dataset(columns)
    groups = group_columns(columns, dataset_name)
    ungrouped = [c for c in columns if c not in {col for cols in groups.values() for col in cols}]
    if ungrouped:
        groups["ungrouped"] = ungrouped
    return groups
