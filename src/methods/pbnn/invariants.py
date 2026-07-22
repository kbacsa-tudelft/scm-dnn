"""Process invariants for the PbNN method (Raman & Mathur 2022, Table III).

Table III in the paper is itself ambiguous for I5/I6 (merged/blank target
cells), and a 7th invariant (I7) is plotted in Fig. 4 but never defined in
the table at all -- I7 is dropped here rather than guessed. I5/I6 targets
below are a best-effort, physically-motivated completion (I5's target is
blank in the source; FIT301 is the only sensor among its listed predictors
that isn't already a predictor elsewhere upstream. I6's target is blank;
LIT401 is chosen as the natural downstream level sensor fed by
MV201/P301), not a verbatim transcription of the paper.

WADI and HAI have no equivalent invariant table in this paper at all -- it
only evaluates SWaT. `infer_invariants` is our own extension of the
*method* (group columns by their subsystem-prefix tag naming, treat
sensor-like tags as targets, other same-subsystem tags as predictors), not
something described in the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Invariant:
    name: str
    target: str
    predictors: list[str]


_SWAT_TABLE: list[tuple[str, str, list[str]]] = [
    ("I1", "FIT101", ["FIT101", "MV101"]),
    ("I2", "LIT101", ["FIT101", "MV101", "FIT201"]),
    ("I3", "FIT201", ["LIT101", "MV101", "P601", "P101"]),
    ("I4", "LIT301", ["FIT201", "MV201", "P101"]),
    ("I5", "FIT301", ["LIT301", "FIT201"]),
    ("I6", "LIT401", ["LIT301", "MV201", "P301"]),
]


def swat_invariants(columns: list[str]) -> list[Invariant]:
    present = set(columns)
    invariants = []
    for name, target, predictors in _SWAT_TABLE:
        preds = [p for p in predictors if p in present and p != target]
        if target in present and preds:
            invariants.append(Invariant(name, target, preds))
    return invariants


_SENSOR_HINTS = ("PV", "FIT", "LIT", "AIT", "PIT", "FT", "LT", "AT", "PT", "TIT")


def _subsystem_prefix(col: str, dataset_name: str) -> str | None:
    if dataset_name == "wadi":
        m = re.match(r"^(\d+[A-Za-z]?)_", col)
        return m.group(1) if m else None
    if dataset_name.startswith("hai"):
        m = re.match(r"^(P\d+)_", col)
        return m.group(1) if m else None
    if dataset_name.startswith("z24"):
        # Z24 columns are "setupNN__channel" (see data/z24.py) -- each setup
        # is its own physically-moved sensor array, the natural "subsystem"
        # grouping here. No sensor-type hints match Z24's location+direction
        # channel codes (e.g. "139V", "DP2V"), so target selection within a
        # group falls back to `infer_invariants`'s arbitrary-first-N choice.
        m = re.match(r"^(setup\d+)__", col)
        return m.group(1) if m else None
    return None


def infer_invariants(
    columns: list[str],
    dataset_name: str,
    max_invariants: int = 8,
    max_predictors: int = 4,
) -> list[Invariant]:
    """Group columns by subsystem-prefix; within each subsystem pick one or two
    sensor-like columns as targets and a few other same-subsystem columns as
    predictors. Our own generalization of the paper's invariant idea to
    datasets it never covers -- not from the paper."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        prefix = _subsystem_prefix(col, dataset_name)
        if prefix is not None:
            groups.setdefault(prefix, []).append(col)

    invariants = []
    for prefix, cols in sorted(groups.items()):
        if len(cols) < 2:
            continue
        sensor_cols = [c for c in cols if any(h in c.upper() for h in _SENSOR_HINTS)]
        target_pool = sensor_cols if sensor_cols else cols
        for i, target in enumerate(target_pool[:2]):
            predictors = [c for c in cols if c != target][:max_predictors]
            if predictors:
                invariants.append(Invariant(f"{prefix}-inv{i + 1}", target, predictors))
        if len(invariants) >= max_invariants:
            break
    return invariants[:max_invariants]


def build_invariants(columns: list[str], dataset_name: str) -> list[Invariant]:
    """SWaT gets the paper's hardcoded table (if enough of its tags are
    present); everything else falls back to `infer_invariants`."""
    swat = swat_invariants(columns)
    if len(swat) >= 3:
        return swat
    return infer_invariants(columns, dataset_name)
