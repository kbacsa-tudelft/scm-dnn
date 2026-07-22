"""Anomaly-detection metrics shared across methods.

`conflict_index_factor` uses the exact PbNN-paper weights (c1=0.4, c2=0.6):
    Dr = Tp/(Tp+Fn), Fr = Fp/(Fp+Tn), CiF = c1*Dr - c2*Fr
"""
from __future__ import annotations

import numpy as np


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn, _ = _confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, fn, _ = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tn = _confusion(y_true, y_pred)
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def conflict_index_factor(y_true: np.ndarray, y_pred: np.ndarray, c1: float = 0.4, c2: float = 0.6) -> float:
    return c1 * detection_rate(y_true, y_pred) - c2 * false_alarm_rate(y_true, y_pred)


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Standard TSAD point-adjustment: if any point in a contiguous true-anomaly
    segment is flagged, treat the whole segment as detected."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            j = i
            while j < n and y_true[j] == 1:
                j += 1
            if y_pred[i:j].any():
                y_pred[i:j] = 1
            i = j
        else:
            i += 1
    return y_pred


def summarize(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Full metric report for one method/dataset run at a given decision threshold."""
    y_pred = (scores > threshold).astype(int)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    y_pred_pa = point_adjust(y_true, y_pred)
    precision_pa, recall_pa, f1_pa = precision_recall_f1(y_true, y_pred_pa)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_pa": precision_pa,
        "recall_pa": recall_pa,
        "f1_pa": f1_pa,
        "detection_rate": detection_rate(y_true, y_pred),
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "conflict_index_factor": conflict_index_factor(y_true, y_pred),
        "threshold": threshold,
    }
