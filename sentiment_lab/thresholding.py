from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD_GRID = np.linspace(0.05, 0.95, 37)


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    y_pred_thr = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_thr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred_thr),
        "precision": precision_score(y_true, y_pred_thr, zero_division=0),
        "recall": recall_score(y_true, y_pred_thr, zero_division=0),
        "f1": f1_score(y_true, y_pred_thr, zero_division=0),
        "specificity": tn / (tn + fp + 1e-9),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "tn": int(tn),
    }
    metrics["cost"] = metrics["fp"] * float(cost_fp) + metrics["fn"] * float(cost_fn)
    return metrics


def compute_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_fp: float,
    cost_fn: float,
    grid: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    rows = [_metrics_at_threshold(y_true, y_proba, float(thr), cost_fp, cost_fn) for thr in grid]
    return pd.DataFrame(rows)


def compute_threshold_view(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _metrics_at_threshold(y_true, y_proba, threshold, cost_fp, cost_fn)
    curve = compute_threshold_curve(y_true, y_proba, cost_fp=cost_fp, cost_fn=cost_fn)
    return metrics, curve[["threshold", "f1", "fp", "fn", "cost"]]


def recommend_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    cost_fp: float,
    cost_fn: float,
    precision_target: float | None = None,
    recall_target: float | None = None,
) -> list[dict[str, Any]]:
    """Return practical threshold recommendations for decision workflows."""
    curve = compute_threshold_curve(y_true, y_proba, cost_fp=cost_fp, cost_fn=cost_fn)
    recs: list[dict[str, Any]] = []

    best_f1 = curve.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    recs.append({"strategy": "best_f1", **best_f1.to_dict()})

    best_cost = curve.sort_values(["cost", "threshold"], ascending=[True, True]).iloc[0]
    recs.append({"strategy": "lowest_cost", **best_cost.to_dict()})

    if precision_target is not None:
        viable = curve[curve["precision"] >= precision_target]
        if not viable.empty:
            row = viable.sort_values(["recall", "cost"], ascending=[False, True]).iloc[0]
            recs.append({"strategy": f"precision>={precision_target:.2f}", **row.to_dict()})

    if recall_target is not None:
        viable = curve[curve["recall"] >= recall_target]
        if not viable.empty:
            row = viable.sort_values(["precision", "cost"], ascending=[False, True]).iloc[0]
            recs.append({"strategy": f"recall>={recall_target:.2f}", **row.to_dict()})

    return recs
