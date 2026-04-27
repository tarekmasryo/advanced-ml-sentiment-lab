from __future__ import annotations

from typing import Any

import numpy as np

from sentiment_lab.thresholding import recommend_thresholds


def labels_from_proba(y_proba: np.ndarray, *, threshold: float) -> np.ndarray:
    return (np.asarray(y_proba, dtype=float) >= float(threshold)).astype(int)


def class_confidence(proba_pos: float, label: int) -> float:
    proba = float(proba_pos)
    return proba if int(label) == 1 else 1.0 - proba


def threshold_for_model(
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    model_name: str,
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> float:
    decision_summary = metadata.get("decision_summary", {}) or {}
    if model_name == decision_summary.get("recommended_model"):
        return _clamp_threshold(decision_summary.get("recommended_threshold", 0.5))

    y_val = np.asarray(metadata.get("y_val", []), dtype=int)
    metrics = results.get(model_name, {}).get("metrics", {})
    y_proba = np.asarray(metrics.get("y_proba", []), dtype=float)
    if len(y_val) == 0 or len(y_val) != len(y_proba):
        return 0.5

    recommendations = recommend_thresholds(y_val, y_proba, cost_fp=cost_fp, cost_fn=cost_fn)
    cost_choice = next(
        (row for row in recommendations if row.get("strategy") == "lowest_cost"),
        recommendations[0] if recommendations else {},
    )
    return _clamp_threshold(cost_choice.get("threshold", 0.5))


def _clamp_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(0.95, max(0.05, threshold))
