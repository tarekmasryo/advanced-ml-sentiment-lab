from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    smin = float(np.min(scores))
    smax = float(np.max(scores))
    if np.isclose(smin, smax):
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - smin) / (smax - smin)


def predict_proba_pos(model: Any, x_matrix) -> np.ndarray:
    """Produce a probability-like positive-class score."""
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        proba = predict_proba(x_matrix)
        return proba[:, 1]

    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        scores = decision_function(x_matrix)
        scores = np.asarray(scores).reshape(-1)
        return minmax_scale(scores)

    raise AttributeError("Model has neither predict_proba nor decision_function.")


def evaluate_model(model: Any, x_val, y_val: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(x_val)
    y_proba = predict_proba_pos(model, x_val)

    try:
        roc_auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        roc_auc = float("nan")

    try:
        brier = brier_score_loss(y_val, y_proba)
    except ValueError:
        brier = float("nan")

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": average_precision_score(y_val, y_proba),
        "brier": brier,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
