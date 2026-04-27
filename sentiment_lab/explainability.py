from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _unwrap_calibrated_estimator(model: Any) -> Any:
    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        calibrated = calibrated_classifiers[0]
        return getattr(calibrated, "estimator", getattr(calibrated, "base_estimator", calibrated))
    return model


def _linear_coef(model: Any) -> np.ndarray | None:
    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        coefs: list[np.ndarray] = []
        for calibrated in calibrated_classifiers:
            estimator = getattr(
                calibrated,
                "estimator",
                getattr(calibrated, "base_estimator", calibrated),
            )
            coef = getattr(estimator, "coef_", None)
            if coef is not None:
                coefs.append(np.asarray(coef, dtype=float).reshape(-1))
        if coefs:
            return np.mean(np.vstack(coefs), axis=0)
        return None

    coef = getattr(model, "coef_", None)
    if coef is None:
        return None
    return np.asarray(coef, dtype=float).reshape(-1)


def _feature_names(vectorizer_or_vectorizers: Any) -> np.ndarray:
    if isinstance(vectorizer_or_vectorizers, (list, tuple)):
        parts: list[np.ndarray] = []
        for idx, vec in enumerate(vectorizer_or_vectorizers):
            if not hasattr(vec, "get_feature_names_out"):
                continue
            prefix = "word" if idx == 0 else "char"
            names = np.asarray(vec.get_feature_names_out(), dtype=object)
            parts.append(np.asarray([f"{prefix}:{name}" for name in names], dtype=object))
        if parts:
            return np.concatenate(parts)
        return np.asarray([], dtype=object)

    if hasattr(vectorizer_or_vectorizers, "get_feature_names_out"):
        return np.asarray(vectorizer_or_vectorizers.get_feature_names_out(), dtype=object)

    return np.asarray([], dtype=object)


def top_linear_terms(
    model: Any,
    vectorizer_or_vectorizers: Any,
    *,
    top_n: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top positive/negative terms for supported linear classifiers."""
    coef = _linear_coef(model)
    if coef is None:
        return pd.DataFrame(), pd.DataFrame()
    names = _feature_names(vectorizer_or_vectorizers)

    if len(coef) != len(names) or len(names) == 0:
        return pd.DataFrame(), pd.DataFrame()

    order = np.argsort(coef)
    negative_idx = order[:top_n]
    positive_idx = order[-top_n:][::-1]

    positive = pd.DataFrame({"term": names[positive_idx], "weight": coef[positive_idx]})
    negative = pd.DataFrame({"term": names[negative_idx], "weight": coef[negative_idx]})
    return positive, negative


def explain_linear_prediction(
    model: Any,
    vectorizer_or_vectorizers: Any,
    x_row: Any,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Return top token contributions for one sparse feature row and a linear model."""
    coef = _linear_coef(model)
    if coef is None:
        return pd.DataFrame(columns=["term", "contribution"])
    names = _feature_names(vectorizer_or_vectorizers)
    if len(coef) != len(names) or len(names) == 0:
        return pd.DataFrame(columns=["term", "contribution"])

    row = x_row.tocsr() if hasattr(x_row, "tocsr") else x_row
    values = np.asarray(row.multiply(coef).toarray()).reshape(-1)
    nonzero = np.flatnonzero(values)
    if len(nonzero) == 0:
        return pd.DataFrame(columns=["term", "contribution"])

    order = nonzero[np.argsort(np.abs(values[nonzero]))[::-1]][:top_n]
    return pd.DataFrame({"term": names[order], "contribution": values[order]})
