from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from sentiment_lab.features import make_feature_union


def repeated_split_stability(
    texts: list[str],
    y: np.ndarray,
    models: dict[str, Any],
    *,
    n_splits: int,
    test_size: float,
    max_word_features: int,
    use_char: bool,
    char_max: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate model stability across repeated leakage-free stratified splits."""
    if n_splits <= 0:
        return pd.DataFrame()

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    texts_arr = np.asarray(texts, dtype=object)
    y_arr = np.asarray(y, dtype=int)

    for split_id, (train_idx, val_idx) in enumerate(splitter.split(texts_arr, y_arr), start=1):
        x_train_text = texts_arr[train_idx]
        x_val_text = texts_arr[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]
        features = make_feature_union(
            max_word_features=max_word_features,
            use_char=use_char,
            char_max=char_max,
        )
        for name, model in models.items():
            pipe = Pipeline([("features", features), ("model", clone(model))])
            pipe.fit(x_train_text, y_train)
            y_pred = pipe.predict(x_val_text)
            try:
                y_score = pipe.predict_proba(x_val_text)[:, 1]
            except AttributeError:
                try:
                    score = pipe.decision_function(x_val_text)
                    score = np.asarray(score).reshape(-1)
                    y_score = (score - score.min()) / ((score.max() - score.min()) + 1e-9)
                except AttributeError:
                    y_score = y_pred
            try:
                roc_auc = roc_auc_score(y_val, y_score)
            except ValueError:
                roc_auc = float("nan")
            rows.append(
                {
                    "split": split_id,
                    "model": name,
                    "f1": f1_score(y_val, y_pred, zero_division=0),
                    "roc_auc": roc_auc,
                    "pr_auc": average_precision_score(y_val, y_score),
                }
            )
    return pd.DataFrame(rows)


def summarize_stability(stability_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated split results into mean/std summary."""
    if stability_df.empty:
        return pd.DataFrame()
    return (
        stability_df.groupby("model")[["f1", "roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
