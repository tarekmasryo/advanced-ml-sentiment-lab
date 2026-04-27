from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sentiment_lab.features import build_advanced_features, transform_advanced_features
from sentiment_lab.runtime_config import default_n_jobs


def make_stratified_subset_indices(
    y: np.ndarray,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    """Return a capped, stratified subset of row positions for responsive training."""
    n_total = len(y)
    train_rows = min(int(max_rows), n_total)
    indices = np.arange(n_total)

    if train_rows >= n_total:
        return indices

    sample_idx, _ = train_test_split(
        indices,
        train_size=train_rows,
        stratify=y,
        random_state=random_state,
    )
    return np.asarray(sample_idx)


def build_leakage_free_training_data(
    dfc: pd.DataFrame,
    y: np.ndarray,
    *,
    max_train_rows: int,
    test_size: float,
    max_word_features: int,
    use_char: bool,
    char_max: int,
    random_state: int,
    holdout_test_size: float = 0.0,
) -> dict[str, Any]:
    """Create a leakage-free train/validation/(optional) held-out test dataset."""
    sample_idx = make_stratified_subset_indices(y, max_train_rows, random_state)
    df_sample = dfc.iloc[sample_idx].copy()
    y_sample = y[sample_idx]

    min_class = int(min((y_sample == 0).sum(), (y_sample == 1).sum()))
    if min_class < 2:
        raise ValueError(
            f"Not enough samples per class for a stratified split (min class count={min_class})."
        )

    local_idx = np.arange(len(df_sample))
    test_loc = np.asarray([], dtype=int)
    y_test = np.asarray([], dtype=int)

    if holdout_test_size > 0:
        if min_class < 3:
            raise ValueError(
                "At least 3 samples per class are required when a held-out test split is enabled."
            )
        dev_loc, test_loc, y_dev, y_test = train_test_split(
            local_idx,
            y_sample,
            test_size=holdout_test_size,
            stratify=y_sample,
            random_state=random_state,
        )
    else:
        dev_loc = local_idx
        y_dev = y_sample

    train_loc_rel, val_loc_rel, y_train, y_val = train_test_split(
        np.arange(len(dev_loc)),
        y_dev,
        test_size=test_size,
        stratify=y_dev,
        random_state=random_state,
    )
    train_loc = dev_loc[train_loc_rel]
    val_loc = dev_loc[val_loc_rel]

    train_texts = df_sample.iloc[train_loc]["text_clean"].tolist()
    val_texts = df_sample.iloc[val_loc]["text_clean"].tolist()
    test_texts = df_sample.iloc[test_loc]["text_clean"].tolist() if len(test_loc) else []

    x_train, vectorizers = build_advanced_features(
        train_texts,
        max_word_features=max_word_features,
        use_char=use_char,
        char_max=char_max,
    )
    x_val = transform_advanced_features(val_texts, vectorizers)
    x_test = transform_advanced_features(test_texts, vectorizers) if test_texts else None

    return {
        "df_sample": df_sample,
        "sample_idx": sample_idx,
        "train_loc": train_loc,
        "val_loc": val_loc,
        "test_loc": test_loc,
        "val_idx_global": df_sample.index[val_loc],
        "test_idx_global": df_sample.index[test_loc] if len(test_loc) else pd.Index([], dtype=int),
        "y_sample": y_sample,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "vectorizers": vectorizers,
    }


def _min_class_count(y: np.ndarray) -> int:
    return int(min(np.bincount(np.asarray(y, dtype=int), minlength=2)))


def _maybe_calibrate(model: Any, y_train: np.ndarray, *, method: str, cv: int = 3) -> Any:
    """Wrap a classifier in CalibratedClassifierCV when the split supports it."""
    if _min_class_count(y_train) < cv:
        return model

    return CalibratedClassifierCV(estimator=model, method=method, cv=cv)


def ensure_non_negative_features(x_train) -> None:
    """Fail fast for models that require non-negative features."""
    try:
        min_value = float(x_train.min())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Could not verify feature non-negativity for Naive Bayes.") from exc

    if min_value < 0:
        raise ValueError(
            "Naive Bayes requires non-negative features. Disable Naive Bayes or use "
            "a non-negative vectorizer/feature pipeline."
        )


def train_multiple_models(
    x_train,
    y_train: np.ndarray,
    models_config: dict[str, dict[str, Any]],
    *,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Train enabled classical baselines/challengers."""
    models: dict[str, Any] = {}
    effective_n_jobs = default_n_jobs() if n_jobs is None else int(n_jobs)

    for name, cfg in models_config.items():
        if not cfg.get("enabled", False):
            continue

        if name == "Logistic Regression":
            base = LogisticRegression(
                C=float(cfg["C"]),
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                random_state=random_state,
            )
            model = (
                _maybe_calibrate(
                    base,
                    y_train,
                    method=str(cfg.get("calibration_method", "sigmoid")),
                )
                if cfg.get("calibrated", False)
                else base
            )

        elif name == "Linear SVM":
            base = LinearSVC(
                C=float(cfg["C"]),
                class_weight="balanced",
                random_state=random_state,
            )
            model = (
                _maybe_calibrate(
                    base,
                    y_train,
                    method=str(cfg.get("calibration_method", "sigmoid")),
                )
                if cfg.get("calibrated", True)
                else base
            )

        elif name == "SGD Logistic":
            model = SGDClassifier(
                loss="log_loss",
                alpha=float(cfg["alpha"]),
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )

        elif name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=int(cfg["n_estimators"]),
                max_depth=int(cfg["max_depth"]),
                min_samples_split=int(cfg["min_samples_split"]),
                n_jobs=effective_n_jobs,
                class_weight="balanced",
                random_state=random_state,
            )

        elif name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=int(cfg["n_estimators"]),
                learning_rate=float(cfg["learning_rate"]),
                max_depth=int(cfg["max_depth"]),
                random_state=random_state,
            )

        elif name == "Naive Bayes":
            ensure_non_negative_features(x_train)
            model = MultinomialNB(alpha=float(cfg["alpha"]))

        else:
            continue

        model.fit(x_train, y_train)
        models[name] = model

    return models
