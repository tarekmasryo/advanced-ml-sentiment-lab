from __future__ import annotations

from sentiment_lab.decision import build_decision_summary, build_model_leaderboard
from sentiment_lab.evaluation import evaluate_model
from sentiment_lab.explainability import top_linear_terms
from sentiment_lab.features import build_advanced_features, transform_advanced_features
from sentiment_lab.prediction import labels_from_proba, threshold_for_model
from sentiment_lab.preprocessing import basic_clean, clean_dataframe
from sentiment_lab.thresholding import compute_threshold_view, recommend_thresholds
from sentiment_lab.training import (
    build_leakage_free_training_data,
    ensure_non_negative_features,
    train_multiple_models,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "basic_clean",
    "clean_dataframe",
    "labels_from_proba",
    "threshold_for_model",
    "build_advanced_features",
    "transform_advanced_features",
    "build_leakage_free_training_data",
    "ensure_non_negative_features",
    "train_multiple_models",
    "evaluate_model",
    "compute_threshold_view",
    "recommend_thresholds",
    "build_decision_summary",
    "build_model_leaderboard",
    "top_linear_terms",
]
