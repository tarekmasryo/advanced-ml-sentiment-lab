import numpy as np
import pandas as pd

from sentiment_lab.explainability import top_linear_terms
from sentiment_lab.features import build_advanced_features, transform_advanced_features
from sentiment_lab.preprocessing import basic_clean
from sentiment_lab.thresholding import compute_threshold_view
from sentiment_lab.training import (
    build_leakage_free_training_data,
    ensure_non_negative_features,
    make_stratified_subset_indices,
    train_multiple_models,
)


def test_basic_clean_normalizes_html_urls_and_symbols():
    raw = "Great<br /> movie!!! Visit https://example.com NOW 😊"

    assert basic_clean(raw) == "great movie visit now"


def test_transform_advanced_features_does_not_learn_validation_only_terms():
    train_texts = [
        "good movie",
        "excellent movie",
        "bad acting",
        "terrible acting",
    ]
    validation_texts = ["uniquevalidationtoken appears only here"]

    x_train, vectorizers = build_advanced_features(
        train_texts,
        max_word_features=100,
        use_char=False,
        char_max=50,
    )
    x_val = transform_advanced_features(validation_texts, vectorizers)

    vocab = vectorizers[0].vocabulary_
    assert "uniquevalidationtoken" not in vocab
    assert x_train.shape[0] == len(train_texts)
    assert x_val.shape[0] == len(validation_texts)
    assert x_train.shape[1] == x_val.shape[1]


def test_make_stratified_subset_indices_caps_rows_without_dropping_classes():
    y = np.array([0] * 20 + [1] * 20)

    idx = make_stratified_subset_indices(y, max_rows=10, random_state=42)

    assert len(idx) == 10
    assert set(np.unique(y[idx])) == {0, 1}


def test_build_leakage_free_training_data_fits_vectorizer_on_train_only():
    df = pd.DataFrame(
        {
            "text_clean": [
                "good movie",
                "excellent film",
                "great acting",
                "nice story",
                "bad movie",
                "terrible film",
                "awful acting",
                "poor story",
                "good nice",
                "bad awful",
            ]
        }
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

    out = build_leakage_free_training_data(
        df,
        y,
        max_train_rows=10,
        test_size=0.3,
        max_word_features=100,
        use_char=True,
        char_max=100,
        random_state=42,
    )

    assert out["X_train"].shape[0] == len(out["y_train"])
    assert out["X_val"].shape[0] == len(out["y_val"])
    assert out["X_train"].shape[1] == out["X_val"].shape[1]
    assert len(out["vectorizers"]) == 2
    assert len(set(out["train_loc"]).intersection(set(out["val_loc"]))) == 0


def test_train_multiple_models_supports_calibrated_linear_baselines():
    df = pd.DataFrame(
        {
            "text_clean": [
                "good movie",
                "excellent film",
                "great acting",
                "nice story",
                "bad movie",
                "terrible film",
                "awful acting",
                "poor story",
                "good nice",
                "bad awful",
                "excellent nice",
                "terrible poor",
            ]
        }
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    out = build_leakage_free_training_data(
        df,
        y,
        max_train_rows=12,
        test_size=0.25,
        max_word_features=100,
        use_char=False,
        char_max=100,
        random_state=42,
    )

    models = train_multiple_models(
        out["X_train"],
        out["y_train"],
        {
            "Logistic Regression": {
                "enabled": True,
                "C": 1.0,
                "calibrated": True,
                "calibration_method": "sigmoid",
            },
            "Linear SVM": {
                "enabled": True,
                "C": 1.0,
                "calibrated": True,
                "calibration_method": "sigmoid",
            },
        },
    )

    assert {"Logistic Regression", "Linear SVM"}.issubset(models.keys())


def test_compute_threshold_view_cost_accounting():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.10, 0.70, 0.80, 0.20])

    metrics, curve = compute_threshold_view(
        y_true,
        y_proba,
        threshold=0.5,
        cost_fp=3.0,
        cost_fn=5.0,
    )

    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["cost"] == 8.0
    assert {"threshold", "f1", "fp", "fn", "cost"}.issubset(curve.columns)


def test_naive_bayes_non_negative_guard_rejects_negative_features():
    x_train = np.array([[0.1, -0.2], [0.3, 0.4]])

    try:
        ensure_non_negative_features(x_train)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative features to be rejected.")


def test_top_linear_terms_supports_multiple_vectorizers():
    train_texts = [
        "good movie",
        "excellent movie",
        "bad acting",
        "terrible acting",
        "good story",
        "bad story",
    ]
    x_train, vectorizers = build_advanced_features(
        train_texts,
        max_word_features=100,
        use_char=True,
        char_max=100,
    )
    y_train = np.array([1, 1, 0, 0, 1, 0])
    models = train_multiple_models(
        x_train,
        y_train,
        {
            "Logistic Regression": {
                "enabled": True,
                "C": 1.0,
                "calibrated": False,
            }
        },
    )

    positive, negative = top_linear_terms(models["Logistic Regression"], vectorizers, top_n=5)

    assert not positive.empty
    assert not negative.empty
    assert set(positive.columns) == {"term", "weight"}
    assert set(negative.columns) == {"term", "weight"}


def test_build_leakage_free_training_data_can_create_heldout_test_split():
    df = pd.DataFrame(
        {
            "text_clean": [
                "good movie",
                "excellent film",
                "great acting",
                "nice story",
                "solid cast",
                "bad movie",
                "terrible film",
                "awful acting",
                "poor story",
                "weak cast",
                "good nice",
                "bad awful",
            ]
        }
    )
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0])

    out = build_leakage_free_training_data(
        df,
        y,
        max_train_rows=12,
        test_size=0.25,
        holdout_test_size=0.25,
        max_word_features=100,
        use_char=False,
        char_max=100,
        random_state=42,
    )

    assert out["X_test"] is not None
    assert len(out["y_test"]) > 0
    assert len(set(out["train_loc"]).intersection(set(out["test_loc"]))) == 0
    assert len(set(out["val_loc"]).intersection(set(out["test_loc"]))) == 0


def test_data_quality_report_flags_duplicates_and_empty_texts():
    from sentiment_lab.data_quality import compute_data_quality_report

    df = pd.DataFrame({"text_clean": ["good movie", "", "good movie", "ok"]})
    y = np.array([1, 0, 1, 0])

    report = compute_data_quality_report(df, text_col="text_clean", label_values=y)

    assert report["empty_texts"] == 1
    assert report["duplicate_texts"] == 1
    assert "empty_texts" in report["quality_flags"]
    assert "duplicate_texts" in report["quality_flags"]


def test_recommend_thresholds_returns_cost_and_f1_strategies():
    from sentiment_lab.thresholding import recommend_thresholds

    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.7, 0.8, 0.4, 0.9])

    recs = recommend_thresholds(y_true, y_proba, cost_fp=1.0, cost_fn=5.0)
    strategies = {row["strategy"] for row in recs}

    assert "best_f1" in strategies
    assert "lowest_cost" in strategies


def test_render_model_report_html_contains_metrics_table():
    from sentiment_lab.reporting import render_model_report_html

    html = render_model_report_html(
        title="Test Report",
        metadata={"evaluation_scope": "test"},
        results={
            "Model A": {
                "metrics": {
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "brier": 0.0,
                }
            }
        },
        data_quality={"quality_flags": []},
        threshold_recommendations=[],
    )

    assert "Test Report" in html
    assert "Model A" in html
    assert "Model Metrics" in html


def test_decision_summary_selects_model_and_action():
    from sentiment_lab.decision import build_decision_summary, decision_summary_table

    y_val = np.array([0, 0, 1, 1])
    results = {
        "Logistic Regression": {
            "metrics": {
                "f1": 0.75,
                "pr_auc": 0.8,
                "brier": 0.2,
                "y_proba": np.array([0.1, 0.4, 0.8, 0.9]),
            }
        }
    }
    metadata = {"y_val": y_val, "data_quality_report": {"quality_flags": []}}

    summary = build_decision_summary(results=results, metadata=metadata)
    table = decision_summary_table(summary)

    assert summary["recommended_model"] == "Logistic Regression"
    assert "recommended_threshold" in summary
    assert not table.empty


def test_data_quality_issues_returns_row_level_flags():
    from sentiment_lab.data_quality import data_quality_issues

    df = pd.DataFrame({"text_clean": ["good movie", "", "good movie", "ok"]})

    issues = data_quality_issues(df, text_col="text_clean")

    assert not issues.empty
    assert "issue_tags" in issues.columns
    assert any("duplicate_text" in tags for tags in issues["issue_tags"])


def test_render_model_card_markdown_contains_decision_summary():
    from sentiment_lab.reporting import render_model_card_markdown

    card = render_model_card_markdown(
        title="Test Model Card",
        metadata={"sample_rows": 100, "holdout_test_split": 0.2},
        results={
            "Model A": {
                "metrics": {
                    "f1": 0.8,
                    "pr_auc": 0.85,
                    "brier": 0.15,
                }
            }
        },
        decision_summary={
            "recommended_model": "Model A",
            "recommended_threshold": 0.6,
            "risk_level": "Moderate",
            "next_action": "Review errors.",
        },
        data_quality={"quality_flags": []},
        threshold_recommendations=[],
    )

    assert "Test Model Card" in card
    assert "Recommended model: Model A" in card
    assert "Evaluation scope" in card


def test_decision_summary_risk_uses_metric_quality_not_only_test_presence():
    from sentiment_lab.decision import build_decision_summary

    y_val = np.array([0, 0, 1, 1])
    results = {
        "Logistic Regression": {
            "metrics": {
                "f1": 0.55,
                "pr_auc": 0.60,
                "brier": 0.32,
                "y_proba": np.array([0.45, 0.55, 0.52, 0.48]),
            },
            "test_metrics": {
                "f1": 0.55,
                "pr_auc": 0.60,
                "brier": 0.32,
            },
        }
    }
    metadata = {
        "y_val": y_val,
        "test_rows": 200,
        "data_quality_report": {"quality_flags": []},
        "stability_summary": [],
    }

    summary = build_decision_summary(results=results, metadata=metadata)

    assert summary["risk_level"] in {"Caution", "High"}
    assert any("F1 < 0.70" in reason for reason in summary["risk_reasons"])


def test_decision_summary_lower_risk_requires_test_and_strong_metrics():
    from sentiment_lab.decision import build_decision_summary

    y_val = np.array([0, 0, 1, 1])
    results = {
        "Logistic Regression": {
            "metrics": {
                "f1": 0.92,
                "pr_auc": 0.95,
                "brier": 0.10,
                "y_proba": np.array([0.05, 0.20, 0.80, 0.95]),
            },
            "test_metrics": {
                "f1": 0.90,
                "pr_auc": 0.94,
                "brier": 0.11,
            },
        }
    }
    metadata = {
        "y_val": y_val,
        "test_rows": 250,
        "data_quality_report": {"quality_flags": []},
        "stability_summary": [],
    }

    summary = build_decision_summary(results=results, metadata=metadata)

    assert summary["risk_level"] == "Lower"
    assert summary["risk_reasons"] == []


def test_model_report_html_contains_visual_sections():
    from sentiment_lab.reporting import render_model_report_html

    html = render_model_report_html(
        title="Visual Report",
        metadata={"y_val": np.array([0, 0, 1, 1])},
        results={
            "Logistic Regression": {
                "metrics": {
                    "accuracy": 1.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "brier": 0.05,
                    "y_pred": np.array([0, 0, 1, 1]),
                    "y_proba": np.array([0.05, 0.1, 0.9, 0.95]),
                }
            }
        },
        data_quality={"quality_flags": []},
        threshold_recommendations=[{"strategy": "lowest_cost", "threshold": 0.5, "cost": 0.0}],
        decision_summary={"recommended_model": "Logistic Regression", "risk_reasons": []},
    )

    assert "<svg" in html
    assert "Validation confusion matrix" in html
    assert "Threshold recommendation cost comparison" in html


def test_deduplicate_clean_texts_removes_exact_duplicates_before_split():
    from sentiment_lab.preprocessing import deduplicate_clean_texts

    df = pd.DataFrame(
        {
            "text_clean": [
                "good movie",
                "bad film",
                "good movie",
                "excellent story",
            ]
        }
    )
    y = np.array([1, 0, 1, 1])

    out_df, out_y, removed = deduplicate_clean_texts(df, y, text_col="text_clean")

    assert removed == 1
    assert out_df["text_clean"].tolist() == ["good movie", "bad film", "excellent story"]
    assert out_y.tolist() == [1, 0, 1]


def test_data_quality_table_includes_duplicate_policy_fields():
    from sentiment_lab.data_quality import data_quality_table

    report = {
        "rows": 3,
        "empty_texts": 0,
        "duplicate_texts": 0,
        "duplicate_texts_before_policy": 2,
        "duplicate_texts_removed_before_split": 2,
        "duplicate_policy": "removed_before_split",
        "short_texts": 0,
        "avg_tokens": 4.0,
        "median_tokens": 4.0,
    }

    table = data_quality_table(report)

    assert "Duplicate policy" in table["check"].tolist()
    assert "Duplicate texts removed before split" in table["check"].tolist()


def test_detect_column_roles_finds_imdb_review_and_sentiment():
    from sentiment_lab.ux import detect_column_roles

    df = pd.DataFrame(
        {
            "review": [
                "This movie was wonderful and the performances were excellent.",
                "Terrible pacing, weak story, and very poor acting.",
                "A charming film with emotional depth and beautiful music.",
            ],
            "sentiment": ["positive", "negative", "positive"],
        }
    )

    roles = detect_column_roles(df)

    assert roles["text_col"] == "review"
    assert roles["label_col"] == "sentiment"


def test_validate_column_mapping_blocks_label_as_text_column():
    from sentiment_lab.ux import validate_column_mapping

    df = pd.DataFrame(
        {
            "review": [
                "This movie was wonderful and the performances were excellent.",
                "Terrible pacing, weak story, and very poor acting.",
            ],
            "sentiment": ["positive", "negative"],
        }
    )

    errors, warnings = validate_column_mapping(
        df,
        text_col="sentiment",
        label_col="review",
    )

    assert errors
    assert any("looks like a label column" in error for error in errors)
    assert isinstance(warnings, list)


def test_basic_clean_handles_none_and_html_tags():
    from sentiment_lab.preprocessing import basic_clean

    assert basic_clean(None) == ""
    assert basic_clean("<b>Great</b><br />Movie!") == "great movie"


def test_labels_from_proba_uses_custom_threshold():
    from sentiment_lab.prediction import labels_from_proba

    labels = labels_from_proba(np.array([0.40, 0.55, 0.80]), threshold=0.60)

    assert labels.tolist() == [0, 0, 1]


def test_threshold_for_model_prefers_saved_decision_summary():
    from sentiment_lab.prediction import threshold_for_model

    results = {
        "Model A": {
            "metrics": {
                "y_proba": np.array([0.1, 0.4, 0.7, 0.9]),
            }
        }
    }
    metadata = {
        "y_val": np.array([0, 0, 1, 1]),
        "decision_summary": {
            "recommended_model": "Model A",
            "recommended_threshold": 0.63,
        },
    }

    assert threshold_for_model(results, metadata, "Model A") == 0.63
