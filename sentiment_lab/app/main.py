from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from sentiment_lab.app.components import (
    plain_next_step,
    quality_status as get_quality_status,
    render_client_metric_card,
    render_decision_cards,
    render_status_chip_row,
    render_workflow_steps,
    show_validation_scope_notice,
)
from sentiment_lab.app.navigation import render_workspace_navigation
from sentiment_lab.app.runtime import (
    AppPaths,
    ReloadSmokeTestWriter,
    ReviewBundleBuilder,
    TrainingStateStore,
)
from sentiment_lab.artifacts import load_run_history, save_training_artifacts
from sentiment_lab.data_quality import (
    compute_data_quality_report,
    data_quality_issues,
    data_quality_table,
)
from sentiment_lab.decision import (
    build_decision_summary,
    build_model_leaderboard,
    decision_summary_table,
)
from sentiment_lab.evaluation import evaluate_model, predict_proba_pos
from sentiment_lab.explainability import explain_linear_prediction, top_linear_terms
from sentiment_lab.features import transform_advanced_features
from sentiment_lab.io import (
    read_uploaded_csv as _read_uploaded_csv,
    safe_read_csv as _safe_read_csv,
)
from sentiment_lab.prediction import class_confidence, labels_from_proba, threshold_for_model
from sentiment_lab.preprocessing import basic_clean, clean_dataframe, deduplicate_clean_texts
from sentiment_lab.reporting import (
    render_model_card_markdown,
    render_model_report_html,
    save_model_card_markdown,
    save_model_report_html,
)
from sentiment_lab.runtime_config import default_n_jobs
from sentiment_lab.stability import repeated_split_stability, summarize_stability
from sentiment_lab.thresholding import compute_threshold_view, recommend_thresholds
from sentiment_lab.training import build_leakage_free_training_data, train_multiple_models
from sentiment_lab.ui_theme import (
    PREMIUM_UI_CSS,
    escape_html,
    render_empty_dataset_state,
    render_hero,
    render_metric_card,
    render_section_header,
)
from sentiment_lab.ux import detect_column_roles, validate_column_mapping

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))).expanduser()
APP_PATHS = AppPaths(base_dir=BASE_DIR, artifacts_dir=ARTIFACTS_DIR)
MODELS_DIR = APP_PATHS.models_dir
STATE_STORE = TrainingStateStore()
BUNDLE_BUILDER = ReviewBundleBuilder()
SMOKE_TEST_WRITER = ReloadSmokeTestWriter()

APP_CSS = PREMIUM_UI_CSS


@st.cache_data(show_spinner=False, ttl=60)
def load_csv_auto() -> pd.DataFrame | None:
    candidates = [
        BASE_DIR / "IMDB Dataset.csv",
        BASE_DIR.parent / "IMDB Dataset.csv",
        BASE_DIR / "data" / "IMDB Dataset.csv",
        BASE_DIR.parent / "data" / "IMDB Dataset.csv",
        BASE_DIR / "imdb.csv",
        BASE_DIR.parent / "imdb.csv",
        BASE_DIR / "data.csv",
        BASE_DIR.parent / "data.csv",
        BASE_DIR / "reviews.csv",
        BASE_DIR.parent / "reviews.csv",
        BASE_DIR / "comments.csv",
        BASE_DIR.parent / "comments.csv",
    ]

    for p in candidates:
        if p.exists():
            df = _safe_read_csv(p)
            if df is not None:
                return df

    return None


@st.cache_data(show_spinner=False)
def clean_df(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    pos_label_str: str,
    neg_label_str: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    return clean_dataframe(
        df,
        text_col=text_col,
        label_col=label_col,
        pos_label_str=pos_label_str,
        neg_label_str=neg_label_str,
    )


def main() -> None:
    st.set_page_config(
        page_title="Advanced ML Sentiment Lab",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    uploaded = st.sidebar.file_uploader(
        "Upload CSV dataset (optional)",
        type=["csv"],
        help="For small custom datasets. For IMDB, keep the CSV in the repo.",
    )

    active_page = render_workspace_navigation()

    if uploaded is not None:
        df = _read_uploaded_csv(uploaded)
        if df is None:
            st.error(
                "Failed to read the uploaded CSV. Try exporting as UTF-8 or a simpler delimiter."
            )
            st.stop()
    else:
        df = load_csv_auto()

    if df is None:
        render_empty_dataset_state()
        st.stop()

    all_cols = list(df.columns)
    if len(all_cols) < 2:
        st.error("Dataset must have at least 2 columns (text + label).")
        st.stop()

    detected_roles = detect_column_roles(df)
    detected_text_col = detected_roles.get("text_col") or all_cols[0]
    detected_label_col = detected_roles.get("label_col") or next(
        (c for c in all_cols if c != detected_text_col), all_cols[0]
    )

    st.sidebar.markdown("### Dataset setup")
    st.sidebar.info(f"Text column: `{detected_text_col}`")
    st.sidebar.info(f"Label column: `{detected_label_col}`")
    manual_mapping = st.sidebar.toggle(
        "Override detected columns",
        value=False,
        help="Use this when the detected text or label column is not correct.",
    )

    if manual_mapping:
        with st.sidebar.expander("Column mapping", expanded=True):
            default_text_idx = (
                all_cols.index(detected_text_col) if detected_text_col in all_cols else 0
            )
            text_col = st.selectbox("Text column", all_cols, index=default_text_idx)

            label_candidates = [c for c in all_cols if c != text_col]
            if not label_candidates:
                st.error("Dataset must have at least 2 columns (text + label).")
                st.stop()

            default_label_idx = (
                label_candidates.index(detected_label_col)
                if detected_label_col in label_candidates
                else 0
            )
            label_col = st.selectbox("Label column", label_candidates, index=default_label_idx)
    else:
        text_col = detected_text_col
        label_col = detected_label_col

    mapping_errors, mapping_warnings = validate_column_mapping(
        df, text_col=text_col, label_col=label_col
    )
    if mapping_errors:
        st.error("Column mapping needs attention before training.")
        for message in mapping_errors:
            st.error(message)
        st.info(
            "For the IMDB dataset, use `review` as Text column and `sentiment` as Label column."
        )
        st.stop()
    for message in mapping_warnings:
        st.sidebar.warning(message, icon="⚠️")

    label_values = df[label_col].astype(str).dropna().value_counts().index.tolist()
    if len(label_values) < 2:
        st.error("Label column must have at least 2 distinct values.")
        st.stop()

    st.sidebar.markdown("### Label setup")
    lower_label_values = {str(v).lower(): v for v in label_values}
    labels_are_standard = {"positive", "negative"}.issubset(lower_label_values)
    if labels_are_standard:
        default_pos_label = lower_label_values["positive"]
        default_neg_label = lower_label_values["negative"]
        st.sidebar.info(f"Labels: `{default_pos_label}` / `{default_neg_label}`")
    else:
        default_pos_label = label_values[0]
        default_neg_label = label_values[1 if len(label_values) > 1 else 0]

    override_labels = st.sidebar.toggle(
        "Override label mapping",
        value=not labels_are_standard,
        help="Use this when the positive and negative labels are not mapped correctly.",
    )
    if override_labels:
        with st.sidebar.expander("Label mapping", expanded=True):
            pos_label_str = st.selectbox(
                "Positive class (1)",
                label_values,
                index=label_values.index(default_pos_label)
                if default_pos_label in label_values
                else 0,
            )
            neg_label_str = st.selectbox(
                "Negative class (0)",
                label_values,
                index=label_values.index(default_neg_label)
                if default_neg_label in label_values
                else 1,
            )
    else:
        pos_label_str = default_pos_label
        neg_label_str = default_neg_label

    if pos_label_str == neg_label_str:
        st.error("Positive and negative labels must be different.")
        st.stop()

    st.sidebar.markdown("### Training subset")
    max_train_rows = st.sidebar.slider(
        "Max rows used for training",
        min_value=5000,
        max_value=50000,
        value=10000,
        step=5000,
        help="Training uses a stratified subset to keep runtime under control.",
    )
    with st.sidebar.expander("Runtime options", expanded=False):
        n_jobs_options = [1, 2, 4, -1]
        default_jobs = default_n_jobs()
        default_jobs_index = (
            n_jobs_options.index(default_jobs) if default_jobs in n_jobs_options else 0
        )
        n_jobs = st.selectbox(
            "Parallel jobs",
            options=n_jobs_options,
            index=default_jobs_index,
            format_func=lambda x: "all available cores" if x == -1 else str(x),
            help="Use 1 for the most stable local/CI behavior; use more cores for faster large runs.",
        )

    dfc, y, dropped = clean_df(
        df,
        text_col=text_col,
        label_col=label_col,
        pos_label_str=pos_label_str,
        neg_label_str=neg_label_str,
    )

    pre_policy_quality_report = compute_data_quality_report(
        dfc, text_col="text_clean", label_values=y
    )
    st.sidebar.markdown("### Data policy")
    remove_duplicate_texts = st.sidebar.checkbox(
        "Remove duplicate cleaned texts before splitting",
        value=True,
        help=(
            "Recommended. This duplicate-safe policy prevents exact "
            "duplicate reviews from leaking across train/validation/test partitions."
        ),
    )
    duplicate_texts_removed = 0
    if remove_duplicate_texts:
        dfc, y, duplicate_texts_removed = deduplicate_clean_texts(dfc, y, text_col="text_clean")
        if duplicate_texts_removed:
            st.sidebar.success(
                f"Removed {duplicate_texts_removed:,} duplicate cleaned texts before splitting."
            )
    n_pos_total = int((y == 1).sum())
    n_neg_total = int((y == 0).sum())
    if min(n_pos_total, n_neg_total) < 2:
        st.error(
            f"Not enough samples per class for stratified training. "
            f"Found: {n_pos_total:,} positive vs {n_neg_total:,} negative."
        )
        st.stop()

    if len(dfc) < 100:
        st.error("Not enough rows after filtering to the selected labels.")
        st.stop()

    if dropped > 0:
        st.sidebar.info(f"Filtered out {dropped:,} rows with labels outside the selected classes.")

    n_rows = len(dfc)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_ratio = n_pos / max(1, n_rows)
    avg_len = float(dfc["text_clean"].str.len().mean())
    sample_vocab = len(set(" ".join(dfc["text_clean"].head(5000)).split()))
    data_quality_report = compute_data_quality_report(dfc, text_col="text_clean", label_values=y)
    data_quality_report["duplicate_policy"] = (
        "removed_before_split" if remove_duplicate_texts else "kept_flagged"
    )
    data_quality_report["duplicate_texts_before_policy"] = int(
        pre_policy_quality_report.get("duplicate_texts", 0)
    )
    data_quality_report["duplicate_texts_removed_before_split"] = int(duplicate_texts_removed)
    data_quality_issue_df = data_quality_issues(dfc, text_col="text_clean", max_rows=200)

    render_hero(
        title="Advanced ML Sentiment Lab",
        subtitle=(
            f"A professional sentiment analysis workspace for {n_rows:,} cleaned samples. "
            "It guides quality review, safe training, model selection, threshold tuning, "
            "prediction checks, and stakeholder-ready export."
        ),
        badges=(f"Text column: {text_col}", f"Label column: {label_col}"),
        soft_badges=(
            f"Binary labels: {pos_label_str} / {neg_label_str}",
            f"Duplicate policy: {data_quality_report['duplicate_policy']}",
        ),
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card(
            icon="📊", label="Total samples", value=f"{n_rows:,}", note="Cleaned for modeling"
        )

    with k2:
        render_metric_card(
            icon="✅",
            label="Positive share",
            value=f"{pos_ratio * 100:.1f}%",
            note=f"{n_pos:,} positive / {n_neg:,} negative",
        )

    with k3:
        render_metric_card(
            icon="📝", label="Avg text length", value=f"{avg_len:.0f}", note="characters per record"
        )

    with k4:
        render_metric_card(
            icon="📚",
            label="Sample vocabulary",
            value=f"{sample_vocab:,}",
            note="unique tokens (first 5k rows)",
        )

    if active_page == "Overview":
        render_section_header(
            "Start here",
            "A clear workflow for reviewers: check the data, train with safe defaults, read the recommendation, test examples, and export the report.",
        )

        quality_status, quality_note = get_quality_status(data_quality_report)
        current_state = STATE_STORE.load(MODELS_DIR)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            render_client_metric_card(
                "Data rows", f"{n_rows:,}", "after cleaning and policy checks"
            )
        with s2:
            render_client_metric_card("Data status", quality_status, quality_note)
        with s3:
            render_client_metric_card(
                "Training status",
                "Ready" if current_state is None else "Trained",
                "use safe defaults first",
            )
        with s4:
            render_client_metric_card(
                "Next step",
                "Train" if current_state is None else "Review",
                plain_next_step(current_state),
            )

        render_status_chip_row(
            [
                ("Text", text_col, "good"),
                ("Label", label_col, "good"),
                ("Samples", f"{n_rows:,}", "good"),
                ("Quality", quality_status, "warn" if quality_status != "Ready" else "good"),
            ]
        )

        st.markdown("### Guided workflow")
        render_workflow_steps()

        if current_state is None:
            st.info(
                "No model has been trained in this session yet. Open Training and click Train models."
            )
        else:
            summary = build_decision_summary(
                results=current_state["results"], metadata=current_state["metadata"]
            )
            if summary:
                st.success(
                    f"Current recommendation: {summary['recommended_model']} at threshold {summary['recommended_threshold']:.3f}. "
                    f"Risk level: {summary['risk_level']}."
                )

        with st.expander("What this app does behind the scenes", expanded=False):
            st.markdown(
                """
                - Removes exact duplicate cleaned texts before splitting by default.
                - Fits TF-IDF on training text only to avoid leakage.
                - Uses validation data for model selection and threshold tuning.
                - Uses a held-out test split for a cleaner final estimate when enabled.
                - Exports reports, metrics, model cards, and review bundles for stakeholders.
                """
            )
    if active_page == "Evaluation":
        render_section_header(
            "Decision summary",
            "A business-facing summary of the active run: recommended model, decision threshold, estimated cost, risk level, and next action.",
        )

        state = STATE_STORE.load(MODELS_DIR)
        if state is None:
            st.info("Train models first to generate an evaluation summary.")
            st.dataframe(
                data_quality_table(data_quality_report), use_container_width=True, hide_index=True
            )
            if data_quality_report.get("quality_flags"):
                st.warning(
                    "Data-quality flags before training: "
                    + ", ".join(data_quality_report["quality_flags"]),
                    icon="⚠️",
                )
        else:
            all_results = state["results"]
            metadata = state["metadata"]
            summary = build_decision_summary(results=all_results, metadata=metadata)
            if not summary:
                st.info("Decision summary could not be created for the active run.")
            else:
                render_decision_cards(summary)

                st.dataframe(
                    decision_summary_table(summary), use_container_width=True, hide_index=True
                )
                st.info(f"Next action: {summary['next_action']}")

                leaderboard = build_model_leaderboard(all_results)
                st.markdown("#### Model leaderboard")
                st.dataframe(leaderboard.round(4), use_container_width=True, hide_index=True)

                if summary.get("quality_flags"):
                    st.info(
                        "Data quality note: " + ", ".join(summary["quality_flags"]),
                        icon="ℹ️",
                    )
                show_validation_scope_notice()
    if active_page == "Data Setup":
        render_section_header(
            "Exploratory data analysis",
            "Quick checks on class balance, text lengths, and token distribution.",
        )

        st.markdown("#### Data quality checks")
        st.dataframe(
            data_quality_table(data_quality_report), use_container_width=True, hide_index=True
        )
        if data_quality_report.get("quality_flags"):
            st.info(
                "Data quality note: " + ", ".join(data_quality_report["quality_flags"]), icon="ℹ️"
            )

        if data_quality_issue_df.empty:
            st.success(
                "No row-level empty/duplicate/short-text issues detected in the review sample."
            )
        else:
            with st.expander("Review row-level data-quality issues", expanded=False):
                st.dataframe(data_quality_issue_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download data-quality issues CSV",
                    data_quality_issue_df.to_csv(index=False).encode("utf-8"),
                    file_name="data_quality_issues.csv",
                    mime="text/csv",
                )

        col1, col2 = st.columns(2)

        with col1:
            len_tokens = dfc["text_clean"].str.split().map(len)

            fig_len = px.histogram(
                x=len_tokens,
                nbins=50,
                title="Token length distribution",
            )
            fig_len.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                xaxis_title="Tokens per text",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_len, use_container_width=True)

            dist_data = pd.DataFrame(
                {
                    "Class": [neg_label_str, pos_label_str],
                    "Count": [n_neg, n_pos],
                }
            )
            fig_class = px.pie(
                dist_data,
                values="Count",
                names="Class",
                title="Class distribution",
            )
            fig_class.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_class, use_container_width=True)

        with col2:
            sample_size = min(10000, len(dfc))
            cnt: Counter[str] = Counter()
            for t in dfc["text_clean"].sample(sample_size, random_state=42):
                cnt.update(t.split())

            top_tokens = pd.DataFrame(cnt.most_common(25), columns=["Token", "Frequency"])
            fig_tokens = px.bar(
                top_tokens,
                x="Frequency",
                y="Token",
                orientation="h",
                title="Top tokens (sample)",
            )
            fig_tokens.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                showlegend=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_tokens, use_container_width=True)

            st.markdown("**Length statistics (tokens)**")
            tmp = pd.DataFrame({"label": dfc[label_col].astype(str), "len_tokens": len_tokens})
            st.dataframe(
                tmp.groupby("label")["len_tokens"].describe().round(2), use_container_width=True
            )
    if active_page == "Training":
        render_section_header(
            "Multi-model training and held-out evaluation",
            "Configure the duplicate-safe workflow: leakage-free TF-IDF, train/validation/test discipline, model comparison, and threshold selection on validation only.",
        )
        show_validation_scope_notice()

        fe1, fe2, fe3 = st.columns(3)
        with fe1:
            max_word_features = st.slider(
                "Max word features",
                min_value=5000,
                max_value=60000,
                value=20000,
                step=5000,
            )
        with fe2:
            use_char = st.checkbox("Add character n-grams", value=True)
        with fe3:
            test_size = st.slider("Validation split (%)", 10, 40, 20, 5) / 100.0
            holdout_test_size = (
                st.slider(
                    "Held-out test split (%)",
                    0,
                    30,
                    15,
                    5,
                    help="Recommended default is 15%. Set to 0 only for very small datasets.",
                )
                / 100.0
            )

        if holdout_test_size <= 0:
            st.warning(
                "Held-out test split is disabled. Results will be validation-only estimates.",
                icon="⚠️",
            )

        with st.expander("Advanced feature and reproducibility settings"):
            char_max = st.slider(
                "Max character n-gram features",
                min_value=5000,
                max_value=50000,
                value=20000,
                step=5000,
                disabled=not use_char,
            )
            random_state = st.number_input(
                "Random seed",
                min_value=0,
                max_value=999_999,
                value=42,
                step=1,
                help="Controls sampling and train/validation/test splits.",
            )
            random_state = int(random_state)

        run_stability_report = st.checkbox(
            "Run repeated split stability report (slower)",
            value=False,
            help="Runs three extra leakage-free repeated splits. This can add several minutes on large datasets.",
        )
        if run_stability_report:
            st.warning(
                "Repeated split stability can be slow on large text datasets because features are refit for each split.",
                icon="⏱️",
            )

        st.markdown("---")
        st.info(
            "The default settings are selected for a strong first run. Open the advanced sections only when you need to tune model behavior."
        )
        st.markdown("#### Model configuration")

        models_config: dict[str, dict[str, Any]] = {}
        mc1, mc2 = st.columns(2)

        with mc1:
            with st.expander("Logistic Regression", expanded=False):
                en = st.checkbox("Enable Logistic Regression", value=True, key="lr_en_ultra")
                C_val = st.slider("Regularization C", 0.1, 10.0, 2.0, 0.5, key="lr_C_ultra")
                calibrate_lr = st.checkbox(
                    "Calibrate probabilities",
                    value=True,
                    key="lr_calibrate_ultra",
                    help="Recommended when using threshold/cost decisions.",
                )
                models_config["Logistic Regression"] = {
                    "enabled": en,
                    "C": C_val,
                    "calibrated": calibrate_lr,
                    "calibration_method": "sigmoid",
                }

            with st.expander("Linear SVM / SGD challengers"):
                svm_en = st.checkbox("Enable calibrated Linear SVM", value=True, key="svm_en_ultra")
                svm_c = st.slider("Linear SVM C", 0.1, 10.0, 1.0, 0.5, key="svm_C_ultra")
                models_config["Linear SVM"] = {
                    "enabled": svm_en,
                    "C": svm_c,
                    "calibrated": True,
                    "calibration_method": "sigmoid",
                }

                sgd_en = st.checkbox("Enable SGD Logistic", value=False, key="sgd_en_ultra")
                sgd_alpha = st.select_slider(
                    "SGD alpha",
                    options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                    value=0.0001,
                    key="sgd_alpha_ultra",
                )
                models_config["SGD Logistic"] = {"enabled": sgd_en, "alpha": sgd_alpha}

        with mc2:
            with st.expander("Naive Bayes", expanded=False):
                en = st.checkbox("Enable Naive Bayes", value=True, key="nb_en_ultra")
                alpha = st.slider("alpha (smoothing)", 0.1, 3.0, 1.0, 0.1, key="nb_alpha_ultra")
                models_config["Naive Bayes"] = {"enabled": en, "alpha": alpha}

            with st.expander("Tree models (optional, slower for sparse text)"):
                rf_en = st.checkbox("Enable Random Forest", value=False, key="rf_en_ultra")
                est = st.slider("RF n_estimators", 50, 300, 120, 50, key="rf_est_ultra")
                depth = st.slider("RF max_depth", 5, 40, 18, 5, key="rf_depth_ultra")
                split = st.slider("RF min_samples_split", 2, 20, 5, 1, key="rf_split_ultra")
                models_config["Random Forest"] = {
                    "enabled": rf_en,
                    "n_estimators": est,
                    "max_depth": depth,
                    "min_samples_split": split,
                }

                gb_en = st.checkbox("Enable Gradient Boosting", value=False, key="gb_en_ultra")
                gb_est = st.slider("GB n_estimators", 50, 300, 120, 50, key="gb_est_ultra")
                gb_lr = st.slider("GB learning_rate", 0.01, 0.3, 0.08, 0.01, key="gb_lr_ultra")
                gb_depth = st.slider("GB max_depth", 2, 8, 3, 1, key="gb_depth_ultra")
                models_config["Gradient Boosting"] = {
                    "enabled": gb_en,
                    "n_estimators": gb_est,
                    "learning_rate": gb_lr,
                    "max_depth": gb_depth,
                }

        st.markdown("---")

        if st.button("Train models", type="primary", use_container_width=True):
            enabled_models = [m for m, cfg in models_config.items() if cfg["enabled"]]
            if not enabled_models:
                st.warning("Enable at least one model before training.", icon="⚠️")
            else:
                progress = st.progress(0)
                status = st.empty()

                progress.progress(5)
                status.markdown("Sampling rows for training (stratified)…")

                status.markdown("Creating duplicate-safe train/validation/test split…")
                progress.progress(20)

                try:
                    training_data = build_leakage_free_training_data(
                        dfc,
                        y,
                        max_train_rows=max_train_rows,
                        test_size=test_size,
                        max_word_features=max_word_features,
                        use_char=use_char,
                        char_max=char_max,
                        random_state=random_state,
                        holdout_test_size=holdout_test_size,
                    )
                except ValueError as e:
                    st.error(
                        "Could not prepare a stratified leakage-free training set. "
                        "Try increasing `Max rows used for training`, lowering `Validation split (%)`, "
                        "or checking your label mapping."
                    )
                    st.exception(e)
                    st.stop()

                df_train = training_data["df_sample"]
                X_train = training_data["X_train"]
                X_val = training_data["X_val"]
                X_test = training_data["X_test"]
                y_train = training_data["y_train"]
                y_val = training_data["y_val"]
                y_test = training_data["y_test"]
                vecs = training_data["vectorizers"]
                val_idx_global = training_data["val_idx_global"]
                test_idx_global = training_data["test_idx_global"]

                status.markdown(
                    "Vectorization complete: duplicate policy applied and TF-IDF fitted on training text only."
                )
                progress.progress(40)

                status.markdown("Training models…")
                progress.progress(65)

                train_started_at = time.perf_counter()
                trained_models = train_multiple_models(
                    X_train,
                    y_train,
                    models_config,
                    random_state=random_state,
                    n_jobs=int(n_jobs),
                )
                train_seconds = time.perf_counter() - train_started_at

                status.markdown("Evaluating models on validation set…")
                progress.progress(80)

                all_results: dict[str, dict[str, Any]] = {}
                for name, model in trained_models.items():
                    metrics = evaluate_model(model, X_val, y_val)
                    result_payload: dict[str, Any] = {"model": model, "metrics": metrics}
                    if X_test is not None and len(y_test):
                        result_payload["test_metrics"] = evaluate_model(model, X_test, y_test)
                    all_results[name] = result_payload

                stability_summary = pd.DataFrame()
                if run_stability_report:
                    status.markdown("Running repeated split stability report…")
                    try:
                        stability_raw = repeated_split_stability(
                            df_train["text_clean"].tolist(),
                            training_data["y_sample"],
                            trained_models,
                            n_splits=3,
                            test_size=test_size,
                            max_word_features=max_word_features,
                            use_char=use_char,
                            char_max=char_max,
                            random_state=random_state,
                        )
                        stability_summary = summarize_stability(stability_raw)
                    except Exception as exc:
                        st.warning(f"Stability report skipped: {exc}", icon="⚠️")

                status.markdown("Saving artifacts…")
                progress.progress(92)

                metadata = {
                    "pos_label": pos_label_str,
                    "neg_label": neg_label_str,
                    "val_idx": val_idx_global,
                    "y_val": y_val,
                    "test_idx": test_idx_global,
                    "y_test": y_test,
                    "text_col": text_col,
                    "label_col": label_col,
                    "pipeline_policy": "duplicate_safe_split_then_tfidf_fit_on_train_only",
                    "remove_duplicate_texts_before_split": bool(remove_duplicate_texts),
                    "duplicate_texts_removed_before_split": int(duplicate_texts_removed),
                    "max_word_features": max_word_features,
                    "use_char_ngrams": use_char,
                    "validation_split": test_size,
                    "holdout_test_split": holdout_test_size,
                    "test_rows": int(len(y_test)) if y_test is not None else 0,
                    "training_seconds": train_seconds,
                    "sample_rows": int(len(df_train)),
                    "char_max_features": int(char_max),
                    "random_state": int(random_state),
                    "n_jobs": int(n_jobs),
                    "data_quality_report": data_quality_report,
                    "stability_summary": stability_summary.to_dict("records")
                    if not stability_summary.empty
                    else [],
                    "evaluation_scope": "validation_plus_optional_heldout_test",
                    "evaluation_limitation": "Validation scores support model selection; held-out test scores are final-estimate only when enabled.",
                    "models_config": models_config,
                }
                decision_summary = build_decision_summary(results=all_results, metadata=metadata)
                metadata["decision_summary"] = decision_summary

                run_dir = save_training_artifacts(
                    MODELS_DIR,
                    vectorizers=vecs,
                    trained_models=trained_models,
                    all_results=all_results,
                    metadata=metadata,
                )
                report_model_name = str(decision_summary.get("recommended_model") or "")
                if report_model_name not in all_results:
                    report_model_name = max(
                        all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"]
                    )

                report_recommendations = recommend_thresholds(
                    y_val,
                    all_results[report_model_name]["metrics"]["y_proba"],
                    cost_fp=1.0,
                    cost_fn=5.0,
                    precision_target=0.90,
                    recall_target=0.90,
                )
                metadata["threshold_recommendation_model"] = report_model_name

                leaderboard_df = build_model_leaderboard(all_results)
                run_history_df = load_run_history(MODELS_DIR)
                data_quality_issue_df.to_csv(run_dir / "data_quality_issues.csv", index=False)
                leaderboard_df.to_csv(run_dir / "model_leaderboard.csv", index=False)
                pd.DataFrame(report_recommendations).to_csv(
                    run_dir / "threshold_recommendations.csv", index=False
                )

                report_html = render_model_report_html(
                    title="Advanced ML Sentiment Lab — Model Report",
                    metadata=metadata,
                    results=all_results,
                    data_quality=data_quality_report,
                    threshold_recommendations=report_recommendations,
                    decision_summary=decision_summary,
                    data_quality_issues=data_quality_issue_df,
                    run_history=run_history_df,
                )
                report_path = save_model_report_html(run_dir, report_html)
                metadata["report_path"] = str(report_path)

                model_card_md = render_model_card_markdown(
                    title="Advanced ML Sentiment Lab — Model Card",
                    metadata=metadata,
                    results=all_results,
                    decision_summary=decision_summary,
                    data_quality=data_quality_report,
                    threshold_recommendations=report_recommendations,
                )
                model_card_path = save_model_card_markdown(run_dir, model_card_md)
                metadata["model_card_path"] = str(model_card_path)
                smoke_path = SMOKE_TEST_WRITER.write(run_dir)
                metadata["reload_smoke_test_path"] = str(smoke_path)

                STATE_STORE.persist_metadata(MODELS_DIR, run_dir, metadata)
                STATE_STORE.save(
                    vectorizers=vecs,
                    trained_models=trained_models,
                    all_results=all_results,
                    metadata=metadata,
                    run_dir=run_dir,
                )

                progress.progress(100)
                status.markdown("Training complete.")

                st.success(f"Trained {len(trained_models)} model(s) on {len(df_train):,} rows.")
                st.caption(f"Saved latest artifacts and immutable run history: `{run_dir}`")

                rows: list[dict[str, Any]] = []
                for name, res in all_results.items():
                    m = res["metrics"]
                    tm = res.get("test_metrics", {})
                    rows.append(
                        {
                            "Model": name,
                            "Val Accuracy": f"{m['accuracy']:.4f}",
                            "Precision": f"{m['precision']:.4f}",
                            "Recall": f"{m['recall']:.4f}",
                            "Val F1": f"{m['f1']:.4f}",
                            "Test F1": f"{tm.get('f1', float('nan')):.4f}" if tm else "not enabled",
                            "ROC-AUC": f"{m['roc_auc']:.4f}"
                            if not np.isnan(m["roc_auc"])
                            else "nan",
                            "PR-AUC": f"{m['pr_auc']:.4f}",
                            "Val Brier": f"{m.get('brier', float('nan')):.4f}",
                            "Test Brier": f"{tm.get('brier', float('nan')):.4f}"
                            if tm
                            else "not enabled",
                        }
                    )
                res_df = pd.DataFrame(rows)
                st.markdown("#### Training summary")
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                if not stability_summary.empty:
                    st.markdown("#### Repeated split stability summary")
                    st.dataframe(stability_summary, use_container_width=True, hide_index=True)
    if active_page == "Evaluation":
        render_section_header(
            "Threshold tuning and business cost",
            "Pick a model, move the decision threshold, and inspect how metrics and expected cost change.",
        )

        state = STATE_STORE.load(MODELS_DIR)

        if state is None:
            st.info("Train models in the previous tab to unlock threshold tuning.")
        else:
            all_results = state["results"]
            metadata = state["metadata"]
            y_val = metadata["y_val"]
            show_validation_scope_notice()

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])

            model_name = st.selectbox(
                "Model to analyse",
                options=list(all_results.keys()),
                index=list(all_results.keys()).index(best_name),
            )

            metrics_base = all_results[model_name]["metrics"]
            y_proba = metrics_base["y_proba"]
            default_threshold = threshold_for_model(all_results, metadata, model_name)

            col_thr, col_cost = st.columns([1.2, 1])
            with col_thr:
                threshold = st.slider(
                    "Decision threshold for positive class",
                    min_value=0.05,
                    max_value=0.95,
                    value=float(default_threshold),
                    step=0.01,
                )
            with col_cost:
                cost_fp = st.number_input(
                    "Cost of a false positive (FP)", min_value=0.0, value=1.0, step=0.5
                )
                cost_fn = st.number_input(
                    "Cost of a false negative (FN)", min_value=0.0, value=5.0, step=0.5
                )

            thr_metrics, df_curve = compute_threshold_view(
                y_true=y_val,
                y_proba=y_proba,
                threshold=threshold,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
            )

            threshold_recs = recommend_thresholds(
                y_val,
                y_proba,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                precision_target=0.90,
                recall_target=0.90,
            )

            st.markdown("##### Recommended thresholds")
            st.dataframe(pd.DataFrame(threshold_recs), use_container_width=True, hide_index=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Accuracy", f"{thr_metrics['accuracy']:.4f}")
            with c2:
                st.metric("Precision", f"{thr_metrics['precision']:.4f}")
            with c3:
                st.metric("Recall", f"{thr_metrics['recall']:.4f}")
            with c4:
                st.metric("F1", f"{thr_metrics['f1']:.4f}")

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.metric("Specificity", f"{thr_metrics['specificity']:.4f}")
            with c6:
                st.metric("FP", thr_metrics["fp"])
            with c7:
                st.metric("FN", thr_metrics["fn"])
            with c8:
                st.metric("Total cost", f"{thr_metrics['cost']:.2f}")

            st.markdown("##### F1 over threshold")
            fig_thr = px.line(df_curve, x="threshold", y="f1", title="F1 vs threshold")
            fig_thr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_thr, use_container_width=True)

            fig_cost = px.line(
                df_curve,
                x="threshold",
                y="cost",
                title=f"Estimated cost (FP cost={cost_fp}, FN cost={cost_fn})",
            )
            fig_cost.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig_cost, use_container_width=True)
    if active_page == "Evaluation":
        render_section_header(
            "Model comparison",
            "Side-by-side comparison of metrics, ROC / PR curves, and confusion matrices.",
        )

        state = STATE_STORE.load(MODELS_DIR)

        if state is None:
            st.info("Train models first to unlock comparison.")
        else:
            all_results = state["results"]
            metadata = state["metadata"]
            y_val = metadata["y_val"]
            show_validation_scope_notice()

            st.markdown("#### Model cards")
            cols = st.columns(len(all_results))
            for (name, res), col in zip(all_results.items(), cols, strict=True):
                m = res["metrics"]
                with col:
                    st.markdown(
                        f"""
                        <div class="model-card">
                            <div class="model-name">{name}</div>
                            <div class="model-metrics">
                                <div class="metric-box">
                                    <div class="metric-label">ACC</div>
                                    <div class="metric-value">{m["accuracy"]:.3f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">F1</div>
                                    <div class="metric-value">{m["f1"]:.3f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">ROC</div>
                                    <div class="metric-value">{m["roc_auc"]:.3f}</div>
                                </div>
                                <div class="metric-box">
                                    <div class="metric-label">PR</div>
                                    <div class="metric-value">{m["pr_auc"]:.3f}</div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("##### ROC curves")
                fig_roc = go.Figure()
                for name, res in all_results.items():
                    fpr, tpr, _ = roc_curve(y_val, res["metrics"]["y_proba"])
                    auc_score = res["metrics"]["roc_auc"]
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name=f"{name} (AUC={auc_score:.3f})",
                        )
                    )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(dash="dash", color="gray"),
                    )
                )
                fig_roc.update_layout(
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with r2:
                st.markdown("##### Precision-Recall curves")
                fig_pr = go.Figure()
                for name, res in all_results.items():
                    prec, rec, _ = precision_recall_curve(y_val, res["metrics"]["y_proba"])
                    pr_auc = res["metrics"]["pr_auc"]
                    fig_pr.add_trace(
                        go.Scatter(
                            x=rec,
                            y=prec,
                            mode="lines",
                            name=f"{name} (AUC={pr_auc:.3f})",
                            fill="tonexty",
                        )
                    )
                fig_pr.update_layout(
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            st.markdown("##### Confusion matrices (validation set, recommended threshold)")
            cm_cols = st.columns(len(all_results))
            for (name, res), col in zip(all_results.items(), cm_cols, strict=True):
                m = res["metrics"]
                model_threshold = threshold_for_model(all_results, metadata, name)
                y_pred_thr = labels_from_proba(m["y_proba"], threshold=model_threshold)
                cm = confusion_matrix(y_val, y_pred_thr, labels=[0, 1])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[metadata["neg_label"], metadata["pos_label"]],
                    y=[metadata["neg_label"], metadata["pos_label"]],
                    text_auto=True,
                    title=f"{name} @ {model_threshold:.3f}",
                )
                fig_cm.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                )
                with col:
                    st.plotly_chart(fig_cm, use_container_width=True)
    if active_page == "Evaluation":
        render_section_header(
            "Error analysis",
            "Browse misclassified texts to see where the model struggles and how confident it was.",
        )

        state = STATE_STORE.load(MODELS_DIR)

        if state is None:
            st.info("Train models first to unlock error analysis.")
        else:
            all_results = state["results"]
            models = state["models"]
            vecs = state["vectorizers"]
            metadata = state["metadata"]
            y_val = metadata["y_val"]
            val_idx = metadata["val_idx"]
            show_validation_scope_notice()

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])
            model_name = st.selectbox(
                "Model to inspect",
                options=list(all_results.keys()),
                index=list(all_results.keys()).index(best_name),
            )

            m = all_results[model_name]["metrics"]
            y_proba = m["y_proba"]
            model_threshold = threshold_for_model(all_results, metadata, model_name)
            y_pred = labels_from_proba(y_proba, threshold=model_threshold)
            st.caption(
                f"Error review uses the recommended threshold for this model: {model_threshold:.3f}"
            )

            val_df = dfc.loc[val_idx].copy()
            val_df["true_label"] = np.where(
                y_val == 1, metadata["pos_label"], metadata["neg_label"]
            )
            val_df["pred_label"] = np.where(
                y_pred == 1, metadata["pos_label"], metadata["neg_label"]
            )
            val_df["proba_pos"] = y_proba
            val_df["correct"] = y_val == y_pred
            val_df["error_type"] = np.where(
                val_df["correct"],
                "Correct",
                np.where(y_val == 1, "False negative", "False positive"),
            )

            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                only_errors = st.checkbox("Show only misclassified samples", value=True)
            with col_f2:
                sort_mode = st.selectbox(
                    "Sort by",
                    options=["Most confident errors", "Least confident predictions", "Random"],
                )

            df_view = val_df.copy()
            if only_errors:
                df_view = df_view[~df_view["correct"]]

            if sort_mode in ["Most confident errors", "Least confident predictions"]:
                df_view["conf"] = np.abs(df_view["proba_pos"] - model_threshold)
                df_view = df_view.sort_values(
                    "conf", ascending=(sort_mode == "Least confident predictions")
                )
            else:
                df_view = df_view.sample(frac=1, random_state=42)

            top_n = st.slider("Rows to show", 10, 200, 50, 10)
            cols_show = ["text_raw", "true_label", "pred_label", "proba_pos", "error_type"]
            st.dataframe(df_view[cols_show].head(top_n), use_container_width=True)

            st.markdown("#### Linear feature explainability")
            st.caption(
                "For linear models, this shows the strongest positive and negative TF-IDF terms. "
                "For non-linear models or unsupported calibrated wrappers, the panel is skipped."
            )
            pos_terms, neg_terms = top_linear_terms(models[model_name], vecs, top_n=20)
            if pos_terms.empty or neg_terms.empty:
                st.info(
                    "Feature-level explanation is available for supported linear models only. "
                    "Try Logistic Regression, calibrated Logistic Regression, Linear SVM, or SGD Logistic."
                )
            else:
                pos_col, neg_col = st.columns(2)
                with pos_col:
                    st.markdown(f"##### Terms pushing toward `{metadata['pos_label']}`")
                    fig_pos = px.bar(
                        pos_terms,
                        x="weight",
                        y="term",
                        orientation="h",
                        title="Top positive terms",
                    )
                    fig_pos.update_layout(
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e5e7eb"),
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
                with neg_col:
                    st.markdown(f"##### Terms pushing toward `{metadata['neg_label']}`")
                    fig_neg = px.bar(
                        neg_terms.sort_values("weight", ascending=False),
                        x="weight",
                        y="term",
                        orientation="h",
                        title="Top negative terms",
                    )
                    fig_neg.update_layout(
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e5e7eb"),
                    )
                    st.plotly_chart(fig_neg, use_container_width=True)
    if active_page == "Prediction Lab":
        render_section_header(
            "Deployment & interactive prediction",
            "Use the recommended model first, test arbitrary texts, and reuse the same logic in an API or batch job.",
        )

        state = STATE_STORE.load(MODELS_DIR)

        if state is None:
            st.info("Train models first to enable deployment.")
        else:
            models = state["models"]
            vecs = state["vectorizers"]
            all_results = state["results"]
            metadata = state["metadata"]

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])
            decision_summary = metadata.get("decision_summary", {}) or {}
            recommended_name = str(decision_summary.get("recommended_model") or "")

            deployment_options = ["Recommended model", "Best by validation F1"] + list(models.keys())
            model_choice = st.selectbox(
                "Model for deployment",
                options=deployment_options,
                index=0,
            )

            if model_choice == "Recommended model":
                deploy_name = recommended_name if recommended_name in models else best_name
                st.info(f"Using {deploy_name} based on the saved decision summary.")
            elif model_choice == "Best by validation F1":
                deploy_name = best_name
                st.info(f"Using {best_name} based on validation F1.")
            else:
                deploy_name = model_choice

            model = models[deploy_name]
            word_vec = vecs[0]
            char_vec = vecs[1] if len(vecs) > 1 else None

            if "deploy_text" not in st.session_state:
                st.session_state["deploy_text"] = ""

            c_in, c_out = st.columns([1.4, 1.1])
            with c_in:
                st.markdown("#### Input text")
                example_col1, example_col2, example_col3 = st.columns(3)
                with example_col1:
                    if st.button("Positive example"):
                        st.session_state["deploy_text"] = (
                            "Absolutely loved this. Great quality, fast delivery, and I would happily buy again."
                        )
                with example_col2:
                    if st.button("Mixed example"):
                        st.session_state["deploy_text"] = (
                            "Some parts were decent, but overall it felt overpriced and a bit disappointing."
                        )
                with example_col3:
                    if st.button("Negative example"):
                        st.session_state["deploy_text"] = (
                            "Terrible experience. Support was unhelpful and the product broke quickly."
                        )

                text_input = st.text_area(
                    "Write or paste any text", height=160, value=st.session_state["deploy_text"]
                )
                predict_btn = st.button("Predict sentiment", type="primary")

            with c_out:
                if predict_btn and text_input.strip():
                    clean_text = basic_clean(text_input)
                    active_vectorizers = tuple(v for v in (word_vec, char_vec) if v is not None)
                    X_test = transform_advanced_features([clean_text], active_vectorizers)

                    try:
                        proba = float(predict_proba_pos(model, X_test)[0])
                    except (AttributeError, ValueError, TypeError) as e:
                        st.error(f"Model scoring failed: {e}")
                        st.stop()

                    decision_threshold = threshold_for_model(all_results, metadata, deploy_name)
                    label_int = int(proba >= decision_threshold)
                    label_str = metadata["pos_label"] if label_int == 1 else metadata["neg_label"]
                    conf_pct = class_confidence(proba, label_int) * 100.0

                    st.markdown(
                        """
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted sentiment</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    cls = "prediction-positive" if label_int == 1 else "prediction-negative"
                    st.markdown(
                        f'<div class="prediction-result {cls}">{escape_html(label_str)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        (
                            f'<div class="prediction-confidence">{conf_pct:.1f}% class confidence · '
                            f"positive probability {proba * 100.0:.1f}% · threshold {decision_threshold:.3f}</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                    width_pct = int(max(0, min(100, conf_pct)))
                    st.markdown(
                        f"""
                        <div class="progress-bar">
                            <div class="progress-fill" style="width:{width_pct}%;"></div>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    explanation = explain_linear_prediction(
                        model, active_vectorizers, X_test, top_n=8
                    )
                    if not explanation.empty:
                        st.markdown("#### Why this prediction?")
                        st.dataframe(explanation, use_container_width=True, hide_index=True)
    if active_page == "Export":
        render_section_header(
            "Export package",
            "Download the active run report, model card, metrics, threshold recommendations, metadata, and stakeholder bundle.",
        )

        state = STATE_STORE.load(MODELS_DIR)
        if state is None:
            st.info("Train models first to create exportable artifacts.")
        else:
            run_dir = state.get("run_dir")
            metadata = state["metadata"]
            if not run_dir:
                st.info(
                    "Active run artifacts are not available yet. Train models again to create an export bundle."
                )
            else:
                st.markdown("#### Active run files")
                st.code(str(run_dir))
                report_path = Path(metadata.get("report_path", run_dir / "model_report.html"))
                if report_path.exists():
                    st.download_button(
                        "Download active HTML model report",
                        report_path.read_bytes(),
                        file_name="model_report.html",
                        mime="text/html",
                    )

                model_card_path = Path(metadata.get("model_card_path", run_dir / "model_card.md"))
                if model_card_path.exists():
                    st.download_button(
                        "Download model card Markdown",
                        model_card_path.read_bytes(),
                        file_name="model_card.md",
                        mime="text/markdown",
                    )

                bundle_bytes = BUNDLE_BUILDER.build(Path(run_dir))
                if bundle_bytes:
                    st.download_button(
                        "Download stakeholder review bundle (.zip)",
                        bundle_bytes,
                        file_name="sentiment_lab_run_review_bundle.zip",
                        mime="application/zip",
                    )

                for filename, mime in [
                    ("metrics.csv", "text/csv"),
                    ("model_leaderboard.csv", "text/csv"),
                    ("threshold_recommendations.csv", "text/csv"),
                    ("data_quality_issues.csv", "text/csv"),
                    ("metadata.json", "application/json"),
                    ("reload_smoke_test.py", "text/x-python"),
                ]:
                    path = Path(run_dir) / filename
                    if path.exists():
                        st.download_button(
                            f"Download {filename}",
                            path.read_bytes(),
                            file_name=filename,
                            mime=mime,
                        )

            st.caption(
                "Artifacts are local trusted outputs. Do not load joblib artifacts from untrusted sources."
            )

    if active_page == "Run History":
        render_section_header(
            "Run history",
            "Review recent local experiment runs and compare their saved validation/test metrics.",
        )

        history_df = load_run_history(MODELS_DIR)
        if history_df.empty:
            st.info("No saved runs yet. Train models to create run artifacts.")
        else:
            st.markdown("#### Recent runs")
            st.dataframe(history_df.round(4), use_container_width=True, hide_index=True)
            st.download_button(
                "Download run history CSV",
                history_df.to_csv(index=False).encode("utf-8"),
                file_name="run_history.csv",
                mime="text/csv",
            )

            st.caption(
                "This app uses a lightweight local run-history layer instead of requiring external experiment-tracking services."
            )


if __name__ == "__main__":
    main()
