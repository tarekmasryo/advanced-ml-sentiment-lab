from __future__ import annotations

import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import hstack
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -----------------------------------------------------------------------------
# Paths / logging
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts"))).expanduser()
MODELS_DIR = ARTIFACTS_DIR / "sentiment_lab"

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# UI styling
# -----------------------------------------------------------------------------
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(circle at top, #151a2f 0, #020617 45%, #020617 100%);
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
}

.hero-premium {
    padding: 30px 34px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        rgba(88, 80, 236, 0.35) 0%,
        rgba(236, 72, 153, 0.22) 55%,
        rgba(15, 23, 42, 0.98) 100%
    );
    border: 1px solid rgba(129, 140, 248, 0.55);
    box-shadow:
        0 20px 70px rgba(15, 23, 42, 0.9),
        0 0 120px rgba(129, 140, 248, 0.4);
    backdrop-filter: blur(16px);
    margin-bottom: 26px;
}

.hero-title-pro {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: 0.02em;
    background: linear-gradient(120deg, #e5e7eb 0%, #e0f2fe 30%, #f9a8d4 60%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle-pro {
    font-size: 15px;
    color: #cbd5e1;
    line-height: 1.7;
    max-width: 840px;
}

.hero-badges {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: radial-gradient(circle at top left, #6366f1, #8b5cf6);
    color: #f9fafb;
    box-shadow: 0 4px 16px rgba(129, 140, 248, 0.5);
}

.badge-soft {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(55, 65, 81, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.6);
    color: #cbd5e1;
    box-shadow: none;
}

.kpi-premium {
    padding: 22px 20px;
    border-radius: 20px;
    background: radial-gradient(circle at top left, rgba(129, 140, 248, 0.16), rgba(15, 23, 42, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.5);
    box-shadow: 0 14px 40px rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(12px);
    transition: all 0.22s ease;
}

.kpi-premium:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 60px rgba(30, 64, 175, 0.7);
}

.kpi-icon {
    font-size: 26px;
    margin-bottom: 6px;
}

.kpi-label-pro {
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 6px;
}

.kpi-value-pro {
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(130deg, #e5e7eb 0%, #bfdbfe 40%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.kpi-trend {
    font-size: 11px;
    color: #22c55e;
    margin-top: 2px;
}

.section-header-pro {
    font-size: 22px;
    font-weight: 800;
    color: #e5e7eb;
    margin: 18px 0 6px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.5);
}

.section-desc-pro {
    font-size: 13px;
    color: #9ca3af;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: radial-gradient(circle at top, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.9));
    padding: 8px;
    border-radius: 999px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 20px;
    background: transparent;
    color: #9ca3af;
    font-size: 13px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    color: #f9fafb !important;
    box-shadow: 0 4px 16px rgba(129, 140, 248, 0.6);
}

.model-card {
    padding: 18px 16px;
    border-radius: 16px;
    background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.97), rgba(15, 23, 42, 0.94));
    border: 1px solid rgba(148, 163, 184, 0.5);
    margin-bottom: 12px;
}

.model-name {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}

.model-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
}

.metric-box {
    padding: 6px 8px;
    border-radius: 10px;
    background: rgba(30, 64, 175, 0.35);
}

.metric-label {
    font-size: 10px;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    font-size: 14px;
    font-weight: 700;
}

.info-box {
    padding: 14px 16px;
    border-radius: 14px;
    background: rgba(37, 99, 235, 0.14);
    border-left: 4px solid #3b82f6;
    margin: 10px 0 16px 0;
}

.info-box-title {
    font-size: 13px;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 4px;
}

.info-box-text {
    font-size: 12px;
    color: #e5e7eb;
    line-height: 1.6;
}

.threshold-card {
    padding: 18px;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.97), rgba(15, 23, 42, 0.94));
    border: 1px solid rgba(148, 163, 184, 0.5);
    box-shadow: 0 12px 36px rgba(15, 23, 42, 0.9);
}

.prediction-card {
    padding: 20px 18px;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.97), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(129, 140, 248, 0.6);
    box-shadow: 0 12px 40px rgba(30, 64, 175, 0.8);
    margin-top: 8px;
}

.prediction-label {
    font-size: 12px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.prediction-result {
    font-size: 26px;
    font-weight: 800;
    margin: 6px 0;
}

.prediction-positive {
    color: #22c55e;
}

.prediction-negative {
    color: #f97373;
}

.prediction-confidence {
    font-size: 14px;
    color: #e5e7eb;
}

.progress-bar {
    width: 100%;
    height: 8px;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.8);
    overflow: hidden;
    margin-top: 6px;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #a3e635 50%, #facc15 100%);
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}

.loading-pulse {
    animation: pulse 1.6s ease-in-out infinite;
}
</style>
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def basic_clean(s: str) -> str:
    import html
    import re

    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s).lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _minmax_scale(scores: np.ndarray) -> np.ndarray:
    smin = float(np.min(scores))
    smax = float(np.max(scores))
    denom = (smax - smin) + 1e-9
    return (scores - smin) / denom


def _predict_proba_pos(model: Any, X) -> np.ndarray:
    """
    Robustly produce a [n] probability-like score for the positive class.
    - prefer predict_proba
    - else decision_function then min-max to [0,1]
    """
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        proba = predict_proba(X)
        return proba[:, 1]

    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        scores = decision_function(X)
        scores = np.asarray(scores).reshape(-1)
        return _minmax_scale(scores)

    raise AttributeError("Model has neither predict_proba nor decision_function.")


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    """
    Try reading CSV with two attempts:
      1) default
      2) with encoding_errors='ignore'
    Catch only expected IO/parse/decode errors.
    """
    read_errors = (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError)

    try:
        return pd.read_csv(path)
    except read_errors as e:
        logger.debug("read_csv failed for %s (default): %s", path, e)

    try:
        return pd.read_csv(path, encoding_errors="ignore")
    except read_errors as e:
        logger.debug("read_csv failed for %s (encoding_errors=ignore): %s", path, e)

    return None


# -----------------------------------------------------------------------------
# Data loading / cleaning
# -----------------------------------------------------------------------------
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
    out = df.copy()
    out["text_raw"] = out[text_col].astype(str)
    out["text_clean"] = out["text_raw"].map(basic_clean)

    lab = out[label_col].astype(str)
    mask = lab.isin([pos_label_str, neg_label_str])
    dropped = int((~mask).sum())

    out = out.loc[mask].copy()
    lab = lab.loc[mask]
    y = np.where(lab == pos_label_str, 1, 0).astype(int)

    return out, y, dropped


# -----------------------------------------------------------------------------
# Feature engineering / training
# -----------------------------------------------------------------------------
def build_advanced_features(
    texts: list[str],
    max_word_features: int,
    use_char: bool,
    char_max: int,
) -> tuple[Any, tuple[TfidfVectorizer, ...]]:
    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_word_features,
        min_df=2,
        max_df=0.95,
    )
    Xw = word_vec.fit_transform(texts)

    vecs: list[TfidfVectorizer] = [word_vec]
    mats = [Xw]

    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            max_features=char_max,
            min_df=2,
        )
        Xc = char_vec.fit_transform(texts)
        vecs.append(char_vec)
        mats.append(Xc)

    X_all = hstack(mats) if len(mats) > 1 else mats[0]
    return X_all, tuple(vecs)


def train_multiple_models(
    X_train, y_train, models_config: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, cfg in models_config.items():
        if not cfg.get("enabled", False):
            continue

        if name == "Logistic Regression":
            model = LogisticRegression(
                C=float(cfg["C"]),
                max_iter=2000,
                solver="saga",
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )
        elif name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=int(cfg["n_estimators"]),
                max_depth=int(cfg["max_depth"]),
                min_samples_split=int(cfg["min_samples_split"]),
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )
        elif name == "Gradient Boosting":
            model = GradientBoostingClassifier(
                n_estimators=int(cfg["n_estimators"]),
                learning_rate=float(cfg["learning_rate"]),
                max_depth=int(cfg["max_depth"]),
                random_state=42,
            )
        elif name == "Naive Bayes":
            model = MultinomialNB(alpha=float(cfg["alpha"]))
        else:
            continue

        model.fit(X_train, y_train)
        models[name] = model

    return models


def evaluate_model(model: Any, X_val, y_val: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_val)
    y_proba = _predict_proba_pos(model, X_val)

    # roc_auc can fail if only 1 class present in y_val
    try:
        roc_auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": average_precision_score(y_val, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def compute_threshold_view(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
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

    grid = np.linspace(0.05, 0.95, 37)
    rows: list[dict[str, Any]] = []
    for t in grid:
        y_pred_g = (y_proba >= t).astype(int)
        cm_g = confusion_matrix(y_true, y_pred_g, labels=[0, 1])
        tn_g, fp_g, fn_g, _tp_g = cm_g.ravel()
        f1_g = f1_score(y_true, y_pred_g, zero_division=0)
        cost_g = fp_g * float(cost_fp) + fn_g * float(cost_fn)
        rows.append(
            {
                "threshold": float(t),
                "f1": float(f1_g),
                "fp": int(fp_g),
                "fn": int(fn_g),
                "cost": float(cost_g),
            }
        )

    df_curve = pd.DataFrame(rows)
    return metrics, df_curve


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore")

    st.set_page_config(
        page_title="Advanced ML Sentiment Lab",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.sidebar.markdown("### 🚀 Advanced ML Sentiment Lab")
    st.sidebar.markdown("---")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV dataset (optional)",
        type=["csv"],
        help="For small custom datasets. For IMDB, keep the CSV in the repo.",
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = load_csv_auto()

    if df is None:
        st.markdown(
            """
            <div class="hero-premium">
                <div class="hero-title-pro">Advanced ML Sentiment Lab</div>
                <div class="hero-subtitle-pro">
                    No dataset found yet.
                    Put <code>IMDB Dataset.csv</code> in the root of this repo
                    (or inside a <code>data/</code> folder), or upload a CSV from the sidebar.
                </div>
                <div class="hero-badges">
                    <span class="badge-pill">TF-IDF word &amp; character features</span>
                    <span class="badge-pill">Logistic / Random Forest / Gradient Boosting / Naive Bayes</span>
                    <span class="badge-soft">Threshold tuning with business cost</span>
                    <span class="badge-soft">Artifacts saved under artifacts/</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    all_cols = list(df.columns)
    if len(all_cols) < 2:
        st.error("Dataset must have at least 2 columns (text + label).")
        st.stop()

    st.sidebar.markdown("### Column mapping")

    default_text_idx = 0
    for i, c in enumerate(all_cols):
        if str(c).lower() in ["review", "text", "comment", "content", "message", "body"]:
            default_text_idx = i
            break

    text_col = st.sidebar.selectbox("Text column", all_cols, index=default_text_idx)

    label_candidates = [c for c in all_cols if c != text_col]
    if not label_candidates:
        st.error("Dataset must have at least 2 columns (text + label).")
        st.stop()

    default_label_idx = 0
    for i, c in enumerate(label_candidates):
        if str(c).lower() in ["sentiment", "label", "target", "y", "class"]:
            default_label_idx = i
            break

    label_col = st.sidebar.selectbox("Label column", label_candidates, index=default_label_idx)

    label_values = df[label_col].astype(str).dropna().value_counts().index.tolist()
    if len(label_values) < 2:
        st.error("Label column must have at least 2 distinct values.")
        st.stop()

    st.sidebar.markdown("### Label mapping")
    pos_label_str = st.sidebar.selectbox("Positive class (1)", label_values, index=0)
    neg_label_str = st.sidebar.selectbox(
        "Negative class (0)", label_values, index=1 if len(label_values) > 1 else 0
    )

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

    dfc, y, dropped = clean_df(
        df,
        text_col=text_col,
        label_col=label_col,
        pos_label_str=pos_label_str,
        neg_label_str=neg_label_str,
    )

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

    st.markdown(
        f"""
        <div class="hero-premium">
            <div class="hero-title-pro">Advanced ML Sentiment Lab</div>
            <div class="hero-subtitle-pro">
                Production-style sentiment analytics on <b>{n_rows:,}</b> samples.
                Configure TF-IDF features, train multiple models, tune the decision threshold
                under custom business costs, and inspect model errors.
            </div>
            <div class="hero-badges">
                <span class="badge-pill">Text column: {text_col}</span>
                <span class="badge-pill">Label column: {label_col}</span>
                <span class="badge-soft">Binary labels: {pos_label_str} / {neg_label_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-premium">
                <div class="kpi-icon">📊</div>
                <div class="kpi-label-pro">Total samples</div>
                <div class="kpi-value-pro">{n_rows:,}</div>
                <div class="kpi-trend">Cleaned for modeling</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-premium">
                <div class="kpi-icon">✅</div>
                <div class="kpi-label-pro">Positive share</div>
                <div class="kpi-value-pro">{pos_ratio * 100:.1f}%</div>
                <div class="kpi-trend">{n_pos:,} positive / {n_neg:,} negative</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-premium">
                <div class="kpi-icon">📝</div>
                <div class="kpi-label-pro">Avg text length</div>
                <div class="kpi-value-pro">{avg_len:.0f}</div>
                <div class="kpi-trend">characters per record</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-premium">
                <div class="kpi-icon">📚</div>
                <div class="kpi-label-pro">Sample vocabulary</div>
                <div class="kpi-value-pro">{sample_vocab:,}</div>
                <div class="kpi-trend">unique tokens (first 5k rows)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_eda, tab_train, tab_threshold, tab_compare, tab_errors, tab_deploy = st.tabs(
        [
            "EDA",
            "Train & Validation",
            "Threshold & Cost",
            "Compare Models",
            "Error Analysis",
            "Deploy",
        ]
    )

    # -------------------------------------------------------------------------
    # EDA
    # -------------------------------------------------------------------------
    with tab_eda:
        st.markdown(
            '<div class="section-header-pro">Exploratory data analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-desc-pro">Quick checks on class balance, text lengths, and token distribution.</div>',
            unsafe_allow_html=True,
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

    # -------------------------------------------------------------------------
    # Train & Validation
    # -------------------------------------------------------------------------
    with tab_train:
        st.markdown(
            '<div class="section-header-pro">Multi-model training (single split)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-desc-pro">Configure TF-IDF, select models, then run a stratified train/validation split on a capped subset for fast turnaround.</div>',
            unsafe_allow_html=True,
        )

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

        st.markdown("---")
        st.markdown("#### Model configuration")

        models_config: dict[str, dict[str, Any]] = {}
        mc1, mc2 = st.columns(2)

        with mc1:
            with st.expander("Logistic Regression", expanded=True):
                en = st.checkbox("Enable Logistic Regression", value=True, key="lr_en_ultra")
                C_val = st.slider("Regularization C", 0.1, 10.0, 2.0, 0.5, key="lr_C_ultra")
                models_config["Logistic Regression"] = {"enabled": en, "C": C_val}

            with st.expander("Random Forest"):
                en = st.checkbox("Enable Random Forest", value=False, key="rf_en_ultra")
                est = st.slider("n_estimators", 50, 300, 120, 50, key="rf_est_ultra")
                depth = st.slider("max_depth", 5, 40, 18, 5, key="rf_depth_ultra")
                split = st.slider("min_samples_split", 2, 20, 5, 1, key="rf_split_ultra")
                models_config["Random Forest"] = {
                    "enabled": en,
                    "n_estimators": est,
                    "max_depth": depth,
                    "min_samples_split": split,
                }

        with mc2:
            with st.expander("Gradient Boosting"):
                en = st.checkbox("Enable Gradient Boosting", value=False, key="gb_en_ultra")
                est = st.slider("n_estimators", 50, 300, 120, 50, key="gb_est_ultra")
                lr = st.slider("learning_rate", 0.01, 0.3, 0.08, 0.01, key="gb_lr_ultra")
                depth = st.slider("max_depth", 2, 8, 3, 1, key="gb_depth_ultra")
                models_config["Gradient Boosting"] = {
                    "enabled": en,
                    "n_estimators": est,
                    "learning_rate": lr,
                    "max_depth": depth,
                }

            with st.expander("Naive Bayes"):
                en = st.checkbox("Enable Naive Bayes", value=True, key="nb_en_ultra")
                alpha = st.slider("alpha (smoothing)", 0.1, 3.0, 1.0, 0.1, key="nb_alpha_ultra")
                models_config["Naive Bayes"] = {"enabled": en, "alpha": alpha}

        st.markdown("---")

        random_state = 42

        if st.button("Train models", type="primary", use_container_width=True):
            enabled_models = [m for m, cfg in models_config.items() if cfg["enabled"]]
            if not enabled_models:
                st.warning("Enable at least one model before training.", icon="⚠️")
            else:
                progress = st.progress(0)
                status = st.empty()

                progress.progress(5)
                status.markdown("Sampling rows for training (stratified)…")

                n_total = len(dfc)
                train_rows = min(max_train_rows, n_total)
                indices = np.arange(n_total)

                if train_rows < n_total:
                    sample_idx, _ = train_test_split(
                        indices,
                        train_size=train_rows,
                        stratify=y,
                        random_state=random_state,
                    )
                else:
                    sample_idx = indices

                df_train = dfc.iloc[sample_idx].copy()
                y_sample = y[sample_idx]

                status.markdown("Cleaning and vectorising text…")
                progress.progress(20)

                texts = df_train["text_clean"].tolist()
                X_all, vecs = build_advanced_features(
                    texts,
                    max_word_features=max_word_features,
                    use_char=use_char,
                    char_max=20000,
                )

                status.markdown("Creating stratified train/validation split…")
                progress.progress(40)

                local_idx = np.arange(len(df_train))
                train_loc, val_loc, y_train, y_val = train_test_split(
                    local_idx,
                    y_sample,
                    test_size=test_size,
                    stratify=y_sample,
                    random_state=random_state,
                )
                X_train = X_all[train_loc]
                X_val = X_all[val_loc]

                status.markdown("Training models…")
                progress.progress(65)

                trained_models = train_multiple_models(X_train, y_train, models_config)

                status.markdown("Evaluating models on validation set…")
                progress.progress(80)

                all_results: dict[str, dict[str, Any]] = {}
                for name, model in trained_models.items():
                    metrics = evaluate_model(model, X_val, y_val)
                    all_results[name] = {"model": model, "metrics": metrics}

                status.markdown("Saving artifacts…")
                progress.progress(92)

                val_idx_global = df_train.index[val_loc]

                joblib.dump(vecs, MODELS_DIR / "vectorizers.joblib")
                joblib.dump(trained_models, MODELS_DIR / "models.joblib")
                joblib.dump(all_results, MODELS_DIR / "results.joblib")
                joblib.dump(
                    {
                        "pos_label": pos_label_str,
                        "neg_label": neg_label_str,
                        "val_idx": val_idx_global,
                        "y_val": y_val,
                        "text_col": text_col,
                        "label_col": label_col,
                    },
                    MODELS_DIR / "metadata.joblib",
                )

                progress.progress(100)
                status.markdown("Training complete.")

                st.success(f"Trained {len(trained_models)} model(s) on {len(df_train):,} rows.")

                rows: list[dict[str, Any]] = []
                for name, res in all_results.items():
                    m = res["metrics"]
                    rows.append(
                        {
                            "Model": name,
                            "Accuracy": f"{m['accuracy']:.4f}",
                            "Precision": f"{m['precision']:.4f}",
                            "Recall": f"{m['recall']:.4f}",
                            "F1 (validation)": f"{m['f1']:.4f}",
                            "ROC-AUC": f"{m['roc_auc']:.4f}"
                            if not np.isnan(m["roc_auc"])
                            else "nan",
                            "PR-AUC": f"{m['pr_auc']:.4f}",
                        }
                    )
                res_df = pd.DataFrame(rows)
                st.markdown("#### Training summary")
                st.dataframe(res_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # Threshold & Cost
    # -------------------------------------------------------------------------
    with tab_threshold:
        st.markdown(
            '<div class="section-header-pro">Threshold tuning and business cost</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-desc-pro">Pick a model, move the decision threshold, and inspect how metrics and expected cost change.</div>',
            unsafe_allow_html=True,
        )

        results_path = MODELS_DIR / "results.joblib"
        meta_path = MODELS_DIR / "metadata.joblib"

        if not results_path.exists() or not meta_path.exists():
            st.info("Train models in the previous tab to unlock threshold tuning.")
        else:
            all_results = joblib.load(results_path)
            metadata = joblib.load(meta_path)
            y_val = metadata["y_val"]

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])

            model_name = st.selectbox(
                "Model to analyse",
                options=list(all_results.keys()),
                index=list(all_results.keys()).index(best_name),
            )

            metrics_base = all_results[model_name]["metrics"]
            y_proba = metrics_base["y_proba"]

            col_thr, col_cost = st.columns([1.2, 1])
            with col_thr:
                threshold = st.slider(
                    "Decision threshold for positive class",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.5,
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

    # -------------------------------------------------------------------------
    # Compare Models
    # -------------------------------------------------------------------------
    with tab_compare:
        st.markdown(
            '<div class="section-header-pro">Model comparison</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-desc-pro">Side-by-side comparison of metrics, ROC / PR curves, and confusion matrices.</div>',
            unsafe_allow_html=True,
        )

        results_path = MODELS_DIR / "results.joblib"
        meta_path = MODELS_DIR / "metadata.joblib"

        if not results_path.exists() or not meta_path.exists():
            st.info("Train models first to unlock comparison.")
        else:
            all_results = joblib.load(results_path)
            metadata = joblib.load(meta_path)
            y_val = metadata["y_val"]

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

            st.markdown("##### Confusion matrices (validation set)")
            cm_cols = st.columns(len(all_results))
            for (name, res), col in zip(all_results.items(), cm_cols, strict=True):
                m = res["metrics"]
                cm = confusion_matrix(y_val, m["y_pred"], labels=[0, 1])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[metadata["neg_label"], metadata["pos_label"]],
                    y=[metadata["neg_label"], metadata["pos_label"]],
                    text_auto=True,
                    title=name,
                )
                fig_cm.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                )
                with col:
                    st.plotly_chart(fig_cm, use_container_width=True)

    # -------------------------------------------------------------------------
    # Error Analysis
    # -------------------------------------------------------------------------
    with tab_errors:
        st.markdown('<div class="section-header-pro">Error analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc-pro">Browse misclassified texts to see where the model struggles and how confident it was.</div>',
            unsafe_allow_html=True,
        )

        results_path = MODELS_DIR / "results.joblib"
        meta_path = MODELS_DIR / "metadata.joblib"

        if not results_path.exists() or not meta_path.exists():
            st.info("Train models first to unlock error analysis.")
        else:
            all_results = joblib.load(results_path)
            metadata = joblib.load(meta_path)
            y_val = metadata["y_val"]
            val_idx = metadata["val_idx"]

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])
            model_name = st.selectbox(
                "Model to inspect",
                options=list(all_results.keys()),
                index=list(all_results.keys()).index(best_name),
            )

            m = all_results[model_name]["metrics"]
            y_pred = m["y_pred"]
            y_proba = m["y_proba"]

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
                df_view["conf"] = np.abs(df_view["proba_pos"] - 0.5)
                df_view = df_view.sort_values(
                    "conf", ascending=(sort_mode == "Least confident predictions")
                )
            else:
                df_view = df_view.sample(frac=1, random_state=42)

            top_n = st.slider("Rows to show", 10, 200, 50, 10)
            cols_show = ["text_raw", "true_label", "pred_label", "proba_pos", "error_type"]
            st.dataframe(df_view[cols_show].head(top_n), use_container_width=True)

    # -------------------------------------------------------------------------
    # Deploy
    # -------------------------------------------------------------------------
    with tab_deploy:
        st.markdown(
            '<div class="section-header-pro">Deployment & interactive prediction</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-desc-pro">Pick the best model, test arbitrary texts, and reuse the same logic in an API or batch job.</div>',
            unsafe_allow_html=True,
        )

        models_path = MODELS_DIR / "models.joblib"
        vecs_path = MODELS_DIR / "vectorizers.joblib"
        results_path = MODELS_DIR / "results.joblib"
        meta_path = MODELS_DIR / "metadata.joblib"

        if not (
            models_path.exists()
            and vecs_path.exists()
            and results_path.exists()
            and meta_path.exists()
        ):
            st.info("Train models first to enable deployment.")
        else:
            models = joblib.load(models_path)
            vecs = joblib.load(vecs_path)
            all_results = joblib.load(results_path)
            metadata = joblib.load(meta_path)

            best_name = max(all_results.keys(), key=lambda n: all_results[n]["metrics"]["f1"])

            model_choice = st.selectbox(
                "Model for deployment",
                options=["Best (by F1)"] + list(models.keys()),
                index=0,
            )

            if model_choice == "Best (by F1)":
                deploy_name = best_name
                st.info(f"Using {best_name} (best F1 on validation).")
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
                    Xw = word_vec.transform([clean_text])
                    if char_vec is not None:
                        Xc = char_vec.transform([clean_text])
                        X_test = hstack([Xw, Xc])
                    else:
                        X_test = Xw

                    try:
                        proba = float(_predict_proba_pos(model, X_test)[0])
                    except (AttributeError, ValueError, TypeError) as e:
                        st.error(f"Model scoring failed: {e}")
                        st.stop()

                    label_int = int(proba >= 0.5)
                    label_str = metadata["pos_label"] if label_int == 1 else metadata["neg_label"]
                    conf_pct = proba * 100.0 if label_int == 1 else (1.0 - proba) * 100.0

                    st.markdown(
                        """
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted sentiment</div>
                        """,
                        unsafe_allow_html=True,
                    )

                    cls = "prediction-positive" if label_int == 1 else "prediction-negative"
                    st.markdown(
                        f'<div class="prediction-result {cls}">{label_str}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="prediction-confidence">{conf_pct:.1f}% confidence</div>',
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


if __name__ == "__main__":
    main()
