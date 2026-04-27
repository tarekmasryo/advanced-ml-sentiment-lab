from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st

PREMIUM_UI_CSS = """
<style>
:root {
    --aml-bg: #070b16;
    --aml-bg-2: #0b1020;
    --aml-surface: rgba(15, 23, 42, 0.88);
    --aml-surface-2: rgba(17, 24, 39, 0.84);
    --aml-border: rgba(148, 163, 184, 0.18);
    --aml-border-strong: rgba(99, 102, 241, 0.28);
    --aml-text: #f1f5f9;
    --aml-muted: #94a3b8;
    --aml-soft: #cbd5e1;
    --aml-blue: #60a5fa;
    --aml-indigo: #818cf8;
    --aml-good: #22c55e;
    --aml-warn: #f59e0b;
    --aml-bad: #ef4444;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.12), transparent 26%),
        radial-gradient(circle at 92% 0%, rgba(99, 102, 241, 0.10), transparent 28%),
        linear-gradient(180deg, #070b16 0%, #0a1020 42%, #070b16 100%);
    color: var(--aml-text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

html, body, [class*="css"], button, input, textarea, select {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.main .block-container {
    max-width: 1520px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d19 0%, #070b16 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

[data-testid="stSidebar"] h3 {
    margin-top: 1.1rem;
    margin-bottom: 0.45rem;
    color: #e2e8f0;
    font-size: 0.92rem;
    font-weight: 750;
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #dbe4ef;
}

.hero-premium {
    margin-bottom: 22px;
    padding: 28px 30px;
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.82)),
        radial-gradient(circle at 88% 20%, rgba(99, 102, 241, 0.16), transparent 32%);
    border: 1px solid var(--aml-border);
    box-shadow: 0 18px 54px rgba(2, 6, 23, 0.45);
}

.hero-kicker {
    display: inline-flex;
    margin-bottom: 12px;
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(96, 165, 250, 0.10);
    border: 1px solid rgba(96, 165, 250, 0.18);
    color: #bfdbfe;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.hero-title-pro {
    max-width: 860px;
    color: #f8fafc;
    font-size: clamp(32px, 3.8vw, 50px);
    line-height: 1.02;
    font-weight: 850;
    letter-spacing: -0.055em;
}

.hero-subtitle-pro {
    max-width: 860px;
    margin-top: 12px;
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.7;
}

.hero-badges,
.status-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 15px;
}

.badge-pill,
.badge-soft,
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    line-height: 1.2;
    color: #dbeafe;
    background: rgba(96, 165, 250, 0.09);
    border: 1px solid rgba(96, 165, 250, 0.14);
}

.badge-soft,
.status-chip {
    color: #cbd5e1;
    background: rgba(148, 163, 184, 0.07);
    border-color: rgba(148, 163, 184, 0.14);
}

.status-chip.good {
    color: #bbf7d0;
    background: rgba(34, 197, 94, 0.10);
    border-color: rgba(34, 197, 94, 0.18);
}

.status-chip.warn {
    color: #fde68a;
    background: rgba(245, 158, 11, 0.10);
    border-color: rgba(245, 158, 11, 0.18);
}

.kpi-premium,
.decision-card,
.model-card,
.threshold-card,
.prediction-card,
.export-tile,
.client-panel,
.empty-state-card {
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(15, 23, 42, 0.72));
    border: 1px solid var(--aml-border);
    box-shadow: 0 12px 32px rgba(2, 6, 23, 0.28);
}

.kpi-premium {
    min-height: 132px;
    padding: 18px 18px;
}

.kpi-icon {
    width: 34px;
    height: 34px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    border-radius: 10px;
    color: #bfdbfe;
    background: rgba(96, 165, 250, 0.10);
    border: 1px solid rgba(96, 165, 250, 0.14);
    font-size: 16px;
}

.kpi-label-pro,
.decision-card-label,
.metric-label,
.prediction-label {
    color: var(--aml-muted);
    font-size: 10.5px;
    line-height: 1.3;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

.kpi-value-pro,
.decision-card-value,
.prediction-result {
    margin-top: 7px;
    color: #f8fafc;
    font-size: 27px;
    line-height: 1.05;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.kpi-trend,
.decision-card-note,
.prediction-confidence,
.export-tile-text,
.client-panel-text,
.empty-state-text,
.info-box-text,
.workflow-step-text {
    margin-top: 7px;
    color: #b6c2d1;
    font-size: 12px;
    line-height: 1.6;
}

.section-header-pro {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 25px 0 6px;
    color: #f8fafc;
    font-size: 22px;
    line-height: 1.2;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.section-header-pro::before {
    content: "";
    width: 4px;
    height: 22px;
    border-radius: 999px;
    background: #60a5fa;
}

.section-desc-pro {
    max-width: 900px;
    margin-bottom: 14px;
    padding-left: 14px;
    color: var(--aml-muted);
    font-size: 13px;
    line-height: 1.6;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    padding: 7px;
    margin-bottom: 12px;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.64);
    border: 1px solid rgba(148, 163, 184, 0.12);
    overflow-x: auto;
}

.stTabs [data-baseweb="tab"] {
    min-height: 38px;
    padding: 0 14px;
    border-radius: 11px;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 700;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: #f8fafc !important;
    background: rgba(96, 165, 250, 0.13);
    box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.20);
}

.model-card,
.threshold-card,
.prediction-card,
.decision-card,
.export-tile,
.client-panel,
.empty-state-card {
    padding: 18px;
}

.model-name,
.export-tile-title,
.client-panel-title,
.empty-state-title,
.info-box-title {
    color: #f8fafc;
    font-size: 16px;
    line-height: 1.35;
    font-weight: 800;
}

.model-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-top: 12px;
}

.metric-box,
.empty-state-step,
.workflow-step {
    padding: 11px 12px;
    border-radius: 14px;
    background: rgba(148, 163, 184, 0.06);
    border: 1px solid rgba(148, 163, 184, 0.12);
}

.metric-value {
    margin-top: 4px;
    color: #f8fafc;
    font-size: 15px;
    font-weight: 800;
}

.info-box,
.soft-note {
    margin: 10px 0 16px;
    padding: 14px 16px;
    border-radius: 15px;
    background: rgba(96, 165, 250, 0.07);
    border: 1px solid rgba(96, 165, 250, 0.13);
}

.prediction-positive { color: #86efac; }
.prediction-negative { color: #fca5a5; }

.progress-bar {
    width: 100%;
    height: 8px;
    margin-top: 10px;
    border-radius: 999px;
    background: rgba(148, 163, 184, 0.14);
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #60a5fa, #818cf8);
}

.empty-state-steps,
.workflow-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.workflow-step-num {
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 10px;
    border-radius: 999px;
    color: #dbeafe;
    background: rgba(96, 165, 250, 0.12);
    font-size: 12px;
    font-weight: 800;
}

.workflow-step-title {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 4px;
}

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    min-height: 41px;
    border-radius: 13px;
    border: 1px solid rgba(96, 165, 250, 0.20);
    background: rgba(96, 165, 250, 0.12);
    color: #eff6ff;
    font-weight: 750;
    box-shadow: none;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: rgba(96, 165, 250, 0.32);
    background: rgba(96, 165, 250, 0.17);
}

[data-testid="stFileUploader"] section,
[data-testid="stExpander"] {
    border-radius: 15px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    background: rgba(15, 23, 42, 0.62);
}

[data-testid="stDataFrame"] {
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.12);
}

[data-baseweb="select"] > div,
[data-baseweb="base-input"] > div,
[data-baseweb="textarea"] > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea textarea {
    border-radius: 13px !important;
    background: rgba(15, 23, 42, 0.72) !important;
    border-color: rgba(148, 163, 184, 0.16) !important;
}

@media (max-width: 1100px) {
    .model-metrics,
    .workflow-grid,
    .empty-state-steps {
        grid-template-columns: 1fr 1fr;
    }
}

@media (max-width: 800px) {
    .hero-premium { padding: 22px 20px; }
    .hero-title-pro { font-size: 32px; }
    .model-metrics,
    .workflow-grid,
    .empty-state-steps {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def escape_html(value: object) -> str:
    """Return a safe string for raw HTML fragments."""
    return escape(str(value), quote=True)


def render_hero(
    *,
    title: str,
    subtitle: str,
    badges: Iterable[str] = (),
    soft_badges: Iterable[str] = (),
    kicker: str = "Decision-ready Sentiment Analysis",
) -> None:
    badge_markup = "".join(
        f'<span class="badge-pill">{escape_html(badge)}</span>' for badge in badges
    )
    badge_markup += "".join(
        f'<span class="badge-soft">{escape_html(badge)}</span>' for badge in soft_badges
    )
    st.markdown(
        f"""
        <div class="hero-premium">
            <div class="hero-kicker">{escape_html(kicker)}</div>
            <div class="hero-title-pro">{escape_html(title)}</div>
            <div class="hero-subtitle-pro">{escape_html(subtitle)}</div>
            <div class="hero-badges">{badge_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    *,
    label: str,
    value: str,
    note: str = "",
    icon: str = "◆",
) -> None:
    st.markdown(
        f"""
        <div class="kpi-premium">
            <div class="kpi-icon">{escape_html(icon)}</div>
            <div class="kpi-label-pro">{escape_html(label)}</div>
            <div class="kpi-value-pro">{escape_html(value)}</div>
            <div class="kpi-trend">{escape_html(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_chips(items: Iterable[tuple[str, str, str]]) -> None:
    chips: list[str] = []
    for label, value, tone in items:
        safe_tone = tone if tone in {"good", "warn"} else ""
        chips.append(
            f'<span class="status-chip {safe_tone}">'
            f"{escape_html(label)}: <b>{escape_html(value)}</b></span>"
        )
    st.markdown(
        '<div class="status-chip-row">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "") -> None:
    description_markup = ""
    if description:
        description_markup = f'<div class="section-desc-pro">{escape_html(description)}</div>'
    st.markdown(
        f"""
        <div class="section-header-pro">{escape_html(title)}</div>
        {description_markup}
        """,
        unsafe_allow_html=True,
    )


def render_empty_dataset_state() -> None:
    render_hero(
        title="Advanced ML Sentiment Lab",
        subtitle=(
            "Upload a CSV from the sidebar or place IMDB Dataset.csv in the repo root or data "
            "folder. The lab will detect the text and label columns, then guide you through "
            "quality review, safe training, threshold tuning, prediction checks, and export."
        ),
        badges=("TF-IDF word + character features", "Calibrated linear baselines"),
        soft_badges=("Threshold tuning", "Prediction lab", "Export bundle"),
    )
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-title">Start with a clean dataset handoff</div>
            <div class="empty-state-text">
                The lab expects a CSV with one free-text review or comment column and one binary label
                column. Keep the first run simple: use the detected columns, train with the safe defaults,
                then review the recommendation before changing advanced settings.
            </div>
            <div class="empty-state-steps">
                <div class="empty-state-step"><b>1.</b> Upload a CSV or add the IMDB dataset locally.</div>
                <div class="empty-state-step"><b>2.</b> Confirm the text and label mapping in the sidebar.</div>
                <div class="empty-state-step"><b>3.</b> Train, inspect errors, then export the review bundle.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
