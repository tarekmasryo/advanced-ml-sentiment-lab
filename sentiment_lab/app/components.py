from __future__ import annotations

from typing import Any

import streamlit as st

from sentiment_lab.ui_theme import escape_html, render_metric_card, render_status_chips


def show_validation_scope_notice() -> None:
    st.info(
        "Validation metrics guide model selection and threshold tuning. "
        "The held-out test split gives a cleaner final estimate, but production use still "
        "requires recent real-world validation.",
        icon="ℹ️",
    )


def render_client_metric_card(label: str, value: str, note: str = "") -> None:
    render_metric_card(label=label, value=value, note=note, icon="▣")


def render_status_chip_row(items: list[tuple[str, str, str]]) -> None:
    render_status_chips(items)


def render_workflow_steps() -> None:
    steps = [
        ("1", "Load data", "Upload a CSV or place IMDB Dataset.csv in the repo."),
        ("2", "Check quality", "Review duplicate, empty, and short-text warnings."),
        ("3", "Train safely", "Use duplicate-safe, leakage-free defaults first."),
        ("4", "Read decision", "Use the recommended model, threshold, and risk level."),
        ("5", "Inspect mistakes", "Look at confident false positives and false negatives."),
        ("6", "Export bundle", "Download the report, model card, and review package."),
    ]

    for row_start in range(0, len(steps), 3):
        cols = st.columns(3)
        for col, (num, title, text) in zip(cols, steps[row_start : row_start + 3], strict=False):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {num}")
                    st.markdown(f"**{title}**")
                    st.caption(text)


def render_decision_cards(summary: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Recommended model", summary.get("recommended_model", "—"), "Best current candidate"),
        ("Threshold", f"{summary.get('recommended_threshold', 0.0):.3f}", "Decision cutoff"),
        ("Estimated cost", f"{summary.get('estimated_cost', 0.0):.2f}", "Based on FP/FN settings"),
        ("Risk level", summary.get("risk_level", "—"), "Use with the notes below"),
    ]
    for col, (label, value, note) in zip((c1, c2, c3, c4), cards, strict=True):
        with col:
            st.markdown(
                f"""
                <div class="decision-card">
                    <div class="decision-card-label">{escape_html(label)}</div>
                    <div class="decision-card-value">{escape_html(value)}</div>
                    <div class="decision-card-note">{escape_html(note)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def quality_status(data_quality_report: dict[str, Any]) -> tuple[str, str]:
    flags = data_quality_report.get("quality_flags", [])
    if not flags:
        return "Ready", "No major quality flags detected"
    if len(flags) <= 2:
        return "Review", ", ".join(flags)
    return "Needs review", f"{len(flags)} quality flags detected"


def plain_next_step(state: dict[str, Any] | None) -> str:
    if state is None:
        return "Open Training, use the safe defaults, then review Evaluation."
    return "Review Evaluation, inspect confident mistakes, then export the stakeholder bundle."
