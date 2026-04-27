from __future__ import annotations

import streamlit as st

WORKSPACE_PAGES = [
    "Overview",
    "Data Setup",
    "Training",
    "Evaluation",
    "Prediction Lab",
    "Export",
    "Run History",
]


def render_workspace_navigation() -> str:
    st.sidebar.markdown("### Workspace")
    return st.sidebar.radio(
        "Workspace page",
        WORKSPACE_PAGES,
        index=0,
        label_visibility="collapsed",
        help="Move through the review workflow one step at a time.",
    )
