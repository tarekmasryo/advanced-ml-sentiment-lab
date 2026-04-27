from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SHORT_TEXT_TOKEN_THRESHOLD = 3


def compute_data_quality_report(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
    label_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute lightweight data quality checks for text classification datasets."""
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    text = df[text_col].fillna("").astype(str)
    token_counts = text.str.split().map(len)
    empty_mask = text.str.strip().eq("")
    duplicate_count = int(text.duplicated().sum())
    short_count = int((token_counts < SHORT_TEXT_TOKEN_THRESHOLD).sum())

    report: dict[str, Any] = {
        "rows": int(len(df)),
        "empty_texts": int(empty_mask.sum()),
        "duplicate_texts": duplicate_count,
        "short_texts": short_count,
        "avg_tokens": float(token_counts.mean()) if len(token_counts) else 0.0,
        "median_tokens": float(token_counts.median()) if len(token_counts) else 0.0,
        "quality_flags": [],
    }

    if label_values is not None:
        labels = np.asarray(label_values)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = {str(k): int(v) for k, v in zip(unique, counts, strict=True)}
        min_count = int(counts.min()) if len(counts) else 0
        max_count = int(counts.max()) if len(counts) else 0
        imbalance_ratio = float(max_count / max(1, min_count)) if min_count else float("inf")
        report.update(
            {
                "label_counts": label_counts,
                "minority_class_count": min_count,
                "class_imbalance_ratio": imbalance_ratio,
            }
        )
        if imbalance_ratio >= 3.0:
            report["quality_flags"].append("class_imbalance")

    if report["empty_texts"] > 0:
        report["quality_flags"].append("empty_texts")
    if report["duplicate_texts"] > 0:
        report["quality_flags"].append("duplicate_texts")
    if report["short_texts"] / max(1, report["rows"]) > 0.05:
        report["quality_flags"].append("many_short_texts")

    return report


def data_quality_table(report: dict[str, Any]) -> pd.DataFrame:
    """Return a compact table for Streamlit display and reports."""
    rows = [
        ("Rows", report.get("rows", 0)),
        ("Empty texts", report.get("empty_texts", 0)),
        ("Duplicate texts after policy", report.get("duplicate_texts", 0)),
        (
            "Duplicate texts before policy",
            report.get("duplicate_texts_before_policy", "not recorded"),
        ),
        (
            "Duplicate texts removed before split",
            report.get("duplicate_texts_removed_before_split", 0),
        ),
        ("Duplicate policy", report.get("duplicate_policy", "not recorded")),
        ("Short texts (<3 tokens)", report.get("short_texts", 0)),
        ("Average tokens", round(float(report.get("avg_tokens", 0.0)), 2)),
        ("Median tokens", round(float(report.get("median_tokens", 0.0)), 2)),
    ]
    if "class_imbalance_ratio" in report:
        rows.append(("Class imbalance ratio", round(float(report["class_imbalance_ratio"]), 2)))
    return pd.DataFrame(rows, columns=["check", "value"])


def data_quality_issues(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
    max_rows: int = 200,
) -> pd.DataFrame:
    """Return row-level lightweight data-quality issues for review/export."""
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    text = df[text_col].fillna("").astype(str)
    token_counts = text.str.split().map(len)
    duplicate_mask = text.duplicated(keep=False)
    empty_mask = text.str.strip().eq("")
    short_mask = token_counts < SHORT_TEXT_TOKEN_THRESHOLD

    issues: list[dict[str, Any]] = []
    for idx in df.index[empty_mask | duplicate_mask | short_mask]:
        tags: list[str] = []
        if bool(empty_mask.loc[idx]):
            tags.append("empty_text")
        if bool(duplicate_mask.loc[idx]):
            tags.append("duplicate_text")
        if bool(short_mask.loc[idx]):
            tags.append("short_text")
        issues.append(
            {
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "issue_tags": ",".join(tags),
                "token_count": int(token_counts.loc[idx]),
                "text_preview": text.loc[idx][:220],
            }
        )
    return pd.DataFrame(issues).head(max_rows)
