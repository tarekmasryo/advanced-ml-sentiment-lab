from __future__ import annotations

from typing import Any

import pandas as pd

TEXT_NAME_HINTS = {
    "review",
    "text",
    "comment",
    "content",
    "message",
    "body",
    "feedback",
    "description",
    "response",
}
LABEL_NAME_HINTS = {
    "sentiment",
    "label",
    "target",
    "class",
    "category",
    "rating",
    "score",
    "y",
}


def profile_column(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Return lightweight statistics used to guide column mapping UX."""
    values = df[column].dropna().astype(str)
    row_count = int(len(values))
    unique_count = int(values.nunique())
    avg_len = float(values.str.len().mean()) if row_count else 0.0
    max_len = int(values.str.len().max()) if row_count else 0
    unique_ratio = unique_count / max(row_count, 1)
    name = str(column).strip().lower()
    return {
        "column": column,
        "name": name,
        "row_count": row_count,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "avg_len": avg_len,
        "max_len": max_len,
    }


def detect_column_roles(df: pd.DataFrame) -> dict[str, Any]:
    """Detect the most likely text and label columns for client-friendly defaults."""
    columns = list(df.columns)
    if len(columns) < 2:
        return {"text_col": columns[0] if columns else None, "label_col": None, "profiles": []}

    profiles = [profile_column(df, c) for c in columns]
    text_profile = max(profiles, key=_text_score)
    label_profiles = [p for p in profiles if p["column"] != text_profile["column"]]
    label_profile = max(label_profiles, key=_label_score)

    return {
        "text_col": text_profile["column"],
        "label_col": label_profile["column"],
        "profiles": profiles,
        "text_profile": text_profile,
        "label_profile": label_profile,
    }


def validate_column_mapping(
    df: pd.DataFrame,
    *,
    text_col: str,
    label_col: str,
) -> tuple[list[str], list[str]]:
    """Return user-facing mapping errors and warnings.

    The checks intentionally catch obvious UX mistakes, such as selecting a two-value
    label column (for example `sentiment`) as the free-text column.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if text_col == label_col:
        errors.append("Text column and label column must be different.")
        return errors, warnings

    text_profile = profile_column(df, text_col)
    label_profile = profile_column(df, label_col)

    if _looks_like_label(text_profile):
        errors.append(
            f"'{text_col}' looks like a label column, not a free-text column. "
            "Choose a longer text column such as 'review', 'text', or 'comment'."
        )

    if _looks_like_text(label_profile):
        errors.append(
            f"'{label_col}' looks like a free-text column, not a label column. "
            "Choose a short categorical column such as 'sentiment', 'label', or 'target'."
        )

    if label_profile["unique_count"] < 2:
        errors.append(f"'{label_col}' must contain at least two distinct labels.")

    if text_profile["avg_len"] < 15 and text_profile["unique_count"] < 50:
        warnings.append(
            f"'{text_col}' has very short values. Confirm this is the actual review/comment text."
        )

    if label_profile["unique_count"] > 20:
        warnings.append(
            f"'{label_col}' has {label_profile['unique_count']:,} unique values. "
            "This app is optimized for binary labels; map only the two target classes."
        )

    return errors, warnings


def _text_score(profile: dict[str, Any]) -> float:
    score = 0.0
    name = profile["name"]
    if name in TEXT_NAME_HINTS:
        score += 6.0
    if name in LABEL_NAME_HINTS:
        score -= 5.0
    score += min(profile["avg_len"] / 30.0, 5.0)
    score += min(profile["unique_ratio"] * 3.0, 3.0)
    if profile["unique_count"] <= 20 and profile["avg_len"] <= 30:
        score -= 4.0
    return score


def _label_score(profile: dict[str, Any]) -> float:
    score = 0.0
    name = profile["name"]
    if name in LABEL_NAME_HINTS:
        score += 6.0
    if name in TEXT_NAME_HINTS:
        score -= 5.0
    if 2 <= profile["unique_count"] <= 20:
        score += 4.0
    if profile["avg_len"] <= 40:
        score += 1.5
    if profile["avg_len"] > 80:
        score -= 4.0
    return score


def _looks_like_label(profile: dict[str, Any]) -> bool:
    name = profile["name"]
    return (name in LABEL_NAME_HINTS and profile["avg_len"] <= 40) or (
        profile["unique_count"] <= 20 and profile["avg_len"] <= 25
    )


def _looks_like_text(profile: dict[str, Any]) -> bool:
    name = profile["name"]
    return (name in TEXT_NAME_HINTS and profile["avg_len"] > 40) or (
        profile["unique_count"] > 50 and profile["avg_len"] > 40
    )
