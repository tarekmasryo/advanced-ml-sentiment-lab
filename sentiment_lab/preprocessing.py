from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd


def basic_clean(s: object) -> str:
    """Normalize English review text for a fast classical baseline."""
    if s is None or pd.isna(s):
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s).lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http\S+|www\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    pos_label_str: str,
    neg_label_str: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Create canonical text columns and binary labels."""
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


def deduplicate_clean_texts(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    text_col: str = "text_clean",
    keep: str = "first",
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Remove exact duplicate cleaned texts before splitting.

    The duplicate policy removes exact duplicate reviews before train/validation/test
    splitting so duplicate text cannot appear across evaluation partitions. This helper
    preserves the original row index for traceability while returning labels aligned to
    the deduplicated dataframe order.
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    duplicate_mask = df[text_col].fillna("").astype(str).duplicated(keep=keep)
    removed = int(duplicate_mask.sum())
    if removed == 0:
        return df.copy(), np.asarray(y).copy(), 0

    keep_mask = ~duplicate_mask.to_numpy()
    out = df.loc[keep_mask].copy()
    y_out = np.asarray(y)[keep_mask]
    return out, y_out, removed
