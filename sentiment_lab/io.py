from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV from disk with conservative fallbacks."""
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


def read_uploaded_csv(uploaded) -> pd.DataFrame | None:
    """Read a Streamlit UploadedFile with a UTF-8 fallback."""
    read_errors = (UnicodeDecodeError, ValueError, pd.errors.ParserError)

    try:
        return pd.read_csv(uploaded)
    except read_errors as e:
        logger.debug("read_csv failed for uploaded file (default): %s", e)

    try:
        uploaded.seek(0)
    except Exception:
        pass

    try:
        return pd.read_csv(uploaded, encoding_errors="ignore")
    except read_errors as e:
        logger.debug("read_csv failed for uploaded file (encoding_errors=ignore): %s", e)

    return None
