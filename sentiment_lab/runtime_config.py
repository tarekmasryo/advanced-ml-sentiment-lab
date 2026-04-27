from __future__ import annotations

import os

_ALLOWED_N_JOBS = {-1, 1, 2, 4, 8}


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def default_n_jobs() -> int:
    value = parse_int_env("SENTIMENT_LAB_N_JOBS", 1)
    return value if value in _ALLOWED_N_JOBS else 1
