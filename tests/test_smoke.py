import importlib
import importlib.util
import os
import sys

import pytest


def _parse_version(v: str) -> tuple[int, int]:
    parts = v.strip().split(".")
    if len(parts) < 2:
        raise ValueError("MIN_PYTHON must be like '3.10' or '3.11'")
    return int(parts[0]), int(parts[1])


def test_smoke_python_version():
    min_py = os.getenv("MIN_PYTHON", "3.11")
    assert sys.version_info >= _parse_version(min_py)


def test_smoke_core_imports():
    """Fast fail if core runtime dependencies are broken."""
    imports = os.getenv("SMOKE_IMPORTS", "numpy,pandas,sklearn")

    for mod in [m.strip() for m in imports.split(",") if m.strip()]:
        importlib.import_module(mod)


def test_smoke_project_module_importable():
    """Ensure the project package imports without optional environment flags."""
    module = os.getenv("PROJECT_MODULE", "sentiment_lab").strip()
    importlib.import_module(module)


def test_smoke_app_importable():
    """
    Importing the app module should not execute heavy work at import-time.
    Override via:
      APP_MODULE="app"  (default)
    """
    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit is not installed in this environment")
    app_module = os.getenv("APP_MODULE", "app").strip()
    importlib.import_module(app_module)
