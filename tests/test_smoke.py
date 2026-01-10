import importlib
import os
import sys


def _parse_version(v: str) -> tuple[int, int]:
    parts = v.strip().split(".")
    if len(parts) < 2:
        raise ValueError("MIN_PYTHON must be like '3.10' or '3.11'")
    return int(parts[0]), int(parts[1])


def test_smoke_python_version():
    min_py = os.getenv("MIN_PYTHON", "3.11")
    assert sys.version_info >= _parse_version(min_py)


def test_smoke_core_imports():
    """
    Fast fail if core dependencies are broken.

    Configure via:
      SMOKE_IMPORTS="numpy,pandas,sklearn,streamlit,plotly"
    """
    imports = os.getenv("SMOKE_IMPORTS", "")
    if not imports.strip():
        return

    for mod in [m.strip() for m in imports.split(",") if m.strip()]:
        importlib.import_module(mod)


def test_smoke_project_module_importable():
    """
    Ensure the main project module imports.
    Configure via:
      PROJECT_MODULE="your_package_name"
    """
    module = os.getenv("PROJECT_MODULE", "").strip()
    if not module:
        return
    importlib.import_module(module)


def test_smoke_app_importable():
    """
    Importing the app module should not execute heavy work at import-time.
    Override via:
      APP_MODULE="app"  (default)
    """
    app_module = os.getenv("APP_MODULE", "app").strip()
    importlib.import_module(app_module)
