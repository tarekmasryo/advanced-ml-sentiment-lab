# scripts/doctor.py
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=str(ROOT))


def has_file(name: str) -> bool:
    return (ROOT / name).exists()


def which(exe: str) -> str | None:
    return shutil.which(exe)


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def info(k: str, v: str) -> None:
    print(f"{k:<24} {v}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def parse_min_python() -> tuple[int, int]:
    # Default policy for the kit; override via env if needed.
    v = os.getenv("MIN_PYTHON", "3.11").strip()
    parts = v.split(".")
    return int(parts[0]), int(parts[1])


def check_python_version() -> bool:
    min_major, min_minor = parse_min_python()
    if sys.version_info >= (min_major, min_minor):
        ok(f"Python version >= {min_major}.{min_minor} ({sys.version.split()[0]})")
        return True
    fail(f"Python too old: {sys.version.split()[0]} < {min_major}.{min_minor}")
    return False


def check_venv() -> None:
    in_venv = sys.prefix != sys.base_prefix
    info("In venv", str(in_venv))
    if not in_venv:
        warn("Not in a virtualenv. Recommended: create .venv and activate it.")


def suggest_install() -> None:
    banner("Suggested setup commands")
    py = sys.executable
    if has_file("requirements-dev.txt"):
        print(f"{py} -m pip install -U pip")
        print(f"{py} -m pip install -r requirements-dev.txt")
    elif has_file("requirements.txt"):
        print(f"{py} -m pip install -U pip")
        print(f"{py} -m pip install -r requirements.txt")
        warn("requirements-dev.txt not found (dev tooling may be missing).")
    else:
        warn("No requirements*.txt found.")


def check_tools() -> None:
    banner("Tooling")
    py = sys.executable
    # Try importing these only if installed; don't crash the script.
    for mod in ["ruff", "pytest"]:
        code = run([py, "-c", f"import {mod}"])
        if code == 0:
            ok(f"import {mod}")
        else:
            warn(f"{mod} not importable (install dev deps).")

    pc = which("pre-commit")
    if pc:
        ok("pre-commit found in PATH")
    else:
        warn("pre-commit not found in PATH (optional, but recommended).")


def run_checks() -> int:
    banner("Running checks")
    py = sys.executable
    steps = [
        ([py, "-m", "ruff", "check", "."], "ruff check"),
        ([py, "-m", "ruff", "format", "--check", "."], "ruff format --check"),
        ([py, "-m", "pytest", "-q"], "pytest"),
    ]
    rc = 0
    for cmd, name in steps:
        print(f"\n$ {' '.join(cmd)}")
        code = run(cmd)
        if code != 0:
            fail(f"{name} failed (exit={code})")
            rc = code
            break
        ok(f"{name} passed")
    return rc


def main() -> int:
    banner("Doctor")
    info("OS", f"{platform.system()} {platform.release()}")
    info("Python", sys.version.splitlines()[0])
    info("Repo", str(ROOT))

    py_ok = check_python_version()
    check_venv()

    banner("Project files")
    for f in ["pyproject.toml", "requirements.txt", "requirements-dev.txt", "Dockerfile"]:
        info(f, "present" if has_file(f) else "missing")

    check_tools()
    suggest_install()

    if not py_ok:
        return 2

    if "--check" in sys.argv:
        return run_checks()

    print("\nTip: run `python scripts/doctor.py --check`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
