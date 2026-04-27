from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from sentiment_lab.artifacts import _jsonable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    artifacts_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "sentiment_lab"


class TrustedArtifactStore:
    manifest_name = "artifact_manifest.json"
    tracked_files = ("models.joblib", "vectorizers.joblib", "results.joblib", "metadata.joblib")

    def write_manifest(self, directory: Path) -> None:
        manifest = {filename: _sha256(directory / filename) for filename in self.tracked_files}
        (directory / self.manifest_name).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def verify_manifest(self, directory: Path) -> bool:
        manifest_path = directory / self.manifest_name
        if not manifest_path.exists():
            logger.warning("Artifact manifest missing: %s", manifest_path)
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Artifact manifest is not valid JSON: %s", manifest_path)
            return False
        for filename in self.tracked_files:
            path = directory / filename
            if not _is_relative_to(path.resolve(), directory.resolve()):
                logger.warning("Artifact path escaped trusted directory: %s", path)
                return False
            if not path.exists() or manifest.get(filename) != _sha256(path):
                logger.warning("Artifact hash mismatch or missing file: %s", path)
                return False
        return True


class TrainingStateStore:
    def __init__(self, key: str = "sentiment_lab_training_state") -> None:
        self.key = key
        self.artifacts = TrustedArtifactStore()

    def save(
        self,
        *,
        vectorizers: tuple[Any, ...],
        trained_models: dict[str, Any],
        all_results: dict[str, dict[str, Any]],
        metadata: dict[str, Any],
        run_dir: Path,
    ) -> None:
        st.session_state[self.key] = {
            "vectorizers": vectorizers,
            "models": trained_models,
            "results": all_results,
            "metadata": metadata,
            "run_dir": run_dir,
        }

    def load(self, models_dir: Path) -> dict[str, Any] | None:
        active_state = st.session_state.get(self.key)
        if active_state is not None:
            return active_state

        required_paths = {
            "models": models_dir / "models.joblib",
            "vectorizers": models_dir / "vectorizers.joblib",
            "results": models_dir / "results.joblib",
            "metadata": models_dir / "metadata.joblib",
        }
        if not all(path.exists() for path in required_paths.values()):
            return None
        if not self.artifacts.verify_manifest(models_dir):
            logger.warning("Skipping artifact auto-load because trust verification failed.")
            return None

        try:
            return {
                "models": joblib.load(required_paths["models"]),
                "vectorizers": joblib.load(required_paths["vectorizers"]),
                "results": joblib.load(required_paths["results"]),
                "metadata": joblib.load(required_paths["metadata"]),
                "run_dir": None,
            }
        except (OSError, EOFError, ValueError) as exc:
            logger.warning("Failed to load training artifacts: %s", exc)
            return None

    def persist_metadata(self, models_dir: Path, run_dir: Path, metadata: dict[str, Any]) -> None:
        joblib.dump(metadata, models_dir / "metadata.joblib")
        joblib.dump(metadata, run_dir / "metadata.joblib")
        (run_dir / "metadata.json").write_text(
            json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.artifacts.write_manifest(models_dir)
        self.artifacts.write_manifest(run_dir)


class ReviewBundleBuilder:
    included_files = (
        "model_report.html",
        "model_card.md",
        "metrics.csv",
        "model_leaderboard.csv",
        "threshold_recommendations.csv",
        "data_quality_issues.csv",
        "metadata.json",
        "artifact_manifest.json",
        "metrics.json",
    )

    def build(self, run_dir: Path) -> bytes:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename in self.included_files:
                path = run_dir / filename
                if path.exists():
                    zf.write(path, arcname=filename)
        buffer.seek(0)
        return buffer.getvalue()


class ReloadSmokeTestWriter:
    script = """from pathlib import Path

import joblib

run_dir = Path(__file__).resolve().parent
models = joblib.load(run_dir / "models.joblib")
vectorizers = joblib.load(run_dir / "vectorizers.joblib")
metadata = joblib.load(run_dir / "metadata.joblib")

assert models, "No trained models found."
assert vectorizers, "No fitted vectorizers found."
assert metadata.get("pipeline_policy"), "Missing pipeline policy metadata."
print("Reload smoke passed.")
"""

    def write(self, run_dir: Path) -> Path:
        path = run_dir / "reload_smoke_test.py"
        path.write_text(self.script, encoding="utf-8")
        return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
