from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def metrics_summary(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        val = result.get("metrics", {})
        test = result.get("test_metrics", {})
        rows.append(
            {
                "model": name,
                "val_accuracy": float(val.get("accuracy", float("nan"))),
                "val_precision": float(val.get("precision", float("nan"))),
                "val_recall": float(val.get("recall", float("nan"))),
                "val_f1": float(val.get("f1", float("nan"))),
                "val_roc_auc": float(val.get("roc_auc", float("nan"))),
                "val_pr_auc": float(val.get("pr_auc", float("nan"))),
                "val_brier": float(val.get("brier", float("nan"))),
                "test_accuracy": float(test.get("accuracy", float("nan"))),
                "test_f1": float(test.get("f1", float("nan"))),
                "test_roc_auc": float(test.get("roc_auc", float("nan"))),
                "test_pr_auc": float(test.get("pr_auc", float("nan"))),
                "test_brier": float(test.get("brier", float("nan"))),
            }
        )
    return rows


def make_run_dir(models_dir: Path) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = models_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_training_artifacts(
    models_dir: Path,
    *,
    vectorizers: Any,
    trained_models: dict[str, Any],
    all_results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    """Save latest artifacts and immutable run-history artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizers, models_dir / "vectorizers.joblib")
    joblib.dump(trained_models, models_dir / "models.joblib")
    joblib.dump(all_results, models_dir / "results.joblib")
    joblib.dump(metadata, models_dir / "metadata.joblib")

    run_dir = make_run_dir(models_dir)
    joblib.dump(vectorizers, run_dir / "vectorizers.joblib")
    joblib.dump(trained_models, run_dir / "models.joblib")
    joblib.dump(all_results, run_dir / "results.joblib")
    joblib.dump(metadata, run_dir / "metadata.joblib")

    (run_dir / "metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_rows = metrics_summary(all_results)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)

    return run_dir


def load_run_history(models_dir: Path, *, limit: int = 25) -> pd.DataFrame:
    """Load a compact run-history table from immutable run directories."""
    runs_dir = models_dir / "runs"
    if not runs_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "metadata.json"
        metrics_path = run_dir / "metrics.csv"
        if not metadata_path.exists() or not metrics_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics_df = pd.read_csv(metrics_path)
        except (OSError, json.JSONDecodeError, pd.errors.ParserError):
            continue
        best_model = "not available"
        best_val_f1 = float("nan")
        if not metrics_df.empty and "val_f1" in metrics_df.columns:
            best_row = metrics_df.sort_values("val_f1", ascending=False).iloc[0]
            best_model = str(best_row.get("model", "not available"))
            best_val_f1 = float(best_row.get("val_f1", float("nan")))
        rows.append(
            {
                "run_id": run_dir.name,
                "best_model": best_model,
                "best_val_f1": best_val_f1,
                "rows": metadata.get("sample_rows", "not recorded"),
                "holdout_test_split": metadata.get("holdout_test_split", 0.0),
                "training_seconds": metadata.get("training_seconds", float("nan")),
                "report_path": str(run_dir / "model_report.html"),
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)
