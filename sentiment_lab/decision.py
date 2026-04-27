from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sentiment_lab.thresholding import recommend_thresholds


def build_model_leaderboard(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a sortable leaderboard across validation and optional test metrics."""
    rows: list[dict[str, Any]] = []
    for name, payload in results.items():
        val = payload.get("metrics", {})
        test = payload.get("test_metrics", {})
        rows.append(
            {
                "model": name,
                "val_f1": _safe_float(val.get("f1")),
                "val_precision": _safe_float(val.get("precision")),
                "val_recall": _safe_float(val.get("recall")),
                "val_roc_auc": _safe_float(val.get("roc_auc")),
                "val_pr_auc": _safe_float(val.get("pr_auc")),
                "val_brier": _safe_float(val.get("brier")),
                "test_f1": _safe_float(test.get("f1")),
                "test_roc_auc": _safe_float(test.get("roc_auc")),
                "test_pr_auc": _safe_float(test.get("pr_auc")),
                "test_brier": _safe_float(test.get("brier")),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["val_f1", "val_pr_auc"], ascending=[False, False])


def choose_recommended_model(
    results: dict[str, dict[str, Any]],
    *,
    prefer_calibration: bool = True,
) -> str | None:
    """Choose a practical default model for decision workflows."""
    leaderboard = build_model_leaderboard(results)
    if leaderboard.empty:
        return None

    if prefer_calibration:
        calibrated_like = leaderboard[
            leaderboard["model"].str.contains("Logistic|SVM|SGD", case=False, regex=True)
        ]
        if not calibrated_like.empty:
            return str(calibrated_like.iloc[0]["model"])

    return str(leaderboard.iloc[0]["model"])


def build_decision_summary(
    *,
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    precision_target: float = 0.90,
    recall_target: float = 0.90,
) -> dict[str, Any]:
    """Create a business-facing summary from metrics, threshold options, and quality flags."""
    recommended_model = choose_recommended_model(results) or ""
    if not recommended_model:
        return {}

    metrics = results[recommended_model]["metrics"]
    recommendations = recommend_thresholds(
        np.asarray(metadata["y_val"], dtype=int),
        np.asarray(metrics["y_proba"], dtype=float),
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        precision_target=precision_target,
        recall_target=recall_target,
    )
    cost_choice = next(
        (row for row in recommendations if row["strategy"] == "lowest_cost"),
        recommendations[0] if recommendations else {},
    )

    test_metrics = results[recommended_model].get("test_metrics", {})
    has_test = bool(test_metrics)
    quality_flags = list(metadata.get("data_quality_report", {}).get("quality_flags", []))
    stability = metadata.get("stability_summary", []) or []
    test_rows = int(metadata.get("test_rows", 0) or 0)

    risk = _score_risk(
        val_metrics=metrics,
        test_metrics=test_metrics,
        has_test=has_test,
        test_rows=test_rows,
        quality_flags=quality_flags,
        stability=stability,
    )

    return {
        "recommended_model": recommended_model,
        "recommended_threshold": float(cost_choice.get("threshold", 0.5)),
        "threshold_strategy": cost_choice.get("strategy", "lowest_cost"),
        "validation_f1": _safe_float(metrics.get("f1")),
        "validation_pr_auc": _safe_float(metrics.get("pr_auc")),
        "validation_brier": _safe_float(metrics.get("brier")),
        "test_f1": _safe_float(test_metrics.get("f1")),
        "test_pr_auc": _safe_float(test_metrics.get("pr_auc")),
        "test_brier": _safe_float(test_metrics.get("brier")),
        "val_test_f1_gap": risk["val_test_f1_gap"],
        "test_rows": test_rows,
        "estimated_fp": int(cost_choice.get("fp", 0)),
        "estimated_fn": int(cost_choice.get("fn", 0)),
        "estimated_cost": _safe_float(cost_choice.get("cost")),
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_reasons": risk["risk_reasons"],
        "quality_flags": quality_flags,
        "next_action": _next_action(risk["risk_reasons"]),
    }


def decision_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Return a compact table for Streamlit display."""
    if not summary:
        return pd.DataFrame()
    rows = [
        ("Recommended model", summary.get("recommended_model", "")),
        ("Recommended threshold", round(float(summary.get("recommended_threshold", 0.5)), 3)),
        ("Threshold strategy", summary.get("threshold_strategy", "")),
        ("Validation F1", _round_or_na(summary.get("validation_f1"))),
        ("Validation PR-AUC", _round_or_na(summary.get("validation_pr_auc"))),
        ("Validation Brier", _round_or_na(summary.get("validation_brier"))),
        ("Held-out test F1", _round_or_na(summary.get("test_f1"))),
        ("Val/Test F1 gap", _round_or_na(summary.get("val_test_f1_gap"))),
        ("Held-out test rows", summary.get("test_rows", 0)),
        (
            "Estimated FP / FN",
            f"{summary.get('estimated_fp', 0)} / {summary.get('estimated_fn', 0)}",
        ),
        ("Estimated cost", _round_or_na(summary.get("estimated_cost"))),
        ("Risk level", summary.get("risk_level", "")),
        ("Risk reasons", "; ".join(summary.get("risk_reasons", [])) or "none"),
        ("Next action", summary.get("next_action", "")),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _score_risk(
    *,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    has_test: bool,
    test_rows: int,
    quality_flags: list[str],
    stability: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score model decision risk using quality, calibration, generalization, and stability signals."""
    score = 0
    reasons: list[str] = []

    val_f1 = _safe_float(val_metrics.get("f1"))
    val_pr_auc = _safe_float(val_metrics.get("pr_auc"))
    val_brier = _safe_float(val_metrics.get("brier"))
    test_f1 = _safe_float(test_metrics.get("f1"))
    test_pr_auc = _safe_float(test_metrics.get("pr_auc"))
    test_brier = _safe_float(test_metrics.get("brier"))
    f1_gap = float("nan")

    if not has_test:
        score += 2
        reasons.append("held-out test disabled")
    else:
        if test_rows < 100:
            score += 1
            reasons.append("small held-out test set")
        if not np.isnan(val_f1) and not np.isnan(test_f1):
            f1_gap = abs(val_f1 - test_f1)
            if f1_gap > 0.08:
                score += 2
                reasons.append("validation/test F1 gap > 0.08")
        if not np.isnan(test_f1) and test_f1 < 0.70:
            score += 2
            reasons.append("held-out test F1 < 0.70")
        if not np.isnan(test_pr_auc) and test_pr_auc < 0.70:
            score += 1
            reasons.append("held-out test PR-AUC < 0.70")
        if not np.isnan(test_brier) and test_brier > 0.25:
            score += 1
            reasons.append("held-out test Brier score > 0.25")

    if not np.isnan(val_f1) and val_f1 < 0.70:
        score += 2
        reasons.append("validation F1 < 0.70")
    if not np.isnan(val_pr_auc) and val_pr_auc < 0.70:
        score += 1
        reasons.append("validation PR-AUC < 0.70")
    if not np.isnan(val_brier) and val_brier > 0.25:
        score += 1
        reasons.append("validation Brier score > 0.25")

    severe_flags = {"empty_texts", "class_imbalance"}
    if quality_flags:
        score += 2 if severe_flags.intersection(quality_flags) else 1
        reasons.append("data-quality flags present")

    max_f1_std = _max_stability_std(stability, metric="f1")
    if not np.isnan(max_f1_std) and max_f1_std > 0.05:
        score += 1
        reasons.append("repeated split F1 std > 0.05")

    if score >= 5:
        risk_level = "High"
    elif score >= 3:
        risk_level = "Caution"
    elif score >= 1:
        risk_level = "Moderate"
    else:
        risk_level = "Lower"

    return {
        "risk_level": risk_level,
        "risk_score": int(score),
        "risk_reasons": reasons,
        "val_test_f1_gap": f1_gap,
    }


def _next_action(risk_reasons: list[str]) -> str:
    if "data-quality flags present" in risk_reasons:
        return "Review data-quality flags and rerun training before relying on the model."
    if "held-out test disabled" in risk_reasons:
        return "Enable the held-out test split before making final performance claims."
    if any("F1 < 0.70" in reason or "PR-AUC < 0.70" in reason for reason in risk_reasons):
        return "Improve data/model quality before using this model for decisions."
    if any("F1 gap" in reason for reason in risk_reasons):
        return "Inspect validation/test gap and rerun with a larger or more representative split."
    if any("Brier" in reason for reason in risk_reasons):
        return "Review probability calibration before using cost-based thresholds."
    if any("repeated split" in reason for reason in risk_reasons):
        return "Collect more data or rerun stability checks before deployment."
    return "Review errors and validate on recent real-world data before deployment."


def _max_stability_std(stability: list[dict[str, Any]], *, metric: str) -> float:
    values: list[float] = []
    for row in stability:
        for key, value in row.items():
            key_text = str(key)
            if metric in key_text and "std" in key_text:
                values.append(_safe_float(value))
    values = [v for v in values if not np.isnan(v)]
    return max(values) if values else float("nan")


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _round_or_na(value: Any) -> str:
    x = _safe_float(value)
    if np.isnan(x):
        return "not available"
    return f"{x:.4f}"
