from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_lab.artifacts import metrics_summary
from sentiment_lab.decision import build_model_leaderboard


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "nan"
        return f"{value:.4f}"
    return html.escape(str(value))


def _table_from_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p>No records.</p>"
    return _table_from_df(pd.DataFrame(records))


def _table_from_df(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "<p>No rows.</p>"
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _summary_cards(decision_summary: dict[str, Any]) -> str:
    if not decision_summary:
        return "<p>No decision summary available.</p>"
    cards = [
        ("Recommended model", decision_summary.get("recommended_model", "not available")),
        (
            "Recommended threshold",
            f"{float(decision_summary.get('recommended_threshold', 0.5)):.3f}",
        ),
        ("Estimated cost", _fmt(decision_summary.get("estimated_cost", "not available"))),
        ("Risk level", decision_summary.get("risk_level", "not available")),
    ]
    html_cards = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(str(value))}</div></div>"
        for label, value in cards
    )
    return f"<div class='cards'>{html_cards}</div>"


def _bar_svg(
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    title: str,
    width: int = 760,
    row_height: int = 34,
) -> str:
    if df.empty or value_col not in df.columns or label_col not in df.columns:
        return "<p>No chart data available.</p>"
    chart_df = df[[label_col, value_col]].dropna().head(8).copy()
    if chart_df.empty:
        return "<p>No chart data available.</p>"
    max_value = float(chart_df[value_col].max()) or 1.0
    left = 180
    bar_max = width - left - 100
    height = 44 + row_height * len(chart_df)
    rows = []
    for i, (_, row) in enumerate(chart_df.iterrows()):
        y = 34 + i * row_height
        label = html.escape(str(row[label_col])[:38])
        value = float(row[value_col])
        bar_width = max(2, int((value / max_value) * bar_max)) if max_value > 0 else 2
        rows.append(
            f"<text x='10' y='{y + 16}' font-size='12' fill='#374151'>{label}</text>"
            f"<rect x='{left}' y='{y}' width='{bar_width}' height='20' rx='5' fill='#6366f1'></rect>"
            f"<text x='{left + bar_width + 8}' y='{y + 15}' font-size='12' fill='#111827'>{value:.3f}</text>"
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        f"<svg width='{width}' height='{height}' role='img' aria-label='{html.escape(title)}'>"
        f"<text x='10' y='18' font-size='13' font-weight='700' fill='#111827'>{html.escape(title)}</text>"
        + "".join(rows)
        + "</svg>"
    )


def _threshold_svg(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "<p>No threshold recommendation chart available.</p>"
    df = pd.DataFrame(recommendations)
    if "strategy" not in df.columns or "cost" not in df.columns:
        return "<p>No threshold recommendation chart available.</p>"
    return _bar_svg(
        df.sort_values("cost", ascending=True),
        label_col="strategy",
        value_col="cost",
        title="Threshold recommendation cost comparison",
    )


def _confusion_matrix_html(
    *,
    metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    decision_summary: dict[str, Any],
) -> str:
    model_name = str(decision_summary.get("recommended_model", ""))
    if not model_name or model_name not in results:
        return "<p>No confusion matrix available.</p>"
    y_true = np.asarray(metadata.get("y_val", []), dtype=int)
    y_pred = np.asarray(results[model_name].get("metrics", {}).get("y_pred", []), dtype=int)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return "<p>No confusion matrix available.</p>"
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return f"""
<h3>Validation confusion matrix — {html.escape(model_name)}</h3>
<table class="matrix">
<thead><tr><th></th><th>Predicted negative</th><th>Predicted positive</th></tr></thead>
<tbody>
<tr><th>Actual negative</th><td>{tn}</td><td>{fp}</td></tr>
<tr><th>Actual positive</th><td>{fn}</td><td>{tp}</td></tr>
</tbody>
</table>
"""


def render_model_report_html(
    *,
    title: str,
    metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    data_quality: dict[str, Any] | None = None,
    threshold_recommendations: list[dict[str, Any]] | None = None,
    decision_summary: dict[str, Any] | None = None,
    data_quality_issues: pd.DataFrame | None = None,
    run_history: pd.DataFrame | None = None,
) -> str:
    """Render a standalone HTML model report for sharing and review."""
    metrics = metrics_summary(results)
    leaderboard = build_model_leaderboard(results)
    dq = data_quality or {}
    threshold_recommendations = threshold_recommendations or []
    decision_summary = decision_summary or {}
    flags = dq.get("quality_flags", [])
    metadata_safe = json.dumps(metadata, indent=2, default=str)

    issue_table = (
        _table_from_df(data_quality_issues)
        if data_quality_issues is not None
        else "<p>No issue export attached.</p>"
    )
    run_history_table = (
        _table_from_df(run_history)
        if run_history is not None
        else "<p>No previous run history attached.</p>"
    )
    leaderboard_chart = _bar_svg(
        leaderboard.sort_values("val_f1", ascending=False),
        label_col="model",
        value_col="val_f1",
        title="Validation F1 leaderboard",
    )
    threshold_chart = _threshold_svg(threshold_recommendations)
    confusion_matrix = _confusion_matrix_html(
        metadata=metadata,
        results=results,
        decision_summary=decision_summary,
    )
    risk_reasons = decision_summary.get("risk_reasons", [])
    risk_list = (
        "".join(f"<li>{html.escape(str(reason))}</li>" for reason in risk_reasons)
        or "<li>none</li>"
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
h1 {{ margin-bottom: 4px; }}
h2 {{ margin-top: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }}
h3 {{ margin-top: 18px; }}
.badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-weight: 700; font-size: 12px; margin-right: 6px; }}
.warning {{ border-left: 4px solid #f59e0b; background: #fffbeb; padding: 12px 14px; border-radius: 8px; margin: 16px 0; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0; }}
.card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; background: #f9fafb; }}
.label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .06em; }}
.value {{ font-size: 18px; font-weight: 800; margin-top: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; }}
th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #f9fafb; }}
.matrix th, .matrix td {{ text-align: center; font-size: 15px; }}
pre {{ background: #f9fafb; padding: 14px; border-radius: 8px; overflow-x: auto; }}
svg {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 10px; background: #ffffff; margin: 8px 0 18px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<span class=\"badge\">Leakage-free TF-IDF training</span>
<span class=\"badge\">Default held-out test split</span>
<span class=\"badge\">Cost-aware thresholding</span>
<span class=\"badge\">Decision summary</span>

<div class=\"warning\">
<strong>Evaluation scope:</strong> validation scores support model selection. Held-out test scores provide a cleaner final estimate when enabled. Any production deployment should still be verified on recent real data.
</div>

<h2>Decision Summary</h2>
{_summary_cards(decision_summary)}
<p><strong>Next action:</strong> {html.escape(str(decision_summary.get("next_action", "Review validation scope and error examples.")))}</p>
<p><strong>Risk reasons:</strong></p>
<ul>{risk_list}</ul>

<h2>Charts</h2>
{leaderboard_chart}
{threshold_chart}
{confusion_matrix}

<h2>Model Leaderboard</h2>
{_table_from_df(leaderboard)}

<h2>Model Metrics</h2>
{_table_from_records(metrics)}

<h2>Threshold Recommendations</h2>
{_table_from_records(threshold_recommendations)}

<h2>Data Quality</h2>
<p><strong>Flags:</strong> {html.escape(", ".join(flags) if flags else "none")}</p>
<pre>{html.escape(json.dumps(dq, indent=2, default=str))}</pre>

<h2>Data Quality Issues Sample</h2>
{issue_table}

<h2>Recent Run History</h2>
{run_history_table}

<h2>Run Metadata</h2>
<pre>{html.escape(metadata_safe)}</pre>
</body>
</html>
"""


def _markdown_table_from_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = [str(c) for c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(row[c]) for c in df.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_model_card_markdown(
    *,
    title: str,
    metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    decision_summary: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    threshold_recommendations: list[dict[str, Any]] | None = None,
) -> str:
    """Render a compact Markdown model card for lightweight review."""
    decision_summary = decision_summary or {}
    data_quality = data_quality or {}
    threshold_recommendations = threshold_recommendations or []
    leaderboard = build_model_leaderboard(results)
    flags = data_quality.get("quality_flags", [])
    risk_reasons = decision_summary.get("risk_reasons", [])

    lines = [
        f"# {title}",
        "",
        "## Decision summary",
        f"- Recommended model: {decision_summary.get('recommended_model', 'not available')}",
        f"- Recommended threshold: {decision_summary.get('recommended_threshold', 'not available')}",
        f"- Risk level: {decision_summary.get('risk_level', 'not available')}",
        f"- Risk reasons: {', '.join(risk_reasons) if risk_reasons else 'none'}",
        f"- Next action: {decision_summary.get('next_action', 'Review validation scope and error examples.')}",
        "",
        "## Evaluation scope",
        "Validation metrics support model selection. Held-out test metrics, when enabled, provide a cleaner final estimate. Production use still requires recent real-world validation.",
        "",
        "## Data quality",
        f"- Flags: {', '.join(flags) if flags else 'none'}",
        f"- Duplicate policy: {data_quality.get('duplicate_policy', 'not recorded')}",
        f"- Duplicate texts before policy: {data_quality.get('duplicate_texts_before_policy', 'not recorded')}",
        f"- Duplicate texts removed before split: {data_quality.get('duplicate_texts_removed_before_split', 0)}",
        "",
        "## Leaderboard",
    ]

    if leaderboard.empty:
        lines.append("No leaderboard available.")
    else:
        cols = [
            c
            for c in ["model", "val_f1", "val_pr_auc", "val_brier", "test_f1", "test_brier"]
            if c in leaderboard.columns
        ]
        lines.append(_markdown_table_from_df(leaderboard[cols].round(4)))

    lines.extend(["", "## Threshold recommendations"])
    if threshold_recommendations:
        lines.append(_markdown_table_from_df(pd.DataFrame(threshold_recommendations).round(4)))
    else:
        lines.append("No threshold recommendations available.")

    lines.extend(
        [
            "",
            "## Reproducibility notes",
            f"- Training rows: {metadata.get('sample_rows', 'not recorded')}",
            f"- Hold-out test split: {metadata.get('holdout_test_split', 'not recorded')}",
            f"- Random state: {metadata.get('random_state', 'not recorded')}",
            f"- Pipeline policy: {metadata.get('pipeline_policy', 'not recorded')}",
            f"- Training seconds: {metadata.get('training_seconds', 'not recorded')}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_model_card_markdown(
    run_dir: Path, markdown_text: str, filename: str = "model_card.md"
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(markdown_text, encoding="utf-8")
    return path


def save_model_report_html(
    run_dir: Path, html_text: str, filename: str = "model_report.html"
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(html_text, encoding="utf-8")
    return path
