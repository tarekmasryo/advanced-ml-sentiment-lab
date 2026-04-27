# Case Study — Advanced ML Sentiment Lab

## Problem

Many sentiment-classification examples stop at a single validation score. That is not enough for a reviewer, stakeholder, or engineering team deciding whether a model is reliable enough to use.

A useful review workflow should answer:

- Is the dataset clean enough to train on?
- Was preprocessing fitted without leakage?
- Which model is the strongest baseline?
- Which threshold should drive the final decision?
- What mistakes are most costly?
- Can the result be exported with enough evidence for review?

## Solution

Advanced ML Sentiment Lab packages the sentiment workflow into a Streamlit application with a single guided path:

```text
Upload or detect data -> review quality -> train models -> evaluate thresholds -> inspect mistakes -> test predictions -> export evidence
```

## Engineering decisions

### Leakage-safe training

TF-IDF vectorizers are fitted only on training text. Validation and test partitions are transformed with the fitted vectorizers.

### Duplicate-aware data policy

Exact duplicate cleaned texts can be removed before splitting. This reduces the chance that repeated reviews appear in both training and evaluation partitions.

### Decision-oriented evaluation

The app reports more than accuracy. It includes F1, ROC-AUC, PR-AUC, Brier score, threshold recommendations, expected FP/FN cost, confusion matrices, and error review.

### Threshold-aware prediction

The Prediction Lab uses the selected or recommended threshold. It does not silently fall back to a fixed 0.5 cutoff.

### Exportable evidence

Each run can produce an HTML report, model card, metrics CSV, threshold recommendations, metadata, and a review bundle.

## Current scope

The project is a practical ML review application focused on inspectable text-classification workflows, decision thresholds, and review-ready artifacts.

Intentional constraints:

- English-oriented preprocessing.
- Binary text classification.
- Classical ML baselines and challengers.
- Local artifacts only.
- Streamlit UI instead of a separate frontend/backend service.

## Future extensions

- Optional embedding or transformer challenger.
- Batch scoring flow.
- External validation dataset support.
- Label-quality audit.
- Drift and monitoring report.
- FastAPI inference service.
