# Case Study — Advanced ML Sentiment Lab

## Problem
Binary sentiment projects often stall at “a notebook that works” without a clear way to compare models, tune thresholds, or inspect mistakes. The goal was a single interactive lab that answers practical questions fast:

- Which classical model performs best on *my* dataset (ROC/PR AUC, F1, Precision/Recall)?
- What decision threshold should we use when FP/FN costs aren’t equal?
- What are the most common failure cases (false positives/negatives), and why?

## Approach
- Streamlit + Plotly dashboard for EDA → training → evaluation → error analysis → live prediction.
- Feature engineering with **TF‑IDF word n‑grams (1–3)** + optional **char n‑grams (3–6)**.
- Train/compare classical baselines: **LogReg / RandomForest / GradientBoosting / Multinomial Naive Bayes**.
- Save trained artifacts for reuse under `artifacts/sentiment_lab/`.

## Key Decisions
- **Flexible input:** upload any CSV, map text/label columns, and choose which label value is positive.
- **Fast iteration:** stratified train/val split on a capped subset to keep the UI responsive.
- **Cost-aware thresholding:** visualize metric tradeoffs and choose threshold using FP/FN business costs.
- **Error-first debugging:** browse FP/FN, sorted by confidence, to quickly spot patterns.

## Results
A decision-friendly sentiment lab that supports:
- EDA (class balance, text length/token stats)
- Model comparison + ROC/PR curves + confusion matrices
- Threshold tuning (F1 vs threshold, cost vs threshold)
- Error analysis (FP/FN explorer)
- Interactive prediction on arbitrary text + persisted artifacts

## Next Steps
- Add CV + calibration (Platt/Isotonic) for better probability quality.
- Add simple explainability (top TF‑IDF features / n‑grams per class).
- Add exportable “model report” (HTML/PDF) + run history under `runs/`.
