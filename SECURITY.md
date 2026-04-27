# Security Policy

## Reporting a vulnerability

Do not open a public issue for a security vulnerability.

Use GitHub private vulnerability reporting or contact the repository owner privately. Include a short description, reproduction steps, affected version or commit, and impact assessment.

## Artifact safety

This project uses `joblib` for local scikit-learn artifacts. Treat `.joblib` files as trusted local outputs only. Do not load model artifacts from unknown users or external sources.

The app writes `artifact_manifest.json` for generated local artifacts and verifies hashes before automatic reload. This protects against accidental stale or modified local files, but it is not a sandbox for untrusted pickle/joblib content.

## Data handling

Uploaded CSV files are processed locally by the Streamlit runtime. Do not deploy the app publicly with sensitive data unless the hosting environment, access controls, storage policy, and artifact directory are configured appropriately.

The stakeholder review bundle intentionally excludes `.joblib` model binaries. Share model binaries only through trusted internal channels when they are required.
