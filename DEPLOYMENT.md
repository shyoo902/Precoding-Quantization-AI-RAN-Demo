# Public deployment

This repository is prepared for a public, read-only demonstration.

## Recommended: Streamlit Community Cloud

Streamlit Community Cloud is the simplest option for this repository.

Use these settings:

- Repository: `shyoo902/Precoding-Quantization-AI-RAN-Demo`
- Branch: `main`
- Main file: `streamlit_app.py`
- Python: `3.12`
- Secrets: none
- Suggested subdomain: `precoding-quantization-ai-ran-demo`

After deployment, set the app to public and copy its `https://<subdomain>.streamlit.app` address.

## Optional: Hugging Face Spaces

The repository also includes a `Dockerfile` that runs the Streamlit app on port `7860`, so it can be used as the source for a Docker-based Hugging Face Space.

For the Space README metadata, use:

```yaml
---
title: AI-RAN Inference-Time Allocation Demo
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
```

## Required experiment data

The app expects the following file at runtime:

`dashboard_data/experiments.json`

The public deployment cannot run until this file is present in the repository. The file should contain the saved simulation results used by the dashboard; do not replace it with synthetic or placeholder results.
