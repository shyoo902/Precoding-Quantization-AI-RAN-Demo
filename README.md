# AI-RAN Inference-Time Allocation Demo

An interactive Streamlit demonstration of joint precoder and fronthaul quantizer
inference-time allocation for cell-free massive MIMO.

Select the number of users and channel correlation, adjust the two inference
budgets, and explore finite-blocklength sum-rate and estimated memory together.
The interface includes an allocation heatmap, individual budget sweeps, a
rotatable 3D rate surface, and comparisons with the best saved allocations.

## Deploy to a public URL

This repository is ready for **Streamlit Community Cloud**. The application runs
entirely from the included JSON data; it requires no GPU, model checkpoints,
API keys, or connection to the original research server.

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect the
   GitHub account that owns this repository.
2. Choose **Create app**, then deploy the following:

   | Setting | Value |
   | --- | --- |
   | Repository | `shyoo902/Precoding-Quantization-AI-RAN-Demo` |
   | Branch | `main` |
   | Main file | `streamlit_app.py` |
   | Python version, under Advanced settings | `3.12` |
   | Secrets | None |
   | Suggested app subdomain | `precoding-quantization-ai-ran-demo` |

3. Choose public access so reviewers can open the app without an invitation.
4. Copy the actual `https://….streamlit.app` URL shown after deployment.

The suggested subdomain is subject to availability; it is not an already
deployed URL. The public address becomes usable after Community Cloud finishes
deployment. The URL persists across ordinary app updates. Community Cloud can
hibernate inactive apps, so open the app before a live presentation.

Once connected, pushes to `main` update the cloud app automatically. The GitHub
Actions workflow separately checks the dataset and Streamlit interactions on
each push and pull request. It does not replace the initial Community Cloud
account connection or deployment step, and it does not gate automatic updates.

See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [GitHub account connection](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account).

## Run locally

Use Python 3.12, matching the cloud and CI configuration:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The theme and headless-server defaults
are included in `.streamlit/config.toml`.

## What the results represent

- The app interactively displays **saved simulation results**. Moving a slider
  does not train a model or initiate a new radio experiment.
- The default fixed-candidate evaluation retains previous candidates as budgets
  increase. Its best candidate rate can improve or plateau under the fixed
  evaluator. This metric does not include an inference-time overhead penalty.
- Legacy allocation and budget sensitivity views are kept separately labeled.
  Their effective/modeled metrics have different timing assumptions.
- “Best under budget” searches saved allocations within the selected total
  budget. “Best on full grid” searches all saved allocations and may use more
  inference time. Neither asserts a continuous-domain optimum.
- Memory includes model tensors and estimated candidate buffers. It excludes
  activations, solver workspaces, and framework overhead, and is not measured
  peak RAM, VRAM, power, or energy.
- The 3D surface connects evaluated points; intermediate surface locations are
  not additional measurements. Available user counts and correlations are
  limited to saved experiments.

The **Experiment details** tab contains provenance and CSV/JSON downloads.
Original result filenames in that tab identify source experiments; those raw
research files are not runtime dependencies of this repository.

## Validate changes

```sh
python -m unittest discover -s tests -p test_dashboard.py -v
```

The suite checks data validity, budget feasibility, rate-surface alignment,
memory calculations, sliders, evaluation modes, best-allocation buttons,
downloads, and the ordering of dashboard sections.

## Update the experiment data

Export updated results in the research environment, replace
`dashboard_data/experiments.json`, run the tests, and commit/push the changes.
The exported file must keep schema version 1 and its provenance metadata.
Training datasets and checkpoints are not needed for dashboard deployment.

## Research context and license

This dashboard accompanies research on test-time scalable AI-RAN and joint
precoding/fronthaul quantization for cell-free MIMO.

The source research workspace builds on H. Hojatian, J. Nadal, J.-F. Frigon,
and F. Leduc-Primeau, “Decentralized Beamforming for Cell-Free Massive MIMO with
Unsupervised Learning,” *IEEE Communications Letters*, 2022,
[doi:10.1109/LCOMM.2022.3157161](https://doi.org/10.1109/LCOMM.2022.3157161).
The workspace's GPL v3 license is retained in [LICENSE](LICENSE).
