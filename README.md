# Autoscaling Playground

Experiments using **Prometheus-style metrics** (CPU, network) to predict **latency** with linear regression—for trend prediction, capacity planning, and early warning before saturation.

Uses the [Microservices Bottleneck Detection Dataset](https://www.kaggle.com/datasets/gagansomashekar/microservices-bottleneck-detection-dataset). Main finding: **linear regression fits best on baseline (pre-saturation) behaviour**; fitting across the full window including saturation spikes is unreliable. See [EXPLORATION_SUMMARY.md](EXPLORATION_SUMMARY.md) for the full story.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Place the **processed** dataset at `processed_dataset/` (or pass `--data-dir`). Layout expected: `processed_dataset/{user,home,compose}/multi-modal-data-separate/*.csv`.

## Scripts

| Script | Purpose |
|--------|--------|
| `visualise_processed_dataset.py` | Plot metrics and latency per scenario CSV (time series, correlations). |
| `run_all_scenarios.py` | Batch visualisation → one PNG per scenario in `scenarios/`. |
| `find_saturation_candidates.py` | Score scenarios by saturation heuristics; output list for tagging candidates. |
| `linear_regression_saturation.py` | Fit linear regression (CPU + network → latency) on `<c-latency>-` scenarios; optional baseline-only fit. |

## Quick run

```bash
# Visualise one scenario (default CSV)
python visualise_processed_dataset.py --csv path/to/scenario_graph_1.csv

# Regression on baseline only (recommended)
python linear_regression_saturation.py --baseline-only --out-dir regression/

# Headless batch
MPLBACKEND=Agg python linear_regression_saturation.py --baseline-only --no-show --out-dir regression/
```

Regression supports `--poly`, `--log`, `--log-target`; use `--baseline-only` (with optional `--baseline-percentile` or `--baseline-pct`) to fit only pre-saturation data.
