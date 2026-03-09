# Notebooks

## traffic_forecasting.ipynb

Interactive walkthrough of:

1. **Gradient descent vs closed-form** linear regression
2. **Model 1**: Predict CPU from time (window index); tune learning rate and epochs with sliders
3. **Model 2**: Predict latency from CPU and network (baseline-only fit)
4. **End-to-end**: Forecast CPU 30 minutes ahead, then estimate latency with Model 2

**Requirements:** Run from project root (or from this folder). Needs `processed_dataset/` and `scenarios/` with at least one `<c-latency>-*` scenario. Install deps: `pip install -r requirements.txt` (includes `ipywidgets` for the interactive sliders).

**Launch:** `jupyter notebook traffic_forecasting.ipynb` or open in VS Code / Cursor.
