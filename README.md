# RegimeGuard — Adaptive Limit Order Book Forecasting

A runnable educational MVP for short-horizon order-book prediction, distribution-shift detection, and shift-triggered model retraining.

## Features
- Predicts whether mid-price will rise over a configurable future horizon.
- Engineers spread, order-book imbalance, total depth, return, and rolling volatility.
- Detects input-distribution shifts with two-sample rolling Kolmogorov–Smirnov tests.
- Uses Bonferroni correction across monitored features.
- Compares a static Random Forest baseline with an adaptive Random Forest.
- Streamlit dashboard with shift timeline, metrics, and downloadable predictions.
- Includes reproducible synthetic LOB-like data so the demo works without API credentials.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# If PowerShell blocks activation, use:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Using your own data

Upload a CSV containing at least 500 valid rows and these columns:

- `bid_price`
- `ask_price`
- `bid_size`
- `ask_size`

An optional `timestamp` column is preserved for display. Rows must be ordered chronologically. Prices and quantities should be numeric. For LOBSTER data, map the best-level bid/ask prices and sizes into these columns before upload.

## Experimental design

- Chronological split: first 60% is initial training data; later observations are evaluated sequentially.
- Target: future mid-price is greater than current mid-price after `horizon` rows.
- Shift detector: each monitored feature compares a reference window with the latest window using a two-sample KS test.
- Multiple comparisons: per-feature threshold is `alpha / number_of_features`.
- Adaptation: on detected shifts, refits on a recent window of rows whose future-horizon labels should already be observable.
- Metrics: accuracy, balanced accuracy, F1, shift alerts, and measured inference latency.

## Limitations and research cautions

This is a prototype, not a production trading system or investment advice. The synthetic data is only for demonstrating the pipeline. Real exchange data has feed-specific quirks, timestamps, event sequencing, and latency constraints. Results depend on the dataset and parameters.

The KS test monitors marginal feature distributions. It does not prove that predictive relationships have changed, and it can generate false alarms. The example uses a basic multiple-testing correction; a stronger study should test false-alarm rates, label/target drift, detection delay, adaptation recovery, and walk-forward performance. Do not use future labels for training until those labels would actually be available.

## Suggested improvements
1. Add a second detector such as Page-Hinkley or ADWIN.
2. Add regime-specific heads and a policy for choosing them.
3. Benchmark p50/p95/p99 latency under a clearly specified hardware/load budget.
4. Test on multiple assets and days, with fees/slippage if evaluating trading utility.
5. Add unit tests and save run configuration for reproducibility.

## Project question
Does shift-aware adaptation improve short-horizon forecasting and recovery after regime changes compared with a static model?
