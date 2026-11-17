# GTA VI Twitch CCU Forecaster

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2 to 24) with full quantile distributions (0.05, 0.25, 0.50, 0.75, 0.95).

## Model Overview

The model uses a Gaussian kernel-weighted local-linear log-ratio regression over the case's usable historical comparables. For each forecast hour, it predicts log-ratio growth relative to hour 1, and calibrates prediction spread using leave-one-out (LOO) residual standard deviation across the training comparables.

## Requirements & Execution

- Python 3.13 (Standard library only)
- Run the forecaster by setting input and output directories via environment variables:

```bash
MAKEFUN_INPUT_DIR=/input/ MAKEFUN_OUTPUT_DIR=/output/ python3 main.py
```
