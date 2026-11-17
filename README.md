# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nmercy12/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Nmercy12/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2 to 24) with full quantile distributions (0.05, 0.25, 0.50, 0.75, 0.95) using a Gaussian kernel-weighted local-linear log-ratio regression over comparable past launches.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input/ MAKEFUN_OUTPUT_DIR=/output/ python3 main.py
```

## Model Overview

The model uses a Gaussian kernel-weighted local-linear log-ratio regression over the case's usable historical comparables. For each forecast hour, it predicts log-ratio growth relative to hour 1, and calibrates prediction spread using leave-one-out (LOO) residual standard deviation across the training comparables.

LOO backtest: 0.4070 mean pinball loss on all 37 training launches, 0.2347 on the 12 largest (baseline persistence: 0.9296 / 0.2617).

## Requirements & Execution

- Python 3.13 (Standard library only)
- Run the forecaster by setting input and output directories via environment variables:

```bash
MAKEFUN_INPUT_DIR=/input/ MAKEFUN_OUTPUT_DIR=/output/ python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=stt5ry3a&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
