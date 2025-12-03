import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
  "name": "momentum_daily",
  "task": "momentum_daily",
  "description": "Daily top-k cross-sectional momentum on ETFs.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "daily",
  "signal": "Rank compounded 63-day return and go long top decile.",
  "rules": {
    "entry": "rank_desc top_k",
    "exit": "hold N days"
  },
  "tools": [
    "returns",
    "sharpe",
    "drawdown",
    "normalize_weights",
    "compute_turnover"
  ],
  "required_metrics": [
    "ann_return",
    "sharpe",
    "max_dd"
  ],
  "params": {
    "lookback": 63,
    "top_k": 3,
    "holding_period": 20
  },
  "costs_bps": 1.0,
  "start_date": "2012-01-03",
  "end_date": "2025-10-31",
  "seed": 42,
  "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    params = spec["params"]
    lookback = params["lookback"]
    top_k = params["top_k"]
    holding_period = params["holding_period"]
    max_leverage = spec["max_leverage"]

    # Calculate returns
    returns = prices.pct_change()

    # Calculate momentum signal
    momentum = returns.rolling(lookback).mean()

    # Rank momentum and select top k
    ranked_momentum = momentum.rank(axis=1, method="min", ascending=False)
    top_k_mask = ranked_momentum <= top_k

    # Smooth positions over holding period
    smoothed_positions = top_k_mask.rolling(holding_period).mean()

    # Normalize weights
    normalized_weights = smoothed_positions / smoothed_positions.sum(axis=1, skipna=True)

    # Scale weights by max leverage
    scaled_weights = normalized_weights * max_leverage

    # Calculate portfolio returns
    portfolio_returns = (scaled_weights.shift(1) * returns).sum(axis=1)

    # Calculate metrics
    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (1 + portfolio_returns).cumprod().rolling(len(portfolio_returns)).apply(lambda x: (x.max() - x.min()) / x.max()).max()
    turnover = (scaled_weights.diff().abs().sum(axis=1)).mean() * 252
    hit_rate = (portfolio_returns > 0).mean()
    profit_factor = (portfolio_returns[portfolio_returns > 0].sum() / -portfolio_returns[portfolio_returns < 0].sum())

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover": turnover,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor
    }

def main():
    data_loader = DataLoader(disk_dir="data")
    prices = data_loader.ensure_symbols(SPEC["universe"], SPEC["start_date"], SPEC["end_date"])

    if SPEC["frequency"] == "weekly":
        prices = prices.resample("W").last()

    metrics = run_strategy(prices, SPEC)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
