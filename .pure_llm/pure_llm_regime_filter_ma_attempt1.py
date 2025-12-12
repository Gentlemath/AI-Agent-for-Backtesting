import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
  "name": "regime_filter_ma",
  "task": "regime_filter_ma",
  "description": "Trend regime filter with smoothed moving-average crossover exposure.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "daily",
  "signal": "Compute fast and slow moving averages of the close (fast = {fast}, slow = {slow}). Define a binary risk-on indicator I_t = 1 if fast_ma_t > slow_ma_t, else 0. Smooth I_t over N days (N = smoothing_window) using a moving average to obtain a continuous exposure weight w_t in [0, 1].",
  "rules": {
    "entry": "increase exposure as the smoothed risk-on indicator w_t moves toward 1",
    "exit": "decrease exposure as w_t moves toward 0; no discrete hold-N-days tickets"
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
    "fast": 50,
    "slow": 200,
    "smoothing_window": 5
  },
  "costs_bps": 1.0,
  "start_date": "2012-01-03",
  "end_date": "2025-10-31",
  "seed": 42,
  "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    fast = spec["params"]["fast"]
    slow = spec["params"]["slow"]
    smoothing_window = spec["params"]["smoothing_window"]
    max_leverage = spec["max_leverage"]

    # Compute fast and slow moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Define a binary risk-on indicator
    risk_on = (fast_ma > slow_ma).astype(int)

    # Smooth the risk-on indicator
    weights = risk_on.rolling(window=smoothing_window).mean()

    # Normalize weights
    weights = weights / weights.max() * max_leverage

    # Compute returns
    returns = prices.pct_change()

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute metrics
    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window=len(portfolio_returns)).max() - (portfolio_returns + 1).cumprod()
    max_dd = max_dd.max()
    turnover = (weights.diff().abs().sum(axis=1)).mean() * 252
    hit_rate = (portfolio_returns > 0).mean()
    profit_factor = (portfolio_returns[portfolio_returns > 0].sum() / -portfolio_returns[portfolio_returns < 0].sum())

    metrics = {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "turnover": turnover,
        "hit_rate": hit_rate,
        "profit_factor": profit_factor
    }

    return metrics

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
