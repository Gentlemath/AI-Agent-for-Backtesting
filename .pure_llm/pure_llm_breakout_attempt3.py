import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader

SPEC = {
  "name": "breakout",
  "task": "breakout",
  "description": "Donchian-style long breakout with a trailing stop.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "daily",
  "signal": "Compute a {window}-day rolling high of the prices (rolling_high_t,the highest close over the past {window} days, excluding today)Compute a {stop_window}-day rolling low of the prices(trailing_stop_t, the lowest close over the past {stop_window} days, excluding today). Enter a long position when the close breaks above this rolling high. Once entry, maintain the position until close_t < trailing_stop_t",
  "rules": {
    "entry": "open a long position when close_t > rolling_high_t.",
    "exit": "exit the long position when close_t < trailing_stop_t.You can only exit long positions! NEGATIVE positions is FORBBIDEN!if not in a position, you must wait for the next entry signal."
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
    "window": 55,
    "stop_window": 20
  },
  "costs_bps": 1.0,
  "start_date": "2012-01-03",
  "end_date": "2025-10-31",
  "seed": 42,
  "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    window = spec["params"]["window"]
    stop_window = spec["params"]["stop_window"]
    max_leverage = spec["max_leverage"]

    rolling_high = prices.rolling(window).max()
    rolling_low = prices.rolling(stop_window).min()

    positions = (prices > rolling_high.shift(1)).astype(int)
    positions = positions.rolling(window).mean()

    weights = positions / positions.sum(axis=1, numeric_only=True) * max_leverage

    returns = prices.pct_change()
    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window).max() / (portfolio_returns + 1).cumprod() - 1
    max_dd = max_dd.max()
    turnover = (weights.diff().abs().sum(axis=1)).mean()
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

    print("Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
