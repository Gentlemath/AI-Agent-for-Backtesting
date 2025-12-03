import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
  "name": "volatility_targeting",
  "task": "volatility_targeting",
  "description": "Vol-targeted exposure using realized vol estimates.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "daily",
  "signal": "Scale exposure inversely with trailing volatility.",
  "rules": {
    "entry": "target_vol / realized_vol",
    "exit": "vol > cap"
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
    "lookback": 20,
    "target_vol": 0.12
  },
  "costs_bps": 1.0,
  "start_date": "2012-01-03",
  "end_date": "2025-10-31",
  "seed": 42,
  "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    if spec["frequency"] == "weekly":
        prices = prices.resample('W').last()
        returns = prices.pct_change()
    else:
        returns = prices.pct_change()

    realized_vol = returns.rolling(spec["params"]["lookback"]).std() * np.sqrt(252)
    target_vol = spec["params"]["target_vol"]
    positions = target_vol / realized_vol

    # Smooth positions over the holding period
    smoothed_positions = positions.rolling(spec["params"]["lookback"]).mean()

    # Normalize weights
    normalized_weights = smoothed_positions / smoothed_positions.abs().sum(axis=1)
    normalized_weights = normalized_weights * spec["max_leverage"]

    # Compute turnover
    turnover = (normalized_weights.diff().abs().sum(axis=1)).mean() * 252

    # Compute metrics
    portfolio_returns = (returns * normalized_weights.shift(1)).sum(axis=1)
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window=252, min_periods=1).apply(lambda x: (x.max() - x.min()) / x.max()).max()
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
    metrics = run_strategy(prices, SPEC)

    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nJSON Summary:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
