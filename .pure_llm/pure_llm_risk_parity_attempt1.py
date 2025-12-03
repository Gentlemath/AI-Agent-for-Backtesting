import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader

SPEC = {
  "name": "risk_parity",
  "task": "risk_parity",
  "description": "Toy risk-parity allocating inverse vol weights.",
  "universe": [
    "SPY",
    "TLT",
    "GLD",
    "IEF"
  ],
  "frequency": "daily",
  "signal": "Inverse volatility weights across asset classes.",
  "rules": {
    "entry": "allocate inverse std",
    "exit": "rebalance monthly"
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
    "lookback": 60
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

    vol = returns.rolling(window=spec["params"]["lookback"]).std()
    inv_vol = 1 / vol
    inv_vol_weights = inv_vol / inv_vol.sum(axis=1, skipna=True).values[:, np.newaxis]

    # Smooth positions over the holding period (rolling mean)
    smoothed_weights = inv_vol_weights.rolling(window=spec["params"]["lookback"]).mean()

    # Normalize weights
    normalized_weights = smoothed_weights / smoothed_weights.sum(axis=1, skipna=True).values[:, np.newaxis]
    normalized_weights *= spec["max_leverage"]

    # Compute turnover
    turnover = (normalized_weights.diff().abs().sum(axis=1, skipna=True)).mean() * 252

    # Compute metrics
    portfolio_returns = (returns * normalized_weights.shift(1)).sum(axis=1, skipna=True)
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window=252, min_periods=1).max() / (portfolio_returns + 1).cumprod() - 1
    max_dd = max_dd.max()
    hit_rate = (portfolio_returns > 0).mean()
    profit_factor = (portfolio_returns[portfolio_returns > 0].sum() / -portfolio_returns[portfolio_returns < 0].sum())

    metrics = {
        "ann_return": ann_return,
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

    print("Annualized Return:", metrics["ann_return"])
    print("Annualized Volatility:", metrics["sharpe"] * metrics["ann_vol"])
    print("Sharpe Ratio:", metrics["sharpe"])
    print("Max Drawdown:", metrics["max_dd"])
    print("Turnover:", metrics["turnover"])
    print("Hit Rate:", metrics["hit_rate"])
    print("Profit Factor:", metrics["profit_factor"])
    print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})

if __name__ == "__main__":
    main()
