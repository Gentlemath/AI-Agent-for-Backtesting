import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
  "name": "momentum_weekly",
  "task": "momentum_weekly",
  "description": "Weekly momentum with slower turnover.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "weekly",
  "signal": "Weekly returns ranked over trailing 26 weeks.",
  "rules": {
    "entry": "top_2",
    "exit": "stop if rank drops below 5"
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
    "lookback": 26,
    "top_k": 2,
    "holding_period": 8
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

    momentum = returns.rolling(window=spec["params"]["lookback"]).mean()
    ranks = momentum.rank(axis=1, ascending=False)

    positions = np.where(ranks <= spec["params"]["top_k"], 1, 0)
    smoothed_positions = positions.rolling(window=spec["params"]["holding_period"]).mean()

    weights = smoothed_positions / smoothed_positions.sum(axis=1).values[:, np.newaxis]
    weights = weights * spec["max_leverage"]

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()

    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window=len(portfolio_returns)).min().min() - 1
    turnover = (weights.diff().abs().sum(axis=1)).mean()
    hit_rate = (portfolio_returns > 0).mean()
    profit_factor = (portfolio_returns[portfolio_returns > 0].sum() + 1) / (portfolio_returns[portfolio_returns < 0].sum() + 1)

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

    print("Annualized Return: {:.2f}%".format(metrics["ann_return"] * 100))
    print("Annualized Volatility: {:.2f}%".format(metrics["ann_vol"] * 100))
    print("Sharpe Ratio: {:.2f}".format(metrics["sharpe"]))
    print("Max Drawdown: {:.2f}%".format(metrics["max_dd"] * 100))
    print("Turnover: {:.2f}%".format(metrics["turnover"] * 100))
    print("Hit Rate: {:.2f}%".format(metrics["hit_rate"] * 100))
    print("Profit Factor: {:.2f}".format(metrics["profit_factor"]))

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
