import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader

SPEC = {
    "name": "pair_trading",
    "task": "pair_trading",
    "description": "Stat-arb pair trading toggling between cointegration & distance tests. You must test correlation >= correlation_threshhold before trading. if all correlation < correlation_threshhold, report and do not trade.",
    "universe": [
        "XOM",
        "CVX"
    ],
    "frequency": "daily",
    "signal": "Spread z-score between highly correlated pair.",
    "rules": {
        "entry": "spread zscore > entry_z",
        "exit": "zscore < exit_z"
    },
    "tools": [
        "returns",
        "sharpe",
        "drawdown",
        "normalize_weights",
        "compute_turnover",
        "returns"
    ],
    "required_metrics": [
        "ann_return",
        "sharpe",
        "max_dd"
    ],
    "params": {
        "lookback": 60,
        "mode": "cointegration",
        "correlation_threshhold": 0.6,
        "entry_z": 1.5,
        "exit_z": 0.5
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
    entry_z = params["entry_z"]
    exit_z = params["exit_z"]
    max_leverage = spec["max_leverage"]

    # Calculate daily returns
    returns = prices.pct_change()

    # Calculate spread
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]

    # Calculate z-score
    z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    # Create positions
    positions = np.where(z_score > entry_z, 1, np.where(z_score < exit_z, -1, 0))

    # Smooth positions
    smoothed_positions = positions.rolling(lookback).mean()

    # Normalize weights
    weights = smoothed_positions / np.abs(smoothed_positions).sum(axis=1) * max_leverage

    # Compute turnover
    turnover = np.abs(weights.diff()).sum(axis=1).mean() * 252

    # Compute metrics
    portfolio_returns = (returns * weights.shift(1)).sum(axis=1)
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min()
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

    print("Annualized Return: {:.2f}%".format(metrics["ann_return"] * 100))
    print("Annualized Volatility: {:.2f}%".format(metrics["ann_vol"] * 100))
    print("Sharpe Ratio: {:.2f}".format(metrics["sharpe"]))
    print("Max Drawdown: {:.2f}%".format(metrics["max_dd"] * 100))
    print("Turnover: {:.2f}%".format(metrics["turnover"] * 100))
    print("Hit Rate: {:.2f}%".format(metrics["hit_rate"] * 100))
    print("Profit Factor: {:.2f}".format(metrics["profit_factor"]))

    print("Metrics: {}".format(metrics))

if __name__ == "__main__":
    main()
