import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
    "name": "mean_reversion",
    "task": "mean_reversion",
    "description": "Short-term cross-sectional contrarian rotation on equities.",
    "universe": [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT"
    ],
    "frequency": "daily",
    "signal": "For each stock, compute a {lookback}-day return z-score (standardized across the universe). Go long names with zscore <= -z_entry and short names with zscore >= +z_entry, with dollar-neutral portfolio weights.",
    "rules": {
        "entry": "open or adjust positions when |zscore| >= z_entry long if zscore <= -z_entry, keep the position until exit condition met; short if zscore >= +z_entry, keep the position until exit condition met.",
        "exit": "close positions when |zscore| <= z_exit or the sign of the zscore reverses, sell long positions when zscore > -z_exitcover short positions when zscore < +z_exit"
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
        "lookback": 10,
        "z_entry": 1.5,
        "z_exit": 0.2
    },
    "costs_bps": 1.0,
    "start_date": "2012-01-03",
    "end_date": "2025-10-31",
    "seed": 42,
    "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    lookback = spec["params"]["lookback"]
    z_entry = spec["params"]["z_entry"]
    z_exit = spec["params"]["z_exit"]
    max_leverage = spec["max_leverage"]

    if spec["frequency"] == "weekly":
        prices = prices.resample('W').last()

    returns = prices.pct_change()
    rolling_returns = returns.rolling(lookback).mean()
    rolling_vol = returns.rolling(lookback).std()
    z_scores = (rolling_returns - rolling_returns.mean(axis=1).values[:, None]) / rolling_vol

    long_mask = z_scores <= -z_entry
    short_mask = z_scores >= z_entry

    positions = np.where(long_mask, 1, np.where(short_mask, -1, 0))
    smoothed_positions = positions.rolling(lookback, min_periods=1).mean()

    weights = smoothed_positions / smoothed_positions.abs().sum(axis=1).values[:, None]
    weights = weights * max_leverage

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()

    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (1 + portfolio_returns).cumprod().rolling(len(portfolio_returns)).apply(lambda x: (x.max() - x.min()) / x.max()).max()
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

    metrics = run_strategy(prices, SPEC)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
