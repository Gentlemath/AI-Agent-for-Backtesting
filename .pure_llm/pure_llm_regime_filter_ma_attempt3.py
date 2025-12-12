import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader

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
    params = spec["params"]
    fast_ma = prices.rolling(window=params["fast"]).mean()
    slow_ma = prices.rolling(window=params["slow"]).mean()
    risk_on = (fast_ma > slow_ma).astype(int)
    weights = risk_on.rolling(window=params["smoothing_window"]).mean()
    
    if spec["frequency"] == "weekly":
        prices_weekly = prices.resample('W').last()
        weights_weekly = weights.resample('W').last()
        returns_weekly = prices_weekly.pct_change()
        weights_weekly = weights_weekly.shift(1)
    else:
        returns_daily = prices.pct_change()
        weights_daily = weights.shift(1)
        returns_weekly = None
        weights_weekly = None
    
    # Normalize weights
    max_leverage = spec["max_leverage"]
    weights = weights / weights.max() * max_leverage
    if weights_weekly is not None:
        weights_weekly = weights_weekly / weights_weekly.max() * max_leverage
    
    # Compute metrics
    if returns_weekly is not None:
        portfolio_returns = (returns_weekly * weights_weekly).sum(axis=1)
    else:
        portfolio_returns = (returns_daily * weights_daily).sum(axis=1)
    
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (portfolio_returns + 1).cumprod().rolling(window=len(portfolio_returns)).min().min() - 1
    turnover = (weights.shift(1) - weights).abs().sum(axis=1).mean() * 252
    hit_rate = (portfolio_returns > 0).mean()
    profit_factor = portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()
    
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
    
    print("Metrics: {}".format(metrics))

if __name__ == "__main__":
    main()
