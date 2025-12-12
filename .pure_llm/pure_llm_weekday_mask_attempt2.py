import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader

SPEC = {
    "name": "weekday_mask",
    "task": "weekday_mask",
    "description": "Hard weekday seasonality mask on exposures (no smoothing).",
    "universe": [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT"
    ],
    "frequency": "daily",
    "signal": "Apply a binary weekday filter to the basket: on allowed weekdays the strategy is fully invested; on all other weekdays position is 0.",
    "rules": {
        "entry": "if today's weekday is in allowed_weekdays, set exposure to 1.0 (or pass through the underlying strategy's signal unchanged)",
        "exit": "if today's weekday is not in allowed_weekdays, set exposure to 0.0 and close all positions",
        "weighting": "Equally weight all allowed assets on allowed weekdays."
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
        "allowed_weekdays": [
            0,
            2,
            4
        ]
    },
    "costs_bps": 1.0,
    "start_date": "2012-01-03",
    "end_date": "2025-10-31",
    "seed": 42,
    "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    allowed_weekdays = spec["params"]["allowed_weekdays"]
    weekday_mask = prices.index.to_series().dt.weekday.isin(allowed_weekdays).values[:, None]
    weights = weekday_mask / weekday_mask.sum(axis=1, keepdims=True)
    weights = weights * spec["max_leverage"]
    
    returns = prices.pct_change().shift(1)
    portfolio_returns = (returns * weights).sum(axis=1)
    
    metrics = {}
    metrics["ann_return"] = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    metrics["ann_vol"] = portfolio_returns.std() * np.sqrt(252)
    metrics["sharpe"] = metrics["ann_return"] / metrics["ann_vol"]
    metrics["max_dd"] = (1 + portfolio_returns).cumprod().rolling(window=len(portfolio_returns), min_periods=1).apply(lambda x: (x.max() - x.min()) / x.max()).max()
    metrics["turnover"] = (weights.diff().abs().sum(axis=1)).mean()
    metrics["hit_rate"] = (portfolio_returns > 0).mean()
    metrics["profit_factor"] = (portfolio_returns[portfolio_returns > 0].sum() / portfolio_returns[portfolio_returns < 0].sum()).abs()
    
    return metrics

def main():
    data_loader = DataLoader(disk_dir="data")
    prices = data_loader.ensure_symbols(SPEC["universe"], SPEC["start_date"], SPEC["end_date"])
    if SPEC["frequency"] == "weekly":
        prices = prices.resample('W').last()
    
    metrics = run_strategy(prices, SPEC)
    
    print("Annualized Return: {:.2f}%".format(metrics["ann_return"] * 100))
    print("Annualized Volatility: {:.2f}%".format(metrics["ann_vol"] * 100))
    print("Sharpe Ratio: {:.2f}".format(metrics["sharpe"]))
    print("Max Drawdown: {:.2f}%".format(metrics["max_dd"] * 100))
    print("Turnover: {:.2f}%".format(metrics["turnover"] * 100))
    print("Hit Rate: {:.2f}%".format(metrics["hit_rate"] * 100))
    print("Profit Factor: {:.2f}".format(metrics["profit_factor"]))
    
    print("Metrics: {}".format({k: "{:.2f}%".format(v * 100) if k in ["ann_return", "ann_vol", "max_dd", "turnover", "hit_rate"] else "{:.2f}".format(v) for k, v in metrics.items()}))

if __name__ == "__main__":
    main()
