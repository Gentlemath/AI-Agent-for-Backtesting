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
    "signal": "Apply a binary weekday filter to the basket: on allowed weekdays the strategy is fully invested; on all other weekdays position is 0.Do NOT use numpy arrays for the mask. Use returns.index.to_series().dt.weekday to compute weekdays, and then replicate to a DataFrame aligned with asset number.Don't shift weight since weekdays are known in advance.",
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
    max_leverage = spec["max_leverage"]

    # Compute weekday mask
    weekday_mask = prices.index.to_series().dt.weekday.isin(allowed_weekdays).values.reshape(-1, 1)
    weekday_mask = pd.DataFrame(weekday_mask, index=prices.index, columns=prices.columns)

    # Compute weights
    weights = weekday_mask / weekday_mask.sum(axis=1).values.reshape(-1, 1)

    # Normalize weights
    weights = weights * max_leverage

    # Compute returns
    returns = prices.pct_change().shift(1) * weights.shift(1)

    # Compute metrics
    ann_return = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (returns.cumsum() - returns.cumsum().cummax()).min()
    turnover = (weights.diff().abs().sum(axis=1)).mean() * 252
    hit_rate = (returns > 0).mean()
    profit_factor = (returns[returns > 0].sum() / returns[returns < 0].sum().abs()) if returns[returns < 0].sum().abs() != 0 else np.nan

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

    print("Annualized Return:", metrics["ann_return"])
    print("Annualized Volatility:", metrics["ann_vol"])
    print("Sharpe Ratio:", metrics["sharpe"])
    print("Max Drawdown:", metrics["max_dd"])
    print("Turnover:", metrics["turnover"])
    print("Hit Rate:", metrics["hit_rate"])
    print("Profit Factor:", metrics["profit_factor"])
    print("Metrics:", {k: v for k, v in metrics.items() if not pd.isna(v)})

if __name__ == "__main__":
    main()
