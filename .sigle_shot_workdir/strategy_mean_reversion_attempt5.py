import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters
    lookback = spec["params"]["lookback"]
    z_entry = spec["params"]["z_entry"]
    z_exit = spec["params"]["z_exit"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]

    # Resample prices if weekly frequency
    if frequency == "weekly":
        prices = prices.resample("W").last()

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute z-scores
    z_scores = (returns.rolling(lookback).mean() - returns.mean()) / returns.rolling(lookback).std()

    # Create long and short masks
    long_mask = (z_scores <= -z_entry).where(z_scores <= -z_exit, False)
    short_mask = (z_scores >= z_entry).where(z_scores >= z_exit, False)

    # Smooth signals over holding period
    long_mask_smooth = long_mask.rolling(lookback).mean()
    short_mask_smooth = short_mask.rolling(lookback).mean()

    # Create positions
    positions = pd.DataFrame(index=long_mask_smooth.index, columns=long_mask_smooth.columns)
    positions[long_mask_smooth] = 1
    positions[short_mask_smooth] = -1

    # Normalize weights
    weights = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "ann_return": portfolio_returns.mean() * (252 if frequency == "daily" else 52),
        "sharpe": sharpe_ratio(portfolio_returns, periods=(252 if frequency == "daily" else 52)),
        "max_dd": max_drawdown(portfolio_returns)
    }

    return diagnostics
