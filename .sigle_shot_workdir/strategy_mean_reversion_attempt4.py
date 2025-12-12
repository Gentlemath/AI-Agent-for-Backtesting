import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from the spec
    lookback = spec["params"]["lookback"]
    z_entry = spec["params"]["z_entry"]
    z_exit = spec["params"]["z_exit"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]
    start_date = spec["start_date"]
    end_date = spec["end_date"]

    # Resample prices to the specified frequency
    if frequency == "weekly":
        prices = prices.resample("W").last()
    else:
        prices = prices

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute rolling returns
    rolling_returns = returns.rolling(lookback).mean()

    # Compute z-scores
    z_scores = (rolling_returns - rolling_returns.mean()) / rolling_returns.std()

    # Create long and short masks
    long_mask = (z_scores <= -z_entry).where(returns.notnull(), False)
    short_mask = (z_scores >= z_entry).where(returns.notnull(), False)

    # Create exit masks
    exit_long_mask = (z_scores > -z_exit) | (z_scores * long_mask.shift(1) < 0)
    exit_short_mask = (z_scores < z_exit) | (z_scores * short_mask.shift(1) > 0)

    # Create positions
    positions = pd.DataFrame(index=returns.index, columns=returns.columns)
    positions[long_mask] = 1
    positions[short_mask] = -1
    positions[exit_long_mask] = 0
    positions[exit_short_mask] = 0

    # Smooth positions over the holding period
    positions = positions.rolling(lookback, min_periods=1).mean()

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
