import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a mean reversion strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """
    # Extract parameters from the spec
    lookback = spec["params"]["lookback"]
    z_entry = spec["params"]["z_entry"]
    z_exit = spec["params"]["z_exit"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]

    # Compute daily returns
    daily_returns = pct_returns(prices)

    # Resample to weekly returns if frequency is weekly
    if frequency == "weekly":
        weekly_returns = daily_returns.resample("W").last()
        weekly_returns = weekly_returns.pct_change().fillna(0)
    else:
        weekly_returns = daily_returns

    # Compute z-scores
    z_scores = (weekly_returns - weekly_returns.mean()) / weekly_returns.std()

    # Create a mask for long and short positions
    long_mask = (z_scores <= -z_entry)
    short_mask = (z_scores >= z_entry)

    # Smooth the signals across the holding period
    long_signal = long_mask.rolling(lookback).mean()
    short_signal = short_mask.rolling(lookback).mean()

    # Create a mask for exit conditions
    exit_mask = ((z_scores.abs() <= z_exit) | (z_scores * long_signal < 0) | (z_scores * short_signal > 0))

    # Create a mask for positions
    positions = (long_signal - short_signal).where(exit_mask, 0)

    # Normalize the positions
    normalized_positions = normalize_weights(positions, max_leverage)

    # Compute the portfolio returns
    portfolio_returns = (daily_returns * normalized_positions).sum(axis=1)

    # Compute the turnover
    turnover = compute_turnover(normalized_positions)

    # Compute the Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute the maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute the annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create the diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return
    }

    return diagnostics
