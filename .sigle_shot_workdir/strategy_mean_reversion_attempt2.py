import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the mean reversion strategy.

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

    # Resample to weekly if frequency is weekly
    if frequency == "weekly":
        weekly_returns = daily_returns.resample("W").last()
        weekly_returns = weekly_returns.pct_change().fillna(0)
    else:
        weekly_returns = daily_returns

    # Compute z-scores
    z_scores = (weekly_returns - weekly_returns.rolling(lookback).mean()) / weekly_returns.rolling(lookback).std()

    # Create long and short masks
    long_mask = (z_scores <= -z_entry).where(z_scores <= -z_exit, False)
    short_mask = (z_scores >= z_entry).where(z_scores >= z_exit, False)

    # Create positions
    positions = pd.DataFrame(0, index=long_mask.index, columns=long_mask.columns)
    positions = positions.where(long_mask, 1).where(short_mask, -1)

    # Smooth positions over the holding period
    positions = positions.rolling(lookback).mean()

    # Normalize positions
    normalized_positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_positions)

    # Compute portfolio returns
    portfolio_returns = (daily_returns * normalized_positions).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
