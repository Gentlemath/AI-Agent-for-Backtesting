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

    # Resample to weekly if frequency is weekly
    if frequency == "weekly":
        weekly_returns = daily_returns.resample("W").last()
        weekly_returns = weekly_returns.pct_change().fillna(0)
    else:
        weekly_returns = daily_returns

    # Compute z-scores
    z_scores = weekly_returns.rolling(lookback).apply(lambda x: (x - x.mean()) / x.std(), raw=False)

    # Create long and short masks
    long_mask = z_scores <= -z_entry
    short_mask = z_scores >= z_exit

    # Smooth signals over the holding period
    smoothed_long_mask = long_mask.rolling(lookback).mean()
    smoothed_short_mask = short_mask.rolling(lookback).mean()

    # Create positions
    positions = pd.DataFrame(index=prices.index, columns=prices.columns)
    positions = positions.where(smoothed_long_mask, 1).where(smoothed_short_mask, -1)

    # Normalize weights
    normalized_weights = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_weights)

    # Compute portfolio returns
    portfolio_returns = (daily_returns * normalized_weights).sum(axis=1)

    # Compute metrics
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)
    sharpe = sharpe_ratio(portfolio_returns, periods=(252 if frequency == "daily" else 52))
    max_dd = max_drawdown(portfolio_returns)

    # Create diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }

    return diagnostics
