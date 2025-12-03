import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a risk parity strategy based on the provided prices and strategy spec.

    Parameters:
    prices (pd.DataFrame): A DataFrame of asset prices.
    spec (dict): A dictionary containing the strategy specification.

    Returns:
    dict: A dictionary containing the strategy diagnostics.
    """

    # Extract the universe and frequency from the strategy spec
    universe = spec["universe"]
    frequency = spec["frequency"]

    # Extract the lookback period from the strategy spec
    lookback = spec["params"]["lookback"]

    # Compute daily returns
    returns = pct_returns(prices)

    # If the frequency is weekly, resample the returns to weekly
    if frequency == "weekly":
        returns = returns.resample("W").last()

    # Compute the volatility of each asset
    vol = returns.rolling(lookback).std()

    # Compute the inverse volatility weights
    weights = 1 / vol

    # Smooth the weights over the holding period
    weights = weights.rolling(lookback).mean()

    # Normalize the weights to ensure unit gross leverage
    weights = normalize_weights(weights, max_leverage=spec["max_leverage"])

    # Compute the portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute the turnover
    turnover = compute_turnover(weights)

    # Compute the Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute the maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute the annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create a diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
