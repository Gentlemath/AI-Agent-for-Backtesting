import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a volatility targeting strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): A DataFrame of asset prices.
    spec (dict): A dictionary containing the strategy specification.

    Returns:
    dict: A dictionary containing the strategy diagnostics.
    """

    # Extract relevant parameters from the spec
    lookback = spec["params"]["lookback"]
    target_vol = spec["params"]["target_vol"]
    max_leverage = spec["max_leverage"]
    costs_bps = spec["costs_bps"]
    frequency = spec["frequency"]

    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly frequency if necessary
    if frequency == "weekly":
        returns = returns.resample("W").last()

    # Compute realized volatility
    realized_vol = returns.rolling(lookback).std() * (252 if frequency == "daily" else 52) ** 0.5

    # Compute target weights
    target_weights = (target_vol / realized_vol).where(realized_vol > 0, 0)

    # Smooth target weights over the holding period
    smoothed_weights = target_weights.rolling(lookback).mean()

    # Normalize weights to ensure unit gross leverage
    weights = normalize_weights(smoothed_weights, max_leverage)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Store diagnostics in a dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
