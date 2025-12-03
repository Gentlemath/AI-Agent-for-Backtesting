import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the momentum_daily strategy.

    Parameters:
    prices (pd.DataFrame): Prices of the assets.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute compounded 63-day return
    lookback = spec['params']['lookback']
    compounded_returns = returns.rolling(lookback).apply(lambda x: (1 + x).prod() - 1)

    # Rank compounded returns in descending order
    ranked_returns = compounded_returns.rank(axis=1, method='min', ascending=False)

    # Select top-k assets
    top_k = spec['params']['top_k']
    top_k_assets = ranked_returns.apply(lambda x: x <= top_k, axis=1)

    # Smooth top-k assets over the holding period
    holding_period = spec['params']['holding_period']
    smoothed_assets = top_k_assets.rolling(holding_period).mean()

    # Normalize weights
    max_leverage = spec['max_leverage']
    weights = normalize_weights(smoothed_assets, max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * 252

    # Store diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
