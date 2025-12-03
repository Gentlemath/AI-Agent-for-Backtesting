import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the momentum daily strategy.

    Parameters:
    prices (pd.DataFrame): Prices of the assets.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute compounded 63-day return
    lookback_returns = returns.rolling(spec['params']['lookback']).apply(lambda x: (1 + x).prod() - 1)

    # Rank compounded returns in descending order
    ranked_returns = lookback_returns.rank(axis=1, method='min', ascending=False)

    # Select top k assets
    top_k_assets = ranked_returns.apply(lambda x: x <= spec['params']['top_k'])

    # Smooth signals over the holding period
    smoothed_signals = top_k_assets.rolling(spec['params']['holding_period']).mean()

    # Normalize weights
    weights = normalize_weights(smoothed_signals, max_leverage=spec['max_leverage'])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, rf=0.0, periods=252)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1

    # Create diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
