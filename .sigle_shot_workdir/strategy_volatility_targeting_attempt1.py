import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the specified strategy on the given prices.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute realized volatility
    realized_vol = returns.rolling(spec['params']['lookback']).std()

    # Compute target volatility
    target_vol = spec['params']['target_vol']

    # Compute signal
    signal = target_vol / realized_vol

    # Smooth signal over the holding period
    smoothed_signal = signal.rolling(spec['params']['lookback']).mean()

    # Normalize weights
    weights = normalize_weights(smoothed_signal, max_leverage=spec['max_leverage'])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec['frequency'] == 'daily' else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec['frequency'] == 'daily' else 52)

    # Create diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
