import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the regime filter moving average strategy.

    Parameters:
    prices (pd.DataFrame): DataFrame of asset prices.
    spec (dict): Strategy specification.

    Returns:
    dict: Dictionary containing diagnostics.
    """

    # Extract parameters from the spec
    fast = spec['params']['fast']
    slow = spec['params']['slow']
    smoothing_window = spec['params']['smoothing_window']
    max_leverage = spec['max_leverage']
    frequency = spec['frequency']

    # Compute moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Compute binary risk-on indicator
    risk_on = (fast_ma > slow_ma).astype(int)

    # Smooth the risk-on indicator
    smoothed_risk_on = risk_on.rolling(window=smoothing_window).mean()

    # Shift signals to avoid lookahead bias
    smoothed_risk_on = smoothed_risk_on.shift(1)

    # Normalize weights
    weights = normalize_weights(smoothed_risk_on, max_leverage=max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute returns
    if frequency == 'daily':
        returns = pct_returns(prices) * weights
        periods = 252
    elif frequency == 'weekly':
        returns = pct_returns(prices.resample('W').last()) * weights.resample('W').last()
        periods = 52

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=periods)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * periods

    # Create diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
