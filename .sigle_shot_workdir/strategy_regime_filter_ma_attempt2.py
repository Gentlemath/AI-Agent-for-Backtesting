import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the regime filter moving average strategy.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Compute fast and slow moving averages
    fast_ma = prices.rolling(spec['params']['fast']).mean()
    slow_ma = prices.rolling(spec['params']['slow']).mean()

    # Define a binary risk-on indicator
    risk_on = (fast_ma > slow_ma).astype(int)

    # Smooth the risk-on indicator over the smoothing window
    smoothed_risk_on = risk_on.rolling(spec['params']['smoothing_window']).mean()

    # Shift signals by one period to avoid lookahead bias
    smoothed_risk_on = smoothed_risk_on.shift(1)

    # Normalize weights
    weights = normalize_weights(smoothed_risk_on, max_leverage=spec['max_leverage'])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute returns
    returns = pct_returns(prices) * weights.shift(1)

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec['frequency'] == 'daily' else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec['frequency'] == 'daily' else 52)

    # Store diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
