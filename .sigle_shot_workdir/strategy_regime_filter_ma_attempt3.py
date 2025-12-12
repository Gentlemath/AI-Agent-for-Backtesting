import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from the spec
    fast = spec['params']['fast']
    slow = spec['params']['slow']
    smoothing_window = spec['params']['smoothing_window']
    max_leverage = spec['max_leverage']
    costs_bps = spec['costs_bps']
    frequency = spec['frequency']

    # Compute moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Create a binary risk-on indicator
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
        returns = pct_returns(prices) * weights.shift(1)
        periods = 252
    elif frequency == 'weekly':
        prices_weekly = prices.resample('W').last()
        returns_weekly = pct_returns(prices_weekly) * weights.shift(1).resample('W').last()
        returns = returns_weekly.resample('D').ffill()
        periods = 52

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'ann_return': portfolio_returns.mean() * periods,
        'sharpe': sharpe_ratio(portfolio_returns, periods=periods),
        'max_dd': max_drawdown(portfolio_returns)
    }

    return diagnostics
