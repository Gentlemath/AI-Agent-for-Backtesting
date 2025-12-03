import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if spec['frequency'] == 'weekly':
        returns = returns.resample('W').last()

    # Compute signal (base weights)
    signal = pd.DataFrame(index=returns.index, columns=returns.columns, data=1.0 / len(returns.columns))

    # Smooth signal across holding period (1 day for daily, 7 days for weekly)
    holding_period = 1 if spec['frequency'] == 'daily' else 7
    signal_smoothed = signal.rolling(window=holding_period, min_periods=1).mean()

    # Normalize weights
    weights = normalize_weights(signal_smoothed, max_leverage=spec['max_leverage'])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec['frequency'] == 'daily' else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute cost elasticity
    costs_bps_grid = spec['params']['costs_bps_grid']
    cost_elasticity = 0.0
    for costs_bps in costs_bps_grid:
        portfolio_returns_cost = portfolio_returns - (weights.abs().sum(axis=1) * costs_bps / 10000)
        sharpe_cost = sharpe_ratio(portfolio_returns_cost, periods=252 if spec['frequency'] == 'daily' else 52)
        cost_elasticity += (sharpe - sharpe_cost) / costs_bps

    # Create diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'cost_elasticity': cost_elasticity
    }

    return diagnostics
