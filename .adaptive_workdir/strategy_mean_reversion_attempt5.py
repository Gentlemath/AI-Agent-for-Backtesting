import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from the spec
    lookback = spec['params']['lookback']
    z_entry = spec['params']['z_entry']
    z_exit = spec['params']['z_exit']
    max_leverage = spec['max_leverage']
    frequency = spec['frequency']

    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly if necessary
    if frequency == 'weekly':
        returns = returns.resample('W').last()

    # Compute rolling returns
    rolling_returns = returns.rolling(lookback).mean()

    # Compute z-scores
    z_scores = (rolling_returns - rolling_returns.mean()) / rolling_returns.std()

    # Create long and short masks
    long_mask = (z_scores <= -z_entry) & (z_scores.abs() >= z_exit)
    short_mask = (z_scores >= z_entry) & (z_scores.abs() >= z_exit)

    # Create positions
    positions = pd.DataFrame(index=returns.index, columns=returns.columns)
    positions = positions.where(long_mask, 1).where(short_mask, -1)

    # Smooth positions over the holding period
    positions = positions.rolling(lookback).mean()

    # Normalize weights
    weights = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == 'daily' else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == 'daily' else 52)

    # Create diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
