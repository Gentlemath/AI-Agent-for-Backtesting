import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from spec
    lookback = spec['params']['lookback']
    z_entry = spec['params']['z_entry']
    z_exit = spec['params']['z_exit']
    max_leverage = spec['max_leverage']
    costs_bps = spec['costs_bps']
    frequency = spec['frequency']

    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if frequency == 'weekly':
        returns = returns.resample('W').last()

    # Compute z-scores
    z_scores = (returns.rolling(lookback).mean() - returns.rolling(lookback).mean().rolling(lookback).mean()) / returns.rolling(lookback).std()
    z_scores = z_scores.fillna(0)

    # Create long and short masks
    long_mask = (z_scores <= -z_entry).where(returns.notnull(), False)
    short_mask = (z_scores >= z_entry).where(returns.notnull(), False)

    # Smooth signals across holding period
    long_mask = long_mask.rolling(lookback).mean()
    short_mask = short_mask.rolling(lookback).mean()

    # Create positions
    positions = long_mask - short_mask

    # Normalize weights
    positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(positions)

    # Compute portfolio returns
    portfolio_returns = (returns * positions.shift(1)).sum(axis=1)

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
