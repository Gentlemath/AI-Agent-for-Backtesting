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
    frequency = spec['frequency']

    # Resample prices to weekly closes if frequency is weekly
    if frequency == 'weekly':
        prices = prices.resample('W').last()

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute rolling returns
    rolling_returns = returns.rolling(window=lookback).mean()

    # Compute z-scores
    z_scores = (rolling_returns - rolling_returns.mean()) / rolling_returns.std()

    # Create long and short masks
    long_mask = (z_scores <= -z_entry) & (z_scores.abs() >= z_exit)
    short_mask = (z_scores >= z_entry) & (z_scores.abs() >= z_exit)

    # Create positions
    positions = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    positions.where(long_mask, 1, inplace=True)
    positions.where(short_mask, -1, inplace=True)

    # Smooth positions over the holding period
    positions = positions.rolling(window=lookback).mean()

    # Normalize positions
    positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(positions)

    # Compute portfolio returns
    portfolio_returns = (positions * returns).sum(axis=1)

    # Compute diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'ann_return': portfolio_returns.mean() * (252 if frequency == 'daily' else 52),
        'sharpe': sharpe_ratio(portfolio_returns, periods=(252 if frequency == 'daily' else 52)),
        'max_dd': max_drawdown(portfolio_returns)
    }

    return diagnostics
