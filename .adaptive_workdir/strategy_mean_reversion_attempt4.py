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
    daily_returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if frequency == 'weekly':
        weekly_returns = daily_returns.resample('W').last()
        weekly_prices = prices.resample('W').last()
    else:
        weekly_returns = daily_returns
        weekly_prices = prices

    # Compute z-scores
    z_scores = (weekly_returns.rolling(lookback).mean() - weekly_returns.rolling(lookback).mean().mean(axis=1)) / weekly_returns.rolling(lookback).std()
    z_scores = z_scores.fillna(0)

    # Create signals
    long_signals = (z_scores <= -z_entry).astype(int)
    short_signals = (z_scores >= z_entry).astype(int)

    # Smooth signals across holding period
    holding_period = lookback
    long_signals_smooth = long_signals.rolling(holding_period).mean()
    short_signals_smooth = short_signals.rolling(holding_period).mean()

    # Create positions
    positions = long_signals_smooth - short_signals_smooth
    positions = positions.fillna(0)

    # Normalize positions
    normalized_positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_positions)

    # Compute returns
    strategy_returns = (normalized_positions * weekly_returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(strategy_returns, periods=52 if frequency == 'weekly' else 252)

    # Compute maximum drawdown
    max_dd = max_drawdown(strategy_returns)

    # Compute annualized return
    ann_return = strategy_returns.mean() * (52 if frequency == 'weekly' else 252)

    # Create diagnostics
    diagnostics = {
        'returns': strategy_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
