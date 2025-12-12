import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Compute daily returns
    returns = pct_returns(prices)

    # Compute z-scores
    lookback = spec['params']['lookback']
    z_entry = spec['params']['z_entry']
    z_exit = spec['params']['z_exit']
    z_scores = (returns.rolling(lookback).mean() - returns.mean()) / returns.rolling(lookback).std()
    z_scores = z_scores.where(z_scores.notnull(), 0)

    # Create long and short masks
    long_mask = (z_scores <= -z_entry)
    short_mask = (z_scores >= z_entry)

    # Smooth signals across holding period
    holding_period = lookback
    long_mask_smooth = long_mask.rolling(holding_period).mean()
    short_mask_smooth = short_mask.rolling(holding_period).mean()

    # Create positions
    positions = long_mask_smooth - short_mask_smooth
    positions = positions.where(positions.notnull(), 0)

    # Normalize positions
    max_leverage = spec['max_leverage']
    positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(positions)

    # Compute portfolio returns
    portfolio_returns = (positions * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * 252

    # Create diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'ann_return': ann_return
    }

    return diagnostics
