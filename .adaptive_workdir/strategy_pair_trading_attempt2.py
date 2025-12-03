import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from spec
    lookback = spec['params']['lookback']
    mode = spec['params']['mode']
    correlation_threshhold = spec['params']['correlation_threshhold']
    entry_z = spec['params']['entry_z']
    exit_z = spec['params']['exit_z']
    max_leverage = spec['max_leverage']
    costs_bps = spec['costs_bps']
    frequency = spec['frequency']

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute correlation between stocks
    correlation = prices.pct_change().corr()

    # Check if correlation is above threshold
    if correlation.iloc[0, 1] < correlation_threshhold:
        print("Correlation is below threshold, not trading.")
        return {
            "returns": pd.Series(index=prices.index, dtype=float),
            "turnover": 0.0,
            "ann_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0
        }

    # Compute spread
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]

    # Compute z-score of spread
    z_score = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    # Create signals
    entry_signal = (z_score > entry_z).astype(int)
    exit_signal = (z_score < exit_z).astype(int)

    # Smooth signals over holding period
    holding_period = lookback
    entry_signal_smooth = entry_signal.rolling(holding_period).mean()
    exit_signal_smooth = exit_signal.rolling(holding_period).mean()

    # Create positions
    positions = pd.DataFrame(index=prices.index, columns=prices.columns)
    positions.iloc[:, 0] = entry_signal_smooth - exit_signal_smooth
    positions.iloc[:, 1] = -positions.iloc[:, 0]

    # Normalize positions
    positions_normalized = normalize_weights(positions, max_leverage=max_leverage)

    # Compute turnover
    turnover = compute_turnover(positions_normalized)

    # Compute portfolio returns
    portfolio_returns = (positions_normalized * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == 'daily' else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == 'daily' else 52)

    # Create diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

    return diagnostics
