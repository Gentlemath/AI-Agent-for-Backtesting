import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a pair trading strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): DataFrame of stock prices.
    spec (dict): Strategy specification.

    Returns:
    dict: Dictionary containing diagnostics, including returns and turnover.
    """

    # Extract parameters from the spec
    lookback = spec['params']['lookback']
    mode = spec['params']['mode']
    correlation_threshhold = spec['params']['correlation_threshhold']
    entry_z = spec['params']['entry_z']
    exit_z = spec['params']['exit_z']
    max_leverage = spec['max_leverage']
    frequency = spec['frequency']

    # Calculate daily returns
    daily_returns = pct_returns(prices)

    # Resample to weekly returns if frequency is weekly
    if frequency == 'weekly':
        weekly_returns = daily_returns.resample('W').last()
        weekly_returns = weekly_returns.pct_change().fillna(0)
        returns = weekly_returns
        periods = 52
    else:
        returns = daily_returns
        periods = 252

    # Calculate correlation between stocks
    correlation = prices.pct_change().corr()

    # Check if correlation is above the threshold
    if correlation.values[0][1] < correlation_threshhold:
        print("Correlation is below the threshold, not trading.")
        return {
            "returns": pd.Series(),
            "turnover": 0.0,
            "ann_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0
        }

    # Calculate spread
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]

    # Calculate z-score of the spread
    spread_z = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    # Create entry and exit signals
    entry_signal = (spread_z > entry_z).astype(int)
    exit_signal = (spread_z < exit_z).astype(int)

    # Smooth signals over the holding period
    smoothed_entry_signal = entry_signal.rolling(lookback).mean()
    smoothed_exit_signal = exit_signal.rolling(lookback).mean()

    # Create positions
    positions = smoothed_entry_signal - smoothed_exit_signal

    # Normalize positions
    normalized_positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_positions)

    # Compute returns
    strategy_returns = (returns * normalized_positions).sum(axis=1)

    # Compute annualized return
    ann_return = strategy_returns.mean() * periods

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(strategy_returns, periods=periods)

    # Compute maximum drawdown
    max_dd = max_drawdown(strategy_returns)

    # Return diagnostics
    return {
        "returns": strategy_returns,
        "turnover": turnover,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd
    }
