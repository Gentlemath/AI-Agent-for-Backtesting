import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a breakout strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """
    # Extract parameters from the spec
    window = spec["params"]["window"]
    stop_window = spec["params"]["stop_window"]
    max_leverage = spec["max_leverage"]
    costs_bps = spec["costs_bps"]
    frequency = spec["frequency"]

    # Resample prices to weekly closes if frequency is weekly
    if frequency == "weekly":
        prices = prices.resample("W").last()

    # Compute rolling high and low
    rolling_high = prices.rolling(window).max().shift(1)
    rolling_low = prices.rolling(stop_window).min().shift(1)

    # Create a signal for entry and exit
    entry_signal = (prices > rolling_high).astype(int)
    exit_signal = (prices < rolling_low).astype(int)

    # Smooth the signals over the holding period
    smoothed_entry_signal = entry_signal.rolling(window).mean()
    smoothed_exit_signal = exit_signal.rolling(window).mean()

    # Create a mask for long positions
    long_mask = (smoothed_entry_signal > 0.5) & (smoothed_exit_signal < 0.5)

    # Normalize weights
    weights = normalize_weights(long_mask, max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute returns
    returns = pct_returns(prices) * weights

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
