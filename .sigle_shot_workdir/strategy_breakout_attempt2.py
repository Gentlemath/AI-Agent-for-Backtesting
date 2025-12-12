import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Extract parameters from the spec
    window = spec["params"]["window"]
    stop_window = spec["params"]["stop_window"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]
    start_date = spec["start_date"]
    end_date = spec["end_date"]

    # Resample prices to the specified frequency
    if frequency == "weekly":
        prices = prices.resample("W").last()

    # Compute rolling high and low
    rolling_high = prices.rolling(window).max().shift(1)
    rolling_low = prices.rolling(stop_window).min().shift(1)

    # Generate entry and exit signals
    entry_signal = (prices > rolling_high).astype(int)
    exit_signal = (prices < rolling_low).astype(int)

    # Smooth signals over the holding period
    smoothed_entry_signal = entry_signal.rolling(window).mean()
    smoothed_exit_signal = exit_signal.rolling(window).mean()

    # Compute positions
    positions = smoothed_entry_signal - smoothed_exit_signal

    # Normalize positions
    normalized_positions = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_positions)

    # Compute returns
    returns = pct_returns(prices) * normalized_positions

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "ann_return": portfolio_returns.mean() * (252 if frequency == "daily" else 52),
        "sharpe": sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52),
        "max_dd": max_drawdown(portfolio_returns + 1).cumprod() - 1
    }

    return diagnostics
