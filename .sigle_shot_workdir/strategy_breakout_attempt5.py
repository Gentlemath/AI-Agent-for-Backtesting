import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a Donchian-style long breakout strategy with a trailing stop.

    Parameters:
    prices (pd.DataFrame): A DataFrame of prices for the specified universe.
    spec (dict): A dictionary containing the strategy specification.

    Returns:
    dict: A dictionary containing the diagnostics for the strategy.
    """

    # Extract parameters from the spec
    window = spec["params"]["window"]
    stop_window = spec["params"]["stop_window"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]

    # Compute rolling high and low
    rolling_high = prices.rolling(window).max().shift(1)
    rolling_low = prices.rolling(stop_window).min().shift(1)

    # Compute entry and exit signals
    entry_signal = (prices > rolling_high).astype(int)
    exit_signal = (prices < rolling_low).astype(int)

    # Compute positions
    positions = entry_signal - exit_signal
    positions = positions.rolling(window).mean()  # Smooth positions over the holding period

    # Normalize weights
    weights = normalize_weights(positions, max_leverage)

    # Compute returns
    if frequency == "weekly":
        prices_weekly = prices.resample("W").last()
        returns = pct_returns(prices_weekly).mul(weights)
        annualization_factor = 52
    else:
        returns = pct_returns(prices).mul(weights)
        annualization_factor = 252

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute diagnostics
    ann_return = returns.mean() * annualization_factor
    sharpe = sharpe_ratio(returns.mean() * annualization_factor, periods=annualization_factor)
    max_dd = max_drawdown(returns.cumsum())

    # Store diagnostics in a dictionary
    diagnostics = {
        "returns": returns.sum(axis=1),
        "turnover": turnover,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

    return diagnostics
