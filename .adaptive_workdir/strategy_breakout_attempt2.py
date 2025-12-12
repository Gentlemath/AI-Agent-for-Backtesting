import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    window = spec["params"]["window"]
    stop_window = spec["params"]["stop_window"]
    max_leverage = spec["max_leverage"]
    costs_bps = spec["costs_bps"]

    # Compute rolling high and low
    rolling_high = prices.rolling(window).max().shift(1)
    rolling_low = prices.rolling(stop_window).min().shift(1)

    # Create entry and exit signals
    entry_signal = (prices > rolling_high).astype(int)
    exit_signal = (prices < rolling_low).astype(int)

    # Smooth signals across the holding period
    smoothed_entry_signal = entry_signal.rolling(window).mean()
    smoothed_exit_signal = exit_signal.rolling(stop_window).mean()

    # Create positions
    positions = smoothed_entry_signal - smoothed_exit_signal
    positions = positions.where(positions > 0, 0)

    # Normalize weights
    normalized_weights = normalize_weights(positions, max_leverage)

    # Compute turnover
    turnover = compute_turnover(normalized_weights)

    # Compute returns
    returns = pct_returns(prices) * normalized_weights

    # Compute portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * 252

    # Create diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return
    }

    return diagnostics
