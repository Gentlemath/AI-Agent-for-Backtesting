import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a trading strategy based on the provided specification.

    Parameters:
    prices (pd.DataFrame): A DataFrame of asset prices.
    spec (dict): A dictionary containing the strategy specification.

    Returns:
    dict: A dictionary containing the strategy diagnostics.
    """

    # Extract parameters from the strategy specification
    universe = spec["universe"]
    frequency = spec["frequency"]
    fast_ma = spec["params"]["fast"]
    slow_ma = spec["params"]["slow"]
    max_leverage = spec["max_leverage"]
    costs_bps = spec["costs_bps"]

    # Resample prices to the specified frequency
    if frequency == "weekly":
        prices = prices.resample("W").last()

    # Calculate moving averages
    fast_ma_values = prices.rolling(window=fast_ma).mean()
    slow_ma_values = prices.rolling(window=slow_ma).mean()

    # Create a signal based on the moving average crossover
    signal = (fast_ma_values > slow_ma_values).astype(int)

    # Smooth the signal over the holding period
    holding_period = max(fast_ma, slow_ma)
    smoothed_signal = signal.rolling(window=holding_period).mean()

    # Normalize the weights
    weights = normalize_weights(smoothed_signal, max_leverage=max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Calculate returns
    returns = pct_returns(prices) * weights

    # Calculate portfolio returns
    portfolio_returns = returns.sum(axis=1)

    # Calculate Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Calculate maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Calculate annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return
    }

    return diagnostics
