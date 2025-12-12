import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the regime filter moving average strategy.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Extract parameters from the spec
    fast = spec["params"]["fast"]
    slow = spec["params"]["slow"]
    smoothing_window = spec["params"]["smoothing_window"]
    max_leverage = spec["max_leverage"]
    costs_bps = spec["costs_bps"]
    frequency = spec["frequency"]

    # Compute daily returns
    daily_returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if frequency == "weekly":
        prices = prices.resample("W").last()
        daily_returns = daily_returns.resample("W").sum()

    # Compute fast and slow moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Compute binary risk-on indicator
    risk_on = (fast_ma > slow_ma).astype(int)

    # Smooth risk-on indicator
    smoothed_risk_on = risk_on.rolling(window=smoothing_window).mean()

    # Shift signals to avoid lookahead bias
    smoothed_risk_on = smoothed_risk_on.shift(1)

    # Normalize weights
    weights = normalize_weights(smoothed_risk_on, max_leverage=max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * daily_returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Store diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
