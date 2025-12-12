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

    # Compute moving averages
    fast_ma = prices.rolling(window=fast).mean()
    slow_ma = prices.rolling(window=slow).mean()

    # Compute the risk-on indicator
    risk_on = (fast_ma > slow_ma).astype(int)

    # Smooth the risk-on indicator
    smoothed_risk_on = risk_on.rolling(window=smoothing_window).mean()

    # Shift the smoothed risk-on indicator to avoid lookahead bias
    smoothed_risk_on = smoothed_risk_on.shift(1)

    # Normalize the weights
    weights = normalize_weights(smoothed_risk_on, max_leverage=max_leverage)

    # Compute returns
    returns = pct_returns(prices) * weights

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute diagnostics
    diagnostics = {
        "returns": returns.sum(axis=1),
        "turnover": turnover,
        "ann_return": returns.sum(axis=1).mean() * 252,
        "sharpe": sharpe_ratio(returns.sum(axis=1), periods=252),
        "max_dd": max_drawdown(returns.sum(axis=1).cumsum())
    }

    return diagnostics
