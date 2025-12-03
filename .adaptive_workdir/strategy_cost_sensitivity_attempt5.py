import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Define the holding period based on the frequency
    if spec["frequency"] == "weekly":
        holding_period = 5
        periods = 52
    else:
        holding_period = 1
        periods = 252

    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly if necessary
    if spec["frequency"] == "weekly":
        returns = returns.resample("W").last()

    # Define the cost grid
    costs_bps_grid = spec["params"]["costs_bps_grid"]

    # Initialize diagnostics
    diagnostics = {}

    # Iterate over the cost grid
    for costs_bps in costs_bps_grid:
        # Compute the signal
        signal = returns.mean() / (costs_bps / 10000)

        # Smooth the signal over the holding period
        signal_smooth = signal.rolling(holding_period).mean()

        # Normalize the weights
        weights = normalize_weights(signal_smooth, max_leverage=spec["max_leverage"])

        # Compute the turnover
        turnover = compute_turnover(weights)

        # Compute the portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Compute the Sharpe ratio
        sharpe = sharpe_ratio(portfolio_returns, periods=periods)

        # Compute the maximum drawdown
        max_dd = max_drawdown(portfolio_returns)

        # Store the diagnostics
        diagnostics[f"costs_bps_{costs_bps}"] = {
            "returns": portfolio_returns,
            "turnover": turnover,
            "sharpe": sharpe,
            "max_dd": max_dd,
        }

    return diagnostics
