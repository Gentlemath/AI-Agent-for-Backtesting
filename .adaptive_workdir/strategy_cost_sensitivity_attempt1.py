import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a strategy based on the provided prices and strategy specification.

    Parameters:
    prices (pd.DataFrame): DataFrame of asset prices.
    spec (dict): Strategy specification.

    Returns:
    dict: Dictionary of diagnostics, including returns and turnover.
    """

    # Resample prices if frequency is weekly
    if spec["frequency"] == "weekly":
        prices = prices.resample("W").last()

    # Compute daily returns
    returns = pct_returns(prices)

    # Create a DataFrame of base weights (equal weights for all assets)
    base_weights = pd.DataFrame(1 / len(spec["universe"]), index=prices.index, columns=spec["universe"])

    # Smooth base weights over the holding period (1 day for daily frequency, 7 days for weekly frequency)
    holding_period = 1 if spec["frequency"] == "daily" else 7
    smoothed_weights = base_weights.rolling(holding_period).mean()

    # Normalize weights to ensure unit gross leverage scaled by max_leverage
    normalized_weights = normalize_weights(smoothed_weights, max_leverage=spec["max_leverage"])

    # Compute portfolio returns
    portfolio_returns = (returns * normalized_weights).sum(axis=1)

    # Compute turnover
    turnover = compute_turnover(normalized_weights)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec["frequency"] == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec["frequency"] == "daily" else 52)

    # Compute cost elasticity
    costs_bps_grid = spec["params"]["costs_bps_grid"]
    cost_elasticity = 0
    for cost in costs_bps_grid:
        cost_returns = portfolio_returns - (cost / 10000) * turnover
        cost_sharpe = sharpe_ratio(cost_returns, periods=252 if spec["frequency"] == "daily" else 52)
        cost_elasticity += (cost_sharpe - sharpe) / cost

    # Create diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
        "cost_elasticity": cost_elasticity
    }

    return diagnostics
