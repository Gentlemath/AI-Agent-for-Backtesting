import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Compute daily returns
    daily_returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if spec["frequency"] == "weekly":
        weekly_returns = daily_returns.resample("W").last()
        periods = 52
    else:
        weekly_returns = daily_returns
        periods = 252

    # Compute base weights
    base_weights = pd.DataFrame(1 / len(spec["universe"]), index=spec["universe"], columns=weekly_returns.index)

    # Smooth base weights across holding period
    holding_period = 5  # Assuming a holding period of 5 days
    smoothed_weights = base_weights.rolling(holding_period).mean()

    # Normalize weights
    normalized_weights = normalize_weights(smoothed_weights, max_leverage=spec["max_leverage"])

    # Compute turnover
    turnover = compute_turnover(normalized_weights)

    # Compute returns
    returns = (weekly_returns * normalized_weights).sum(axis=0)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(returns, periods=periods)

    # Compute maximum drawdown
    max_dd = max_drawdown(returns)

    # Compute annualized return
    ann_return = returns.mean() * periods

    # Compute cost elasticity
    costs_bps_grid = spec["params"]["costs_bps_grid"]
    cost_elasticity = []
    for costs_bps in costs_bps_grid:
        returns_with_costs = returns - (weekly_returns * normalized_weights * costs_bps / 10000).sum(axis=0)
        sharpe_with_costs = sharpe_ratio(returns_with_costs, periods=periods)
        cost_elasticity.append(sharpe_with_costs / sharpe)

    # Create diagnostics dictionary
    diagnostics = {
        "returns": returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
        "cost_elasticity": cost_elasticity
    }

    return diagnostics
