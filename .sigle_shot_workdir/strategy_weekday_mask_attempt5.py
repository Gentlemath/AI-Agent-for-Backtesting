import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): Prices of the assets.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """
    # Extract parameters from the spec
    allowed_weekdays = spec["params"]["allowed_weekdays"]
    max_leverage = spec["max_leverage"]
    frequency = spec["frequency"]

    # Compute daily returns
    daily_returns = pct_returns(prices)

    # Create a weekday mask
    weekday_mask = daily_returns.index.to_series().dt.weekday.isin(allowed_weekdays)

    # Replicate the weekday mask to a DataFrame aligned with asset number
    weekday_mask_df = weekday_mask.values[:, None] * pd.ones((len(daily_returns), len(daily_returns.columns)))

    # Apply the weekday mask to the returns
    masked_returns = daily_returns.where(weekday_mask_df, 0.0)

    # Normalize weights
    weights = normalize_weights(masked_returns, max_leverage)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * masked_returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if frequency == "daily" else 52)

    # Compute maximum drawdown
    max_drawdown_value = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if frequency == "daily" else 52)

    # Create diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_drawdown_value,
        "ann_return": ann_return,
    }

    return diagnostics
