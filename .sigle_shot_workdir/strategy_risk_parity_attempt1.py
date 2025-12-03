import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a risk parity strategy based on the provided prices and strategy specification.

    Args:
    - prices (pd.DataFrame): A DataFrame of asset prices.
    - spec (dict): A dictionary containing the strategy specification.

    Returns:
    - diagnostics (dict): A dictionary containing the strategy's diagnostics, including returns and turnover.
    """

    # Resample prices to weekly closes if the strategy frequency is weekly
    if spec["frequency"] == "weekly":
        prices = prices.resample("W").last()

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute volatility
    volatility = returns.rolling(spec["params"]["lookback"]).std()

    # Compute inverse volatility weights
    weights = 1 / volatility

    # Smooth weights over the holding period
    weights = weights.rolling(spec["params"]["lookback"]).mean()

    # Normalize weights to ensure unit gross leverage
    weights = normalize_weights(weights, spec["max_leverage"])

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec["frequency"] == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec["frequency"] == "daily" else 52)

    # Create diagnostics dictionary
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return
    }

    return diagnostics
