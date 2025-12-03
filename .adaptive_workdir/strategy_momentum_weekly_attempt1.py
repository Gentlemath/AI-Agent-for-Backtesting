import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Resample prices to weekly closes if frequency is weekly
    if spec["frequency"] == "weekly":
        prices = prices.resample("W").last()
        periods = 52
    else:
        periods = 252

    # Compute weekly returns
    returns = pct_returns(prices)

    # Compute signal (weekly returns ranked over trailing 26 weeks)
    signal = returns.rolling(spec["params"]["lookback"]).mean().rank(axis=1, ascending=False)

    # Create top-k mask
    top_k_mask = signal <= spec["params"]["top_k"]

    # Smooth raw signals across the holding period
    smoothed_signal = top_k_mask.rolling(spec["params"]["holding_period"]).mean()

    # Normalize weights
    weights = normalize_weights(smoothed_signal, max_leverage=spec["max_leverage"])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=periods)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * periods

    # Create diagnostics dict
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return
    }

    return diagnostics
