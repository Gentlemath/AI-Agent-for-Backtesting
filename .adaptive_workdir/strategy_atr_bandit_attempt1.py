import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run the ATR-driven stop/take-profit allocation bandit strategy.

    Parameters:
    prices (pd.DataFrame): Daily close prices for the universe of assets.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Compute daily returns
    returns = pct_returns(prices)

    # Resample to weekly if frequency is weekly
    if spec["frequency"] == "weekly":
        returns = returns.resample("W").last()

    # Compute ATR
    atr_window = spec["params"]["atr_window"]
    atr = returns.rolling(atr_window).std() * (252 ** 0.5)

    # Compute signals
    risk_budget = spec["params"]["risk_budget"]
    tp_mult = spec["params"]["tp_mult"]
    sl_mult = spec["params"]["sl_mult"]
    signals = (atr * risk_budget) / (atr * tp_mult + atr * sl_mult)

    # Smooth signals over the holding period
    signals_smooth = signals.rolling(atr_window).mean()

    # Normalize weights
    weights = normalize_weights(signals_smooth, max_leverage=spec["max_leverage"])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * returns).sum(axis=1)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec["frequency"] == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec["frequency"] == "daily" else 52)

    # Store diagnostics
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": turnover,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "ann_return": ann_return,
    }

    return diagnostics
