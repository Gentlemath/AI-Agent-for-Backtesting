import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Compute daily returns
    returns = pct_returns(prices)

    # Create a weekday mask
    weekday_mask = returns.index.to_series().dt.weekday.isin(spec['params']['allowed_weekdays']).values[:, None]
    weekday_mask = pd.DataFrame(weekday_mask, index=returns.index, columns=returns.columns)

    # Apply the weekday mask to the returns
    masked_returns = returns.where(weekday_mask, 0.0)

    # Normalize weights
    weights = normalize_weights(masked_returns, max_leverage=spec['max_leverage'])

    # Compute turnover
    turnover = compute_turnover(weights)

    # Compute portfolio returns
    portfolio_returns = (weights * masked_returns).sum(axis=1)

    # Compute diagnostics
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'ann_return': portfolio_returns.mean() * 252,
        'sharpe': sharpe_ratio(portfolio_returns, periods=252),
        'max_dd': max_drawdown(portfolio_returns)
    }

    return diagnostics
