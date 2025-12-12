import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    # Compute daily returns
    daily_returns = pct_returns(prices)

    # Create a weekday mask
    weekdays = daily_returns.index.to_series().dt.weekday
    weekday_mask = weekdays.isin(spec['params']['allowed_weekdays'])

    # Replicate the weekday mask to a DataFrame with column number = asset number
    weekday_mask_df = weekday_mask.values[:, None] * pd.ones((len(weekday_mask), len(spec['universe'])))

    # Apply the weekday mask to the returns
    masked_returns = daily_returns.where(weekday_mask_df)

    # Compute the portfolio returns
    portfolio_returns = (masked_returns * (1 / len(spec['universe']))).sum(axis=1)

    # Compute the portfolio weights
    weights = pd.DataFrame(weekday_mask_df, index=daily_returns.index, columns=daily_returns.columns)
    weights = normalize_weights(weights, max_leverage=spec['max_leverage'])

    # Compute the turnover
    turnover = compute_turnover(weights)

    # Compute the Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252)

    # Compute the maximum drawdown
    max_drawdown_value = max_drawdown(portfolio_returns)

    # Compute the annualized return
    ann_return = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1

    # Create the diagnostics dictionary
    diagnostics = {
        'returns': portfolio_returns,
        'turnover': turnover,
        'sharpe': sharpe,
        'max_dd': max_drawdown_value,
        'ann_return': ann_return
    }

    return diagnostics
