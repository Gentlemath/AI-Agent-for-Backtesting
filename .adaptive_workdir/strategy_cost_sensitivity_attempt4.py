import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a cost sensitivity strategy.

    Parameters:
    prices (pd.DataFrame): Asset prices.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, Sharpe ratio, max drawdown, and cost elasticity.
    """

    # Compute daily returns
    returns = pct_returns(prices)

    # Define a holding period (1 day for daily frequency, 7 days for weekly frequency)
    holding_period = 7 if spec['frequency'] == 'weekly' else 1

    # Resample returns to weekly frequency if necessary
    if spec['frequency'] == 'weekly':
        returns = returns.resample('W').last()

    # Compute base weights (equal weights for simplicity)
    base_weights = pd.Series(1.0 / len(spec['universe']), index=spec['universe'])

    # Initialize diagnostics
    diagnostics = {}

    # Iterate over the cost grid
    for cost in spec['params']['costs_bps_grid']:
        # Compute costs as a fraction of the base weights
        costs = base_weights * cost / 10000

        # Smooth costs over the holding period
        smoothed_costs = costs.rolling(holding_period).mean()

        # Normalize weights to ensure unit gross leverage
        weights = normalize_weights(smoothed_costs, max_leverage=spec['max_leverage'])

        # Compute turnover
        turnover = compute_turnover(weights)

        # Compute portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Compute Sharpe ratio
        sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec['frequency'] == 'daily' else 52)

        # Compute max drawdown
        max_dd = max_drawdown(portfolio_returns)

        # Compute cost elasticity (simple example: cost elasticity is the ratio of turnover to cost)
        cost_elasticity = turnover / cost if cost > 0 else 0

        # Store diagnostics
        diagnostics[f'cost_{cost}_returns'] = portfolio_returns
        diagnostics[f'cost_{cost}_turnover'] = turnover
        diagnostics[f'cost_{cost}_sharpe'] = sharpe
        diagnostics[f'cost_{cost}_max_dd'] = max_dd
        diagnostics[f'cost_{cost}_cost_elasticity'] = cost_elasticity

    return diagnostics
