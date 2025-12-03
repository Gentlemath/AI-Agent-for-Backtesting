import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Run a pair trading strategy based on the provided spec.

    Parameters:
    prices (pd.DataFrame): Prices of the assets in the universe.
    spec (dict): Strategy specification.

    Returns:
    dict: Diagnostics containing returns, turnover, and other metrics.
    """

    # Resample prices to weekly closes if frequency is weekly
    if spec["frequency"] == "weekly":
        prices = prices.resample("W").last()

    # Compute daily returns
    returns = pct_returns(prices)

    # Compute correlation between assets
    correlation = prices.pct_change().corr()

    # Check if correlation is above threshold
    if correlation.iloc[0, 1] < spec["params"]["correlation_threshhold"]:
        print("Correlation is below threshold, not trading.")
        return {
            "returns": pd.Series(index=prices.index),
            "turnover": 0.0,
            "ann_return": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
        }

    # Compute spread
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]

    # Compute z-score of spread
    z_score = (spread - spread.rolling(spec["params"]["lookback"]).mean()) / spread.rolling(spec["params"]["lookback"]).std()

    # Create entry and exit signals
    entry_signal = (z_score > spec["params"]["entry_z"]).astype(int)
    exit_signal = (z_score < spec["params"]["exit_z"]).astype(int)

    # Smooth signals over the holding period
    entry_signal = entry_signal.rolling(spec["params"]["lookback"]).mean()
    exit_signal = exit_signal.rolling(spec["params"]["lookback"]).mean()

    # Create positions
    positions = pd.DataFrame(index=prices.index, columns=prices.columns)
    positions.iloc[:, 0] = entry_signal - exit_signal
    positions.iloc[:, 1] = -positions.iloc[:, 0]

    # Normalize positions
    positions = normalize_weights(positions, max_leverage=spec["max_leverage"])

    # Compute turnover
    turnover = compute_turnover(positions)

    # Compute portfolio returns
    portfolio_returns = (returns * positions).sum(axis=1)

    # Compute annualized return
    ann_return = portfolio_returns.mean() * (252 if spec["frequency"] == "daily" else 52)

    # Compute Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, periods=252 if spec["frequency"] == "daily" else 52)

    # Compute maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    # Return diagnostics
    return {
        "returns": portfolio_returns,
        "turnover": turnover,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }
