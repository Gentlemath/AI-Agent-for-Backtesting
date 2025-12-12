import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover


def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    """
    Execute the weekday mask strategy defined by ``spec`` on the provided ``prices`` DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data indexed by date with asset symbols as columns.
    spec : dict
        Strategy specification adhering to the StrategySpec schema.

    Returns
    -------
    dict
        Diagnostics containing at least:
        - "returns": pd.Series of daily portfolio returns
        - "turnover": float average daily turnover
        - "ann_return": float annualized return
        - "sharpe": float annualized Sharpe ratio
        - "max_dd": float maximum drawdown
    """
    # ------------------------------------------------------------------
    # 1. Prepare universe and price data
    # ------------------------------------------------------------------
    universe = spec.get("universe", list(prices.columns))
    prices = prices[universe].copy()

    # ------------------------------------------------------------------
    # 2. Compute asset returns (daily, as spec.frequency == "daily")
    # ------------------------------------------------------------------
    asset_returns = pct_returns(prices)  # already fills NaNs with 0.0

    # ------------------------------------------------------------------
    # 3. Build weekday mask (no smoothing, as per signal description)
    # ------------------------------------------------------------------
    allowed_weekdays = set(spec.get("params", {}).get("allowed_weekdays", []))
    weekdays = asset_returns.index.to_series().dt.weekday  # Monday=0

    # start with a DataFrame of ones and mask out disallowed days
    raw_weights = pd.DataFrame(1.0, index=asset_returns.index, columns=asset_returns.columns)
    raw_weights = raw_weights.where(weekdays.isin(allowed_weekdays), other=0.0)

    # ------------------------------------------------------------------
    # 4. Normalize weights respecting max leverage
    # ------------------------------------------------------------------
    max_leverage = spec.get("max_leverage", 1.0)
    weights = normalize_weights(raw_weights, max_leverage=max_leverage)

    # ------------------------------------------------------------------
    # 5. Compute turnover on the normalized weights
    # ------------------------------------------------------------------
    turnover = compute_turnover(weights)

    # ------------------------------------------------------------------
    # 6. Portfolio returns (weights applied to same‑day returns, no shift)
    # ------------------------------------------------------------------
    portfolio_returns = (weights * asset_returns).sum(axis=1)
    portfolio_returns = portfolio_returns.fillna(0.0)

    # ------------------------------------------------------------------
    # 7. Compute diagnostics
    # ------------------------------------------------------------------
    # Annualized return (simple mean * periods per year)
    periods_per_year = 252 if spec.get("frequency", "daily") == "daily" else 52
    ann_return = portfolio_returns.mean() * periods_per_year

    # Sharpe ratio (annualized)
    sharpe = sharpe_ratio(portfolio_returns, rf=0.0, periods=periods_per_year)

    # Maximum drawdown
    max_dd = max_drawdown(portfolio_returns)

    diagnostics = {
        "returns": portfolio_returns,
        "turnover": float(turnover),
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

    return diagnostics
