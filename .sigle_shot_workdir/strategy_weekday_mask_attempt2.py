import pandas as pd
from datetime import datetime

from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover


def run_strategy(prices: pd.DataFrame, spec: dict):
    """
    Execute the weekday mask strategy defined in the provided spec.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price data indexed by date with asset symbols as columns.
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
    # 1. Basic preparation
    # ------------------------------------------------------------------
    universe = spec.get("universe", [])
    if not universe:
        raise ValueError("Strategy spec must define a non‑empty universe.")

    # Filter price data to the universe and to the date range
    start_date = pd.to_datetime(spec["start_date"])
    end_date = pd.to_datetime(spec["end_date"])
    price_data = prices.loc[start_date:end_date, universe].copy()

    # Ensure the index is sorted
    price_data = price_data.sort_index()

    # ------------------------------------------------------------------
    # 2. Compute asset returns
    # ------------------------------------------------------------------
    asset_returns = pct_returns(price_data)

    # ------------------------------------------------------------------
    # 3. Build weekday mask (no smoothing, equal weighting on allowed days)
    # ------------------------------------------------------------------
    allowed_weekdays = set(spec.get("params", {}).get("allowed_weekdays", []))
    if not allowed_weekdays:
        raise ValueError("allowed_weekdays must be provided in spec.params")

    # Weekday series: Monday=0, Sunday=6
    weekdays = asset_returns.index.to_series().dt.weekday

    # Boolean mask where True = allowed trading day
    day_mask = weekdays.isin(allowed_weekdays)

    # Expand mask to DataFrame matching assets
    mask_df = pd.DataFrame(
        data=day_mask.astype(float).values[:, None],
        index=asset_returns.index,
        columns=universe,
    )

    # Equal weight across all assets on allowed days
    raw_weights = mask_df * (1.0 / len(universe))

    # ------------------------------------------------------------------
    # 4. Normalize weights respecting max leverage
    # ------------------------------------------------------------------
    max_leverage = spec.get("max_leverage", 1.0)
    weights = normalize_weights(raw_weights, max_leverage=max_leverage)

    # ------------------------------------------------------------------
    # 5. Compute portfolio returns (shifted by one period to avoid look‑ahead)
    # ------------------------------------------------------------------
    shifted_weights = weights.shift(1).fillna(0.0)
    portfolio_returns = (shifted_weights * asset_returns).sum(axis=1)

    # ------------------------------------------------------------------
    # 6. Turnover calculation (using the normalized, unshifted weights)
    # ------------------------------------------------------------------
    turnover = compute_turnover(weights)

    # ------------------------------------------------------------------
    # 7. Performance diagnostics
    # ------------------------------------------------------------------
    # Annualized return (simple mean * periods)
    periods_per_year = 252 if spec.get("frequency", "daily") == "daily" else 52
    ann_return = portfolio_returns.mean() * periods_per_year

    # Sharpe ratio
    sharpe = sharpe_ratio(portfolio_returns, rf=0.0, periods=periods_per_year)

    # Maximum drawdown (computed on equity curve)
    equity_curve = (1.0 + portfolio_returns).cumprod()
    max_dd = max_drawdown(equity_curve)

    diagnostics = {
        "returns": portfolio_returns,
        "turnover": float(turnover),
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

    return diagnostics
