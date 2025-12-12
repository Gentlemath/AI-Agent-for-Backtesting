import pandas as pd
from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover


def run_strategy(prices: pd.DataFrame, spec: dict):
    """
    Execute a weekday mask strategy.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data indexed by date with columns as assets.
    spec : dict
        Strategy specification adhering to the StrategySpec schema.

    Returns
    -------
    dict
        Diagnostics containing at least:
        - "returns": pd.Series of portfolio daily returns
        - "turnover": float average daily turnover
        Additional metrics (annual return, sharpe, max drawdown) are also provided.
    """
    # ------------------------------------------------------------------
    # 1. Basic preparation: universe, date range, frequency handling
    # ------------------------------------------------------------------
    universe = spec.get("universe", [])
    if universe:
        prices = prices[universe]

    start_date = pd.to_datetime(spec["start_date"])
    end_date = pd.to_datetime(spec["end_date"])
    prices = prices.loc[start_date:end_date]

    # Frequency handling (spec is daily; keep as‑is)
    if spec.get("frequency") == "weekly":
        # Resample to weekly close (Friday) and forward‑fill missing days
        prices = prices.resample("W-FRI").last().ffill()

    # ------------------------------------------------------------------
    # 2. Build raw (un‑normalized) weight matrix based on weekday mask
    # ------------------------------------------------------------------
    allowed_weekdays = set(spec.get("params", {}).get("allowed_weekdays", []))
    weekdays = prices.index.to_series().dt.weekday
    mask_series = weekdays.isin(allowed_weekdays).astype(float)  # 1.0 on allowed days, 0.0 otherwise

    # Replicate mask across all assets
    raw_weights = pd.DataFrame(
        mask_series.values[:, None],
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # 3. Normalize weights respecting max leverage
    # ------------------------------------------------------------------
    max_leverage = spec.get("max_leverage", 1.0)
    weights = normalize_weights(raw_weights, max_leverage=max_leverage)

    # ------------------------------------------------------------------
    # 4. Compute asset returns and portfolio returns
    # ------------------------------------------------------------------
    asset_returns = pct_returns(prices)  # already fills NaNs with 0.0
    portfolio_returns = (weights * asset_returns).sum(axis=1)

    # ------------------------------------------------------------------
    # 5. Turnover calculation (using normalized weights)
    # ------------------------------------------------------------------
    turnover = compute_turnover(weights)

    # ------------------------------------------------------------------
    # 6. Additional diagnostics
    # ------------------------------------------------------------------
    # Annualized return (geometric)
    cumulative = (1.0 + portfolio_returns).cumprod()
    total_periods = len(portfolio_returns)
    ann_return = (cumulative.iloc[-1] ** (252.0 / total_periods) - 1.0) if total_periods > 0 else 0.0

    # Sharpe ratio (annualized, risk‑free rate = 0)
    sharpe = sharpe_ratio(portfolio_returns, rf=0.0, periods=252)

    # Maximum drawdown (apply to cumulative return series)
    max_dd = max_drawdown(cumulative - 1.0)

    # ------------------------------------------------------------------
    # 7. Assemble diagnostics dictionary
    # ------------------------------------------------------------------
    diagnostics = {
        "returns": portfolio_returns,
        "turnover": float(turnover),
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
    }

    return diagnostics
