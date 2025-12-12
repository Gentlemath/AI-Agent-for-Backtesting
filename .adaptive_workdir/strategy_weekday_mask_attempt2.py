import pandas as pd
from typing import Dict, Any

from backtester.kb.returns import pct_returns
from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.kb.strategies import normalize_weights, compute_turnover


def run_strategy(prices: pd.DataFrame, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the weekday_mask strategy.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data indexed by date with asset symbols as columns.
    spec : dict
        Strategy specification (see problem description).

    Returns
    -------
    dict
        Diagnostics containing at least:
        - "returns" : pd.Series of daily portfolio returns (after costs)
        - "turnover": float average daily turnover
        - "sharpe"  : float annualized Sharpe ratio
        - "max_dd"  : float maximum drawdown
        - "ann_return": float annualized arithmetic return
    """
    # ------------------------------------------------------------------
    # 1. Prepare price data for the defined universe and date range
    # ------------------------------------------------------------------
    universe = spec.get("universe", [])
    price_df = prices[universe].copy()

    start_date = pd.to_datetime(spec["start_date"])
    end_date = pd.to_datetime(spec["end_date"])
    price_df = price_df.loc[start_date:end_date]

    # ------------------------------------------------------------------
    # 2. Compute asset returns (daily pct change, NaNs filled with 0.0)
    # ------------------------------------------------------------------
    asset_returns = pct_returns(price_df)

    # ------------------------------------------------------------------
    # 3. Build weekday mask (allowed weekdays are known in advance)
    # ------------------------------------------------------------------
    allowed_weekdays = set(spec.get("params", {}).get("allowed_weekdays", []))
    weekdays_series = asset_returns.index.to_series().dt.weekday
    mask_series = weekdays_series.isin(allowed_weekdays).astype(float)

    # Replicate the mask across all assets without using numpy arrays
    mask_df = pd.DataFrame(
        {col: mask_series.copy() for col in universe},
        index=asset_returns.index,
    )

    # ------------------------------------------------------------------
    # 4. (No smoothing required for this strategy – holding period = 1)
    # ------------------------------------------------------------------
    raw_weights = mask_df

    # ------------------------------------------------------------------
    # 5. Normalize weights respecting max leverage
    # ------------------------------------------------------------------
    max_leverage = spec.get("max_leverage", 1.0)
    norm_weights = normalize_weights(raw_weights, max_leverage=max_leverage)

    # ------------------------------------------------------------------
    # 6. Compute turnover (average daily absolute change in weights)
    # ------------------------------------------------------------------
    turnover = compute_turnover(norm_weights)

    # ------------------------------------------------------------------
    # 7. Portfolio returns before costs
    # ------------------------------------------------------------------
    portfolio_ret = (norm_weights * asset_returns).sum(axis=1)

    # ------------------------------------------------------------------
    # 8. Apply transaction costs (costs expressed in basis points)
    # ------------------------------------------------------------------
    costs_bps = spec.get("costs_bps", 0.0)
    cost_per_day = turnover * (costs_bps / 10_000.0)
    net_ret = portfolio_ret - cost_per_day

    # ------------------------------------------------------------------
    # 9. Compute diagnostics
    # ------------------------------------------------------------------
    periods_per_year = 252 if spec.get("frequency", "daily") == "daily" else 52

    ann_return = net_ret.mean() * periods_per_year
    sharpe = sharpe_ratio(net_ret, periods=periods_per_year)
    cum_curve = (1.0 + net_ret).cumprod()
    max_dd = max_drawdown(cum_curve)

    diagnostics = {
        "returns": net_ret,
        "turnover": float(turnover),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "ann_return": float(ann_return),
    }

    return diagnostics
