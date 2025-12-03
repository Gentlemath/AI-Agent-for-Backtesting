from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtester.kb.sharpe import sharpe_ratio
from backtester.kb.drawdown import max_drawdown
from backtester.schemas import StrategySpec, BacktestResult

TRADING_DAYS = 252

def build_backtest_result(
    returns: pd.Series,
    turnover: float,
    spec: StrategySpec,
    diagnostics: Dict[str, float],
    file_path: str,
) -> BacktestResult:
    """Compute performance metrics and wrap them inside BacktestResult."""
    if returns.empty:
        raise ValueError("Strategy produced no returns for evaluation window.")

    gross = (1 + returns).prod()
    ann_factor = TRADING_DAYS / max(len(returns), 1)
    ann_ret = gross**ann_factor - 1
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = sharpe_ratio(returns)
    mdd = max_drawdown((1 + returns).cumprod())

    cost_drag = turnover * (spec.costs_bps / 10000.0)
    net_returns = returns - cost_drag
    trades = int(turnover * len(returns))
    hit_rate = float((net_returns > 0).mean())
    loss_sum = net_returns[net_returns < 0].sum()
    pf = (
        float(net_returns[net_returns > 0].sum() / abs(loss_sum))
        if loss_sum != 0
        else float("inf")
    )

    diagnostics = {**diagnostics, "turnover": turnover, "cost_drag": cost_drag}

    return BacktestResult(
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_dd=mdd,
        turnover=turnover,
        trades=trades,
        hit_rate=hit_rate,
        pf=pf,
        seed=spec.seed,
        period_start=spec.start_date,
        period_end=spec.end_date,
        artifact_paths=[file_path],
        diagnostics=diagnostics,
    )
