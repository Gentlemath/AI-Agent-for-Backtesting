from __future__ import annotations
import numpy as np
from functools import partial

from ..schemas import BacktestResult

def _finite_metrics(result: BacktestResult) -> bool:
    metrics = np.array(
        [result.ann_return, result.ann_vol, result.sharpe, result.max_dd], dtype=float
    )
    return np.isfinite(metrics).all()

def _non_trivial_performance(result: BacktestResult, tol: float = 1e-1) -> bool:
    score = abs(result.ann_return) + result.ann_vol + abs(result.max_dd) + result.turnover
    return score > tol

def _rerun_guard(limit: float, result: BacktestResult) -> bool:
    return result.ann_return >= limit

def _turnover_guard(limit: float, result: BacktestResult) -> bool:
    return result.turnover <= limit

def _sharpe_guard(result: BacktestResult) -> bool:
    return -5.0 <= result.sharpe <= 5.0

def _hit_rate_guard(result: BacktestResult) -> bool:
    return 0.0 <= result.hit_rate <= 1.0

CHECKS = {
    "finite_metrics": _finite_metrics,
    "non_trivial_performance": _non_trivial_performance,
    "return_reasonable": partial(_rerun_guard, 0.0),
    "turnover_reasonable": partial(_turnover_guard, 5.0),
    "sharpe_in_range": _sharpe_guard,
    "hit_rate_bounds": _hit_rate_guard,
}

class BTVerifierAgent:
    def __init__(self, checks=None):
        self.checks = checks or CHECKS

    def evaluate(self, result: BacktestResult) -> tuple[bool, list[str]]:
        fails = [name for name, fn in self.checks.items() if not fn(result)]
        return (len(fails) == 0, fails)
