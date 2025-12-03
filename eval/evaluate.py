from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from backtester.schemas import WorkflowState
from backtester.tasks import allowed_tasks

def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, iters: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return 0.0, (0.0, 0.0)
    diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        diffs.append((a[idx] - b[idx]).mean())
    diffs = np.array(diffs)
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    return diffs.mean(), ci

@dataclass
class RunsetSummary:
    success_rate: float
    spec_compliance: float
    error_rate: float
    sharpe_stability: float
    avg_attempts: float
    avg_tools: float

def _extract(states: Iterable[WorkflowState], key: str) -> np.ndarray:
    values = [float(state.metrics.get(key, np.nan)) for state in states]
    return np.array(values, dtype=float)

def summarize(states: List[WorkflowState]) -> RunsetSummary:
    if not states:
        return RunsetSummary(0, 0, 0, 0, 0, 0)
    success = np.array([state.verdict == "pass" for state in states], dtype=float)
    compliance = np.array(
        [1.0 if state.spec.task in allowed_tasks() else 0.0 for state in states],
        dtype=float,
    )
    sharpe_values = _extract(states, "sharpe")
    sharpe_stability = float(np.nanvar(sharpe_values))
    attempts = np.array([state.attempts for state in states], dtype=float)
    tool_counts = np.array([len(state.tool_refs) for state in states], dtype=float)
    success_rate = float(success.mean())
    spec_compliance = float(compliance.mean())
    error_rate = float(1.0 - success_rate)
    return RunsetSummary(
        success_rate=success_rate,
        spec_compliance=spec_compliance,
        error_rate=error_rate,
        sharpe_stability=sharpe_stability,
        avg_attempts=float(attempts.mean()),
        avg_tools=float(tool_counts.mean()),
    )

def compare(agentic: List[WorkflowState], baseline: List[WorkflowState]) -> dict:
    agentic_sharpe = _extract(agentic, "sharpe")
    baseline_sharpe = _extract(baseline, "sharpe")
    sharpe_diff, sharpe_ci = paired_bootstrap_diff(agentic_sharpe, baseline_sharpe)
    success_diff, success_ci = paired_bootstrap_diff(
        np.array([s.verdict == "pass" for s in agentic], dtype=float),
        np.array([s.verdict == "pass" for s in baseline], dtype=float),
    )
    return {
        "sharpe_diff": sharpe_diff,
        "sharpe_ci": sharpe_ci,
        "success_diff": success_diff,
        "success_ci": success_ci,
    }

if __name__ == "__main__":
    print("Use summarize() and compare() by importing this module inside eval notebooks.")
