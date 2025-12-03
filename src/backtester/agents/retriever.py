from __future__ import annotations

from dataclasses import dataclass
from typing import List

TOOLKIT = {
    "returns": {
        "module": "backtester.kb.returns",
        "symbol": "pct_returns",
        "file": "returns.py",
        "description": "pct_returns(prices: pd.DataFrame) -> pd.DataFrame of daily pct changes (NaNs filled with 0.0).",
    },
    "sharpe": {
        "module": "backtester.kb.sharpe",
        "symbol": "sharpe_ratio",
        "file": "sharpe.py",
        "description": "sharpe_ratio(r: pd.Series, rf: float = 0.0, periods: int = 252) -> float annualized Sharpe.",
    },
    "drawdown": {
        "module": "backtester.kb.drawdown",
        "symbol": "max_drawdown",
        "file": "drawdown.py",
        "description": "max_drawdown(r: pd.Series) -> float maximum peak-to-trough drawdown.",
    },
    "normalize_weights": {
        "module": "backtester.kb.strategies",
        "symbol": "normalize_weights",
        "file": "strategies.py",
        "description": "normalize_weights(weights: pd.DataFrame, max_leverage: float = 1.0) -> pd.DataFrame with unit gross leverage scaled by max_leverage.",
    },
    "compute_turnover": {
        "module": "backtester.kb.strategies",
        "symbol": "compute_turnover",
        "file": "strategies.py",
        "description": "compute_turnover(weights: pd.DataFrame) -> float average daily absolute weight change.",
    },
    "walk_forward": {
        "module": "backtester.kb.walk_forward",
        "symbol": "walk_forward_validate",
        "file": "walk_forward.py",
        "description": "walk_forward_validate(prices, spec, *, train_window, test_window, min_periods) -> dict of robustness metrics.",
    },
}

@dataclass(frozen=True)
class ToolSpec:
    name: str
    module: str
    symbol: str
    path: str
    description: str

class RetrieverAgent:
    """Returns names/paths of vetted snippet modules implementing metrics & utilities."""

    def __init__(self, kb_root: str):
        self.kb_root = kb_root

    def fetch(self, needs: list[str]) -> List[ToolSpec]:
        specs: List[ToolSpec] = []
        seen: set[str] = set()
        for name in needs:
            if name in seen:
                continue
            info = TOOLKIT.get(name)
            if not info:
                continue
            specs.append(
                ToolSpec(
                    name=name,
                    module=info["module"],
                    symbol=info["symbol"],
                    path=f"{self.kb_root}/{info['file']}",
                    description=info.get("description", ""),
                )
            )
            seen.add(name)
        return specs
