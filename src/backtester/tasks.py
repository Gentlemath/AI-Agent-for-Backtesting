from __future__ import annotations

"""Canonical library of frozen strategy tasks and shared constraints."""

from typing import Any, Dict, List, Tuple

DATA_BOUNDS = {"start_date": "2005-01-03", "end_date": "2025-10-31"}
DEFAULT_DATES = {"start_date": "2012-01-03", "end_date": DATA_BOUNDS["end_date"]}

BASE_UNIVERSE: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "GLD",
    "USO",
    "VNQ",
    "HYG",
    "LQD",
    "DBC",
    "XLK",
    "XLF",
    "XLE",
    "XLU",
    "XLY",
    "XLI",
    "XLB",
    "XLP",
    "XOM", "CVX"
]

DEFAULT_TOOLS = ["returns", "sharpe", "drawdown", "normalize_weights", "compute_turnover"]
ROBUSTNESS_TOOLS = DEFAULT_TOOLS + ["walk_forward"]


def _base_spec(name: str, description: str, **overrides: Any) -> Dict[str, Any]:
    spec = {
        "task": name,
        "name": name,
        "description": description,
        "universe": BASE_UNIVERSE[:6],
        "frequency": "daily",
        "signal": "",
        "rules": {},
        "tools": DEFAULT_TOOLS[:],
        "required_metrics": ["ann_return", "sharpe", "max_dd"],
        "params": {},
        "costs_bps": 1.0,
        "start_date": DEFAULT_DATES["start_date"],
        "end_date": DEFAULT_DATES["end_date"],
        "seed": 42,
        "max_leverage": 1.0,
    }
    spec.update(overrides)
    return spec


TASK_LIBRARY: Dict[str, Dict[str, Any]] = {
    "momentum_daily": _base_spec(
        "momentum_daily",
        "Daily top-k cross-sectional momentum on ETFs.",
        signal="Rank compounded 63-day return and go long top decile.",
        rules={"entry": "rank_desc top_k", "exit": "hold N days"},
        params={"lookback": 63, "top_k": 3, "holding_period": 20},
    ),
    "momentum_weekly": _base_spec(
        "momentum_weekly",
        "Weekly momentum with slower turnover.",
        frequency="weekly",
        signal="Weekly returns ranked over trailing 26 weeks.",
        rules={"entry": "top_2", "exit": "stop if rank drops below 5"},
        params={"lookback": 26, "top_k": 2, "holding_period": 8},
    ),
    "mean_reversion": _base_spec(
        "mean_reversion",
        "Short-term contrarian rotation on equities.",
        signal="Fade 5-day z-score; dollar-neutral weights.",
        rules={"entry": "zscore below -1 buys, above +1 sells", "exit": "revert to mean"},
        params={"lookback": 5, "z_entry": 1.0, "z_exit": 0.2},
    ),
    "breakout": _base_spec(
        "breakout",
        "Donchian-style breakout with trailing stop.",
        signal="Price crossing above 55-day high triggers entry.",
        rules={"entry": "close > rolling_high", "exit": "close < trailing_stop"},
        params={"window": 55, "stop_window": 20},
    ),
    "pair_trading": _base_spec(
        "pair_trading",
        "Stat-arb pair trading toggling between cointegration & distance tests. You must test correlation >= correlation_threshhold before trading. if all correlation < correlation_threshhold, report and do not trade.",
        universe=["XOM", "CVX"],
        signal="Spread z-score between highly correlated pair.",
        rules={"entry": "spread zscore > entry_z", "exit": "zscore < exit_z"},
        params={"lookback": 60, "mode": "cointegration", "correlation_threshhold": 0.6, "entry_z": 1.5, "exit_z": 0.5},
        tools=DEFAULT_TOOLS + ["returns"],
    ),
    "volatility_targeting": _base_spec(
        "volatility_targeting",
        "Vol-targeted exposure using realized vol estimates.",
        signal="Scale exposure inversely with trailing volatility.",
        rules={"entry": "target_vol / realized_vol", "exit": "vol > cap"},
        params={"lookback": 20, "target_vol": 0.12},
    ),
    "risk_parity": _base_spec(
        "risk_parity",
        "Toy risk-parity allocating inverse vol weights.",
        universe=["SPY", "TLT", "GLD", "IEF"],
        signal="Inverse volatility weights across asset classes.",
        rules={"entry": "allocate inverse std", "exit": "rebalance monthly"},
        params={"lookback": 60},
    ),
    "atr_bandit": _base_spec(
        "atr_bandit",
        "ATR-driven stop/take-profit allocation bandit.",
        signal="Allocate risk budget via ATR sizing.",
        rules={"entry": "atr < threshold", "exit": "stop loss atr multiple"},
        params={"atr_window": 14, "risk_budget": 0.02, "tp_mult": 1.5, "sl_mult": 1.0},
    ),
    "weekday_mask": _base_spec(
        "weekday_mask",
        "Mask exposures by weekday seasonality.",
        signal="Hold only on weekdays with positive expected alpha.",
        rules={"entry": "allowed weekdays long the basket", "exit": "otherwise flat"},
        params={"allowed_weekdays": [0, 2, 4]},  # Monday, Wednesday, Friday
    ),
    "regime_filter_ma": _base_spec(
        "regime_filter_ma",
        "Regime filter using moving average crossover.",
        signal="Stay invested when fast MA > slow MA.",
        rules={"entry": "fast_ma > slow_ma", "exit": "fast_ma <= slow_ma"},
        params={"fast": 50, "slow": 200},
    ),
    "cost_sensitivity": _base_spec(
        "cost_sensitivity",
        "Stress test strategy under multiple cost assumptions.",
        signal="Replay base strategy under multiple cost grids.",
        rules={"entry": "base weights", "exit": "cost grid analysis"},
        params={"costs_bps_grid": [0, 1, 5, 10, 25]},
        tools=ROBUSTNESS_TOOLS,
        required_metrics=["ann_return", "sharpe", "max_dd", "turnover", "cost_elasticity"],
    ),

}

def allowed_tasks() -> Tuple[str, ...]:
    return tuple(TASK_LIBRARY.keys())


def allowed_universe() -> Tuple[str, ...]:
    return tuple(BASE_UNIVERSE)
