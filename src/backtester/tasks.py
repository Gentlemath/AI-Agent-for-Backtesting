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
        signal=(
            "Compute 63-day compounded returns and rank assets cross-sectionally from highest to lowest each day, going long the top-k names."
            "Assign long weights based on an N-day moving average of top-k membership."
        ),
        rules={
            "entry": "rank_desc top_k; signal = N-day average of top_k indicator",
            "exit": "continuous rebalancing; no discrete hold-N-days tickets"
        },
        params={"lookback": 63, "top_k": 3, "holding_period": 20},
    ),
    "momentum_weekly": _base_spec(
        "momentum_weekly",
        "Weekly top-k cross-sectional momentum on ETFs with slower turnover.",
        frequency="weekly",
        signal=(
            "Compute trailing 26-week compounded returns and rank assets "
            "cross-sectionally in descending order each week. "
            "Define a binary top-k membership indicator and smooth it over N weeks "
            "(N = holding_period) to obtain continuous momentum weights."
        ),
        rules={
            "entry": "weekly rank_desc top_k; signal = N-week moving average of top_k indicator",
            "exit": "rebalance weekly using the smoothed signal; no discrete hold-N-weeks tickets",
        },
        params={"lookback": 26, "top_k": 2, "holding_period": 8},
    ),
    "mean_reversion": _base_spec(
        "mean_reversion",
        "Short-term cross-sectional contrarian rotation on equities.",
        universe = ["SPY"],
        signal=(
            "For each stock, compute a {lookback}-day return z-score (standardized across the universe). "
            "Go long names with zscore <= -z_entry and short names with zscore >= +z_entry, "
            "with dollar-neutral portfolio weights."
        ),
        rules={
            "entry": (
                "open or adjust positions when |zscore| >= z_entry "
                "long if zscore <= -z_entry, keep the position until exit condition met; " 
                "short if zscore >= +z_entry, keep the position until exit condition met."
            ),
            "exit": (
                "close positions when |zscore| <= z_exit or the sign of the zscore reverses, "
                "sell long positions when zscore > -z_exit"
                "cover short positions when zscore < +z_exit"
            ),
        },
        params={
            "lookback": 10,
            "z_entry": 1.5,
            "z_exit": 0.2,
        },
    ),
    "breakout": _base_spec(
        "breakout",
        "Donchian-style long breakout with a trailing stop.",
        signal=(
            "Compute a {window}-day rolling high of the prices (rolling_high_t,the highest close over the past {window} days, excluding today)" 
            "Compute a {stop_window}-day rolling low of the prices(trailing_stop_t, the lowest close over the past {stop_window} days, excluding today). "
            "Enter a long position when the close breaks above this rolling high. "
            "Once entry, maintain the position until close_t < trailing_stop_t"
        ),
        rules={
            "entry": (
                "open a long position when close_t > rolling_high_t."
            ),
            "exit": (
                "exit the long position when close_t < trailing_stop_t."
                "You can only exit long positions! NEGATIVE positions is FORBBIDEN!"
                "if not in a position, you must wait for the next entry signal."
            ),
        },
        params={
            "window": 55,
            "stop_window": 20,
        },
    ),
    "pair_trading": _base_spec(
        "pair_trading",
        "Stat-arb pair trading toggling between cointegration & distance tests. You must test correlation >= correlation_threshhold before trading. if all correlation < correlation_threshhold, report and do not trade.",
        universe=["XOM", "CVX"],
        signal="Spread z-score between highly correlated pair.",
        rules={"entry": "spread zscore > entry_z", "exit": "zscore < exit_z"},
        params={"lookback": 60, "mode": "cointegration", "correlation_threshhold": 0.6, "entry_z": 1.5, "exit_z": 0.5},
        tools=DEFAULT_TOOLS,
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

    "weekday_mask": _base_spec(
        "weekday_mask",
        "Hard weekday seasonality mask on exposures (no smoothing).",
        signal=(
            "Apply a binary weekday filter to the basket: on allowed weekdays the strategy "
            "is fully invested; on all other weekdays position is 0."
            "Do NOT use numpy arrays for the mask. "
            "Use returns.index.to_series().dt.weekday to compute weekdays, and then replicate to a DataFrame with column number = asset number."
            "Don't shift weight since weekdays are known in advance."
        ),
        rules={
            "entry": (
                "if today's weekday is in allowed_weekdays, set exposure to 1.0 "
                "(or pass through the underlying strategy's signal unchanged)"
            ),
            "exit": (
                "if today's weekday is not in allowed_weekdays, set exposure to 0.0 "
                "and close all positions"
            ),
            "weighting": "Equally weight all allowed assets on allowed weekdays."
        },
        params={
            "allowed_weekdays": [0, 2, 4],  # Monday=0, ..., Friday=4 → trade Mon/Wed/Fri
        },
    ),


    "regime_filter_ma": _base_spec(
        "regime_filter_ma",
        "Trend regime filter with smoothed moving-average crossover exposure.",
        signal=(
            "Compute fast and slow moving averages of the close (fast = {fast}, slow = {slow}). "
            "Define a binary risk-on indicator I_t = 1 if fast_ma_t > slow_ma_t, else 0. "
            "Smooth I_t over N days (N = smoothing_window) using a moving average to obtain a "
            "continuous exposure weight w_t in [0, 1]."
        ),
        rules={
            "entry": "increase exposure as the smoothed risk-on indicator w_t moves toward 1",
            "exit": "decrease exposure as w_t moves toward 0; no discrete hold-N-days tickets",
        },
        params={
            "fast": 50,
            "slow": 200,
            "smoothing_window": 5,  # or whatever N you like
        },
    )



}

def allowed_tasks() -> Tuple[str, ...]:
    return tuple(TASK_LIBRARY.keys())


def allowed_universe() -> Tuple[str, ...]:
    return tuple(BASE_UNIVERSE)
