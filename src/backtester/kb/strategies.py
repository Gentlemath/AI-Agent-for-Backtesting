from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

def _sanitize_prices(prices: pd.DataFrame, universe: Iterable[str]) -> pd.DataFrame:
    cols = list(universe) if universe else prices.columns.tolist()
    frame = prices.loc[:, [c for c in cols if c in prices.columns]]
    frame = frame.sort_index().ffill().dropna(how="all")
    return frame


def _returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def _normalize(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return weights
    denom = weights.abs().sum(axis=1).replace(0, np.nan)
    return weights.div(denom, axis=0).fillna(0.0)

def normalize_weights(weights: pd.DataFrame, max_leverage: float = 1.0) -> pd.DataFrame:
    """Public helper to enforce bounded gross exposure each period."""
    scaled = _normalize(weights)
    limit = max(float(max_leverage), 0.0)
    if limit == 0.0:
        return scaled * 0.0
    return scaled * limit

def _estimate_turnover(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    diffs = weights.fillna(0.0).diff().abs().sum(axis=1)
    return float(diffs.mean())

def compute_turnover(weights: pd.DataFrame) -> float:
    """Public helper mirroring the toolkit's turnover definition."""
    return _estimate_turnover(weights)


def _equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.shape[1] == 0:
        raise ValueError("Strategy universe is empty.")
    weight = 1.0 / prices.shape[1]
    return pd.DataFrame(weight, index=prices.index, columns=prices.columns)


def _package(returns: pd.Series, weights: pd.DataFrame, extra: Dict[str, float] | None = None) -> Dict[str, object]:
    returns = returns.fillna(0.0)
    weights = weights.fillna(0.0)
    diagnostics = {
        "returns": returns,
        "weights": weights,
        "turnover": _estimate_turnover(weights),
    }
    if extra:
        diagnostics.update(extra)
    return diagnostics


def momentum_daily(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    lookback = int(spec.get("params", {}).get("lookback", 63))
    top_k = int(spec.get("params", {}).get("top_k", 3))
    signal = px.pct_change(lookback)
    ranks = signal.rank(axis=1, ascending=False, method="first")
    weights = (ranks <= top_k).astype(float)
    weights = _normalize(weights)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"lookback": float(lookback)})


def momentum_weekly(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    weekly = px.resample("W-FRI").last().dropna(how="all")
    rets = _returns(px)
    lookback = int(spec.get("params", {}).get("lookback", 26))
    top_k = int(spec.get("params", {}).get("top_k", 2))
    signal = weekly.pct_change(lookback)
    ranks = signal.rank(axis=1, ascending=False, method="first")
    weights_weekly = (ranks <= top_k).astype(float)
    weights_weekly = _normalize(weights_weekly)
    weights = weights_weekly.reindex(px.index).ffill().fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"lookback": float(lookback)})


def mean_reversion(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    lookback = int(spec.get("params", {}).get("lookback", 5))
    z_entry = float(spec.get("params", {}).get("z_entry", 1.0))
    cum_dev = rets.rolling(lookback).sum()
    rolling_std = rets.rolling(lookback).std().replace(0, np.nan)
    zscore = cum_dev.divide(rolling_std).fillna(0.0)
    signals = -zscore / (z_entry + 1e-6)
    signals = signals.clip(-1.0, 1.0)
    weights = signals.sub(signals.mean(axis=1), axis=0)
    weights = _normalize(weights)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"z_entry": z_entry})


def breakout(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    window = int(spec.get("params", {}).get("window", 55))
    stop_window = int(spec.get("params", {}).get("stop_window", 20))
    roll_high = px.rolling(window).max()
    roll_low = px.rolling(stop_window).min()
    breakout_signal = px.gt(roll_high.shift(1))
    stop_signal = px.lt(roll_low.shift(1))
    weights = breakout_signal.astype(float)
    weights = _normalize(weights)
    weights = weights.mask(stop_signal, 0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"window": float(window)})


def pair_trading(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", [])[:2])
    if px.shape[1] < 2:
        raise ValueError("Pair trading requires at least two assets.")
    asset_a, asset_b = px.columns[:2]
    rets = _returns(px)
    params = spec.get("params", {})
    lookback = int(params.get("lookback", 60))
    entry_z = float(params.get("entry_z", 2.0))
    exit_z = float(params.get("exit_z", 0.5))
    mode = params.get("mode", "cointegration")
    if mode == "cointegration":
        beta = (
            px[asset_a].rolling(lookback).cov(px[asset_b])
            / px[asset_b].rolling(lookback).var().replace(0, np.nan)
        ).fillna(1.0)
    else:
        beta = pd.Series(1.0, index=px.index)
    spread = px[asset_a] - beta * px[asset_b]
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std().replace(0, np.nan)
    zscore = (spread - spread_mean) / (spread_std + 1e-6)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    long_signal = zscore < -entry_z
    short_signal = zscore > entry_z
    flat_signal = zscore.abs() < exit_z
    weights.loc[long_signal, asset_a] = 0.5
    weights.loc[long_signal, asset_b] = -0.5
    weights.loc[short_signal, asset_a] = -0.5
    weights.loc[short_signal, asset_b] = 0.5
    weights.loc[flat_signal, :] = 0.0
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(
        port_ret,
        weights,
        {"mode_cointegration": float(mode == "cointegration"), "entry_z": entry_z},
    )


def volatility_targeting(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    eq_weights = _equal_weight(px)
    base_ret = (eq_weights.shift(1) * rets).sum(axis=1)
    lookback = int(spec.get("params", {}).get("lookback", 20))
    target_vol = float(spec.get("params", {}).get("target_vol", 0.12))
    realized = base_ret.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / (realized + 1e-6)).clip(0.0, spec.get("max_leverage", 2.0))
    weights = eq_weights.mul(scale, axis=0)
    weights = _normalize(weights)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"target_vol": target_vol})


def risk_parity(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    window = int(spec.get("params", {}).get("lookback", 60))
    rolling_vol = rets.rolling(window).std().replace(0, np.nan)
    inv_vol = 1.0 / (rolling_vol + 1e-6)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"lookback": float(window)})


def atr_bandit(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    params = spec.get("params", {})
    atr_window = int(params.get("atr_window", 14))
    risk_budget = float(params.get("risk_budget", 0.02))
    tr = px.diff().abs()
    atr = tr.rolling(atr_window).mean() / px.shift(1)
    inv_atr = (risk_budget / (atr + 1e-6)).clip(0.0, 5.0)
    weights = inv_atr.div(inv_atr.sum(axis=1), axis=0).fillna(0.0)
    position_ret = (weights.shift(1) * rets).sum(axis=1)
    tp = float(params.get("tp_mult", 1.5))
    sl = float(params.get("sl_mult", 1.0))
    pnl = position_ret.copy()
    pnl[pnl > tp * risk_budget] = tp * risk_budget
    pnl[pnl < -sl * risk_budget] = -sl * risk_budget
    return _package(pnl, weights, {"risk_budget": risk_budget})


def weekday_mask(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    allowed = set(spec.get("params", {}).get("allowed_weekdays", [0, 2, 4]))
    eq_weights = _equal_weight(px)
    mask = pd.Series(px.index.weekday, index=px.index).isin(allowed).astype(float)
    weights = eq_weights.mul(mask, axis=0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"active_days": float(len(allowed))})


def regime_filter_ma(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    rets = _returns(px)
    params = spec.get("params", {})
    fast = int(params.get("fast", 50))
    slow = int(params.get("slow", 200))
    benchmark = px.iloc[:, 0]
    fast_ma = benchmark.rolling(fast).mean()
    slow_ma = benchmark.rolling(slow).mean()
    regime = fast_ma > slow_ma
    eq_weights = pd.DataFrame(1.0 / len(px.columns), index=px.index, columns=px.columns)
    weights = eq_weights.where(regime[:, None], 0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    return _package(port_ret, weights, {"fast": float(fast), "slow": float(slow)})


def cost_sensitivity(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    base = momentum_daily(prices, spec)
    rets = base["returns"]
    turnover = base.get("turnover", 0.0)
    grid = spec.get("params", {}).get("costs_bps_grid", [0, 1, 5, 10, 25])
    grid = [float(x) for x in grid]
    annualized = []
    for cost in grid:
        slip = turnover * (cost / 10000.0)
        net = rets - slip
        ann = (1 + net).prod() ** (252 / max(len(net), 1)) - 1
        annualized.append(ann)
    elasticity = float(np.polyfit(grid, annualized, deg=1)[0]) if len(grid) >= 2 else 0.0
    extra = {
        "cost_elasticity": elasticity,
        "turnover": turnover,
    }
    return _package(base["returns"], base["weights"], extra)


def walk_forward_robustness(prices: pd.DataFrame, spec: Dict) -> Dict[str, object]:
    px = _sanitize_prices(prices, spec.get("universe", []))
    params = spec.get("params", {})
    train_window = int(params.get("train_window", 252 * 2))
    test_window = int(params.get("test_window", 252))
    min_periods = int(params.get("min_periods", train_window + test_window))
    rets = _returns(px)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    if len(px) < min_periods:
        zeros = pd.Series(0.0, index=px.index)
        return _package(zeros, weights, {"train_window": float(train_window)})
    index = px.index
    for start in range(0, len(index) - min_periods + 1, test_window):
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window
        if test_end > len(index):
            break
        train_px = px.iloc[train_start:train_end]
        if train_px.empty:
            continue
        momentum_signal = train_px.pct_change(63).iloc[-1].sort_values(ascending=False)
        top_k = max(1, int(spec.get("params", {}).get("top_k", 3)))
        top_assets = momentum_signal.index[:top_k]
        weights.loc[index[test_start:test_end], :] = 0.0
        weights.loc[index[test_start:test_end], top_assets] = 1.0 / len(top_assets)
    weights = _normalize(weights).fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    extra = {
        "train_window": float(train_window),
        "test_window": float(test_window),
    }
    return _package(port_ret, weights, extra)


STRATEGY_REGISTRY = {
    "momentum_daily": momentum_daily,
    "momentum_weekly": momentum_weekly,
    "mean_reversion": mean_reversion,
    "breakout": breakout,
    "pair_trading": pair_trading,
    "volatility_targeting": volatility_targeting,
    "risk_parity": risk_parity,
    "atr_bandit": atr_bandit,
    "weekday_mask": weekday_mask,
    "regime_filter_ma": regime_filter_ma,
    "cost_sensitivity": cost_sensitivity,
    "walk_forward_robustness": walk_forward_robustness,
}
