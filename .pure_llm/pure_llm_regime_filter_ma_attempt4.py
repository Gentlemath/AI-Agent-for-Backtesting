import pandas as pd
import numpy as np
from backtester.utils.data_loader import DataLoader
import json

SPEC = {
  "name": "regime_filter_ma",
  "task": "regime_filter_ma",
  "description": "Trend regime filter with smoothed moving-average crossover exposure.",
  "universe": [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT"
  ],
  "frequency": "daily",
  "signal": "Compute fast and slow moving averages of the close (fast = {fast}, slow = {slow}). Define a binary risk-on indicator I_t = 1 if fast_ma_t > slow_ma_t, else 0. Smooth I_t over N days (N = smoothing_window) using a moving average to obtain a continuous exposure weight w_t in [0, 1].",
  "rules": {
    "entry": "increase exposure as the smoothed risk-on indicator w_t moves toward 1",
    "exit": "decrease exposure as w_t moves toward 0; no discrete hold-N-days tickets"
  },
  "tools": [
    "returns",
    "sharpe",
    "drawdown",
    "normalize_weights",
    "compute_turnover"
  ],
  "required_metrics": [
    "ann_return",
    "sharpe",
    "max_dd"
  ],
  "params": {
    "fast": 50,
    "slow": 200,
    "smoothing_window": 5
  },
  "costs_bps": 1.0,
  "start_date": "2012-01-03",
  "end_date": "2025-10-31",
  "seed": 42,
  "max_leverage": 1.0
}

def run_strategy(prices: pd.DataFrame, spec: dict) -> dict:
    params = spec['params']
    fast_ma = prices['Close'].rolling(window=params['fast']).mean()
    slow_ma = prices['Close'].rolling(window=params['slow']).mean()
    risk_on = (fast_ma > slow_ma).astype(int)
    weights = risk_on.rolling(window=params['smoothing_window']).mean()
    weights = weights / weights.max() * spec['max_leverage']

    if spec['frequency'] == 'weekly':
        prices_weekly = prices.resample('W').last()
        weights_weekly = weights.resample('W').last()
        returns_weekly = prices_weekly['Close'].pct_change()
        returns_weekly = returns_weekly * weights_weekly.shift(1)
    else:
        returns_daily = prices['Close'].pct_change()
        returns_daily = returns_daily * weights.shift(1)

    if spec['frequency'] == 'weekly':
        returns = returns_weekly
    else:
        returns = returns_daily

    ann_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = (returns + 1).cumprod().rolling(window=len(returns)).max() / (returns + 1).cumprod() - 1
    max_dd = max_dd.max()
    turnover = (weights.diff().abs().sum() / len(weights)) * 252
    hit_rate = (returns > 0).mean()
    profit_factor = (returns[returns > 0].sum() / returns[returns < 0].sum().abs()) if returns[returns < 0].sum() != 0 else np.nan

    metrics = {
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'turnover': turnover,
        'hit_rate': hit_rate,
        'profit_factor': profit_factor
    }

    return metrics

def main():
    data_loader = DataLoader(disk_dir="data")
    prices = data_loader.ensure_symbols(SPEC['universe'], SPEC['start_date'], SPEC['end_date'])
    prices = prices['Close'].unstack()

    metrics = run_strategy(prices, SPEC)

    for metric, value in metrics.items():
        print(f'{metric}: {value}')

    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
