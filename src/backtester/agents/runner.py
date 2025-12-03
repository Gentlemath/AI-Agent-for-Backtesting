from __future__ import annotations
from importlib.util import spec_from_file_location, module_from_spec
from typing import Any, Dict

import pandas as pd

from ..schemas import StrategySpec, BacktestResult
from ..utils.metrics import build_backtest_result
from ..utils.data_loader import DataLoader

class RunnerAgent:
    def __init__(self, data_loader: DataLoader | None = None):
        self.loader = data_loader
        self._price_cache: pd.DataFrame | None = None

    def _load_prices(self, spec: StrategySpec) -> pd.DataFrame:
        if self._price_cache is not None:
            return self._price_cache.copy()

        if not self.loader:
            raise FileNotFoundError("No DataLoader configured to fetch prices.")

        fetched = self.loader.ensure_symbols(spec.universe, spec.start_date, spec.end_date)
        if fetched.empty:
            raise ValueError(f"DataLoader returned empty frame for symbols {spec.universe}.")
        self._price_cache = fetched.sort_index()
        return self._price_cache.copy()

    def _load_module(self, file_path: str):
        spec = spec_from_file_location("gen_strategy", file_path)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def _prepare_returns(
        self, payload: Any
    ) -> tuple[pd.Series, Dict[str, Any], float]:
        diagnostics: Dict[str, Any] = {}
        turnover = 0.0
        if isinstance(payload, pd.Series):
            returns = payload
        elif isinstance(payload, dict):
            if "returns" not in payload:
                raise ValueError("Strategy output dict must contain 'returns'.")
            returns = payload["returns"]
            raw_diag = {k: v for k, v in payload.items() if k != "returns"}
            turnover = float(raw_diag.get("turnover", 0.0))
            diagnostics = {
                key: float(value)
                for key, value in raw_diag.items()
                if isinstance(value, (int, float))
            }
        else:
            raise TypeError("Strategy output must be a pandas Series or dict.")
        if not isinstance(returns, pd.Series):
            raise TypeError("Strategy returns must be a pandas Series.")
        return returns.astype(float), diagnostics, turnover

    def run(self, file_path: str, spec: StrategySpec) -> BacktestResult:
        prices = self._load_prices(spec)
        start_ts = pd.Timestamp(spec.start_date)
        end_ts = pd.Timestamp(spec.end_date)
        px = prices.loc[start_ts:end_ts, spec.universe].dropna(how="all")
        if px.empty:
            # retry from source once in case cache was stale
            self._price_cache = None
            prices = self._load_prices(spec)
            px = prices.loc[start_ts:end_ts, spec.universe].dropna(how="all")
            if px.empty:
                raise ValueError("No price data in requested window, even after fetch.")
            
        print(f"Running backtest from {start_ts.date()} to {end_ts.date()} on {len(spec.universe)} symbols.")
        print(f"Price data shape: {px.shape}")
        print(f"Price data date range: {px.index.min().date()} to {px.index.max().date()}")
        print(f"Price data columns: {px.columns.tolist()}")


        mod = self._load_module(file_path)
        payload = mod.run_strategy(px, spec.model_dump())
        raw_returns, diagnostics, turnover = self._prepare_returns(payload)
        returns = raw_returns.loc[start_ts:end_ts].fillna(0.0)
        return build_backtest_result(returns, turnover, spec, diagnostics, file_path)
