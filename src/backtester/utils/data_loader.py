from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests


class DataLoader:
    """Utility for sourcing price data from public APIs (yfinance, AlphaVantage) with caching."""

    def __init__(
        self,
        disk_dir: str = "data/etf",  # small typo fix: etf not eft
        cache_dir: str = "data/cache",  # <- DEPRECATED
        alpha_vantage_key: str | None = None,
        prefer_source: str = "auto",
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ):
        self.disk_dir = Path(disk_dir)
        self.disk_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.alpha_key = alpha_vantage_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        self.prefer_source = prefer_source
        self.session = session or requests.Session()
        self.timeout = timeout

    # ------------------------------
    # Public interface
    # ------------------------------
    def ensure_symbols(
        self,
        symbols: Sequence[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        sym_list = [s.upper() for s in symbols]

        permanent_frames: list[pd.DataFrame] = []
        missing: list[str] = []

        # load what we already have on disk
        for sym in sym_list:
            permanent = self._read_disk(sym)
            if permanent is not None:
                permanent_frames.append(permanent)
            else:
                missing.append(sym)

        panel = (
            pd.concat(permanent_frames, axis=1) if permanent_frames else pd.DataFrame()
        )

        # fetch the missing stuff
        if missing:
            fetched = self._fetch_panel(missing, start, end)
            if not fetched.empty:
                # cache symbol by symbol
                for sym in missing:
                    if sym in fetched.columns:
                        self._write_cache(sym, fetched[[sym]])
                # merge permanent + fetched
                panel = fetched if panel.empty else panel.combine_first(fetched)

        if panel.empty:
            return panel

        panel = panel.sort_index()

        start_ts, end_ts = self._to_timestamp(start), self._to_timestamp(end)
        window = panel.loc[start_ts:end_ts]

        # ensure column order matches requested symbols
        window = window.reindex(columns=sym_list)

        return window

    # ------------------------------
    # Fetchers
    # ------------------------------
    def _fetch_panel(
        self,
        symbols: Iterable[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        sym_list = [s.upper() for s in symbols]
        frames: list[pd.DataFrame] = []

        # try yfinance first
        if self.prefer_source in {"auto", "yfinance"}:
            try:
                frames.append(self._fetch_yfinance(sym_list, start, end))
            except ImportError:
                if self.prefer_source == "yfinance":
                    raise
            except Exception:
                # fall through to AlphaVantage
                pass

        # find which symbols are still missing after yfinance
        if frames:
            combined = pd.concat(frames, axis=1)
            have = set(combined.columns)
            missing = [s for s in sym_list if s not in have]
        else:
            missing = sym_list

        # try AlphaVantage for the rest
        if missing and self.prefer_source in {"auto", "alphavantage"}:
            alpha_frame = self._fetch_alpha_vantage(missing, start, end)
            if not alpha_frame.empty:
                frames.append(alpha_frame)

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=1)
        panel = panel.loc[:, ~panel.columns.duplicated()]  # drop dups if any
        panel = panel.sort_index()
        return panel

    def _fetch_yfinance(
        self,
        symbols: Sequence[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for DataLoader._fetch_yfinance; install it via pip."
            ) from exc

        data = yf.download(
            tickers=" ".join(symbols),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",  # <- important
        )

        if data.empty:
            return pd.DataFrame()

        # if MultiIndex (e.g. ('Adj Close', 'SPY')), pick the appropriate level
        if isinstance(data.columns, pd.MultiIndex):
            base_cols = data.columns.get_level_values(0)
            target = "Adj Close" if "Adj Close" in base_cols else "Close"
            close = data[target]
        else:
            # single ticker case: columns are just OHLCV
            if "Adj Close" in data.columns:
                close = data[["Adj Close"]]
            elif "Close" in data.columns:
                close = data[["Close"]]
            else:
                # fallback: assume it's already prices
                close = data

        # ensure 2D
        if isinstance(close, pd.Series):
            close = close.to_frame(name=symbols[0])

        close.index = pd.to_datetime(close.index)
        close = close.sort_index()

        # yfinance with group_by="column" gives columns as tickers on the second level
        # After selecting "Adj Close" / "Close", columns should be tickers.
        close.columns = [c.upper() if isinstance(c, str) else c for c in close.columns]

        return close

    def _fetch_alpha_vantage(
        self,
        symbols: Sequence[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        if not symbols or not self.alpha_key:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        start_ts, end_ts = self._to_timestamp(start), self._to_timestamp(end)

        for sym in symbols:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": sym,
                "outputsize": "full",
                "apikey": self.alpha_key,
            }
            resp = self.session.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=self.timeout,
            )
            payload = resp.json()
            series = payload.get("Time Series (Daily)")
            if not series:
                continue

            df = pd.DataFrame(series).T
            df.index = pd.to_datetime(df.index)

            col = "5. adjusted close"
            if col not in df.columns:
                continue

            prices = df[col].astype(float).sort_index()
            prices = prices.loc[start_ts:end_ts]
            frames.append(prices.to_frame(sym))

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=1).sort_index()
        return panel

    # ------------------------------
    # Path helpers
    # ------------------------------
    def _disk_path(self, symbol: str) -> Path:
        sym = symbol.upper()
        return self.disk_dir / f"{sym}.parquet"
    
    def _cache_path(self, symbol: str) -> Path:
        sym = symbol.upper()
        return self.cache_dir / f"{sym}.parquet"

    def _read_disk(self, symbol: str) -> pd.DataFrame | None:
        sym = symbol.upper()
        path = self._disk_path(sym)

        if not path.exists():
            return None

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        cols = list(df.columns)

        # if the symbol column is there, use it
        if sym in cols:
            series = df[sym]
        # otherwise, try typical names
        elif "Adj Close" in cols:
            series = df["Adj Close"]
        elif "Close" in cols:
            series = df["Close"]
        # fallback: single-column file, just use that
        elif len(cols) == 1:
            series = df[cols[0]]
        else:
            # cannot interpret – treat as missing
            return None

        out = series.to_frame(sym)
        return out

    def _write_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        sym = symbol.upper()
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()

        # ensure single col named by symbol
        if len(frame.columns) == 1 and frame.columns[0] != sym:
            frame = frame.rename(columns={frame.columns[0]: sym})
        elif len(frame.columns) > 1:
            frame = frame[[sym]]

        frame.to_parquet(self._cache_path(sym))

    @staticmethod
    def _to_timestamp(value: date | str) -> pd.Timestamp:
        return pd.Timestamp(value)
