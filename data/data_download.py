import os
import yfinance as yf
import pandas as pd
from typing import List

def get_etf_history(
    tickers: List[str],
    start: str = "2005-01-01",
    end: str = "2025-12-31",
    data_dir: str = "data/etf",
) -> pd.DataFrame:
    """
    Download ETF OHLCV data and cache each ticker separately as a parquet file.
    Returns a long-format DataFrame with columns:
    ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    os.makedirs(data_dir, exist_ok=True)

    for ticker in tickers:
        data_path = os.path.join(data_dir, f"{ticker}.parquet")

        if os.path.exists(data_path):
            print(f"[{ticker}] Exist!")
        else:
            print(f"[{ticker}] Downloading from Yahoo Finance...")
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            df.columns = df.columns.droplevel(1)
            df.to_parquet(data_path)
            print(f"[{ticker}] Saved to {data_path}")



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
]

# Usage
get_etf_history(BASE_UNIVERSE)

SPY = pd.read_parquet("data/etf/SPY.parquet")
print(SPY.head())


