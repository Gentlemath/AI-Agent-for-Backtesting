import pandas as pd

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)
