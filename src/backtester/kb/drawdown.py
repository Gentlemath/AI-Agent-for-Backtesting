import pandas as pd  # lazy import

def max_drawdown(cum: pd.Series) -> float:

    roll_max = cum.cummax()
    dd = (cum / (roll_max + 1e-12)) - 1.0
    return float(dd.min())
