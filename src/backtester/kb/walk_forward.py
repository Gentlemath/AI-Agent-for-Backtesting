from __future__ import annotations

import numpy as np
import pandas as pd

def walk_forward_validate(series: pd.Series, splits: int = 5) -> float:
    """Simple placeholder that quantifies walk-forward stability."""
    if series.empty:
        return 0.0
    splits = max(1, splits)
    chunk = max(5, len(series) // splits)
    rolling = series.rolling(chunk).std().dropna()
    return float(np.nanmean(rolling.values)) if not rolling.empty else 0.0
