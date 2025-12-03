import numpy as np
import pandas as pd

def sharpe_ratio(r: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    import pandas as pd  # lazy import
    excess = r - (rf / periods)
    return float(np.nan_to_num(excess.mean() / (excess.std() + 1e-12)) * np.sqrt(periods))
