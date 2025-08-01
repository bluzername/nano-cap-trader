import numpy as np
import pandas as pd

def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)