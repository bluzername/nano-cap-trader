from app.utils import zscore
import pandas as pd

def test_zscore():
    s = pd.Series([1, 2, 3, 4, 5])
    zs = zscore(s)
    assert zs.mean().round(6) == 0
    assert zs.std(ddof=0).round(6) == 1