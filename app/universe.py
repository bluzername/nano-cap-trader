"""Dynamic universe builder."""
import pandas as pd
from sqlalchemy import text
from .config import get_settings
from .data import _engine

_settings = get_settings()

_QUERY = """
SELECT DISTINCT ticker, floatShares, marketCap, avgVolume FROM fundamentals
WHERE marketCap BETWEEN 10000000 AND 350000000
  AND avgVolume * closePrice > 50000
"""

def current_universe() -> pd.DataFrame:
    return pd.read_sql(text(_QUERY), _engine)