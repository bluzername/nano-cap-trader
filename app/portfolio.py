"""Position sizing & risk."""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from .config import get_settings
_settings = get_settings()

@dataclass
class Position:
    ticker: str
    qty: int
    entry: float

class Portfolio:
    def __init__(self):
        self.cash = _settings.max_portfolio_value
        self.positions: dict[str, Position] = {}

    def target_weights(self, signals: pd.Series) -> pd.Series:
        longs = signals.nlargest(40)
        shorts = signals.nsmallest(20)
        n_long, n_short = len(longs), len(shorts)
        w_long = 0.5 / n_long if n_long else 0
        w_short = -0.5 / n_short if n_short else 0
        w = pd.concat([pd.Series(w_long, longs.index), pd.Series(w_short, shorts.index)])
        return w