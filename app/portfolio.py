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
        """Generate long-only target weights for realistic $50k nano-cap portfolio."""
        from .config import get_settings
        settings = get_settings()
        
        # Long-only strategy - no shorts for nano-caps
        if settings.enable_short_selling:
            # Keep old logic if shorts are enabled
            longs = signals.nlargest(40)
            shorts = signals.nsmallest(20)
            n_long, n_short = len(longs), len(shorts)
            w_long = 0.5 / n_long if n_long else 0
            w_short = -0.5 / n_short if n_short else 0
            w = pd.concat([pd.Series(w_long, longs.index), pd.Series(w_short, shorts.index)])
            return w
        else:
            # Long-only: 8-12 positions for $50k portfolio
            max_positions = min(12, len(signals))  # Cap at 12 positions
            longs = signals.nlargest(max_positions)
            
            if len(longs) == 0:
                return pd.Series(dtype=float)
            
            # Equal weight across selected positions (100% invested)
            weight_per_position = 1.0 / len(longs)
            return pd.Series(weight_per_position, index=longs.index)