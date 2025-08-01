import pandas as pd
from .portfolio import Portfolio

class Backtester:
    def __init__(self, price_panel: pd.DataFrame, signals: pd.DataFrame):
        self.prices = price_panel  # multi‑index (date, ticker)
        self.signals = signals
        self.portfolio = Portfolio()

    def run(self):
        # Simplified loop through dates.
        returns = []
        for date, daily_prices in self.prices.groupby(level=0):
            if date in self.signals.index:
                w = self.portfolio.target_weights(self.signals.loc[date])
                # Here we simply compute pnl as weight * next‑day return.
            next_idx = date + pd.Timedelta(days=1)
            if next_idx in self.prices.index.get_level_values(0):
                next_prices = self.prices.loc[next_idx]
                daily_ret = (next_prices.close / daily_prices.close - 1)
                returns.append((next_idx, (w * daily_ret).sum()))
        return pd.Series(dict(returns))