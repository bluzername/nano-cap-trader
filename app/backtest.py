import pandas as pd
from .portfolio import Portfolio

class Backtester:
    def __init__(self, price_panel: pd.DataFrame, signals: pd.DataFrame):
        self.prices = price_panel  # multi-index (date, ticker)
        self.signals = signals
        self.portfolio = Portfolio()

    def run(self):
        # Simplified loop through dates: pnl = weight * next-day return.
        returns = []
        weights = None
        for date, daily_prices in self.prices.groupby(level=0):
            if date in self.signals.index:
                weights = self.portfolio.target_weights(self.signals.loc[date])
            if weights is None:
                continue
            next_idx = date + pd.Timedelta(days=1)
            if next_idx in self.prices.index.get_level_values(0):
                next_prices = self.prices.loc[next_idx]
                today_prices = daily_prices.droplevel(0)  # index by ticker only
                daily_ret = next_prices.close / today_prices.close - 1
                returns.append((next_idx, (weights * daily_ret).sum()))
        return pd.Series(dict(returns))
