import pandas as pd
import numpy as np
from app.backtest import Backtester

def test_backtester_runs():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    tickers = ["XXX", "YYY"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    price_panel = pd.DataFrame({"close": 100 + np.random.randn(len(idx)), "open": 100}, index=idx)
    sig = pd.DataFrame(0.1, index=dates, columns=tickers)
    bt = Backtester(price_panel, sig)
    series = bt.run()
    assert not series.empty