import pandas as pd
from app.portfolio import Portfolio

def test_portfolio_target_weights():
    portfolio = Portfolio()
    signals = pd.Series([0.1, 0.2, -0.1, 0.3], index=["A", "B", "C", "D"])
    weights = portfolio.target_weights(signals)
    assert not weights.empty
    assert weights.sum() <= 1.0