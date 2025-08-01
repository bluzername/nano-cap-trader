import pandas as pd
from app.signals import insider_buy_score

def test_insider_buy_score():
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "transactionType": ["P", "P", "S"],
        "netTransactionValue": [10000, 15000, -2000],
    })
    s = insider_buy_score(df)
    assert "AAA" in s.index and s.loc["AAA"] > 0