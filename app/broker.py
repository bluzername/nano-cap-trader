"""Abstract broker API. IBKR implementation via ib_insync stub."""
from ib_insync import IB, util, Contract, MarketOrder

class IBKRBroker:
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)

    def order(self, ticker: str, qty: int, side: str):
        contract = Contract(symbol=ticker, secType="STK", currency="USD", exchange="SMART")
        order = MarketOrder("BUY" if side == "long" else "SELL", qty)
        trade = self.ib.placeOrder(contract, order)
        trade.waitUntilSettled()
        return trade.orderStatus.status