from fastapi import APIRouter, Depends
from .portfolio import Portfolio

router = APIRouter(prefix="/api", tags=["core"])
_portfolio = Portfolio()

@router.get("/status")
def status():
    return {
        "cash": _portfolio.cash,
        "positions": {k: vars(p) for k, p in _portfolio.positions.items()},
    }