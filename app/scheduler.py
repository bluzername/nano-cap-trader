from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .data import nightly_etl
from .signals import composite_signal, insider_buy_score, gap_reversion_score, alt_growth_score, squeeze_score, momo_score
from .portfolio import Portfolio
from .broker import IBKRBroker

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("cron", hour=22, minute=30)
async def nightly_pipeline():
    await nightly_etl()
    # assemble signals and rebalance (pseudo) ...

scheduler.start()