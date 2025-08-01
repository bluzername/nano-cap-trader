"""ETL layer: nightly refresh of raw data (prices, trades, fundamentals, insider, events)"""
from __future__ import annotations
import datetime as dt
import httpx
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Float, Integer, Date, String
from .config import get_settings

_settings = get_settings()
_engine = create_engine(_settings.db_url, echo=False, future=True)

_POLYGON = "https://api.polygon.io"

# -- 1. Insider filings -------------------------------------------------------
async def fetch_form4(session: httpx.AsyncClient, since: dt.datetime) -> pd.DataFrame:
    params = {
        "apiKey": _settings.polygon_api_key,
        "type": "4",
        "limit": 1000,
        "sort": "timestamp",
        "order": "asc",
        "timestamp.gte": int(since.timestamp() * 1000),
    }
    r = await session.get(f"{_POLYGON}/vX/reference/sec_filings", params=params)
    r.raise_for_status()
    items = r.json()["results"]
    return pd.DataFrame(items)

# -- 2. Prices ---------------------------------------------------------------
async def fetch_daily_prices(session: httpx.AsyncClient, ticker: str, start: str, end: str) -> pd.DataFrame:
    r = await session.get(
        f"{_POLYGON}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
        params={"apiKey": _settings.polygon_api_key},
    )
    r.raise_for_status()
    data = r.json()["results"]
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"t": "date", "o": "open", "c": "close", "h": "high", "l": "low", "v": "volume"})
    return df

# -- 3. Loader orchestrator ---------------------------------------------------
async def nightly_etl() -> None:
    """Fetch & persist yesterday's data"""
    async with httpx.AsyncClient(timeout=30) as session:
        since = dt.datetime.utcnow() - dt.timedelta(days=1)
        form4 = await fetch_form4(session, since)
        form4.to_sql("form4", _engine, if_exists="append", index=False)
    # price loading for tracked universe omitted for brevity.