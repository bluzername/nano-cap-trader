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
        
        # Fetch insider filings
        form4 = await fetch_form4(session, since)
        if not form4.empty:
            form4.to_sql("form4", _engine, if_exists="append", index=False)
        
        # Fetch short interest data (free sources or Ortex)
        await fetch_short_interest_data(session)
        
        # price loading for tracked universe omitted for brevity.

# -- 4. Short Interest Data ETL -----------------------------------------------
async def fetch_short_interest_data(session: httpx.AsyncClient) -> None:
    """Fetch short interest data from free sources and store to database."""
    from .alt_data import (
        fetch_finra_short_sale_volume, 
        analyze_daily_short_pressure,
        get_short_interest_data
    )
    from .universe import current_universe
    
    try:
        # Get current universe of symbols
        universe_df = current_universe()
        if universe_df.empty:
            print("No symbols in universe, skipping short interest ETL")
            return
            
        symbols = universe_df["ticker"].tolist()
        yesterday = (dt.datetime.utcnow() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Fetch daily short sale volume from FINRA
        short_sale_data = await fetch_finra_short_sale_volume(session, yesterday)
        if not short_sale_data.empty:
            short_sale_data.to_sql("finra_short_sales", _engine, if_exists="append", index=False)
            print(f"Stored {len(short_sale_data)} FINRA short sale records")
        
        # Fetch aggregated short interest data
        short_interest_data = await get_short_interest_data(session, symbols[:50])  # Limit to avoid rate limits
        if not short_interest_data.empty:
            short_interest_data["date"] = dt.datetime.utcnow().date()
            short_interest_data.to_sql("short_interest", _engine, if_exists="append", index=False)
            print(f"Stored short interest data for {len(short_interest_data)} symbols")
        
        # Analyze short pressure
        pressure_data = await analyze_daily_short_pressure(session, yesterday)
        if not pressure_data.empty:
            pressure_data.to_sql("short_pressure", _engine, if_exists="append", index=False)
            print(f"Stored short pressure data for {len(pressure_data)} symbols")
            
    except Exception as e:
        print(f"Error in short interest ETL: {e}")

# -- 5. Fetch latest short interest for signals -------------------------------
async def get_latest_short_data(symbols: list[str]) -> pd.DataFrame:
    """Get latest short interest data for signal generation."""
    try:
        from sqlalchemy import text
        
        # Try to get recent data from database first
        query = text("""
            SELECT symbol, shares_short_consensus, days_to_cover_consensus, 
                   short_pressure_score, utilization_proxy
            FROM short_interest 
            WHERE date >= date('now', '-7 days')
            AND symbol IN ({})
            ORDER BY date DESC
        """.format(','.join(f"'{s}'" for s in symbols)))
        
        df = pd.read_sql(query, _engine)
        
        if not df.empty:
            # Get most recent record per symbol
            return df.groupby("symbol").first().reset_index()
        else:
            # If no recent data, return empty DataFrame
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching latest short data: {e}")
        return pd.DataFrame()