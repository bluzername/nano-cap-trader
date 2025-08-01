"""Alternative data sources for short interest using free/open APIs."""
from __future__ import annotations
import datetime as dt
import httpx
import pandas as pd
from typing import Optional
from .config import get_settings

_settings = get_settings()

# -- 1. FINRA Reg SHO Daily Short Sale Volume ----------------------------------
async def fetch_finra_short_sale_volume(session: httpx.AsyncClient, date: str) -> pd.DataFrame:
    """Fetch daily short sale volume from FINRA RegSHO data.
    
    Args:
        session: HTTP client session
        date: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with columns: [symbol, shortVolume, totalVolume, shortExemptVolume]
    """
    params = {
        "limit": 5000,
        "offset": 0,
        "date": date,
    }
    
    try:
        r = await session.get(_settings.finra_short_sale_url, params=params)
        r.raise_for_status()
        data = r.json()
        
        if not data or len(data) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(date)
        df["short_ratio"] = df["shortVolume"] / df["totalVolume"].replace(0, 1)
        return df
        
    except Exception as e:
        print(f"Error fetching FINRA short sale data: {e}")
        return pd.DataFrame()

# -- 2. FINRA Bi-weekly Short Interest ----------------------------------------
async def fetch_finra_short_interest(session: httpx.AsyncClient, symbol: str) -> pd.DataFrame:
    """Fetch bi-weekly short interest from FINRA.
    
    Args:
        session: HTTP client session
        symbol: Stock symbol
        
    Returns:
        DataFrame with short interest data
    """
    params = {
        "symbol": symbol,
        "limit": 100,
    }
    
    try:
        r = await session.get(_settings.finra_short_interest_url, params=params)
        r.raise_for_status()
        data = r.json()
        
        if not data or len(data) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df["settlementDate"] = pd.to_datetime(df["settlementDate"])
        return df.sort_values("settlementDate").tail(1)  # Latest only
        
    except Exception as e:
        print(f"Error fetching FINRA short interest for {symbol}: {e}")
        return pd.DataFrame()

# -- 3. Finnhub Short Interest (Free Tier) -----------------------------------
async def fetch_finnhub_short_interest(session: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Fetch short interest from Finnhub free tier.
    
    Args:
        session: HTTP client session
        symbol: Stock symbol
        
    Returns:
        Dict with short interest data or None
    """
    if not _settings.finnhub_api_key:
        return None
        
    params = {
        "symbol": symbol,
        "token": _settings.finnhub_api_key,
    }
    
    try:
        r = await session.get(f"{_settings.finnhub_base_url}/stock/short-interest", params=params)
        r.raise_for_status()
        data = r.json()
        return data if data else None
        
    except Exception as e:
        print(f"Error fetching Finnhub short interest for {symbol}: {e}")
        return None

# -- 4. Financial Modeling Prep Short Interest --------------------------------
async def fetch_fmp_short_interest(session: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Fetch short interest from Financial Modeling Prep.
    
    Args:
        session: HTTP client session
        symbol: Stock symbol
        
    Returns:
        Dict with short interest data or None
    """
    if not _settings.fmp_api_key:
        return None
        
    params = {
        "symbol": symbol,
        "apikey": _settings.fmp_api_key,
    }
    
    try:
        r = await session.get(f"{_settings.fmp_base_url}/short-interest", params=params)
        r.raise_for_status()
        data = r.json()
        return data[0] if data and len(data) > 0 else None
        
    except Exception as e:
        print(f"Error fetching FMP short interest for {symbol}: {e}")
        return None

# -- 5. Calculate Days to Cover -----------------------------------------------
def calculate_days_to_cover(shares_short: float, avg_volume: float, price: float) -> float:
    """Calculate days to cover using free data sources.
    
    Formula: days_to_cover = shares_short / max(1, 5-day_avg_dollar_volume / price)
    
    Args:
        shares_short: Number of shares sold short
        avg_volume: 5-day average volume
        price: Current stock price
        
    Returns:
        Days to cover ratio
    """
    if shares_short <= 0 or avg_volume <= 0 or price <= 0:
        return 0.0
        
    daily_share_volume = max(1, avg_volume)
    return shares_short / daily_share_volume

# -- 6. Aggregate Short Interest Data -----------------------------------------
async def get_short_interest_data(session: httpx.AsyncClient, symbols: list[str]) -> pd.DataFrame:
    """Aggregate short interest data from multiple free sources.
    
    Args:
        session: HTTP client session
        symbols: List of stock symbols
        
    Returns:
        DataFrame with consolidated short interest metrics
    """
    results = []
    
    for symbol in symbols:
        row = {"symbol": symbol}
        
        # Try Finnhub first (if API key available)
        finnhub_data = await fetch_finnhub_short_interest(session, symbol)
        if finnhub_data:
            row.update({
                "shares_short_finnhub": finnhub_data.get("shortInterest", 0),
                "days_to_cover_finnhub": finnhub_data.get("daysToCover", 0),
            })
        
        # Try Financial Modeling Prep (if API key available)
        fmp_data = await fetch_fmp_short_interest(session, symbol)
        if fmp_data:
            row.update({
                "shares_short_fmp": fmp_data.get("shortInterest", 0),
                "days_to_cover_fmp": fmp_data.get("daysToCover", 0),
            })
        
        # Try FINRA short interest
        finra_si = await fetch_finra_short_interest(session, symbol)
        if not finra_si.empty:
            row.update({
                "shares_short_finra": finra_si.iloc[0].get("shortInterest", 0),
                "short_interest_ratio_finra": finra_si.iloc[0].get("shortInterestRatio", 0),
            })
        
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Create consensus metrics (average across sources)
    short_cols = [col for col in df.columns if "shares_short" in col]
    if short_cols:
        df["shares_short_consensus"] = df[short_cols].mean(axis=1, skipna=True)
    
    dtc_cols = [col for col in df.columns if "days_to_cover" in col]
    if dtc_cols:
        df["days_to_cover_consensus"] = df[dtc_cols].mean(axis=1, skipna=True)
    
    return df

# -- 7. Daily Short Sale Volume Analysis --------------------------------------
async def analyze_daily_short_pressure(session: httpx.AsyncClient, date: str) -> pd.DataFrame:
    """Analyze daily short sale pressure using FINRA data.
    
    Args:
        session: HTTP client session
        date: Date in YYYY-MM-DD format
        
    Returns:
        DataFrame with short pressure metrics
    """
    # Get daily short sale volume
    short_vol_df = await fetch_finra_short_sale_volume(session, date)
    
    if short_vol_df.empty:
        return pd.DataFrame()
    
    # Calculate short pressure metrics
    short_vol_df["short_pressure_score"] = short_vol_df["short_ratio"] * 100
    
    # Flag high short pressure (>40% as suggested)
    short_vol_df["high_short_pressure"] = short_vol_df["short_ratio"] > 0.40
    
    # Rank by short pressure
    short_vol_df["short_pressure_rank"] = short_vol_df["short_pressure_score"].rank(
        method="dense", ascending=False
    )
    
    return short_vol_df.sort_values("short_pressure_score", ascending=False)

# -- 8. Ortex Alternative Wrapper ---------------------------------------------
async def get_ortex_alternative_data(session: httpx.AsyncClient, symbols: list[str]) -> pd.DataFrame:
    """Get Ortex alternative data using free sources.
    
    This function mimics Ortex's key metrics using free data sources:
    - Short interest from FINRA/Finnhub/FMP
    - Short sale volume from FINRA RegSHO
    - Days to cover calculation
    - Short pressure indicators
    
    Args:
        session: HTTP client session
        symbols: List of stock symbols
        
    Returns:
        DataFrame with Ortex-like metrics
    """
    # Get short interest data
    si_data = await get_short_interest_data(session, symbols)
    
    # Get latest daily short sale data
    yesterday = (dt.datetime.now() - dt.timedelta(days=1)).strftime("%Y-%m-%d")
    daily_data = await analyze_daily_short_pressure(session, yesterday)
    
    # Merge the datasets
    if not daily_data.empty:
        merged = si_data.merge(
            daily_data[["symbol", "short_ratio", "short_pressure_score", "high_short_pressure"]],
            on="symbol",
            how="left"
        )
    else:
        merged = si_data
        merged["short_ratio"] = 0
        merged["short_pressure_score"] = 0
        merged["high_short_pressure"] = False
    
    # Calculate utilization proxy (short interest / float)
    # Note: This requires float data from your universe module
    merged["utilization_proxy"] = merged.get("shares_short_consensus", 0) / 1_000_000  # Placeholder
    
    return merged