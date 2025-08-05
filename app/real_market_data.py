"""Real market data fetching with comprehensive warnings and filter validation."""
from __future__ import annotations
import datetime as dt
import httpx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .config import get_settings
# Import fetch_daily_prices directly to avoid SQLAlchemy dependency
import httpx
import pandas as pd
from datetime import datetime, timedelta

async def fetch_daily_prices(session: httpx.AsyncClient, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily price data from Polygon.io without SQLAlchemy dependency."""
    try:
        from .config import get_settings
        settings = get_settings()
        
        if not settings.polygon_api_key:
            return pd.DataFrame()
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "apiKey": settings.polygon_api_key,
            "adjusted": "true",
            "sort": "asc"
        }
        
        response = await session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("resultsCount", 0) == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date")
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "o": "open",
            "h": "high", 
            "l": "low",
            "c": "close",
            "v": "volume"
        })
        
        return df[["open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

_settings = get_settings()

@dataclass
class FilterWarning:
    """Represents a filter warning with details."""
    symbol: str
    filter_name: str
    message: str
    severity: str  # "WARNING", "ERROR", "INFO"
    current_value: Optional[float] = None
    required_value: Optional[float] = None

@dataclass
class MarketDataResult:
    """Result of market data fetching with warnings."""
    market_data: Dict[str, pd.DataFrame]
    warnings: List[FilterWarning]
    data_quality_score: float  # 0.0 to 1.0
    symbols_with_data: List[str]
    symbols_missing_data: List[str]
    is_real_data: bool
    
    def get_warning_summary(self) -> Dict[str, Any]:
        """Get a summary of all warnings for display."""
        if not self.warnings:
            return {
                "has_warnings": False,
                "total_warnings": 0,
                "error_count": 0,
                "warning_count": 0,
                "info_count": 0,
                "summary": "All data quality checks passed"
            }
        
        error_count = sum(1 for w in self.warnings if w.severity == "ERROR")
        warning_count = sum(1 for w in self.warnings if w.severity == "WARNING")
        info_count = sum(1 for w in self.warnings if w.severity == "INFO")
        
        warnings_by_symbol = {}
        for warning in self.warnings:
            if warning.symbol not in warnings_by_symbol:
                warnings_by_symbol[warning.symbol] = []
            warnings_by_symbol[warning.symbol].append(warning)
        
        return {
            "has_warnings": True,
            "total_warnings": len(self.warnings),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "warnings_by_symbol": warnings_by_symbol,
            "summary": f"{error_count} errors, {warning_count} warnings, {info_count} info messages"
        }

class RealMarketDataFetcher:
    """Fetches real market data and validates against strategy filters."""
    
    def __init__(self):
        self.settings = get_settings()
        self.warnings = []
        
    async def fetch_market_data(self, symbols: List[str], days: int = 60) -> MarketDataResult:
        """Fetch real market data for symbols with comprehensive validation."""
        self.warnings = []
        market_data = {}
        symbols_with_data = []
        symbols_missing_data = []
        
        # Check if Polygon API key is configured
        if not self.settings.polygon_api_key:
            for symbol in symbols:
                self.warnings.append(FilterWarning(
                    symbol=symbol,
                    filter_name="API_CONFIGURATION",
                    message="Polygon.io API key not configured - cannot fetch real market data",
                    severity="ERROR"
                ))
            symbols_missing_data = symbols
            return MarketDataResult(
                market_data={},
                warnings=self.warnings,
                data_quality_score=0.0,
                symbols_with_data=[],
                symbols_missing_data=symbols_missing_data,
                is_real_data=False
            )
        
        # Calculate date range
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=days)
        
        async with httpx.AsyncClient(timeout=30) as session:
            for symbol in symbols:
                try:
                    # Fetch real price data from Polygon
                    df = await fetch_daily_prices(
                        session, 
                        symbol, 
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                    
                    if not df.empty and len(df) >= 30:  # Minimum 30 days of data
                        quality_warnings = self._validate_data_quality(symbol, df)
                        self.warnings.extend(quality_warnings)
                        filter_warnings = self._apply_strategy_filters(symbol, df)
                        self.warnings.extend(filter_warnings)
                        market_data[symbol] = df
                        symbols_with_data.append(symbol)
                    else:
                        symbols_missing_data.append(symbol)
                        self.warnings.append(FilterWarning(
                            symbol=symbol,
                            filter_name="DATA_AVAILABILITY",
                            message=f"Insufficient data: {len(df) if not df.empty else 0} days available (need 30+)",
                            severity="ERROR"
                        ))
                        
                except Exception as e:
                    symbols_missing_data.append(symbol)
                    self.warnings.append(FilterWarning(
                        symbol=symbol,
                        filter_name="API_ERROR",
                        message=f"Failed to fetch data: {str(e)}",
                        severity="ERROR"
                    ))
        
        data_quality_score = len(symbols_with_data) / len(symbols) if symbols else 0.0
        
        return MarketDataResult(
            market_data=market_data,
            warnings=self.warnings,
            data_quality_score=data_quality_score,
            symbols_with_data=symbols_with_data,
            symbols_missing_data=symbols_missing_data,
            is_real_data=True
        )
    
    def _validate_data_quality(self, symbol: str, df: pd.DataFrame) -> List[FilterWarning]:
        """Validate basic data quality metrics."""
        warnings = []
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_pct > 5:
            warnings.append(FilterWarning(
                symbol=symbol,
                filter_name="DATA_QUALITY",
                message=f"High missing data: {missing_pct:.1f}% missing values",
                severity="WARNING",
                current_value=missing_pct,
                required_value=5.0
            ))
        
        # Check for zero volumes
        zero_volume_days = (df['volume'] == 0).sum()
        if zero_volume_days > len(df) * 0.1:
            warnings.append(FilterWarning(
                symbol=symbol,
                filter_name="LIQUIDITY",
                message=f"Low liquidity: {zero_volume_days} zero-volume days",
                severity="WARNING",
                current_value=zero_volume_days,
                required_value=len(df) * 0.1
            ))
        
        return warnings
    
    def _apply_strategy_filters(self, symbol: str, df: pd.DataFrame) -> List[FilterWarning]:
        """Apply strategy-specific filters and generate warnings."""
        warnings = []
        
        avg_volume = df['volume'].mean()
        avg_price = df['close'].mean()
        price_volatility = df['close'].pct_change().std()
        
        # Volume filter
        min_volume = 100000
        if avg_volume < min_volume:
            warnings.append(FilterWarning(
                symbol=symbol,
                filter_name="VOLUME_FILTER",
                message=f"Low average volume: {avg_volume:,.0f} shares/day (min: {min_volume:,})",
                severity="WARNING",
                current_value=avg_volume,
                required_value=min_volume
            ))
        
        # Price filter
        min_price = 1.0
        if avg_price < min_price:
            warnings.append(FilterWarning(
                symbol=symbol,
                filter_name="PRICE_FILTER",
                message=f"Low average price: ${avg_price:.2f} (min: ${min_price})",
                severity="WARNING",
                current_value=avg_price,
                required_value=min_price
            ))
        
        return warnings

_real_data_fetcher = RealMarketDataFetcher()

async def get_real_market_data(symbols: List[str], days: int = 60) -> MarketDataResult:
    """Convenience function to get real market data."""
    return await _real_data_fetcher.fetch_market_data(symbols, days) 