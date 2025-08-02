"""Correlation and cointegration data provider using multiple sources."""
from __future__ import annotations
import asyncio
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import httpx
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
import yfinance as yf
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class CorrelationDataProvider:
    """Multi-source correlation and cointegration data provider."""
    
    def __init__(self):
        self.alpha_vantage_key = getattr(_settings, 'alpha_vantage_key', None)
        self.polygon_key = _settings.polygon_api_key
        
        # Cache for computed correlations
        self.correlation_cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, dt.datetime] = {}
        self.cache_duration = dt.timedelta(hours=1)  # Cache for 1 hour
        
    async def get_correlation_matrix(
        self,
        symbols: List[str],
        lookback_days: int = 60,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get correlation matrix for given symbols."""
        try:
            cache_key = f"corr_{len(symbols)}_{lookback_days}"
            
            # Check cache
            if use_cache and self._is_cache_valid(cache_key):
                return self.correlation_cache[cache_key]
            
            # Fetch price data
            price_data = await self._fetch_price_data(symbols, lookback_days)
            
            if price_data.empty:
                logger.warning("No price data available for correlation calculation")
                return pd.DataFrame()
            
            # Calculate returns
            returns = price_data.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            # Cache the result
            if use_cache:
                self.correlation_cache[cache_key] = correlation_matrix
                self.cache_timestamps[cache_key] = dt.datetime.now()
            
            logger.info(f"Calculated correlation matrix for {len(symbols)} symbols")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    async def get_cointegration_pairs(
        self,
        symbols: List[str],
        lookback_days: int = 60,
        p_value_threshold: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """Find cointegrated pairs using Engle-Granger test."""
        try:
            price_data = await self._fetch_price_data(symbols, lookback_days)
            
            if price_data.empty:
                return []
            
            cointegrated_pairs = []
            
            # Test all pairs
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    if symbol1 in price_data.columns and symbol2 in price_data.columns:
                        series1 = price_data[symbol1].dropna()
                        series2 = price_data[symbol2].dropna()
                        
                        # Align series
                        aligned = pd.concat([series1, series2], axis=1).dropna()
                        if len(aligned) < 30:  # Need minimum data points
                            continue
                        
                        # Engle-Granger cointegration test
                        p_value = self._engle_granger_test(
                            aligned.iloc[:, 0], 
                            aligned.iloc[:, 1]
                        )
                        
                        if p_value < p_value_threshold:
                            cointegrated_pairs.append((symbol1, symbol2, p_value))
            
            # Sort by p-value (stronger cointegration first)
            cointegrated_pairs.sort(key=lambda x: x[2])
            
            logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
            return cointegrated_pairs
            
        except Exception as e:
            logger.error(f"Error finding cointegration pairs: {e}")
            return []
    
    def hierarchical_clustering(
        self,
        correlation_matrix: pd.DataFrame,
        n_clusters: int = 5,
        method: str = 'ward'
    ) -> Dict[str, int]:
        """Perform hierarchical clustering on correlation matrix."""
        try:
            if correlation_matrix.empty:
                return {}
            
            # Convert correlation to distance matrix
            distance_matrix = 1 - correlation_matrix.abs()
            
            # Fill NaN with maximum distance
            distance_matrix = distance_matrix.fillna(1.0)
            
            # Ensure symmetric matrix
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Hierarchical clustering
            linkage_matrix = linkage(distance_matrix.values, method=method)
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Map symbols to clusters
            symbol_clusters = dict(zip(correlation_matrix.index, clusters))
            
            logger.info(f"Created {n_clusters} clusters from {len(correlation_matrix)} symbols")
            return symbol_clusters
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            return {}
    
    def get_cluster_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        clusters: Dict[str, int],
        min_correlation: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Get strongly correlated pairs within clusters."""
        try:
            pairs = []
            
            # Group symbols by cluster
            cluster_groups = {}
            for symbol, cluster_id in clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(symbol)
            
            # Find pairs within each cluster
            for cluster_id, symbols in cluster_groups.items():
                for i, symbol1 in enumerate(symbols):
                    for symbol2 in symbols[i+1:]:
                        if symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                            correlation = correlation_matrix.loc[symbol1, symbol2]
                            
                            if abs(correlation) >= min_correlation:
                                pairs.append((symbol1, symbol2, correlation))
            
            # Sort by correlation strength
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            logger.info(f"Found {len(pairs)} high-correlation pairs in clusters")
            return pairs
            
        except Exception as e:
            logger.error(f"Error finding cluster pairs: {e}")
            return []
    
    async def _fetch_price_data(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Fetch price data from multiple sources with fallback."""
        try:
            # Try Alpha Vantage first (if available)
            if self.alpha_vantage_key:
                try:
                    return await self._fetch_from_alpha_vantage(symbols, lookback_days)
                except Exception as e:
                    logger.warning(f"Alpha Vantage failed, trying Yahoo Finance: {e}")
            
            # Fallback to Yahoo Finance
            try:
                return await self._fetch_from_yahoo(symbols, lookback_days)
            except Exception as e:
                logger.warning(f"Yahoo Finance failed, trying Polygon: {e}")
            
            # Final fallback to Polygon
            return await self._fetch_from_polygon(symbols, lookback_days)
            
        except Exception as e:
            logger.error(f"All price data sources failed: {e}")
            return pd.DataFrame()
    
    async def _fetch_from_alpha_vantage(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Fetch price data from Alpha Vantage."""
        price_data = {}
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)
        
        async with httpx.AsyncClient(timeout=60) as client:
            for symbol in symbols:
                try:
                    params = {
                        'function': 'TIME_SERIES_DAILY_ADJUSTED',
                        'symbol': symbol,
                        'apikey': self.alpha_vantage_key,
                        'outputsize': 'compact'
                    }
                    
                    response = await client.get(
                        "https://www.alphavantage.co/query",
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        
                        prices = []
                        dates = []
                        
                        for date_str, values in time_series.items():
                            date = dt.datetime.strptime(date_str, '%Y-%m-%d')
                            if start_date <= date <= end_date:
                                prices.append(float(values['5. adjusted close']))
                                dates.append(date)
                        
                        if prices:
                            price_series = pd.Series(prices, index=dates, name=symbol)
                            price_data[symbol] = price_series
                    
                    # Rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.warning(f"Alpha Vantage error for {symbol}: {e}")
                    continue
        
        if price_data:
            return pd.DataFrame(price_data).sort_index()
        return pd.DataFrame()
    
    async def _fetch_from_yahoo(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Fetch price data from Yahoo Finance."""
        try:
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=lookback_days)
            
            # Use asyncio to run the synchronous yfinance call
            def fetch_yahoo_data():
                try:
                    data = yf.download(
                        symbols,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        show_errors=False
                    )
                    
                    if data.empty:
                        return pd.DataFrame()
                    
                    # Extract adjusted close prices
                    if len(symbols) == 1:
                        # Single symbol
                        return pd.DataFrame({symbols[0]: data['Adj Close']})
                    else:
                        # Multiple symbols
                        if 'Adj Close' in data.columns:
                            return data['Adj Close']
                        else:
                            return pd.DataFrame()
                            
                except Exception as e:
                    logger.error(f"Yahoo Finance fetch error: {e}")
                    return pd.DataFrame()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            price_data = await loop.run_in_executor(None, fetch_yahoo_data)
            
            return price_data.dropna()
            
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
            return pd.DataFrame()
    
    async def _fetch_from_polygon(
        self,
        symbols: List[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Fetch price data from Polygon.io."""
        price_data = {}
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=lookback_days)
        
        async with httpx.AsyncClient(timeout=60) as client:
            for symbol in symbols:
                try:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                    params = {'apiKey': self.polygon_key}
                    
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'results' in data and data['results']:
                        prices = []
                        dates = []
                        
                        for result in data['results']:
                            date = dt.datetime.fromtimestamp(result['t'] / 1000)
                            prices.append(result['c'])  # Close price
                            dates.append(date)
                        
                        if prices:
                            price_series = pd.Series(prices, index=dates, name=symbol)
                            price_data[symbol] = price_series
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Polygon error for {symbol}: {e}")
                    continue
        
        if price_data:
            return pd.DataFrame(price_data).sort_index()
        return pd.DataFrame()
    
    def _engle_granger_test(self, series1: pd.Series, series2: pd.Series) -> float:
        """Perform Engle-Granger cointegration test."""
        try:
            # Step 1: Run OLS regression
            from sklearn.linear_model import LinearRegression
            
            X = series1.values.reshape(-1, 1)
            y = series2.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Step 2: Get residuals
            residuals = y - model.predict(X)
            
            # Step 3: ADF test on residuals
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(residuals, maxlag=1)
            p_value = adf_result[1]
            
            return p_value
            
        except Exception as e:
            logger.error(f"Error in Engle-Granger test: {e}")
            return 1.0  # Return high p-value on error
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = dt.datetime.now() - self.cache_timestamps[cache_key]
        return age < self.cache_duration
    
    async def get_sector_correlations(
        self,
        symbols: List[str],
        sector_etfs: Dict[str, str] = None
    ) -> pd.DataFrame:
        """Get correlations with sector ETFs for factor attribution."""
        try:
            if sector_etfs is None:
                # Default sector ETFs
                sector_etfs = {
                    'Technology': 'XLK',
                    'Healthcare': 'XLV',
                    'Financials': 'XLF',
                    'Energy': 'XLE',
                    'Industrials': 'XLI',
                    'Consumer Discretionary': 'XLY',
                    'Consumer Staples': 'XLP',
                    'Utilities': 'XLU',
                    'Materials': 'XLB',
                    'Real Estate': 'XLRE',
                    'Communication': 'XLC'
                }
            
            all_symbols = symbols + list(sector_etfs.values())
            correlation_matrix = await self.get_correlation_matrix(all_symbols, 60)
            
            if correlation_matrix.empty:
                return pd.DataFrame()
            
            # Extract correlations between symbols and sector ETFs
            sector_correlations = pd.DataFrame()
            
            for symbol in symbols:
                if symbol in correlation_matrix.index:
                    row = {}
                    for sector, etf in sector_etfs.items():
                        if etf in correlation_matrix.columns:
                            row[sector] = correlation_matrix.loc[symbol, etf]
                    
                    if row:
                        sector_correlations = pd.concat([
                            sector_correlations,
                            pd.DataFrame([row], index=[symbol])
                        ])
            
            return sector_correlations
            
        except Exception as e:
            logger.error(f"Error calculating sector correlations: {e}")
            return pd.DataFrame()