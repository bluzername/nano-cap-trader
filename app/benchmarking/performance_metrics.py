"""Advanced performance metrics and benchmarking."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import asyncio
import httpx
import logging
import hashlib

# Add color formatting for errors and warnings
RED = "\033[91m"
RESET = "\033[0m"
YELLOW = "\033[93m"
CYAN = "\033[96m"

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkData:
    """Benchmark data container."""
    name: str
    symbol: str
    returns: pd.Series
    prices: pd.Series
    start_date: dt.datetime
    end_date: dt.datetime


class PerformanceAnalyzer:
    """Advanced performance analysis with multiple benchmarks."""
    
    def __init__(self):
        # Available benchmarks
        self.benchmarks = {
            'russell_2000': 'IWM',  # Russell 2000 ETF
            'sp_600': 'IJR',        # S&P SmallCap 600 ETF
            'sp_500': 'SPY',        # S&P 500 ETF
            'nasdaq': 'QQQ',        # Nasdaq 100 ETF
            'vti': 'VTI',          # Total Stock Market ETF
        }
        
        # Cache for benchmark data
        self.benchmark_cache: Dict[str, BenchmarkData] = {}
        self.cache_duration = dt.timedelta(hours=6)
        self.cache_timestamps: Dict[str, dt.datetime] = {}
    
    async def get_benchmark_data(
        self,
        benchmark_name: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        use_cache: bool = True
    ) -> Optional[BenchmarkData]:
        """Get benchmark data for comparison."""
        try:
            cache_key = f"{benchmark_name}_{start_date.date()}_{end_date.date()}"
            
            # Check cache
            if use_cache and self._is_cache_valid(cache_key):
                return self.benchmark_cache[cache_key]
            
            if benchmark_name not in self.benchmarks:
                logger.error(f"{RED}Unknown benchmark: {benchmark_name}{RESET}")
                return None
            
            symbol = self.benchmarks[benchmark_name]
            
            # Fetch data using multiple sources
            benchmark_data = await self._fetch_benchmark_data(symbol, start_date, end_date)
            
            if benchmark_data is not None:
                # Cache the result
                if use_cache:
                    self.benchmark_cache[cache_key] = benchmark_data
                    self.cache_timestamps[cache_key] = dt.datetime.now()
                
                logger.info(f"Loaded benchmark {benchmark_name} ({symbol}) with {len(benchmark_data.returns)} data points")
            
            return benchmark_data
            
        except Exception as e:
            logger.error(f"{RED}Error getting benchmark data for {benchmark_name}: {e}{RESET}")
            return None
    
    async def _fetch_benchmark_data(
        self,
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime
    ) -> Optional[BenchmarkData]:
        """Fetch benchmark data from multiple sources."""
        try:
            # Try Yahoo Finance first
            try:
                import yfinance as yf
                # Explicitly set auto_adjust to keep expected columns stable
                data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
                
                if not data.empty:
                    # Prefer Adj Close when available; fallback to Close
                    if 'Adj Close' in data.columns:
                        prices = data['Adj Close']
                    elif 'Close' in data.columns:
                        prices = data['Close']
                    else:
                        raise KeyError("Neither 'Adj Close' nor 'Close' present in Yahoo Finance data")
                    returns = prices.pct_change().dropna()
                    logger.info(f"{CYAN}Yahoo Finance loaded for {symbol}: prices={len(prices)}, returns={len(returns)}{RESET}")
                    logger.debug(f"{CYAN}Yahoo Finance {symbol} returns head: {returns.head(5).to_dict()} index: {list(returns.head(5).index)}{RESET}")
                    logger.debug(f"{CYAN}Yahoo Finance {symbol} returns hash: {hashlib.md5(returns.values.tobytes()).hexdigest()}{RESET}")
                    
                    return BenchmarkData(
                        name=symbol,
                        symbol=symbol,
                        returns=returns,
                        prices=prices,
                        start_date=start_date,
                        end_date=end_date
                    )
            except Exception as e:
                logger.warning(f"{RED}Yahoo Finance failed for {symbol}: {e}{RESET}")
            
            # Fallback to Alpha Vantage if available
            from ..config import get_settings
            settings = get_settings()
            
            if hasattr(settings, 'alpha_vantage_key') and settings.alpha_vantage_key:
                try:
                    data = await self._fetch_from_alpha_vantage(symbol, start_date, end_date, settings.alpha_vantage_key)
                    if data is not None:
                        logger.info(f"Alpha Vantage loaded for {symbol}: returns={len(data.returns)}")
                        logger.debug(f"Alpha Vantage {symbol} returns head: {data.returns.head(5).to_dict()} index: {list(data.returns.head(5).index)}")
                        logger.debug(f"Alpha Vantage {symbol} returns hash: {hashlib.md5(data.returns.values.tobytes()).hexdigest()}")
                        return data
                except Exception as e:
                    logger.warning(f"{RED}Alpha Vantage failed for {symbol}: {e}{RESET}")
            
            # Fallback to Polygon
            if hasattr(settings, 'polygon_api_key') and settings.polygon_api_key:
                try:
                    data = await self._fetch_from_polygon(symbol, start_date, end_date, settings.polygon_api_key)
                    if data is not None:
                        logger.info(f"Polygon loaded for {symbol}: returns={len(data.returns)}")
                        logger.debug(f"Polygon {symbol} returns head: {data.returns.head(5).to_dict()} index: {list(data.returns.head(5).index)}")
                        logger.debug(f"Polygon {symbol} returns hash: {hashlib.md5(data.returns.values.tobytes()).hexdigest()}")
                        return data
                except Exception as e:
                    logger.warning(f"{RED}Polygon failed for {symbol}: {e}{RESET}")
            
            logger.error(f"{RED}All data sources failed for benchmark {symbol}{RESET}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")
            return None
    
    async def _fetch_from_alpha_vantage(
        self,
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        api_key: str
    ) -> Optional[BenchmarkData]:
        """Fetch data from Alpha Vantage."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                params = {
                    'function': 'TIME_SERIES_DAILY_ADJUSTED',
                    'symbol': symbol,
                    'apikey': api_key,
                    'outputsize': 'full'
                }
                
                response = await client.get("https://www.alphavantage.co/query", params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'Time Series (Daily)' not in data:
                    return None
                
                time_series = data['Time Series (Daily)']
                
                prices = []
                dates = []
                
                for date_str, values in time_series.items():
                    date = dt.datetime.strptime(date_str, '%Y-%m-%d')
                    if start_date <= date <= end_date:
                        prices.append(float(values['5. adjusted close']))
                        dates.append(date)
                
                if not prices:
                    return None
                
                price_series = pd.Series(prices, index=dates).sort_index()
                returns = price_series.pct_change().dropna()
                logger.info(f"Alpha Vantage parsed for {symbol}: prices={len(price_series)}, returns={len(returns)}")
                logger.debug(f"Alpha Vantage parsed {symbol} returns head: {returns.head(5).to_dict()} index: {list(returns.head(5).index)}")
                logger.debug(f"Alpha Vantage parsed {symbol} returns hash: {hashlib.md5(returns.values.tobytes()).hexdigest()}")
                
                return BenchmarkData(
                    name=symbol,
                    symbol=symbol,
                    returns=returns,
                    prices=price_series,
                    start_date=start_date,
                    end_date=end_date
                )
                
        except Exception as e:
            logger.error(f"{RED}Alpha Vantage error for {symbol}: {e}{RESET}")
            return None
    
    async def _fetch_from_polygon(
        self,
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        api_key: str
    ) -> Optional[BenchmarkData]:
        """Fetch data from Polygon."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                params = {'apiKey': api_key}
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'results' not in data or not data['results']:
                    return None
                
                prices = []
                dates = []
                
                for result in data['results']:
                    date = dt.datetime.fromtimestamp(result['t'] / 1000)
                    prices.append(result['c'])  # Close price
                    dates.append(date)
                
                if not prices:
                    return None
                
                price_series = pd.Series(prices, index=dates).sort_index()
                returns = price_series.pct_change().dropna()
                logger.info(f"Polygon parsed for {symbol}: prices={len(price_series)}, returns={len(returns)}")
                logger.debug(f"Polygon parsed {symbol} returns head: {returns.head(5).to_dict()} index: {list(returns.head(5).index)}")
                logger.debug(f"Polygon parsed {symbol} returns hash: {hashlib.md5(returns.values.tobytes()).hexdigest()}")
                
                return BenchmarkData(
                    name=symbol,
                    symbol=symbol,
                    returns=returns,
                    prices=price_series,
                    start_date=start_date,
                    end_date=end_date
                )
                
        except Exception as e:
            logger.error(f"{RED}Polygon error for {symbol}: {e}{RESET}")
            return None
    
    def calculate_comprehensive_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_data: BenchmarkData,
        risk_free_rate: float = 0.02  # 2% annual risk-free rate
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            # Align returns with benchmark
            aligned_returns, benchmark_returns = strategy_returns.align(
                benchmark_data.returns, join='inner'
            )
            
            if len(aligned_returns) < 2:
                logger.warning(f"{RED}Insufficient data for performance calculation{RESET}")
                return {}
            
            # Convert to numpy arrays
            strategy_ret = aligned_returns.values
            benchmark_ret = benchmark_returns.values
            
            # Basic metrics
            total_return = (1 + strategy_ret).prod() - 1
            benchmark_total_return = (1 + benchmark_ret).prod() - 1
            
            # Annualized metrics
            trading_days = len(strategy_ret)
            annual_factor = 252 / trading_days if trading_days > 0 else 1
            
            annualized_return = (1 + total_return) ** annual_factor - 1
            annualized_benchmark_return = (1 + benchmark_total_return) ** annual_factor - 1
            
            # Risk metrics
            strategy_vol = np.std(strategy_ret) * np.sqrt(252)
            benchmark_vol = np.std(benchmark_ret) * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (annualized_return - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0
            
            # Alpha and Beta
            covariance = np.cov(strategy_ret, benchmark_ret)[0][1]
            benchmark_variance = np.var(benchmark_ret)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = annualized_return - risk_free_rate - beta * (annualized_benchmark_return - risk_free_rate)
            
            # Information ratio
            active_returns = strategy_ret - benchmark_ret
            tracking_error = np.std(active_returns) * np.sqrt(252)
            information_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + aligned_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Downside deviation (Sortino ratio)
            downside_returns = strategy_ret[strategy_ret < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Win rate
            winning_days = np.sum(strategy_ret > 0)
            total_days = len(strategy_ret)
            win_rate = winning_days / total_days if total_days > 0 else 0
            
            # Profit factor
            positive_returns = strategy_ret[strategy_ret > 0]
            negative_returns = strategy_ret[strategy_ret < 0]
            
            gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
            gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(strategy_ret, 5)
            
            # Expected Shortfall (Conditional VaR)
            shortfall_returns = strategy_ret[strategy_ret <= var_95]
            expected_shortfall = np.mean(shortfall_returns) if len(shortfall_returns) > 0 else 0
            
            # Correlation with benchmark
            correlation = np.corrcoef(strategy_ret, benchmark_ret)[0, 1]
            
            return {
                # Returns
                'total_return': total_return,
                'annualized_return': annualized_return,
                'benchmark_total_return': benchmark_total_return,
                'benchmark_annualized_return': annualized_benchmark_return,
                'excess_return': total_return - benchmark_total_return,
                
                # Risk-adjusted metrics
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                
                # Risk metrics
                'volatility': strategy_vol,
                'benchmark_volatility': benchmark_vol,
                'max_drawdown': max_drawdown,
                'tracking_error': tracking_error,
                'downside_deviation': downside_deviation,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                
                # Alpha and Beta
                'alpha': alpha,
                'beta': beta,
                'correlation': correlation,
                
                # Trading metrics
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': trading_days,
                
                # Benchmark info
                'benchmark_name': benchmark_data.name,
                'analysis_period_days': trading_days,
            }
            
        except Exception as e:
            logger.error(f"{RED}Error calculating performance metrics: {e}{RESET}")
            return {}
    
    def calculate_rolling_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_data: BenchmarkData,
        window_days: int = 60
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        try:
            # Align returns
            aligned_returns, benchmark_returns = strategy_returns.align(
                benchmark_data.returns, join='inner'
            )
            
            if len(aligned_returns) < window_days:
                return pd.DataFrame()
            
            # Calculate rolling metrics
            rolling_metrics = []
            
            for i in range(window_days, len(aligned_returns)):
                window_strategy = aligned_returns.iloc[i-window_days:i]
                window_benchmark = benchmark_returns.iloc[i-window_days:i]
                
                # Calculate metrics for this window
                metrics = self.calculate_comprehensive_metrics(
                    window_strategy,
                    BenchmarkData(
                        name=benchmark_data.name,
                        symbol=benchmark_data.symbol,
                        returns=window_benchmark,
                        prices=pd.Series(),
                        start_date=window_strategy.index[0],
                        end_date=window_strategy.index[-1]
                    )
                )
                
                metrics['date'] = aligned_returns.index[i]
                rolling_metrics.append(metrics)
            
            return pd.DataFrame(rolling_metrics).set_index('date')
            
        except Exception as e:
            logger.error(f"{RED}Error calculating rolling metrics: {e}{RESET}")
            return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = dt.datetime.now() - self.cache_timestamps[cache_key]
        return age < self.cache_duration
    
    def get_available_benchmarks(self) -> Dict[str, str]:
        """Get available benchmarks."""
        return self.benchmarks.copy()
    
    async def compare_multiple_benchmarks(
        self,
        strategy_returns: pd.Series,
        benchmark_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare strategy against multiple benchmarks."""
        if benchmark_names is None:
            benchmark_names = list(self.benchmarks.keys())
        
        results = {}
        
        if not strategy_returns.empty:
            start_date = strategy_returns.index[0]
            end_date = strategy_returns.index[-1]
            
            for benchmark_name in benchmark_names:
                # Bypass cache during multi-benchmark comparison to avoid stale/identical series
                benchmark_data = await self.get_benchmark_data(benchmark_name, start_date, end_date, use_cache=False)
                if benchmark_data:
                    if benchmark_data.returns is None or benchmark_data.returns.empty:
                        logger.warning(f"{RED}Benchmark {benchmark_name} returned empty series for {benchmark_data.symbol}{RESET}")
                    else:
                        logger.debug(f"{YELLOW}Benchmark {benchmark_name} ({benchmark_data.symbol}) returns head: {benchmark_data.returns.head(5).to_dict()} index: {list(benchmark_data.returns.head(5).index)} hash: {hashlib.md5(benchmark_data.returns.values.tobytes()).hexdigest()}{RESET}")
                    metrics = self.calculate_comprehensive_metrics(strategy_returns, benchmark_data)
                    results[benchmark_name] = metrics
        
        return results