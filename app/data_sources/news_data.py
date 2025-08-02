"""News data aggregation from multiple sources with fallback to Polygon.io."""
from __future__ import annotations
import asyncio
import datetime as dt
from typing import Dict, List, Optional, Any
import httpx
import pandas as pd
from dataclasses import dataclass
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class NewsItem:
    """Structured news item."""
    title: str
    description: str
    source: str
    published_at: dt.datetime
    url: str
    sentiment: Optional[float] = None  # -1 to 1
    relevance: Optional[float] = None  # 0 to 1
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


class NewsDataProvider:
    """Multi-source news data provider with intelligent fallback."""
    
    def __init__(self):
        self.newsapi_key = getattr(_settings, 'newsapi_key', None)
        self.alpha_vantage_key = getattr(_settings, 'alpha_vantage_key', None)
        self.polygon_key = _settings.polygon_api_key
        
        # API endpoints
        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.polygon_url = "https://api.polygon.io/v2/reference/news"
        
        # Rate limiting
        self.last_newsapi_call = None
        self.last_alpha_vantage_call = None
        self.newsapi_calls_today = 0
        self.alpha_vantage_calls_today = 0
        self.max_calls_per_day = 1000
        
    async def get_news_for_symbols(
        self,
        symbols: List[str],
        hours_back: int = 24,
        max_items: int = 100
    ) -> Dict[str, List[NewsItem]]:
        """Get news items for specific symbols from all available sources."""
        try:
            news_by_symbol = {symbol: [] for symbol in symbols}
            
            # Try each source in order of preference
            sources_tried = []
            
            # 1. Try NewsAPI first (best for general news)
            if self.newsapi_key and self._can_make_newsapi_call():
                try:
                    newsapi_items = await self._fetch_from_newsapi(symbols, hours_back)
                    self._merge_news_items(news_by_symbol, newsapi_items)
                    sources_tried.append("NewsAPI")
                except Exception as e:
                    logger.warning(f"NewsAPI failed: {e}")
            
            # 2. Try Alpha Vantage (good for financial news)
            if self.alpha_vantage_key and self._can_make_alpha_vantage_call():
                try:
                    av_items = await self._fetch_from_alpha_vantage(symbols, hours_back)
                    self._merge_news_items(news_by_symbol, av_items)
                    sources_tried.append("Alpha Vantage")
                except Exception as e:
                    logger.warning(f"Alpha Vantage failed: {e}")
            
            # 3. Fallback to Polygon.io (always available)
            try:
                polygon_items = await self._fetch_from_polygon(symbols, hours_back)
                self._merge_news_items(news_by_symbol, polygon_items)
                sources_tried.append("Polygon.io")
            except Exception as e:
                logger.error(f"Polygon news fallback failed: {e}")
            
            # Limit items per symbol
            for symbol in news_by_symbol:
                news_by_symbol[symbol] = news_by_symbol[symbol][:max_items]
            
            logger.info(f"Fetched news from sources: {sources_tried}")
            return news_by_symbol
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return {symbol: [] for symbol in symbols}
    
    async def _fetch_from_newsapi(
        self,
        symbols: List[str],
        hours_back: int
    ) -> Dict[str, List[NewsItem]]:
        """Fetch news from NewsAPI.org."""
        news_items = {}
        
        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in symbols:
                try:
                    since = dt.datetime.now() - dt.timedelta(hours=hours_back)
                    params = {
                        'q': f'"{symbol}" OR "{symbol} stock" OR "{symbol} shares"',
                        'from': since.strftime('%Y-%m-%d'),
                        'sortBy': 'publishedAt',
                        'apiKey': self.newsapi_key,
                        'language': 'en',
                        'pageSize': 50
                    }
                    
                    response = await client.get(self.newsapi_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    items = []
                    for article in data.get('articles', []):
                        if article['title'] and article['description']:
                            item = NewsItem(
                                title=article['title'],
                                description=article['description'],
                                source=article['source']['name'],
                                published_at=dt.datetime.fromisoformat(
                                    article['publishedAt'].replace('Z', '+00:00')
                                ),
                                url=article['url'],
                                symbols=[symbol]
                            )
                            items.append(item)
                    
                    news_items[symbol] = items
                    self.newsapi_calls_today += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"NewsAPI error for {symbol}: {e}")
                    news_items[symbol] = []
        
        return news_items
    
    async def _fetch_from_alpha_vantage(
        self,
        symbols: List[str],
        hours_back: int
    ) -> Dict[str, List[NewsItem]]:
        """Fetch news from Alpha Vantage."""
        news_items = {}
        
        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in symbols:
                try:
                    params = {
                        'function': 'NEWS_SENTIMENT',
                        'tickers': symbol,
                        'apikey': self.alpha_vantage_key,
                        'limit': 50
                    }
                    
                    response = await client.get(self.alpha_vantage_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    items = []
                    for article in data.get('feed', []):
                        try:
                            published_at = dt.datetime.strptime(
                                article['time_published'], '%Y%m%dT%H%M%S'
                            )
                            
                            # Filter by time
                            cutoff = dt.datetime.now() - dt.timedelta(hours=hours_back)
                            if published_at < cutoff:
                                continue
                            
                            # Extract sentiment
                            sentiment = None
                            if 'overall_sentiment_score' in article:
                                sentiment = float(article['overall_sentiment_score'])
                            
                            item = NewsItem(
                                title=article['title'],
                                description=article.get('summary', article['title']),
                                source=article['source'],
                                published_at=published_at,
                                url=article['url'],
                                sentiment=sentiment,
                                symbols=[symbol]
                            )
                            items.append(item)
                            
                        except Exception as e:
                            logger.warning(f"Error parsing Alpha Vantage article: {e}")
                            continue
                    
                    news_items[symbol] = items
                    self.alpha_vantage_calls_today += 1
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Alpha Vantage error for {symbol}: {e}")
                    news_items[symbol] = []
        
        return news_items
    
    async def _fetch_from_polygon(
        self,
        symbols: List[str],
        hours_back: int
    ) -> Dict[str, List[NewsItem]]:
        """Fetch news from Polygon.io (fallback)."""
        news_items = {}
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Polygon allows batch requests
                since = dt.datetime.now() - dt.timedelta(hours=hours_back)
                params = {
                    'ticker': ','.join(symbols),
                    'published_utc.gte': since.strftime('%Y-%m-%d'),
                    'order': 'desc',
                    'limit': 100,
                    'apiKey': self.polygon_key
                }
                
                response = await client.get(self.polygon_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Initialize all symbols
                for symbol in symbols:
                    news_items[symbol] = []
                
                for article in data.get('results', []):
                    try:
                        published_at = dt.datetime.fromisoformat(
                            article['published_utc'].replace('Z', '+00:00')
                        )
                        
                        # Extract relevant symbols
                        article_symbols = []
                        for ticker_info in article.get('tickers', []):
                            if ticker_info in symbols:
                                article_symbols.append(ticker_info)
                        
                        if not article_symbols:
                            continue
                        
                        item = NewsItem(
                            title=article['title'],
                            description=article.get('description', article['title']),
                            source=article.get('publisher', {}).get('name', 'Polygon'),
                            published_at=published_at,
                            url=article.get('article_url', ''),
                            symbols=article_symbols
                        )
                        
                        # Add to relevant symbols
                        for symbol in article_symbols:
                            news_items[symbol].append(item)
                            
                    except Exception as e:
                        logger.warning(f"Error parsing Polygon article: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Polygon news error: {e}")
                for symbol in symbols:
                    news_items[symbol] = []
        
        return news_items
    
    def _merge_news_items(
        self,
        target: Dict[str, List[NewsItem]],
        source: Dict[str, List[NewsItem]]
    ) -> None:
        """Merge news items from source into target, avoiding duplicates."""
        for symbol, items in source.items():
            if symbol not in target:
                target[symbol] = []
            
            # Simple deduplication by title
            existing_titles = {item.title for item in target[symbol]}
            for item in items:
                if item.title not in existing_titles:
                    target[symbol].append(item)
                    existing_titles.add(item.title)
    
    def _can_make_newsapi_call(self) -> bool:
        """Check if we can make a NewsAPI call without hitting limits."""
        return self.newsapi_calls_today < self.max_calls_per_day
    
    def _can_make_alpha_vantage_call(self) -> bool:
        """Check if we can make an Alpha Vantage call without hitting limits."""
        return self.alpha_vantage_calls_today < self.max_calls_per_day
    
    def calculate_news_momentum(
        self,
        news_items: List[NewsItem],
        decay_hours: float = 24.0
    ) -> float:
        """Calculate momentum score based on news volume and recency."""
        if not news_items:
            return 0.0
        
        try:
            now = dt.datetime.now(dt.timezone.utc)
            score = 0.0
            
            for item in news_items:
                # Time decay factor
                hours_ago = (now - item.published_at.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600
                time_factor = max(0, 1 - (hours_ago / decay_hours))
                
                # Base score (number of articles is momentum indicator)
                base_score = 1.0
                
                # Sentiment boost if available
                sentiment_factor = 1.0
                if item.sentiment is not None:
                    sentiment_factor = 1 + abs(item.sentiment) * 0.5  # 0.5-1.5x multiplier
                
                score += base_score * time_factor * sentiment_factor
            
            return min(score, 10.0)  # Cap at 10
            
        except Exception as e:
            logger.error(f"Error calculating news momentum: {e}")
            return 0.0