"""Data sources for advanced trading strategies."""

from .news_data import NewsDataProvider
from .correlation_data import CorrelationDataProvider

__all__ = ["NewsDataProvider", "CorrelationDataProvider"]