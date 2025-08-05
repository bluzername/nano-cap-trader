"""
Advanced Backtesting Module for Insider Trading Strategies
"""

from .insider_backtest_engine import (
    InsiderBacktestEngine,
    BacktestConfig,
    BacktestResults,
    Trade,
    MarketRegime,
    run_strategy_comparison,
    quick_backtest
)

__all__ = [
    'InsiderBacktestEngine',
    'BacktestConfig', 
    'BacktestResults',
    'Trade',
    'MarketRegime',
    'run_strategy_comparison',
    'quick_backtest'
]