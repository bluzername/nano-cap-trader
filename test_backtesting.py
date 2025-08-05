#!/usr/bin/env python3
"""
Test script for the new backtesting framework

This script validates the backtesting engine, performance attribution,
and strategy comparison tools for insider trading strategies.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import logging

from app.backtesting import (
    quick_backtest,
    BacktestConfig,
    InsiderBacktestEngine,
    run_strategy_comparison,
    StrategyComparisonEngine
)
from app.backtesting.performance_attribution import (
    PerformanceAttributionAnalyzer,
    quick_attribution
)
from app.strategies.strategy_factory import StrategyFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_single_strategy():
    """Test backtesting of a single insider strategy"""
    
    print("\n" + "="*60)
    print("Testing Single Strategy Backtest")
    print("="*60)
    
    try:
        # Test the ML predictor strategy
        strategy_name = "insider_ml_predictor"
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        print(f"Running backtest for: {strategy_name}")
        print(f"Universe: {universe}")
        print(f"Period: 2023-01-01 to 2024-01-01")
        
        # Run backtest
        results = await quick_backtest(
            strategy_name=strategy_name,
            universe=universe,
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        
        # Display results
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annual Return: {results.annual_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"Total Trades: {results.total_trades}")
        
        if hasattr(results, 'insider_hit_rate'):
            print(f"Insider Hit Rate: {results.insider_hit_rate:.1%}")
        
        # Generate attribution report
        print(f"\n📈 PERFORMANCE ATTRIBUTION:")
        attribution_report = quick_attribution(results, strategy_name)
        print(attribution_report[:500] + "..." if len(attribution_report) > 500 else attribution_report)
        
        print("✅ Single strategy test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Single strategy test failed: {e}")
        logger.exception("Single strategy test error")
        return False


async def test_strategy_comparison():
    """Test comparison of multiple strategies"""
    
    print("\n" + "="*60)
    print("Testing Strategy Comparison")
    print("="*60)
    
    try:
        # Compare insider strategies with traditional strategies
        strategies = [
            "momentum",
            "insider_momentum_advanced",
            "insider_options_flow"
        ]
        universe = ["AAPL", "MSFT", "GOOGL"]
        
        print(f"Comparing strategies: {strategies}")
        print(f"Universe: {universe}")
        
        # Create backtest config
        config = BacktestConfig(
            start_date=datetime(2023, 6, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000.0
        )
        
        # Run comparison
        print("Running backtests...")
        results_dict = await run_strategy_comparison(strategies, universe, config)
        
        # Analyze results
        comparison_engine = StrategyComparisonEngine()
        rankings = comparison_engine.rank_strategies(results_dict)
        
        # Display rankings
        print(f"\n🏆 STRATEGY RANKINGS:")
        for ranking in rankings:
            print(f"{ranking.rank}. {ranking.strategy_name}")
            print(f"   Score: {ranking.overall_score:.3f}")
            print(f"   Recommendation: {ranking.recommendation}")
            if ranking.strengths:
                print(f"   Strengths: {', '.join(ranking.strengths[:2])}")
        
        # Show detailed results
        print(f"\n📊 DETAILED RESULTS:")
        for strategy_name, results in results_dict.items():
            print(f"\n{strategy_name}:")
            print(f"  Annual Return: {results.annual_return:.2%}")
            print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {results.max_drawdown:.2%}")
            print(f"  Trades: {results.total_trades}")
        
        # Generate comparison report
        print(f"\n📋 COMPARISON REPORT (excerpt):")
        report = comparison_engine.generate_comparison_report(rankings, results_dict)
        print(report[:800] + "..." if len(report) > 800 else report)
        
        print("✅ Strategy comparison test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy comparison test failed: {e}")
        logger.exception("Strategy comparison test error")
        return False


async def test_performance_attribution():
    """Test performance attribution analysis"""
    
    print("\n" + "="*60)
    print("Testing Performance Attribution")
    print("="*60)
    
    try:
        # Run a quick backtest
        results = await quick_backtest(
            strategy_name="insider_momentum_advanced",
            universe=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-06-01",
            end_date="2023-12-31"
        )
        
        print("Running attribution analysis...")
        
        # Analyze performance attribution
        analyzer = PerformanceAttributionAnalyzer()
        attribution = analyzer.analyze_performance(results, "insider_momentum_advanced")
        
        # Display attribution results
        print(f"\n🔍 ATTRIBUTION ANALYSIS:")
        print(f"Insider Alpha: {attribution.insider_alpha:.2%}")
        print(f"Cluster Alpha: {attribution.cluster_alpha:.2%}")
        print(f"Technical Alpha: {attribution.technical_alpha:.2%}")
        print(f"Market Beta: {attribution.market_beta:.2f}")
        print(f"Size Beta: {attribution.size_beta:.2f}")
        
        # Show insider type performance
        if attribution.insider_type_performance:
            print(f"\n👔 INSIDER TYPE PERFORMANCE:")
            for insider_type, performance in attribution.insider_type_performance.items():
                print(f"  {insider_type}: {performance:.1f}%")
        
        # Show holding period analysis
        if attribution.holding_period_analysis:
            print(f"\n⏰ HOLDING PERIOD ANALYSIS:")
            for period, performance in attribution.holding_period_analysis.items():
                print(f"  {period.replace('_', ' ').title()}: {performance:.1f}%")
        
        print("✅ Performance attribution test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Performance attribution test failed: {e}")
        logger.exception("Performance attribution test error")
        return False


async def test_strategy_factory_integration():
    """Test integration with strategy factory"""
    
    print("\n" + "="*60)
    print("Testing Strategy Factory Integration")
    print("="*60)
    
    try:
        # Get available strategies
        available_strategies = StrategyFactory.get_available_strategies()
        print(f"Available strategies: {available_strategies}")
        
        # Check insider strategies are available
        insider_strategies = [s for s in available_strategies if 'insider' in s]
        print(f"Insider strategies: {insider_strategies}")
        
        if len(insider_strategies) != 3:
            print(f"❌ Expected 3 insider strategies, found {len(insider_strategies)}")
            return False
        
        # Test strategy creation
        for strategy_name in insider_strategies:
            print(f"Testing strategy creation: {strategy_name}")
            
            strategy = StrategyFactory.create_strategy(strategy_name, ["AAPL", "MSFT"])
            if not strategy:
                print(f"❌ Failed to create strategy: {strategy_name}")
                return False
            
            print(f"  ✅ Created: {strategy.strategy_id}")
            
            # Get strategy info
            info = StrategyFactory.get_strategy_info(strategy_name)
            expected_perf = info.get('expected_performance', {})
            
            if expected_perf:
                print(f"  Expected Alpha: {expected_perf.get('annual_alpha', 0):.1%}")
                print(f"  Expected Sharpe: {expected_perf.get('sharpe_ratio', 0):.2f}")
        
        print("✅ Strategy factory integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Strategy factory integration test failed: {e}")
        logger.exception("Strategy factory integration test error")
        return False


async def main():
    """Run all backtesting tests"""
    
    print("🚀 NanoCap Trader - Backtesting Framework Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Strategy Factory Integration", test_strategy_factory_integration),
        ("Single Strategy Backtest", test_single_strategy),
        ("Performance Attribution", test_performance_attribution),
        ("Strategy Comparison", test_strategy_comparison),
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Backtesting framework is ready for production.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)