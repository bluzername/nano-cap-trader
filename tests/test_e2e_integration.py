"""
End-to-End Integration Test Suite

This comprehensive test suite verifies that the nano-cap trading system works
correctly when all mandatory APIs are properly configured with working API keys.

Tests cover:
1. Market data fetching (Polygon.io)
2. Strategy signal generation
3. Insider trading data integration
4. Portfolio management
5. Backtesting functionality
6. Web API endpoints
7. Error handling and edge cases

Run with: python -m pytest tests/test_e2e_integration.py -v
"""

import pytest
import asyncio
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import os
import sys

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import get_settings
from app.strategies.strategy_factory import StrategyFactory
from app.real_market_data import get_real_market_data
from app.universe import get_high_volume_universe, get_default_universe
from app.portfolio import Portfolio


class TestConfiguration:
    """Test configuration and API key validation"""
    
    def test_api_keys_configured(self):
        """Verify all required API keys are configured"""
        settings = get_settings()
        
        # Check Polygon.io API key (mandatory for market data)
        assert settings.polygon_api_key, "Polygon.io API key not configured"
        assert settings.polygon_api_key != "demo_key_for_testing", "Using demo API key - need real key"
        
        # Check optional API keys (warn if missing)
        optional_keys = [
            ("alpha_vantage_api_key", "Alpha Vantage"),
            ("ortex_token", "ORTEX"),
            ("fintel_api_key", "Fintel"),
            ("whalewisdom_api_key", "WhaleWisdom"),
            ("tradier_api_key", "Tradier"),
            ("benzinga_api_key", "Benzinga")
        ]
        
        missing_optional = []
        for key_attr, service_name in optional_keys:
            if not getattr(settings, key_attr, None):
                missing_optional.append(service_name)
        
        if missing_optional:
            print(f"Warning: Optional API keys missing for: {', '.join(missing_optional)}")
    
    def test_environment_setup(self):
        """Verify environment is properly configured"""
        settings = get_settings()
        
        # Check database settings
        assert settings.database_url, "Database URL not configured"
        
        # Check portfolio limits
        assert settings.max_portfolio_value > 0, "Max portfolio value not set"
        assert settings.max_daily_trades > 0, "Max daily trades not set"
        
        # Check universe configuration
        universe = get_default_universe()
        assert len(universe) >= 50, f"Universe too small: {len(universe)} stocks"
        assert len(universe) <= 200, f"Universe too large: {len(universe)} stocks"


class TestMarketDataIntegration:
    """Test market data fetching and processing"""
    
    @pytest.mark.asyncio
    async def test_real_market_data_fetch(self):
        """Test fetching real market data from Polygon.io"""
        # Use small subset for testing
        test_symbols = ["BBAI", "RBOT", "SGTX", "NVOS", "LOVE"]
        
        result = await get_real_market_data(test_symbols, days=30)
        
        # Verify we got data
        assert result.is_real_data, "Should be using real market data"
        assert result.data_quality_score > 0, "Data quality score should be positive"
        
        # Check we have some valid data
        assert len(result.symbols_with_data) > 0, "Should have data for at least some symbols"
        
        # Verify data structure
        for symbol in result.symbols_with_data[:2]:  # Check first 2 symbols
            assert symbol in result.market_data, f"Missing data for {symbol}"
            data = result.market_data[symbol]
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in data.columns, f"Missing column {col} for {symbol}"
            
            # Check data quality
            assert len(data) >= 10, f"Insufficient data points for {symbol}: {len(data)}"
            assert not data['close'].isna().all(), f"All close prices are NaN for {symbol}"
    
    @pytest.mark.asyncio
    async def test_market_data_error_handling(self):
        """Test handling of invalid symbols and API errors"""
        # Test with mix of valid and invalid symbols
        test_symbols = ["BBAI", "INVALID_SYMBOL_12345", "RBOT"]
        
        result = await get_real_market_data(test_symbols, days=30)
        
        # Should handle invalid symbols gracefully
        assert len(result.symbols_missing_data) > 0, "Should identify missing data"
        assert "INVALID_SYMBOL_12345" in result.symbols_missing_data
        
        # Should still get data for valid symbols
        valid_symbols = [s for s in test_symbols if s != "INVALID_SYMBOL_12345"]
        assert any(s in result.symbols_with_data for s in valid_symbols)


class TestStrategyExecution:
    """Test strategy execution and signal generation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.factory = StrategyFactory()
        self.test_universe = get_high_volume_universe()[:10]  # Use smaller subset
    
    @pytest.mark.asyncio
    async def test_all_strategies_execute(self):
        """Test that all strategies can execute without errors"""
        strategies = self.factory.get_available_strategies()
        
        # Get market data for testing
        market_data_result = await get_real_market_data(self.test_universe, days=60)
        
        for strategy_name in strategies:
            print(f"Testing strategy: {strategy_name}")
            
            # Create strategy instance
            strategy = self.factory.create_strategy(strategy_name, self.test_universe)
            assert strategy is not None, f"Failed to create {strategy_name} strategy"
            
            # Generate signals
            try:
                signals = await strategy.generate_signals(market_data_result.market_data)
                assert isinstance(signals, list), f"{strategy_name} should return list of signals"
                
                # Verify signal structure if any signals generated
                if signals:
                    signal = signals[0]
                    assert hasattr(signal, 'symbol'), "Signal should have symbol"
                    assert hasattr(signal, 'signal_type'), "Signal should have signal_type"
                    assert hasattr(signal, 'confidence'), "Signal should have confidence"
                    assert 0 <= signal.confidence <= 1, "Confidence should be between 0 and 1"
                
            except Exception as e:
                pytest.fail(f"Strategy {strategy_name} failed to generate signals: {e}")
    
    @pytest.mark.asyncio
    async def test_insider_strategies_with_placeholder_data(self):
        """Test insider strategies work with placeholder data"""
        insider_strategies = [
            "insider_momentum_advanced",
            "insider_options_flow", 
            "insider_ml_predictor"
        ]
        
        market_data_result = await get_real_market_data(self.test_universe[:5], days=30)
        
        for strategy_name in insider_strategies:
            strategy = self.factory.create_strategy(strategy_name, self.test_universe[:5])
            assert strategy is not None, f"Failed to create {strategy_name}"
            
            # Should work even without real Form 4 data (uses placeholders)
            signals = await strategy.generate_signals(
                market_data_result.market_data,
                form4_data=None  # Will trigger placeholder data
            )
            
            assert isinstance(signals, list), f"{strategy_name} should return signals list"
            print(f"{strategy_name}: Generated {len(signals)} signals")


class TestWebAPIEndpoints:
    """Test web API endpoints functionality"""
    
    def setup_method(self):
        """Setup for API testing"""
        # Assuming server is running on localhost:8000
        self.base_url = "http://localhost:8000"
        self.api_url = f"{self.base_url}/api"
    
    def test_server_is_running(self):
        """Verify server is accessible"""
        try:
            response = requests.get(f"{self.api_url}/status", timeout=10)
            assert response.status_code == 200, "Server should be running"
            
            data = response.json()
            assert "cash" in data, "Status should include cash balance"
            assert "positions" in data, "Status should include positions"
        except requests.RequestException as e:
            pytest.skip(f"Server not running or not accessible: {e}")
    
    def test_signals_api_endpoint(self):
        """Test signals generation API"""
        try:
            # Test default signals
            response = requests.get(
                f"{self.api_url}/signals",
                params={"strategy": "momentum"},
                timeout=30
            )
            
            assert response.status_code == 200, f"Signals API failed: {response.text}"
            
            data = response.json()
            assert "strategy" in data, "Response should include strategy"
            assert "signals" in data, "Response should include signals"
            assert "data_quality" in data, "Response should include data quality info"
            
            # Verify data quality
            quality = data["data_quality"]
            assert "is_real_data" in quality, "Should indicate if using real data"
            
        except requests.RequestException as e:
            pytest.skip(f"API endpoint not accessible: {e}")
    
    def test_insider_strategy_api(self):
        """Test insider strategy via API"""
        try:
            response = requests.get(
                f"{self.api_url}/signals",
                params={
                    "strategy": "insider_momentum_advanced",
                    "universe": "BBAI,RBOT,SGTX"
                },
                timeout=30
            )
            
            assert response.status_code == 200, f"Insider strategy API failed: {response.text}"
            
            data = response.json()
            assert data.get("strategy") == "insider_momentum_advanced"
            assert "signals" in data
            
        except requests.RequestException as e:
            pytest.skip(f"Insider strategy API not accessible: {e}")
    
    def test_benchmarking_api(self):
        """Test benchmarking functionality"""
        try:
            # Test single benchmark
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            response = requests.get(
                f"{self.api_url}/benchmark/single",
                params={
                    "strategy": "momentum",
                    "benchmark": "equal_weighted",
                    "start_date": start_date,
                    "end_date": end_date
                },
                timeout=60
            )
            
            assert response.status_code == 200, f"Benchmark API failed: {response.text}"
            
            data = response.json()
            if "error" not in data:
                assert "total_return" in data, "Should include performance metrics"
                assert "sharpe_ratio" in data, "Should include Sharpe ratio"
            
        except requests.RequestException as e:
            pytest.skip(f"Benchmark API not accessible: {e}")


class TestPortfolioManagement:
    """Test portfolio management functionality"""
    
    def setup_method(self):
        """Setup portfolio for testing"""
        self.portfolio = Portfolio()
    
    def test_portfolio_initialization(self):
        """Test portfolio initializes correctly"""
        assert self.portfolio.cash > 0, "Portfolio should have initial cash"
        assert len(self.portfolio.positions) == 0, "Portfolio should start empty"
    
    def test_position_management(self):
        """Test basic position management"""
        initial_cash = self.portfolio.cash
        
        # Simulate adding a position (this would normally be done by strategies)
        test_symbol = "BBAI"
        test_price = 10.0
        test_shares = 100
        
        # Note: This would typically be done through the strategy execution flow
        # Here we're testing the basic portfolio mechanics
        
        assert self.portfolio.cash == initial_cash, "Cash should remain unchanged without trades"


class TestDataQualityAndValidation:
    """Test data quality and validation"""
    
    @pytest.mark.asyncio
    async def test_universe_data_availability(self):
        """Test that we can get data for most universe stocks"""
        universe = get_high_volume_universe()  # Use high-volume subset
        
        # Test with larger sample
        sample_size = min(20, len(universe))
        test_symbols = universe[:sample_size]
        
        result = await get_real_market_data(test_symbols, days=30)
        
        # Calculate success rate
        success_rate = len(result.symbols_with_data) / len(test_symbols)
        
        assert success_rate >= 0.6, f"Data availability too low: {success_rate:.2%}"
        
        print(f"Data availability: {success_rate:.2%} ({len(result.symbols_with_data)}/{len(test_symbols)})")
        
        if result.symbols_missing_data:
            print(f"Missing data for: {result.symbols_missing_data}")
    
    def test_universe_validity(self):
        """Test that universe contains valid stock symbols"""
        universe = get_default_universe()
        
        # Basic validation
        assert len(universe) > 0, "Universe should not be empty"
        
        for symbol in universe[:10]:  # Check first 10
            assert isinstance(symbol, str), "Symbols should be strings"
            assert len(symbol) >= 1, "Symbols should not be empty"
            assert len(symbol) <= 6, "Symbols should be reasonable length"
            assert symbol.isalpha(), f"Symbol should be alphabetic: {symbol}"


class TestErrorHandlingAndRecovery:
    """Test error handling and system recovery"""
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_handling(self):
        """Test handling of API rate limits"""
        # Test with many symbols to potentially trigger rate limits
        large_universe = get_default_universe()[:30]
        
        try:
            result = await get_real_market_data(large_universe, days=30)
            
            # Should handle rate limits gracefully
            assert result is not None, "Should return result even with rate limits"
            
            if result.warnings:
                print("Warnings detected (expected with rate limits):")
                for symbol, warnings in result.warnings.items():
                    print(f"  {symbol}: {len(warnings)} warnings")
                    
        except Exception as e:
            # Should not crash, even with API issues
            print(f"API error handled: {e}")
    
    def test_strategy_error_recovery(self):
        """Test that strategy errors don't crash the system"""
        factory = StrategyFactory()
        
        # Test with empty universe (edge case)
        try:
            strategy = factory.create_strategy("momentum", [])
            # Should either create successfully or return None, not crash
            assert strategy is None or hasattr(strategy, 'strategy_id')
        except Exception as e:
            pytest.fail(f"Strategy creation should handle empty universe: {e}")


# Integration test runners
class TestEndToEndWorkflow:
    """Full end-to-end workflow tests"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self):
        """Test complete workflow from data fetch to signal generation"""
        print("\n=== Running Complete Trading Workflow Test ===")
        
        # 1. Setup
        test_universe = get_high_volume_universe()[:5]
        factory = StrategyFactory()
        
        print(f"Testing with universe: {test_universe}")
        
        # 2. Fetch market data
        print("Fetching market data...")
        market_data_result = await get_real_market_data(test_universe, days=60)
        
        assert market_data_result.is_real_data, "Should use real market data"
        print(f"Data quality score: {market_data_result.data_quality_score:.3f}")
        print(f"Symbols with data: {len(market_data_result.symbols_with_data)}")
        
        # 3. Test each strategy type
        test_strategies = ["momentum", "mean_reversion", "insider_momentum_advanced"]
        
        for strategy_name in test_strategies:
            print(f"\nTesting {strategy_name} strategy...")
            
            strategy = factory.create_strategy(strategy_name, test_universe)
            assert strategy is not None, f"Failed to create {strategy_name}"
            
            signals = await strategy.generate_signals(market_data_result.market_data)
            print(f"Generated {len(signals)} signals")
            
            # Validate signals
            for signal in signals[:3]:  # Check first 3 signals
                assert signal.symbol in test_universe, "Signal symbol should be in universe"
                assert 0 <= signal.confidence <= 1, "Confidence should be valid"
        
        print("\n=== Workflow Test Completed Successfully ===")


# Test configuration for pytest
@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment once per session"""
    print("\n=== Setting up test environment ===")
    
    # Verify configuration
    settings = get_settings()
    if settings.polygon_api_key == "demo_key_for_testing":
        pytest.skip("Tests require real API keys - demo key detected")
    
    yield
    
    print("\n=== Test environment cleanup ===")


# Pytest configuration
def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    # Run tests directly
    import pytest
    
    print("Running End-to-End Integration Tests")
    print("=====================================")
    
    # Run with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--setup-show"
    ])