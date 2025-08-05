#!/usr/bin/env python3
"""
System Health Check Script

Quick health check to verify the nano-cap trading system is working correctly.
This script performs essential tests to validate:

1. Configuration and API keys
2. Market data connectivity  
3. Strategy execution
4. Web server availability
5. Universe integrity

Usage:
    python tests/health_check.py
    
Environment Variables Required:
    POLYGON_API_KEY - Your Polygon.io API key
"""

import os
import sys
import asyncio
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.config import get_settings
    from app.strategies.strategy_factory import StrategyFactory
    from app.real_market_data import get_real_market_data
    from app.universe import get_high_volume_universe, get_default_universe
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class HealthChecker:
    """System health checker"""
    
    def __init__(self):
        self.results = []
        self.server_url = "http://localhost:8000"
    
    def check(self, name: str, func, *args, **kwargs):
        """Run a health check and record result"""
        print(f"🔍 Checking {name}...", end=" ")
        
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            
            if result:
                print("✅ PASS")
                self.results.append({"name": name, "status": "PASS", "details": result})
            else:
                print("⚠️  WARN")
                self.results.append({"name": name, "status": "WARN", "details": "Check returned False"})
                
        except Exception as e:
            print(f"❌ FAIL - {str(e)[:100]}")
            self.results.append({"name": name, "status": "FAIL", "details": str(e)})
    
    def run_all_checks(self):
        """Run all health checks"""
        print("🚀 Starting Nano-Cap Trading System Health Check")
        print("=" * 60)
        
        # Configuration checks
        self.check("API Keys Configuration", self.check_api_keys)
        self.check("Environment Setup", self.check_environment)
        
        # Data and connectivity checks
        self.check("Market Data Connectivity", self.check_market_data)
        self.check("Universe Integrity", self.check_universe)
        
        # Strategy checks
        self.check("Strategy Factory", self.check_strategy_factory)
        self.check("Core Strategies", self.check_core_strategies)
        self.check("Insider Strategies", self.check_insider_strategies)
        
        # Server checks
        self.check("Web Server", self.check_web_server)
        self.check("API Endpoints", self.check_api_endpoints)
        
        # Summary
        self.print_summary()
    
    def check_api_keys(self) -> Dict[str, Any]:
        """Check API key configuration"""
        settings = get_settings()
        
        results = {
            "polygon_configured": bool(settings.polygon_api_key),
            "polygon_valid": settings.polygon_api_key != "demo_key_for_testing",
            "optional_keys": {}
        }
        
        # Check optional keys
        optional_keys = {
            "alpha_vantage": settings.alpha_vantage_api_key,
            "ortex": settings.ortex_token,
            "fintel": settings.fintel_api_key,
            "whalewisdom": settings.whalewisdom_api_key,
            "tradier": settings.tradier_api_key,
            "benzinga": settings.benzinga_api_key
        }
        
        for key, value in optional_keys.items():
            results["optional_keys"][key] = bool(value)
        
        # Must have valid Polygon key
        if not results["polygon_configured"] or not results["polygon_valid"]:
            raise Exception("Polygon.io API key not configured or using demo key")
        
        return results
    
    def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration"""
        settings = get_settings()
        
        results = {
            "database_url": bool(settings.database_url),
            "max_portfolio_value": settings.max_portfolio_value,
            "max_daily_trades": settings.max_daily_trades
        }
        
        if settings.max_portfolio_value <= 0:
            raise Exception("Max portfolio value not configured")
        
        return results
    
    async def check_market_data(self) -> Dict[str, Any]:
        """Check market data connectivity"""
        test_symbols = ["BBAI", "RBOT", "SGTX"]
        
        result = await get_real_market_data(test_symbols, days=5)
        
        return {
            "is_real_data": result.is_real_data,
            "data_quality_score": result.data_quality_score,
            "symbols_with_data": len(result.symbols_with_data),
            "symbols_missing_data": len(result.symbols_missing_data),
            "total_symbols": len(test_symbols),
            "success_rate": len(result.symbols_with_data) / len(test_symbols)
        }
    
    def check_universe(self) -> Dict[str, Any]:
        """Check stock universe integrity"""
        default_universe = get_default_universe()
        high_volume_universe = get_high_volume_universe()
        
        results = {
            "default_universe_size": len(default_universe),
            "high_volume_size": len(high_volume_universe),
            "valid_symbols": True
        }
        
        # Basic symbol validation
        for symbol in default_universe[:10]:
            if not isinstance(symbol, str) or not symbol.isalpha():
                results["valid_symbols"] = False
                break
        
        if len(default_universe) < 50:
            raise Exception(f"Universe too small: {len(default_universe)} stocks")
        
        return results
    
    def check_strategy_factory(self) -> Dict[str, Any]:
        """Check strategy factory functionality"""
        factory = StrategyFactory()
        strategies = factory.get_available_strategies()
        
        results = {
            "available_strategies": len(strategies),
            "strategy_list": strategies
        }
        
        if len(strategies) < 5:
            raise Exception(f"Too few strategies available: {len(strategies)}")
        
        return results
    
    async def check_core_strategies(self) -> Dict[str, Any]:
        """Check core trading strategies"""
        factory = StrategyFactory()
        test_universe = get_high_volume_universe()[:5]
        
        core_strategies = ["momentum", "mean_reversion", "statistical_arbitrage", "multi_strategy"]
        results = {"tested": [], "failed": []}
        
        # Get market data for testing
        market_data_result = await get_real_market_data(test_universe, days=30)
        
        for strategy_name in core_strategies:
            try:
                strategy = factory.create_strategy(strategy_name, test_universe)
                if strategy is None:
                    results["failed"].append(strategy_name)
                    continue
                
                signals = await strategy.generate_signals(market_data_result.market_data)
                results["tested"].append({
                    "name": strategy_name,
                    "signals_generated": len(signals)
                })
                
            except Exception as e:
                results["failed"].append(f"{strategy_name}: {str(e)}")
        
        if results["failed"]:
            raise Exception(f"Core strategies failed: {results['failed']}")
        
        return results
    
    async def check_insider_strategies(self) -> Dict[str, Any]:
        """Check insider trading strategies"""
        factory = StrategyFactory()
        test_universe = get_high_volume_universe()[:3]
        
        insider_strategies = [
            "insider_momentum_advanced",
            "insider_options_flow", 
            "insider_ml_predictor"
        ]
        results = {"tested": [], "failed": []}
        
        # Get market data
        market_data_result = await get_real_market_data(test_universe, days=30)
        
        for strategy_name in insider_strategies:
            try:
                strategy = factory.create_strategy(strategy_name, test_universe)
                if strategy is None:
                    results["failed"].append(strategy_name)
                    continue
                
                # Test with placeholder data
                signals = await strategy.generate_signals(
                    market_data_result.market_data,
                    form4_data=None  # Will use placeholder data
                )
                
                results["tested"].append({
                    "name": strategy_name,
                    "signals_generated": len(signals)
                })
                
            except Exception as e:
                results["failed"].append(f"{strategy_name}: {str(e)}")
        
        if results["failed"]:
            raise Exception(f"Insider strategies failed: {results['failed']}")
        
        return results
    
    def check_web_server(self) -> Dict[str, Any]:
        """Check if web server is running"""
        try:
            response = requests.get(f"{self.server_url}/api/status", timeout=5)
            
            results = {
                "server_running": response.status_code == 200,
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }
            
            if response.status_code == 200:
                data = response.json()
                results["has_cash"] = "cash" in data
                results["has_positions"] = "positions" in data
            
            return results
            
        except requests.RequestException as e:
            raise Exception(f"Web server not accessible: {e}")
    
    def check_api_endpoints(self) -> Dict[str, Any]:
        """Check key API endpoints"""
        endpoints = [
            ("/api/status", "Status API"),
            ("/api/signals/dashboard", "Signals Dashboard"),
            ("/api/benchmark", "Benchmarking Dashboard")
        ]
        
        results = {"tested": [], "failed": []}
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.server_url}{endpoint}", timeout=10)
                
                results["tested"].append({
                    "endpoint": endpoint,
                    "name": name,
                    "status": response.status_code,
                    "success": response.status_code == 200
                })
                
            except requests.RequestException as e:
                results["failed"].append(f"{name}: {str(e)}")
        
        return results
    
    def print_summary(self):
        """Print health check summary"""
        print("\n" + "=" * 60)
        print("🏥 HEALTH CHECK SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        warned = sum(1 for r in self.results if r["status"] == "WARN")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        total = len(self.results)
        
        print(f"✅ Passed: {passed}/{total}")
        print(f"⚠️  Warnings: {warned}/{total}")
        print(f"❌ Failed: {failed}/{total}")
        
        if failed > 0:
            print(f"\n🚨 CRITICAL ISSUES DETECTED:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"   ❌ {result['name']}: {result['details']}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"   1. Check API key configuration in .env file")
            print(f"   2. Verify server is running: uvicorn main:app --host 0.0.0.0 --port 8000")
            print(f"   3. Check network connectivity")
            print(f"   4. Review error logs for detailed information")
        
        elif warned > 0:
            print(f"\n⚠️  WARNINGS DETECTED:")
            for result in self.results:
                if result["status"] == "WARN":
                    print(f"   ⚠️  {result['name']}: {result['details']}")
        
        else:
            print(f"\n🎉 ALL SYSTEMS OPERATIONAL!")
            print(f"   The nano-cap trading system is ready for use.")
        
        print(f"\n📊 SYSTEM STATUS: {'🟢 HEALTHY' if failed == 0 else '🔴 NEEDS ATTENTION'}")


def main():
    """Main health check runner"""
    print("Nano-Cap Trading System Health Check")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Run health checks
    checker = HealthChecker()
    checker.run_all_checks()
    
    # Exit with appropriate code
    failed_count = sum(1 for r in checker.results if r["status"] == "FAIL")
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()