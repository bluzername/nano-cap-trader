#!/usr/bin/env python3
"""
Comprehensive debug test suite that actually tests for the 'Loading...' issue
and other real problems that were missed by previous tests.
"""
import time
import requests
import subprocess
import sys
import os
from urllib.parse import urljoin

class ProperDebugTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.failures = []
        self.successes = []
        
    def log_result(self, test_name, success, details=""):
        """Log test results with proper tracking"""
        if success:
            self.successes.append(test_name)
            print(f"✅ {test_name}")
            if details:
                print(f"   └─ {details}")
        else:
            self.failures.append((test_name, details))
            print(f"❌ {test_name}")
            if details:
                print(f"   └─ {details}")
                
    def test_server_connectivity(self):
        """Basic server connectivity test"""
        try:
            response = requests.get(self.base_url, timeout=5)
            self.log_result("Server connectivity", response.status_code == 200, 
                          f"Status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            self.log_result("Server connectivity", False, f"Error: {e}")
            return False
            
    def test_dash_loading_issue(self):
        """THE CRITICAL TEST: Check if Dash shows 'Loading...' indicating broken JS"""
        try:
            response = requests.get(f"{self.base_url}/dash/", timeout=10)
            
            if response.status_code != 200:
                self.log_result("Dash page accessibility", False, 
                              f"HTTP {response.status_code}")
                return False
                
            content = response.text.lower()
            
            # Check for the actual loading issue
            has_loading_text = 'loading...' in content or 'loading' in content
            has_dash_title = 'dash' in content
            has_react_entry = 'react-entry-point' in content
            has_our_content = any(text in content for text in [
                'nanocap trader', 'system online', 'test-button', 'interactive test'
            ])
            
            # This is the key test that was missing!
            if has_loading_text and not has_our_content:
                self.log_result("Dash Loading Issue Detected", False, 
                              "Page shows 'Loading...' - JavaScript assets not loading!")
                return False
            elif has_our_content:
                self.log_result("Dash content rendering", True, "Custom content visible")
                return True
            else:
                self.log_result("Dash content rendering", False, "No content visible")
                return False
                
        except Exception as e:
            self.log_result("Dash loading test", False, f"Error: {e}")
            return False
            
    def test_javascript_asset_failures(self):
        """Test for the specific JavaScript asset 404 errors"""
        # These are the assets that are failing based on server logs
        failing_assets = [
            "/_dash-component-suites/dash/deps/polyfill%407.v2_17_0m1754147341.12.1.min.js",
            "/_dash-component-suites/dash/deps/react%4016.v2_17_0m1754147341.14.0.min.js",
            "/_dash-component-suites/dash/dash-renderer/build/dash_renderer.v2_17_0m1754147341.min.js"
        ]
        
        asset_failures = 0
        for asset in failing_assets:
            try:
                url = urljoin(self.base_url, asset)
                response = requests.get(url, timeout=5)
                if response.status_code == 404:
                    asset_failures += 1
                    print(f"   ❌ Asset 404: {asset}")
                else:
                    print(f"   ✅ Asset OK: {asset}")
            except Exception as e:
                asset_failures += 1
                print(f"   ❌ Asset Error: {asset} - {e}")
        
        if asset_failures > 0:
            self.log_result("JavaScript Asset Loading", False, 
                          f"{asset_failures}/{len(failing_assets)} assets failing")
            return False
        else:
            self.log_result("JavaScript Asset Loading", True, "All assets loading")
            return True
            
    def test_dash_internal_endpoints(self):
        """Test Dash's internal endpoints that are required for functionality"""
        dash_endpoints = [
            "/dash/_dash-layout",
            "/dash/_dash-dependencies", 
            "/dash/_dash-update-component"
        ]
        
        all_working = True
        for endpoint in dash_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ {endpoint}: HTTP 200")
                else:
                    print(f"   ❌ {endpoint}: HTTP {response.status_code}")
                    all_working = False
            except Exception as e:
                print(f"   ❌ {endpoint}: Error {e}")
                all_working = False
                
        self.log_result("Dash Internal Endpoints", all_working)
        return all_working
        
    def test_with_browser_simulation(self):
        """Simulate what a real browser would do"""
        print("\n🌐 BROWSER SIMULATION TEST")
        print("-" * 30)
        
        # Step 1: Get main page
        try:
            main_response = requests.get(self.base_url, timeout=5)
            if main_response.status_code == 200:
                print("✅ Main page loads")
                
                # Step 2: Follow Dash link (like clicking in browser)
                dash_response = requests.get(f"{self.base_url}/dash/", timeout=10)
                if dash_response.status_code == 200:
                    print("✅ Dash page accessible")
                    
                    # Step 3: Check what browser would see
                    content = dash_response.text
                    
                    # This simulates browser JavaScript loading
                    if "Loading..." in content and "test-button" not in content:
                        print("❌ CRITICAL: Browser would see 'Loading...' stuck screen!")
                        print("   └─ This means JavaScript failed to load")
                        return False
                    elif "NanoCap Trader" in content or "System Online" in content:
                        print("✅ Browser would see working interface")
                        return True
                    else:
                        print("❌ Browser would see empty/broken page")
                        return False
                else:
                    print(f"❌ Dash page returns HTTP {dash_response.status_code}")
                    return False
            else:
                print(f"❌ Main page returns HTTP {main_response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Browser simulation failed: {e}")
            return False
            
    def run_comprehensive_debug_tests(self):
        """Run all debug tests to identify real issues"""
        print("🔍 COMPREHENSIVE DEBUG TEST SUITE")
        print("=" * 50)
        print("Testing for actual issues that cause 'Loading...' problems\n")
        
        # Test 1: Basic connectivity
        server_ok = self.test_server_connectivity()
        if not server_ok:
            print("\n❌ FATAL: Server not responding. Start server first.")
            return False
            
        # Test 2: The critical loading issue test
        print("\n🎯 CRITICAL TEST: Dash Loading Issue")
        loading_ok = self.test_dash_loading_issue()
        
        # Test 3: JavaScript asset failures
        print("\n📦 JavaScript Asset Tests:")
        assets_ok = self.test_javascript_asset_failures()
        
        # Test 4: Dash internal endpoints
        print("\n⚛️  Dash Internal Endpoints:")
        endpoints_ok = self.test_dash_internal_endpoints()
        
        # Test 5: Browser simulation
        browser_ok = self.test_with_browser_simulation()
        
        # Results
        print("\n" + "=" * 50)
        print("🔍 DEBUG TEST RESULTS")
        print("=" * 50)
        
        print(f"✅ Successes: {len(self.successes)}")
        print(f"❌ Failures: {len(self.failures)}")
        
        if self.failures:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for test_name, details in self.failures:
                print(f"   ❌ {test_name}")
                if details:
                    print(f"      └─ {details}")
                    
        if not loading_ok:
            print("\n🎯 ROOT CAUSE: Dash JavaScript assets are not loading properly!")
            print("   This causes the 'Loading...' issue you observed.")
            print("   Need to fix Dash asset routing in FastAPI mount configuration.")
            
        return len(self.failures) == 0

if __name__ == "__main__":
    print("🧪 PROPER DEBUG TEST SUITE")
    print("This will detect the actual 'Loading...' issue\n")
    
    tester = ProperDebugTester()
    success = tester.run_comprehensive_debug_tests()
    
    if success:
        print("\n✅ ALL TESTS PASSED - System working properly!")
        sys.exit(0)
    else:
        print("\n❌ ISSUES DETECTED - Need to fix before system is functional!")
        sys.exit(1)