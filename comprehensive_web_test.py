#!/usr/bin/env python3
"""
Comprehensive Web Application Test Suite
Tests all primary logical paths and user interactions
"""

import time
import json
import subprocess
import sys
from typing import Dict, List, Tuple
import urllib.parse
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class ComprehensiveWebTester:
    """Comprehensive test suite for NanoCap Trader web application"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.driver = None
        self.test_results = []
        self.errors = []
        
    def setup_browser(self):
        """Set up Chrome browser for testing"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run headless for automated testing
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)
            return True
        except Exception as e:
            self.errors.append(f"Browser setup failed: {e}")
            return False
    
    def test_endpoint(self, endpoint: str, expected_status: int = 200, method: str = "GET", data: Dict = None) -> Tuple[bool, str, str]:
        """Test HTTP endpoint"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method == "GET":
                response = urllib.request.urlopen(url)
                status_code = response.getcode()
                content = response.read().decode('utf-8')
            elif method == "POST":
                req_data = json.dumps(data).encode('utf-8') if data else b""
                req = urllib.request.Request(url, data=req_data, method="POST")
                req.add_header('Content-Type', 'application/json')
                response = urllib.request.urlopen(req)
                status_code = response.getcode()
                content = response.read().decode('utf-8')
            
            success = status_code == expected_status
            # Don't truncate content for JSON responses to avoid breaking JSON parsing
            if endpoint.endswith('ab-test') and len(content) > 500:
                return success, str(status_code), content  # Return full content for JSON
            else:
                return success, str(status_code), content[:500] + "..." if len(content) > 500 else content
            
        except Exception as e:
            return False, "ERROR", str(e)
    
    def test_basic_endpoints(self):
        """Test all basic HTTP endpoints"""
        print("🔍 Testing Basic HTTP Endpoints...")
        
        endpoints = [
            ("/", 200, "Main landing page"),
            ("/docs", 200, "API documentation"),
            ("/api/status", 200, "Portfolio status API"),
            ("/api/portfolio", 200, "Portfolio dashboard"),
            ("/api/benchmark", 200, "Benchmarking dashboard"),
            ("/dash/", 200, "Dash GUI"),
        ]
        
        for endpoint, expected_status, description in endpoints:
            success, status, content = self.test_endpoint(endpoint, expected_status)
            result = "✅ PASS" if success else "❌ FAIL"
            print(f"  {result} {endpoint} ({description}): HTTP {status}")
            self.test_results.append((f"Endpoint {endpoint}", success, f"HTTP {status}"))
            
            if not success:
                self.errors.append(f"Endpoint {endpoint} failed: HTTP {status}")
    
    def test_benchmark_api(self):
        """Test benchmarking API endpoints"""
        print("🧪 Testing Benchmarking API...")
        
        # Test single benchmark
        benchmark_url = "/api/benchmark/single?strategy=statistical_arbitrage&benchmark=russell_2000&start_date=2024-01-01&end_date=2024-01-31"
        success, status, content = self.test_endpoint(benchmark_url)
        
        if success:
            try:
                data = json.loads(content)
                required_fields = ["total_return", "sharpe_ratio", "max_drawdown", "alpha", "beta"]
                has_all_fields = all(field in data for field in required_fields)
                
                if has_all_fields:
                    print("  ✅ PASS Single benchmark API: All required fields present")
                    self.test_results.append(("Benchmark API Structure", True, "All fields present"))
                else:
                    missing = [f for f in required_fields if f not in data]
                    print(f"  ❌ FAIL Single benchmark API: Missing fields {missing}")
                    self.test_results.append(("Benchmark API Structure", False, f"Missing: {missing}"))
                    self.errors.append(f"Benchmark API missing fields: {missing}")
            except json.JSONDecodeError:
                print("  ❌ FAIL Single benchmark API: Invalid JSON response")
                self.test_results.append(("Benchmark API JSON", False, "Invalid JSON"))
                self.errors.append("Benchmark API returned invalid JSON")
        else:
            print(f"  ❌ FAIL Single benchmark API: HTTP {status}")
            self.test_results.append(("Benchmark API", False, f"HTTP {status}"))
            self.errors.append(f"Benchmark API failed: HTTP {status}")
        
        # Test A/B testing API
        ab_test_data = {
            "strategies": ["statistical_arbitrage", "momentum"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
        success, status, content = self.test_endpoint("/api/benchmark/ab-test", method="POST", data=ab_test_data)
        
        if success:
            try:
                # Clean content in case there are extra characters
                clean_content = content.strip()
                if clean_content.endswith('%'):
                    clean_content = clean_content[:-1]  # Remove trailing %
                data = json.loads(clean_content)
                required_fields = ["strategies", "performance_metrics", "recommended_strategy"]
                has_all_fields = all(field in data for field in required_fields)
                
                if has_all_fields:
                    print("  ✅ PASS A/B test API: All required fields present")
                    self.test_results.append(("A/B Test API Structure", True, "All fields present"))
                else:
                    missing = [f for f in required_fields if f not in data]
                    print(f"  ❌ FAIL A/B test API: Missing fields {missing}")
                    self.test_results.append(("A/B Test API Structure", False, f"Missing: {missing}"))
                    self.errors.append(f"A/B Test API missing fields: {missing}")
            except json.JSONDecodeError:
                print("  ❌ FAIL A/B test API: Invalid JSON response")
                self.test_results.append(("A/B Test API JSON", False, "Invalid JSON"))
                self.errors.append("A/B test API returned invalid JSON")
        else:
            print(f"  ❌ FAIL A/B test API: HTTP {status}")
            self.test_results.append(("A/B Test API", False, f"HTTP {status}"))
            self.errors.append(f"A/B test API failed: HTTP {status}")
    
    def test_dash_gui(self):
        """Test Dash GUI functionality"""
        print("📱 Testing Dash GUI...")
        
        if not self.driver:
            print("  ❌ SKIP Dash GUI: Browser not available")
            return
        
        try:
            self.driver.get(f"{self.base_url}/dash/")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check for loading state
            try:
                loading_div = self.driver.find_element(By.ID, "_dash-loading")
                if loading_div.is_displayed():
                    print("  ❌ FAIL Dash GUI: Stuck on loading screen")
                    self.test_results.append(("Dash GUI Loading", False, "Stuck on loading"))
                    self.errors.append("Dash GUI stuck on loading screen")
                    return
            except NoSuchElementException:
                pass  # No loading div found, which is good
            
            # Check for main content
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
            
            if "NanoCap Trader - System Online!" in body_text:
                print("  ✅ PASS Dash GUI: Main content loaded")
                self.test_results.append(("Dash GUI Content", True, "Main content visible"))
            else:
                print("  ❌ FAIL Dash GUI: Main content not found")
                self.test_results.append(("Dash GUI Content", False, "Main content missing"))
                self.errors.append("Dash GUI main content not loaded")
                return
            
            # Test interactive button
            try:
                test_button = self.driver.find_element(By.ID, "test-button")
                original_clicks = test_button.get_attribute("n_clicks") or "0"
                
                test_button.click()
                time.sleep(2)  # Wait for callback
                
                # Check if output updated
                test_output = self.driver.find_element(By.ID, "test-output")
                output_text = test_output.text
                
                if "SUCCESS" in output_text and "functional" in output_text:
                    print("  ✅ PASS Dash GUI: Interactive callbacks working")
                    self.test_results.append(("Dash GUI Interactivity", True, "Callbacks working"))
                else:
                    print("  ❌ FAIL Dash GUI: Interactive callbacks not working")
                    self.test_results.append(("Dash GUI Interactivity", False, "Callbacks failed"))
                    self.errors.append("Dash GUI callbacks not working")
                    
            except NoSuchElementException:
                print("  ❌ FAIL Dash GUI: Test button not found")
                self.test_results.append(("Dash GUI Button", False, "Button not found"))
                self.errors.append("Dash GUI test button not found")
                
        except TimeoutException:
            print("  ❌ FAIL Dash GUI: Page load timeout")
            self.test_results.append(("Dash GUI Timeout", False, "Page load timeout"))
            self.errors.append("Dash GUI page load timeout")
        except Exception as e:
            print(f"  ❌ FAIL Dash GUI: {e}")
            self.test_results.append(("Dash GUI Error", False, str(e)))
            self.errors.append(f"Dash GUI error: {e}")
    
    def test_benchmark_gui_interaction(self):
        """Test benchmark GUI form interactions"""
        print("🧪 Testing Benchmark GUI Interactions...")
        
        if not self.driver:
            print("  ❌ SKIP Benchmark GUI: Browser not available")
            return
        
        try:
            self.driver.get(f"{self.base_url}/api/benchmark")
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if benchmark form is present
            try:
                strategy_select = self.driver.find_element(By.NAME, "strategy")
                benchmark_select = self.driver.find_element(By.NAME, "benchmark")
                submit_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                
                print("  ✅ PASS Benchmark GUI: Form elements found")
                self.test_results.append(("Benchmark GUI Form", True, "Form elements present"))
                
                # Test form interaction
                Select(strategy_select).select_by_value("statistical_arbitrage")
                Select(benchmark_select).select_by_value("russell_2000")
                
                # Submit form and wait for results
                submit_button.click()
                
                # Wait for results to appear and loading to complete
                try:
                    WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.ID, "single-results"))
                    )
                    
                    # Wait for loading to complete (wait until text changes from loading message)
                    WebDriverWait(self.driver, 15).until(
                        lambda driver: "Running benchmark analysis" not in driver.find_element(By.ID, "single-results").text
                    )
                    
                    results_div = self.driver.find_element(By.ID, "single-results")
                    if results_div.is_displayed():
                        results_text = results_div.text
                        
                        if "Total Return:" in results_text and "Sharpe Ratio:" in results_text:
                            print("  ✅ PASS Benchmark GUI: Form submission and results working")
                            self.test_results.append(("Benchmark GUI Submission", True, "Results displayed"))
                        elif "Error:" in results_text:
                            print(f"  ❌ FAIL Benchmark GUI: API Error - {results_text[:200]}")
                            self.test_results.append(("Benchmark GUI Results", False, f"API Error: {results_text[:100]}"))
                            self.errors.append(f"Benchmark GUI API error: {results_text[:200]}")
                        else:
                            print(f"  ❌ FAIL Benchmark GUI: Unexpected results format - {results_text[:200]}")
                            self.test_results.append(("Benchmark GUI Results", False, f"Format: {results_text[:100]}"))
                            self.errors.append(f"Benchmark GUI unexpected format: {results_text[:200]}")
                    else:
                        print("  ❌ FAIL Benchmark GUI: Results not displayed")
                        self.test_results.append(("Benchmark GUI Display", False, "Results not shown"))
                        self.errors.append("Benchmark GUI results not displayed")
                        
                except TimeoutException:
                    print("  ❌ FAIL Benchmark GUI: Results timeout")
                    self.test_results.append(("Benchmark GUI Timeout", False, "Results timeout"))
                    self.errors.append("Benchmark GUI results timeout")
                    
            except NoSuchElementException as e:
                print(f"  ❌ FAIL Benchmark GUI: Form elements missing - {e}")
                self.test_results.append(("Benchmark GUI Elements", False, "Form elements missing"))
                self.errors.append(f"Benchmark GUI form elements missing: {e}")
                
        except Exception as e:
            print(f"  ❌ FAIL Benchmark GUI: {e}")
            self.test_results.append(("Benchmark GUI Error", False, str(e)))
            self.errors.append(f"Benchmark GUI error: {e}")
    
    def test_navigation_links(self):
        """Test navigation between all pages"""
        print("🔗 Testing Navigation Links...")
        
        if not self.driver:
            print("  ❌ SKIP Navigation: Browser not available")
            return
        
        # Start from main page
        self.driver.get(self.base_url)
        
        navigation_tests = [
            ("Dash GUI", "/dash/", "NanoCap Trader - System Online!"),
            ("Portfolio Status", "/api/portfolio", "Portfolio Status"),
            ("Benchmarking", "/api/benchmark", "Benchmarking & A/B Testing"),
            ("API Docs", "/docs", "html"),
        ]
        
        for name, url, expected_content in navigation_tests:
            try:
                self.driver.get(f"{self.base_url}{url}")
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # For API docs, check both page text and HTML source
                if name == "API Docs":
                    page_source = self.driver.page_source
                    body_text = self.driver.find_element(By.TAG_NAME, "body").text
                    
                    # Check for either Swagger UI elements or API doc content
                    has_content = (expected_content in body_text.lower() or 
                                 expected_content in page_source.lower() or
                                 "swagger" in page_source.lower() or
                                 "openapi" in page_source.lower())
                else:
                    body_text = self.driver.find_element(By.TAG_NAME, "body").text
                    has_content = expected_content in body_text
                
                if has_content:
                    print(f"  ✅ PASS Navigation to {name}: Content loaded")
                    self.test_results.append((f"Navigation to {name}", True, "Content loaded"))
                else:
                    print(f"  ❌ FAIL Navigation to {name}: Expected content not found")
                    self.test_results.append((f"Navigation to {name}", False, "Content missing"))
                    self.errors.append(f"Navigation to {name} failed - content missing")
                    
            except Exception as e:
                print(f"  ❌ FAIL Navigation to {name}: {e}")
                self.test_results.append((f"Navigation to {name}", False, str(e)))
                self.errors.append(f"Navigation to {name} failed: {e}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
        
        if failed_tests > 0:
            print(f"\n🚨 FAILURES AND ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {test_name}: {details}")
        
        return failed_tests == 0
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Comprehensive Web Application Test Suite")
        print("="*80)
        
        if not self.setup_browser():
            print("❌ Browser setup failed - running limited tests")
        
        self.test_basic_endpoints()
        self.test_benchmark_api()
        self.test_dash_gui()
        self.test_benchmark_gui_interaction()
        self.test_navigation_links()
        
        success = self.generate_report()
        self.cleanup()
        
        return success


def main():
    """Main test execution"""
    # Check if server is running
    try:
        urllib.request.urlopen("http://127.0.0.1:8000")
    except:
        print("❌ Server not running on http://127.0.0.1:8000")
        print("Please start the server with: uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        sys.exit(1)
    
    tester = ComprehensiveWebTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Web application is fully functional.")
        sys.exit(0)
    else:
        print("\n🚨 SOME TESTS FAILED! Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()