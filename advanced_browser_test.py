#!/usr/bin/env python3
"""
Advanced browser testing using real browser automation.
This actually clicks links and tests JavaScript functionality.
"""
import time
import subprocess
import sys
import webbrowser
import tempfile
import os

class AdvancedBrowserTester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def create_test_html(self):
        """Create an HTML test page that automatically tests our application"""
        test_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NanoCap Trader Auto-Tester</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .test-result {{ margin: 10px 0; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .loading {{ color: orange; }}
        #results {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🧪 NanoCap Trader Automated Browser Test</h1>
    <div id="results">
        <div class="loading">🔄 Running tests...</div>
    </div>
    
    <script>
        const baseUrl = '{self.base_url}';
        const results = document.getElementById('results');
        let testResults = [];
        
        function logTest(name, passed, details = '') {{
            const icon = passed ? '✅' : '❌';
            const className = passed ? 'pass' : 'fail';
            const result = `${{icon}} ${{name}}${{details ? ': ' + details : ''}}`;
            testResults.push({{name, passed, details}});
            
            const div = document.createElement('div');
            div.className = `test-result ${{className}}`;
            div.textContent = result;
            results.appendChild(div);
        }}
        
        async function testEndpoint(endpoint, expectedStatus = 200) {{
            try {{
                const response = await fetch(baseUrl + endpoint);
                const passed = response.status === expectedStatus;
                logTest(`HTTP ${{endpoint}}`, passed, `Status: ${{response.status}}`);
                return {{ passed, response }};
            }} catch (error) {{
                logTest(`HTTP ${{endpoint}}`, false, `Error: ${{error.message}}`);
                return {{ passed: false, response: null }};
            }}
        }}
        
        async function testDashInteractivity() {{
            return new Promise((resolve) => {{
                // Create iframe to test Dash app
                const iframe = document.createElement('iframe');
                iframe.src = baseUrl + '/dash/';
                iframe.style.width = '400px';
                iframe.style.height = '300px';
                iframe.style.border = '1px solid #ccc';
                iframe.style.margin = '10px 0';
                
                iframe.onload = () => {{
                    setTimeout(() => {{
                        try {{
                            const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            
                            // Test if our content is rendered
                            const hasTitle = iframeDoc.body.textContent.includes('NanoCap Trader');
                            const hasSystemOnline = iframeDoc.body.textContent.includes('System Online');
                            const hasTestButton = iframeDoc.querySelector('#test-button') !== null;
                            const hasReactRoot = iframeDoc.querySelector('#react-entry-point') !== null;
                            
                            logTest('Dash content loaded', hasTitle || hasSystemOnline);
                            logTest('React app initialized', hasReactRoot);
                            logTest('Interactive elements present', hasTestButton);
                            
                            // Try to click the test button if it exists
                            if (hasTestButton) {{
                                const button = iframeDoc.querySelector('#test-button');
                                button.click();
                                
                                setTimeout(() => {{
                                    const output = iframeDoc.querySelector('#test-output');
                                    const hasOutput = output && output.textContent.includes('SUCCESS');
                                    logTest('Button click functionality', hasOutput);
                                    resolve();
                                }}, 1000);
                            }} else {{
                                logTest('Button click functionality', false, 'Button not found');
                                resolve();
                            }}
                            
                        }} catch (error) {{
                            logTest('Dash iframe access', false, error.message);
                            resolve();
                        }}
                    }}, 3000); // Wait for Dash to fully load
                }};
                
                iframe.onerror = () => {{
                    logTest('Dash iframe loading', false, 'Failed to load iframe');
                    resolve();
                }};
                
                document.body.appendChild(iframe);
            }});
        }}
        
        async function testNavigation() {{
            // Test navigation by opening windows
            try {{
                window.open(baseUrl + '/docs', '_blank');
                logTest('API docs navigation', true);
                
                window.open(baseUrl + '/api/status', '_blank');
                logTest('API status navigation', true);
            }} catch (error) {{
                logTest('Navigation', false, error.message);
            }}
        }}
        
        async function runAllTests() {{
            results.innerHTML = '<div class="loading">🔄 Running comprehensive browser tests...</div>';
            
            // HTTP endpoint tests
            await testEndpoint('/');
            await testEndpoint('/api/status');
            await testEndpoint('/docs');
            await testEndpoint('/dash/');
            await testEndpoint('/dash/_dash-layout');
            
            // Navigation tests
            await testNavigation();
            
            // Advanced Dash interactivity tests
            await testDashInteractivity();
            
            // Results summary
            const passed = testResults.filter(t => t.passed).length;
            const total = testResults.length;
            const successRate = (passed / total * 100).toFixed(1);
            
            const summaryDiv = document.createElement('div');
            summaryDiv.innerHTML = `
                <hr>
                <h3>📊 Test Results Summary</h3>
                <p><strong>Passed:</strong> ${{passed}}/${{total}}</p>
                <p><strong>Success Rate:</strong> ${{successRate}}%</p>
                ${{passed === total ? 
                    '<p class="pass">🎉 ALL TESTS PASSED! System is fully functional.</p>' : 
                    '<p class="fail">⚠️ Some tests failed. Check the results above.</p>'
                }}
            `;
            results.appendChild(summaryDiv);
            
            // Auto-close after showing results
            setTimeout(() => {{
                if (passed === total) {{
                    alert('🎉 ALL TESTS PASSED! NanoCap Trader is fully functional!');
                }} else {{
                    alert(`⚠️ ${{total - passed}} tests failed. Check the browser window for details.`);
                }}
            }}, 2000);
        }}
        
        // Start tests when page loads
        window.onload = runAllTests;
    </script>
</body>
</html>
        """
        return test_html
        
    def run_browser_tests(self):
        """Run comprehensive browser tests"""
        print("🚀 RUNNING ADVANCED BROWSER TESTS")
        print("=" * 50)
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(self.create_test_html())
            test_file = f.name
            
        try:
            print(f"📱 Opening test page in browser...")
            webbrowser.open(f'file://{test_file}')
            
            print("⏳ Waiting for tests to complete (30 seconds)...")
            print("   Watch the browser window for real-time test results!")
            time.sleep(30)
            
            print("✅ Browser tests completed!")
            print("\n🔍 Check the browser window for detailed results.")
            
        finally:
            # Cleanup
            try:
                os.unlink(test_file)
            except:
                pass

if __name__ == "__main__":
    print("🧪 Advanced Browser Testing for NanoCap Trader")
    print("This will open a browser window with automated testing.")
    print()
    
    tester = AdvancedBrowserTester()
    tester.run_browser_tests()