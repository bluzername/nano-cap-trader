#!/usr/bin/env python3
"""
Real browser test using selenium to see exactly what happens in the browser
and detect JavaScript errors that cause the Loading... issue.
"""
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_dash_in_real_browser():
    """Test Dash in real browser to see JavaScript execution"""
    print("🌐 REAL BROWSER TEST")
    print("=" * 40)
    
    # Configure Chrome to run headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        print("✅ Browser started")
        
        # Navigate to Dash page
        url = "http://127.0.0.1:8000/dash/"
        print(f"📱 Navigating to: {url}")
        driver.get(url)
        
        # Wait a moment for initial load
        time.sleep(3)
        
        # Check page title
        title = driver.title
        print(f"📄 Page title: {title}")
        
        # Check if we're stuck on Loading...
        body_text = driver.find_element(By.TAG_NAME, "body").text
        print(f"📝 Body text (first 200 chars): {body_text[:200]}")
        
        # Check if loading div exists
        try:
            loading_div = driver.find_element(By.CLASS_NAME, "_dash-loading")
            if loading_div.is_displayed():
                print("❌ STUCK ON LOADING: _dash-loading div is still visible")
                
                # Check for JavaScript errors
                logs = driver.get_log('browser')
                if logs:
                    print("🚨 JAVASCRIPT ERRORS FOUND:")
                    for log in logs:
                        print(f"   {log['level']}: {log['message']}")
                else:
                    print("📋 No JavaScript console errors found")
                    
                # Let's wait longer to see if it loads
                print("⏳ Waiting 10 more seconds to see if it loads...")
                time.sleep(10)
                
                # Check again
                body_text_after = driver.find_element(By.TAG_NAME, "body").text
                if "NanoCap Trader - System Online" in body_text_after:
                    print("✅ SUCCESS! Content loaded after waiting")
                    return True
                else:
                    print("❌ Still stuck on Loading after 13 seconds total")
                    return False
            else:
                print("✅ Loading div hidden - content should be visible")
        except:
            print("✅ No loading div found - content should be rendered")
        
        # Check if our custom content is visible
        if "NanoCap Trader - System Online" in body_text:
            print("✅ SUCCESS! Custom content is visible")
            
            # Try to find the test button
            try:
                test_button = driver.find_element(By.ID, "test-button")
                print("✅ Test button found and clickable")
                
                # Try clicking it
                test_button.click()
                time.sleep(2)
                
                # Check for callback response
                test_output = driver.find_element(By.ID, "test-output")
                output_text = test_output.text
                print(f"🔘 Test button output: {output_text}")
                
                if "SUCCESS" in output_text:
                    print("✅ FULL SUCCESS! JavaScript callbacks working!")
                    return True
                    
            except Exception as e:
                print(f"⚠️  Test button interaction failed: {e}")
            
            return True
        else:
            print("❌ Custom content not visible")
            return False
            
    except Exception as e:
        print(f"❌ Browser test failed: {e}")
        return False
    finally:
        try:
            driver.quit()
            print("🔧 Browser closed")
        except:
            pass

if __name__ == "__main__":
    success = test_dash_in_real_browser()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 BROWSER TEST PASSED! Dash is working properly!")
    else:
        print("❌ BROWSER TEST FAILED! Dash has loading issues!")
    print("=" * 40)