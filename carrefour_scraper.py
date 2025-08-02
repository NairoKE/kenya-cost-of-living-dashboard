from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from datetime import datetime
import os
import random

# Define essential items and their categories
ESSENTIAL_ITEMS = {
    'fresh milk': {'category': 'Dairy', 'search': 'fresh+milk', 'unit': '500ml'},
    'bread': {'category': 'Grains', 'search': 'bread', 'unit': '400g'},
    'maize flour': {'category': 'Grains', 'search': 'maize+flour', 'unit': '2kg'},
    'sugar': {'category': 'Essentials', 'search': 'white+sugar', 'unit': '2kg'},
    'rice': {'category': 'Grains', 'search': 'pishori+rice', 'unit': '2kg'},
    'cooking oil': {'category': 'Essentials', 'search': 'cooking+oil', 'unit': '2L'},
    'tea leaves': {'category': 'Beverages', 'search': 'tea+leaves', 'unit': '500g'},
    'eggs': {'category': 'Proteins', 'search': 'eggs', 'unit': '30pc'}
}

def setup_driver():
    """Set up Chrome driver with appropriate options."""
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Run in visible mode for debugging
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Add random user agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    ]
    chrome_options.add_argument(f'--user-agent={random.choice(user_agents)}')
    
    # Add additional preferences
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Execute CDP commands to mask automation
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": random.choice(user_agents)
    })
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    
    return driver

def clean_price(price_text):
    """Clean price text to float."""
    try:
        return float(price_text.strip().replace("KES", "").replace(",", "").strip())
    except:
        return None

def standardize_name(name, item_type, unit):
    """Standardize item names."""
    name = name.lower()
    if any(keyword in name for keyword in item_type.split()):
        return f"{item_type.title()} ({unit})"
    return name

def random_sleep(min_seconds=2, max_seconds=5):
    """Sleep for a random amount of time."""
    time.sleep(random.uniform(min_seconds, max_seconds))

def scrape_carrefour(driver, search_term, item_details, max_retries=3):
    """Scrape Carrefour website for a specific item using Selenium."""
    url = f"https://www.carrefourkenya.com/mafken/en/search/?text={search_term}"
    out = []
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            driver.get(url)
            random_sleep(3, 6)  # Initial wait after page load
            
            # Wait for product tiles to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "product-tile"))
            )
            
            # Scroll down the page to load all products
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                random_sleep(1, 2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            # Get all product tiles
            tiles = driver.find_elements(By.CLASS_NAME, "product-tile")
            
            if not tiles:
                print("  No product tiles found, retrying...")
                continue
            
            for tile in tiles:
                try:
                    name = tile.find_element(By.CLASS_NAME, "product-title").text
                    price = tile.find_element(By.CLASS_NAME, "now-price").text
                    
                    price_val = clean_price(price)
                    if price_val:
                        item_name = standardize_name(
                            name,
                            search_term.replace("+", " "),
                            item_details['unit']
                        )
                        out.append({
                            "Date": datetime.now().date(),
                            "Item Name": item_name,
                            "Price (KES)": price_val,
                            "Category": item_details['category'],
                            "Unit": item_details['unit'],
                            "Store": "Carrefour",
                            "Location": "Nairobi"
                        })
                except NoSuchElementException:
                    continue
            
            if out:  # If we got some results, break the retry loop
                break
                
        except TimeoutException:
            print(f"  ⚠️ Timeout while loading {search_term} products")
        except Exception as e:
            print(f"  Error scraping {search_term}: {str(e)}")
        
        if attempt < max_retries - 1:  # Don't sleep after the last attempt
            random_sleep(5, 10)  # Longer wait between retries
    
    return out

def main():
    """Main function to scrape all essential items."""
    print("🛒 Starting Carrefour price scraping...")
    
    driver = setup_driver()
    all_rows = []
    
    try:
        for item_name, details in ESSENTIAL_ITEMS.items():
            print(f"\n📍 Scraping {item_name}...")
            results = scrape_carrefour(driver, details['search'], details)
            if results:
                print(f"✅ Found {len(results)} products for {item_name}")
                all_rows.extend(results)
            else:
                print(f"⚠️ No products found for {item_name}")
            random_sleep(4, 8)  # Longer pause between different items
    finally:
        driver.quit()
    
    if not all_rows:
        print("❌ No data collected")
        return
    
    # Create DataFrame and clean data
    df = pd.DataFrame(all_rows)
    
    # Remove duplicates keeping lowest price for each standardized item
    df = df.sort_values('Price (KES)').drop_duplicates('Item Name', keep='first')
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"data/carrefour_prices_{timestamp}.csv", index=False)
    
    # Also save as latest version
    df.to_csv("data/carrefour_prices_latest.csv", index=False)
    
    print("\n📊 Summary of collected prices:")
    summary = df.groupby('Category')[['Price (KES)']].agg(['count', 'mean', 'min', 'max'])
    print(summary)
    
    print("\n💾 Data saved to:")
    print(f"- data/carrefour_prices_{timestamp}.csv")
    print("- data/carrefour_prices_latest.csv")

if __name__ == "__main__":
    main()
