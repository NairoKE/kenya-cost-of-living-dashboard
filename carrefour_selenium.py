from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from datetime import datetime
import pandas as pd
import time
import os

def init_driver():
    """Initialize Chrome driver with appropriate options."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
    
    service = Service()
    return webdriver.Chrome(service=service, options=options)

def scrape_category(driver, category_name, search_term, max_retries=3):
    """Scrape products for a specific category with retries."""
    products = []
    attempt = 0
    
    while attempt < max_retries:
        try:
            print(f"🔍 Scraping {category_name} products (attempt {attempt + 1})...")
            url = f"https://www.carrefour.ke/mafken/en/search?keyword={search_term}"
            driver.get(url)
            
            # Wait for products to load
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "css-1p01izn")))
            time.sleep(5)  # Additional wait for dynamic content
            
            # Find all product cards
            product_cards = driver.find_elements(By.CLASS_NAME, "css-1p01izn")
            
            if not product_cards:
                print(f"⚠️ No products found for {category_name}, retrying...")
                attempt += 1
                continue
            
            for card in product_cards:
                try:
                    # Get product name
                    name = card.find_element(By.CLASS_NAME, "css-1dhe6a").text.strip()
                    
                    # Get price
                    price_text = card.find_element(By.CLASS_NAME, "css-1jeqb").text
                    # Clean price text (remove 'KES' and commas)
                    price = float(price_text.replace("KES", "").replace(",", "").strip())
                    
                    products.append({
                        "Item Name": name,
                        "Price (KES)": price,
                        "Store": "Carrefour",
                        "Date": datetime.now().date(),
                        "Category": category_name
                    })
                except Exception as e:
                    print(f"⚠️ Error processing product: {str(e)}")
                    continue
            
            print(f"✅ Successfully scraped {len(products)} {category_name} products")
            break  # Success, exit retry loop
            
        except TimeoutException:
            print(f"⚠️ Timeout while loading {category_name} products")
            attempt += 1
        except WebDriverException as e:
            print(f"⚠️ WebDriver error: {str(e)}")
            attempt += 1
            # Reinitialize driver on failure
            try:
                driver.quit()
            except:
                pass
            driver = init_driver()
        except Exception as e:
            print(f"❌ Error scraping {category_name}: {str(e)}")
            attempt += 1
    
    return products, driver

def scrape_carrefour():
    """
    Scrape Carrefour products with improved error handling and retry logic.
    """
    categories = {
        'milk': {
            'search': 'fresh+milk',
            'category': 'Dairy'
        },
        'bread': {
            'search': 'bread',
            'category': 'Grains'
        },
        'unga': {
            'search': 'maize+flour',
            'category': 'Grains'
        },
        'sugar': {
            'search': 'sugar',
            'category': 'Essentials'
        },
        'rice': {
            'search': 'rice',
            'category': 'Grains'
        },
        'cooking oil': {
            'search': 'cooking+oil',
            'category': 'Essentials'
        },
        'tea': {
            'search': 'tea+leaves',
            'category': 'Beverages'
        },
        'eggs': {
            'search': 'eggs',
            'category': 'Proteins'
        }
    }
    
    all_products = []
    driver = init_driver()
    
    try:
        for item_name, details in categories.items():
            products, driver = scrape_category(driver, item_name, details['search'])
            # Add category to products
            for product in products:
                product['Category'] = details['category']
            all_products.extend(products)
            
            # Short delay between categories
            time.sleep(2)
    finally:
        try:
            driver.quit()
        except:
            pass
    
    if not all_products:
        print("❌ No products found")
        return None
        
    df = pd.DataFrame(all_products)
    
    # Clean and standardize the data
    df['Item Name'] = df['Item Name'].str.lower()
    df['Category'] = df['Category'].str.title()
    
    # Filter out irrelevant items and standardize names
    def clean_item_name(name):
        name = name.lower()
        # Standardize milk names
        if 'milk' in name and 'fresh' in name:
            return 'Fresh Milk 500ml'
        # Standardize bread names
        elif 'bread' in name and ('white' in name or 'brown' in name):
            return 'Bread 400g'
        # Standardize unga names
        elif 'maize' in name and 'flour' in name:
            return 'Maize Flour 2kg'
        # Standardize sugar names
        elif 'sugar' in name and 'white' in name:
            return 'White Sugar 2kg'
        # Standardize rice names
        elif 'rice' in name and 'pishori' in name:
            return 'Pishori Rice 2kg'
        # Standardize cooking oil names
        elif 'cooking oil' in name:
            return 'Cooking Oil 2L'
        # Standardize tea names
        elif 'tea' in name and 'leaves' in name:
            return 'Tea Leaves 500g'
        # Standardize eggs names
        elif 'eggs' in name:
            return 'Eggs (Tray of 30)'
        return name
    
    df['Item Name'] = df['Item Name'].apply(clean_item_name)
    
    # Remove duplicates and keep the lowest price for each standardized item
    df = df.sort_values('Price (KES)').drop_duplicates('Item Name', keep='first')
    
    # Group by category and count products
    summary = df.groupby('Category').size()
    print("\n✅ All products scraped successfully:")
    print(summary)
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/carrefour_prices.csv", index=False)
    print("📁 Saved to data/carrefour_prices.csv")
    
    return df

if __name__ == "__main__":
    scrape_carrefour()
