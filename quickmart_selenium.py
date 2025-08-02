import requests
import pandas as pd
from datetime import datetime
import os
import time
import random

def get_random_user_agent():
    """Get a random user agent to avoid detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/125.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    ]
    return random.choice(user_agents)

def scrape_quickmart():
    """
    Scrape Quickmart products using their search API.
    """
    categories = {
        'milk': 'milk',
        'bread': 'bread',
        'flour': 'flour',
        'sugar': 'sugar',
        'rice': 'rice',
        'cooking oil': 'cooking+oil',
        'maize flour': 'maize+flour'
    }
    
    all_products = []
    
    for category_name, search_term in categories.items():
        try:
            print(f"🔍 Scraping {category_name} products...")
            
            # Add random delay between requests (2-5 seconds)
            time.sleep(random.uniform(2, 5))
            
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': 'https://www.quickmart.co.ke/',
                'Origin': 'https://www.quickmart.co.ke'
            }
            
            # Use their search API endpoint
            url = f"https://www.quickmart.co.ke/api/v1/products/search"
            params = {
                'keyword': search_term,
                'pagesize': 50,
                'page': 1
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code != 200:
                print(f"❌ Error accessing {category_name}: Status code {response.status_code}")
                continue
                
            data = response.json()
            products = data.get('products', [])
            
            for product in products:
                try:
                    name = product.get('name', '').strip()
                    price = float(product.get('price', 0))
                    
                    if name and price > 0:
                        all_products.append({
                            "Item Name": name,
                            "Price (KES)": price,
                            "Store": "Quickmart",
                            "Date": datetime.now().date(),
                            "Category": category_name
                        })
                except Exception as e:
                    print(f"⚠️ Error processing product: {str(e)}")
                    continue
                    
            print(f"✅ Successfully scraped {len(products)} {category_name} products")
            
        except Exception as e:
            print(f"❌ Error scraping {category_name}: {str(e)}")
            continue
    
    if not all_products:
        print("❌ No products found")
        return None
        
    df = pd.DataFrame(all_products)
    print(f"\n✅ Successfully scraped {len(df)} products from Quickmart")
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/quickmart_prices.csv", index=False)
    print("📁 Saved to data/quickmart_prices.csv")
    
    return df

if __name__ == "__main__":
    scrape_quickmart()
