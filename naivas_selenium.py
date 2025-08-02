import requests
from bs4 import BeautifulSoup
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

def scrape_naivas():
    """
    Scrape Naivas products using their online platform with rotating user agents and delayed requests.
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
            url = f"https://naivas.online/search?term={search_term}"
            
            # Add random delay between requests (2-5 seconds)
            time.sleep(random.uniform(2, 5))
            
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"❌ Error accessing {category_name}: Status code {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all product cards
            product_cards = soup.find_all('div', class_='product-card')
            
            for card in product_cards:
                try:
                    name = card.find('h3', class_='product-title').text.strip()
                    price_text = card.find('div', class_='product-price').text.strip()
                    # Clean price text (remove 'KES' and commas)
                    price = float(price_text.replace("KES", "").replace(",", "").strip())
                    
                    all_products.append({
                        "Item Name": name,
                        "Price (KES)": price,
                        "Store": "Naivas",
                        "Date": datetime.now().date(),
                        "Category": category_name
                    })
                except Exception as e:
                    print(f"⚠️ Error processing product: {str(e)}")
                    continue
                    
            print(f"✅ Successfully scraped {len(product_cards)} {category_name} products")
            
        except Exception as e:
            print(f"❌ Error scraping {category_name}: {str(e)}")
            continue
    
    if not all_products:
        print("❌ No products found")
        return None
        
    df = pd.DataFrame(all_products)
    print(f"\n✅ Successfully scraped {len(df)} products from Naivas")
    
    # Save to file
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/naivas_prices.csv", index=False)
    print("📁 Saved to data/naivas_prices.csv")
    
    return df

if __name__ == "__main__":
    scrape_naivas()
