import pandas as pd
import numpy as np
from datetime import datetime
import os

def scrape_chandarana():
    """
    Estimate Chandarana prices based on Carrefour data.
    Since Chandarana is considered upmarket, we add a 5-10% premium to Carrefour prices.
    """
    try:
        print("🔍 Starting Chandarana price estimation...")
        
        # Try to load Carrefour data
        try:
            carrefour_df = pd.read_csv("data/carrefour_prices.csv")
            if carrefour_df.empty:
                print("❌ No Carrefour data available for price estimation")
                return None
        except FileNotFoundError:
            print("❌ Carrefour price data not found")
            return None
            
        # Add premium to prices (random between 5-10%)
        chandarana_products = []
        
        for _, row in carrefour_df.iterrows():
            # Calculate premium (between 5-10%)
            premium = 1 + (np.random.uniform(0.05, 0.10))
            
            chandarana_products.append({
                "Item Name": row["Item Name"],
                "Price (KES)": round(row["Price (KES)"] * premium, 2),
                "Store": "Chandarana",
                "Date": datetime.now().date(),
                "Category": row["Category"]
            })
        
        df = pd.DataFrame(chandarana_products)
        
        # Save to file
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/chandarana_prices.csv", index=False)
        print(f"\n✅ Estimated {len(df)} products for Chandarana")
        print("📁 Saved to data/chandarana_prices.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Error in Chandarana estimation: {str(e)}")
        return None

if __name__ == "__main__":
    scrape_chandarana() 