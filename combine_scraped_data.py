import pandas as pd
from datetime import datetime
import os

from carrefour_selenium import scrape_carrefour
from naivas_selenium import scrape_naivas
from quickmart_selenium import scrape_quickmart
from chandarana_selenium import scrape_chandarana

def combine_scraped_data():
    """
    Run all scrapers and combine their data into a single file.
    """
    print("🚀 Starting data scraping from all supermarkets...\n")
    
    # Initialize empty list to store all dataframes
    dfs = []
    
    # Scrape Carrefour
    print("📍 Scraping Carrefour...")
    try:
        carrefour_df = scrape_carrefour()
        if carrefour_df is not None:
            dfs.append(carrefour_df)
        else:
            print("⚠️ No data returned from Carrefour")
    except Exception as e:
        print(f"❌ Error scraping Carrefour: {str(e)}")
    
    # Scrape Naivas
    print("\n📍 Scraping Naivas...")
    try:
        naivas_df = scrape_naivas()
        if naivas_df is not None:
            dfs.append(naivas_df)
        else:
            print("⚠️ No data returned from Naivas")
    except Exception as e:
        print(f"❌ Error scraping Naivas: {str(e)}")
    
    # Scrape Quickmart
    print("\n📍 Scraping Quickmart...")
    try:
        quickmart_df = scrape_quickmart()
        if quickmart_df is not None:
            dfs.append(quickmart_df)
        else:
            print("⚠️ No data returned from Quickmart")
    except Exception as e:
        print(f"❌ Error scraping Quickmart: {str(e)}")
    
    # Scrape/Estimate Chandarana
    print("\n📍 Scraping Chandarana...")
    try:
        chandarana_df = scrape_chandarana()
        if chandarana_df is not None:
            dfs.append(chandarana_df)
        else:
            print("⚠️ No data returned from Chandarana")
    except Exception as e:
        print(f"❌ Error in Chandarana estimation: {str(e)}")
    
    if not dfs:
        print("\n❌ No data collected from any store")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined data with timestamp
    os.makedirs("data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = f"data/cost_data_{timestamp}.csv"
    latest_file = "data/cost_data.csv"
    
    combined_df.to_csv(timestamped_file, index=False)
    combined_df.to_csv(latest_file, index=False)
    
    print(f"\n✅ Successfully saved {len(combined_df)} products to:")
    print(f"  - {timestamped_file}")
    print(f"  - {latest_file}")
    
    # Print summary statistics
    print("\n📊 Summary:")
    summary = combined_df.groupby(['Store', 'Category'])['Item Name'].agg(['count'])
    summary['mean'] = combined_df.groupby(['Store', 'Category'])['Price (KES)'].mean()
    summary['min'] = combined_df.groupby(['Store', 'Category'])['Price (KES)'].min()
    summary['max'] = combined_df.groupby(['Store', 'Category'])['Price (KES)'].max()
    print(summary)

if __name__ == "__main__":
    combine_scraped_data()
