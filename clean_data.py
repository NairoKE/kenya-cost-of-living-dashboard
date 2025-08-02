import os
import pandas as pd

# Base folders
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_cpi(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "kenya_cpi_forecast.csv")
    df_cpi = pd.read_csv(path, parse_dates=["Date"])
    return df_cpi[["Date", "CPI"]]

def load_and_clean_data(path=None, adjust_for_inflation=True):
    # Default to the newly named cost_Data.csv
    if path is None:
        path = os.path.join(DATA_DIR, "cost_data.csv")

    # Load price data
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip()

    # Add Category based on Item Name if not present
    if 'Category' not in df.columns:
        # Define category keywords
        category_keywords = {
            'milk': ['milk', 'lala', 'dairy'],
            'bread': ['bread', 'loaf'],
            'flour': ['flour', 'unga'],
            'sugar': ['sugar'],
            'rice': ['rice'],
            'cooking oil': ['oil', 'cooking oil'],
            'maize flour': ['maize flour', 'unga']
        }
        
        def get_category(item_name):
            item_name = item_name.lower()
            for category, keywords in category_keywords.items():
                if any(keyword in item_name for keyword in keywords):
                    return category
            return 'Other'
        
        df['Category'] = df['Item Name'].apply(get_category)

    # Merge CPI & compute real price
    if adjust_for_inflation:
        cpi_df = load_cpi()
        df = df.merge(cpi_df, on="Date", how="left")
        base_cpi = cpi_df["CPI"].iloc[0]
        df["Real_Price"] = df["Price (KES)"] * base_cpi / df["CPI"]
    else:
        df["Real_Price"] = df["Price (KES)"]

    # Drop any rows missing critical data
    df.dropna(subset=["Date", "Price (KES)", "Real_Price"], inplace=True)
    return df
