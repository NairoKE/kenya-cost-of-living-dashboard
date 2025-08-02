import wbgapi as wb
import pandas as pd
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
import logging
import os

class EconomicDataCollector:
    """
    Collects economic indicators from various sources including World Bank,
    Central Bank of Kenya, and Kenya National Bureau of Statistics
    """
    
    # Key economic indicators from World Bank
    INDICATORS = {
        # Price & Inflation Indicators
        'FP.CPI.TOTL.ZG': 'Inflation (consumer prices, annual %)',
        'FP.CPI.TOTL': 'Consumer Price Index (2010 = 100)',
        'PA.NUS.FCRF': 'Official exchange rate (KES per US$)',
        
        # Economic Growth & Production
        'NY.GDP.MKTP.KD.ZG': 'GDP growth (annual %)',
        'NV.AGR.TOTL.KD.ZG': 'Agriculture, value added (annual % growth)',
        'NV.IND.TOTL.KD.ZG': 'Industry, value added (annual % growth)',
        
        # Trade & External Sector
        'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
        'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
        'BN.CAB.XOKA.GD.ZS': 'Current account balance (% of GDP)',
        
        # Money & Banking
        'FM.LBL.BMNY.GD.ZS': 'Broad money (% of GDP)',
        'FR.INR.LEND': 'Lending interest rate (%)',
        
        # Employment & Income
        'SL.UEM.TOTL.ZS': 'Unemployment rate',
        'NY.GNP.PCAP.CD': 'GNI per capita (current US$)',
        
        # Additional Context Indicators
        'SP.POP.GROW': 'Population growth (annual %)',
    }
    
    def __init__(self):
        self.country_code = 'KEN'  # ISO3 code for Kenya
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_world_bank_data(self, start_year: Optional[int] = None, 
                            end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetches economic indicators from World Bank API
        """
        if not start_year:
            start_year = datetime.now().year - 10  # Default to 10 years of data
        if not end_year:
            end_year = datetime.now().year
        
        self.logger.info(f"Fetching World Bank data for Kenya from {start_year} to {end_year}")
        
        # Create a base dataframe with all years
        years = list(range(start_year, end_year + 1))
        base_df = pd.DataFrame({'Year': years})
        
        # Set up the World Bank API
        wb.db = 2  # Use World Development Indicators database
        
        for indicator_code, indicator_name in self.INDICATORS.items():
            try:
                # Get the data for this indicator
                data = []
                for entry in wb.data.get(indicator_code, self.country_code, range(start_year, end_year + 1)):
                    if entry['value'] is not None:
                        data.append({
                            'Year': int(entry['time']),
                            indicator_name: float(entry['value'])
                        })
                
                if data:
                    # Convert to DataFrame and merge
                    indicator_df = pd.DataFrame(data)
                    base_df = base_df.merge(indicator_df, on='Year', how='left')
                    self.logger.info(f"Successfully fetched {indicator_name}")
                else:
                    self.logger.warning(f"No data found for {indicator_name}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {indicator_name}: {str(e)}")
        
        if len(base_df.columns) <= 1:  # Only has 'Year' column
            self.logger.error("No data was collected from World Bank API")
            return pd.DataFrame()
        
        # Sort by year and fill missing values with forward fill then backward fill
        base_df = base_df.sort_values('Year')
        base_df = base_df.fillna(method='ffill').fillna(method='bfill')
        
        return base_df
    
    def fetch_cbk_exchange_rate(self) -> pd.DataFrame:
        """
        Fetches latest exchange rates from Central Bank of Kenya
        Note: Implementation depends on CBK API access
        """
        # TODO: Implement CBK API integration when available
        pass
    
    def fetch_knbs_data(self) -> pd.DataFrame:
        """
        Fetches data from Kenya National Bureau of Statistics
        Note: Implementation depends on KNBS API access
        """
        # TODO: Implement KNBS API integration when available
        pass
    
    def get_all_economic_indicators(self, start_year: Optional[int] = None,
                                  end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Collects all available economic indicators from various sources
        """
        # Get World Bank data
        wb_data = self.fetch_world_bank_data(start_year, end_year)
        
        if wb_data.empty:
            self.logger.error("Failed to collect any economic indicators")
            return pd.DataFrame()
        
        # Save the collected data
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/economic_indicators_{timestamp}.csv"
        wb_data.to_csv(output_file, index=False)
        
        self.logger.info(f"Saved economic indicators to {output_file}")
        
        return wb_data

if __name__ == "__main__":
    # Test the data collection
    collector = EconomicDataCollector()
    data = collector.get_all_economic_indicators(start_year=2014)  # Start from 2014 for better historical context
    
    if not data.empty:
        print("\nCollected Indicators:")
        print(data.columns.tolist())
        print("\nData Preview:")
        print(data.head())
        print("\nData Shape:", data.shape)
        
        # Print some basic statistics
        print("\nSummary Statistics for 2023 (or latest available year):")
        latest_year = data['Year'].max()
        latest_data = data[data['Year'] == latest_year]
        
        for col in data.columns:
            if col != 'Year':
                print(f"\n{col}:")
                print(f"Latest Value ({latest_year}): {latest_data[col].iloc[0]:.2f}")
                print(f"Historical Average: {data[col].mean():.2f}")
                print(f"Min: {data[col].min():.2f}")
                print(f"Max: {data[col].max():.2f}") 