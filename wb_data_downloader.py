import pandas as pd
import requests
from pathlib import Path
import zipfile
import io
import logging
from datetime import datetime
import os
from typing import Optional, Dict, List

class WorldBankDataDownloader:
    """
    Downloads and processes World Bank data from their bulk download service.
    This approach is more reliable than API calls and provides complete historical data.
    """
    
    # Base URL for World Bank data downloads
    WB_DATA_URL = "https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv"
    
    # Key economic indicators and their codes
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
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the downloader with a data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def download_indicator_data(self, indicator_code: str) -> Optional[pd.DataFrame]:
        """
        Download data for a single indicator from World Bank.
        
        Args:
            indicator_code: The World Bank indicator code
            
        Returns:
            DataFrame containing the indicator data if successful, None otherwise
        """
        try:
            # Construct the download URL
            url = self.WB_DATA_URL.format(indicator_code=indicator_code)
            
            # Download the ZIP file
            self.logger.info(f"Downloading data for indicator {indicator_code}")
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract the ZIP file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the CSV file (usually the largest file in the ZIP)
                csv_file = max(z.filelist, key=lambda x: x.file_size)
                
                # Read the CSV file
                with z.open(csv_file.filename) as f:
                    df = pd.read_csv(f, skiprows=4)  # World Bank CSVs have 4 header rows
                    
                    # Keep only Kenya data
                    df = df[df['Country Code'] == 'KEN']
                    
                    # Melt the year columns into a single column
                    year_columns = [col for col in df.columns if str(col).isdigit()]
                    df = df.melt(
                        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                        value_vars=year_columns,
                        var_name='Year',
                        value_name='Value'
                    )
                    
                    # Clean up the data
                    df['Year'] = pd.to_numeric(df['Year'])
                    df = df[['Year', 'Value']].copy()
                    df = df.sort_values('Year')
                    
                    return df
                    
        except Exception as e:
            self.logger.error(f"Error downloading {indicator_code}: {str(e)}")
            return None
    
    def download_all_indicators(self, start_year: Optional[int] = None,
                              end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Download and combine data for all indicators.
        
        Args:
            start_year: Start year for filtering data (optional)
            end_year: End year for filtering data (optional)
            
        Returns:
            DataFrame containing all indicators
        """
        # Create base DataFrame with years
        if not start_year:
            start_year = 2000  # Default to year 2000
        if not end_year:
            end_year = datetime.now().year - 1  # Previous year (as current year might be incomplete)
        
        base_df = pd.DataFrame({'Year': range(start_year, end_year + 1)})
        
        # Download each indicator
        for indicator_code, indicator_name in self.INDICATORS.items():
            df = self.download_indicator_data(indicator_code)
            
            if df is not None:
                # Filter years and rename Value column to indicator name
                df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
                df = df.rename(columns={'Value': indicator_name})
                
                # Merge with base DataFrame
                base_df = base_df.merge(df, on='Year', how='left')
                self.logger.info(f"Successfully processed {indicator_name}")
            else:
                self.logger.warning(f"Skipping {indicator_name} due to download error")
        
        # Handle missing values
        base_df = base_df.sort_values('Year')
        base_df = base_df.fillna(method='ffill').fillna(method='bfill')
        
        # Save the data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f"economic_indicators_{timestamp}.csv"
        base_df.to_csv(output_file, index=False)
        
        # Also save as latest version
        latest_file = self.data_dir / "economic_indicators_latest.csv"
        base_df.to_csv(latest_file, index=False)
        
        self.logger.info(f"Saved data to {output_file} and {latest_file}")
        
        return base_df

def main():
    """Main function to test the downloader."""
    downloader = WorldBankDataDownloader()
    data = downloader.download_all_indicators(start_year=2014)
    
    if not data.empty:
        print("\nCollected Indicators:")
        print(data.columns.tolist())
        print("\nData Preview:")
        print(data.head())
        print("\nData Shape:", data.shape)
        
        # Print some basic statistics
        print("\nSummary Statistics for latest available year:")
        latest_year = data['Year'].max()
        latest_data = data[data['Year'] == latest_year]
        
        for col in data.columns:
            if col != 'Year':
                print(f"\n{col}:")
                print(f"Latest Value ({latest_year}): {latest_data[col].iloc[0]:.2f}")
                print(f"Historical Average: {data[col].mean():.2f}")
                print(f"Min: {data[col].min():.2f}")
                print(f"Max: {data[col].max():.2f}")

if __name__ == "__main__":
    main() 