from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class EconomicIndicators:
    """
    Handles economic indicators and their impact on price predictions.
    Includes inflation rates, GDP growth, and other economic factors.
    """
    
    def __init__(self):
        # Load historical and projected inflation data
        self.inflation_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', end='2024-12-31', freq='ME'),
            'Inflation_Rate': [
                # 2023 data
                8.5, 9.2, 9.4, 8.8, 8.0, 7.9, 7.3, 6.7, 6.8, 6.9, 6.8, 6.6,
                # 2024 data and projections
                6.4, 6.2, 5.8, 5.2, 4.8, 4.5, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8
            ]
        })
        
        # Sort and set index for asof operation
        self.inflation_data = self.inflation_data.sort_values('Date')
        self.inflation_data.set_index('Date', inplace=True)
        
        # GDP growth projections
        self.gdp_growth: Dict[str, float] = {
            '2023': 5.6,
            '2024': 4.7
        }
        
        # Historical election impacts (percentage price increases observed)
        self.election_impacts: Dict[str, Dict[str, float]] = {
            '2017': {
                'pre_election': 15.3,  # % increase 3 months before
                'post_election': 8.7,  # % increase 3 months after
                'duration': 180.0  # days of impact
            },
            '2022': {
                'pre_election': 12.8,
                'post_election': 7.2,
                'duration': 160.0
            }
        }
        
        # Economic indicators
        self.economic_indicators: Dict[str, float] = {
            'public_debt_ratio': 73.0,  # % of GDP as of Q4 2023
            'current_account_deficit': 4.0,  # % of GDP as of Q3 2024
            'forex_reserves': 9.2,  # billion USD
            'import_cover_months': 4.7,
            'fiscal_deficit': 4.4  # % of GDP projected for FY 2024/25
        }
        
        # Interest rate changes
        self.interest_rate_changes: Dict[str, float] = {
            'total_increase': 6.0,  # 600 basis points from May 2022 to Feb 2024
            'current_rate': 10.5  # %
        }
        
        # Tax changes
        self.tax_changes: Dict[str, Dict[str, float]] = {
            '2024-07-01': {
                'VAT_change': 2.0,  # percentage points
                'expected_price_impact': 3.5  # %
            }
        }

    def get_inflation_adjusted_price(self, price: float, date: datetime) -> float:
        """
        Adjust price based on inflation rate.
        """
        current_inflation = self.get_inflation_rate(date)
        return price * (1 + current_inflation/100)

    def get_inflation_rate(self, date: datetime) -> float:
        """
        Get inflation rate for a specific date.
        """
        closest_date = self.inflation_data.index.asof(date)
        return float(self.inflation_data.loc[closest_date, 'Inflation_Rate'])

    def calculate_election_impact(self, date: datetime, base_price: float) -> Dict[str, Union[float, int]]:
        """
        Calculate expected price impact during election periods.
        """
        next_election = datetime(2027, 8, 9)
        days_to_election = (next_election - date).days
        
        # Calculate average historical impact
        avg_pre_election = float(np.mean([v['pre_election'] for v in self.election_impacts.values()]))
        avg_post_election = float(np.mean([v['post_election'] for v in self.election_impacts.values()]))
        avg_duration = float(np.mean([v['duration'] for v in self.election_impacts.values()]))
        
        if 0 <= days_to_election <= avg_duration/2:
            # We're in pre-election period
            impact_factor = avg_pre_election * (1 - days_to_election/(avg_duration/2))
        elif -avg_duration/2 <= days_to_election < 0:
            # We're in post-election period
            impact_factor = avg_post_election * (1 + days_to_election/(avg_duration/2))
        else:
            impact_factor = 0.0
            
        return {
            'impact_factor': float(impact_factor),
            'adjusted_price': float(base_price * (1 + impact_factor/100)),
            'days_to_election': days_to_election
        }

    def calculate_tax_impact(self, date: datetime, base_price: float) -> Dict[str, Any]:
        """
        Calculate expected price impact from tax changes.
        """
        impact = 0.0
        explanation: List[str] = []
        
        for tax_date, details in self.tax_changes.items():
            tax_date_dt = datetime.strptime(tax_date, '%Y-%m-%d')
            days_to_tax_change = (tax_date_dt - date).days
            
            if 0 <= days_to_tax_change <= 90:  # Consider impact up to 90 days after tax change
                impact += details['expected_price_impact'] * (1 - days_to_tax_change/90)
                explanation.append(f"Tax change on {tax_date}: +{details['expected_price_impact']}%")
        
        return {
            'impact_factor': float(impact),
            'adjusted_price': float(base_price * (1 + impact/100)),
            'explanation': explanation
        }

    def get_combined_forecast(self, 
                            base_price: float, 
                            date: datetime,
                            include_inflation: bool = True,
                            include_election: bool = True,
                            include_tax: bool = True) -> Dict[str, Any]:
        """
        Get combined price forecast considering all factors.
        """
        adjusted_price = base_price
        impacts: List[str] = []
        
        if include_inflation:
            inflation_adjustment = self.get_inflation_adjusted_price(base_price, date)
            inflation_impact = ((inflation_adjustment - base_price) / base_price) * 100
            adjusted_price = inflation_adjustment
            impacts.append(f"Inflation: {inflation_impact:.1f}%")
        
        if include_election:
            election_result = self.calculate_election_impact(date, adjusted_price)
            adjusted_price = float(election_result['adjusted_price'])
            if abs(float(election_result['impact_factor'])) > 0.1:
                impacts.append(f"Election: {election_result['impact_factor']:.1f}%")
        
        if include_tax:
            tax_result = self.calculate_tax_impact(date, adjusted_price)
            adjusted_price = float(tax_result['adjusted_price'])
            if isinstance(tax_result['explanation'], list):
                impacts.extend(tax_result['explanation'])
        
        return {
            'original_price': float(base_price),
            'adjusted_price': float(adjusted_price),
            'total_impact_percent': float(((adjusted_price - base_price) / base_price) * 100),
            'impact_breakdown': impacts
        }

def load_kenya_cpi_data() -> pd.DataFrame:
    """
    Load and process Kenya's Consumer Price Index (CPI) data.
    Returns DataFrame with dates and CPI values.
    """
    try:
        # Try to load from CSV first
        df = pd.read_csv('data/kenya_cpi_forecast.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        # If file doesn't exist, create synthetic data
        # This is placeholder data - replace with actual historical CPI data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        base_cpi = 100
        trend = np.linspace(0, 15, len(dates))  # Upward trend
        seasonal = 2 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Seasonal pattern
        noise = np.random.normal(0, 0.5, len(dates))  # Random variations
        
        cpi = base_cpi + trend + seasonal + noise
        
        df = pd.DataFrame({
            'Date': dates,
            'CPI': cpi
        })
        
        # Save for future use
        df.to_csv('data/kenya_cpi_forecast.csv', index=False)
        return df

if __name__ == "__main__":
    # Test the economic indicators
    indicators = EconomicIndicators()
    test_date = datetime.now()
    test_price = 100.0
    
    # Test combined forecast
    forecast = indicators.get_combined_forecast(test_price, test_date)
    print("\nCombined Forecast Results:")
    print(f"Original Price: KES {test_price:.2f}")
    print(f"Adjusted Price: KES {forecast['adjusted_price']:.2f}")
    print(f"Total Impact: {forecast['total_impact_percent']:.1f}%")
    print("\nImpact Breakdown:")
    for impact in forecast['impact_breakdown']:
        print(f"- {impact}") 