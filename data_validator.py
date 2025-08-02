import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

class DataValidator:
    """
    Validates economic indicator data and performs quality checks.
    """
    
    # Define expected ranges for each indicator
    VALIDATION_RULES = {
        'Inflation (consumer prices, annual %)': {
            'min': -5,
            'max': 50,
            'std_dev_threshold': 3,  # Number of standard deviations for outlier detection
        },
        'Consumer Price Index (2010 = 100)': {
            'min': 0,
            'max': 500,
            'strictly_increasing': True,
        },
        'Official exchange rate (KES per US$)': {
            'min': 50,
            'max': 200,
            'std_dev_threshold': 3,
        },
        'GDP growth (annual %)': {
            'min': -10,
            'max': 15,
            'std_dev_threshold': 3,
        },
        'Agriculture, value added (annual % growth)': {
            'min': -20,
            'max': 20,
            'std_dev_threshold': 3,
        },
        'Industry, value added (annual % growth)': {
            'min': -20,
            'max': 20,
            'std_dev_threshold': 3,
        },
        'Imports of goods and services (% of GDP)': {
            'min': 0,
            'max': 100,
            'std_dev_threshold': 3,
        },
        'Exports of goods and services (% of GDP)': {
            'min': 0,
            'max': 100,
            'std_dev_threshold': 3,
        },
        'Current account balance (% of GDP)': {
            'min': -20,
            'max': 20,
            'std_dev_threshold': 3,
        },
        'Broad money (% of GDP)': {
            'min': 0,
            'max': 200,
            'std_dev_threshold': 3,
        },
        'Lending interest rate (%)': {
            'min': 0,
            'max': 30,
            'std_dev_threshold': 3,
        },
        'Unemployment rate': {
            'min': 0,
            'max': 30,
            'std_dev_threshold': 3,
        },
        'GNI per capita (current US$)': {
            'min': 0,
            'max': 5000,
            'strictly_increasing': True,
        },
        'Population growth (annual %)': {
            'min': -2,
            'max': 5,
            'std_dev_threshold': 3,
        },
    }
    
    def __init__(self):
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate the economic indicators data.
        
        Args:
            df: DataFrame containing the economic indicators
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        validation_messages = {}
        is_valid = True
        
        # Check if required columns are present
        required_columns = list(self.VALIDATION_RULES.keys()) + ['Year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            is_valid = False
            validation_messages['missing_columns'] = missing_columns
            self.logger.error(f"Missing columns: {missing_columns}")
            return is_valid, validation_messages
        
        # Validate each indicator
        for indicator, rules in self.VALIDATION_RULES.items():
            messages = []
            
            # Check value ranges
            min_val = df[indicator].min()
            max_val = df[indicator].max()
            if min_val < rules['min']:
                messages.append(f"Values below minimum threshold: {min_val:.2f} < {rules['min']}")
                is_valid = False
            if max_val > rules['max']:
                messages.append(f"Values above maximum threshold: {max_val:.2f} > {rules['max']}")
                is_valid = False
            
            # Check for outliers using standard deviation
            if 'std_dev_threshold' in rules:
                mean = df[indicator].mean()
                std = df[indicator].std()
                threshold = rules['std_dev_threshold'] * std
                outliers = df[abs(df[indicator] - mean) > threshold]
                if not outliers.empty:
                    messages.append(f"Found {len(outliers)} outlier(s)")
                    for _, row in outliers.iterrows():
                        messages.append(f"Outlier in {int(row['Year'])}: {row[indicator]:.2f}")
            
            # Check if values should be strictly increasing
            if rules.get('strictly_increasing', False):
                is_increasing = df[indicator].is_monotonic_increasing
                if not is_increasing:
                    messages.append("Values are not strictly increasing")
                    is_valid = False
            
            if messages:
                validation_messages[indicator] = messages
                self.logger.warning(f"Validation issues for {indicator}: {messages}")
        
        return is_valid, validation_messages
    
    def generate_validation_report(self, df: pd.DataFrame, output_dir: str = "data") -> str:
        """
        Generate a detailed validation report.
        
        Args:
            df: DataFrame containing the economic indicators
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        is_valid, messages = self.validate_data(df)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_shape': list(df.shape),  # Convert tuple to list for JSON
            'year_range': f"{int(df['Year'].min())} - {int(df['Year'].max())}",
            'is_valid': is_valid,
            'validation_messages': messages,
            'summary_statistics': {
                indicator: {
                    'mean': float(df[indicator].mean()),
                    'std': float(df[indicator].std()),
                    'min': float(df[indicator].min()),
                    'max': float(df[indicator].max()),
                    'missing_values': int(df[indicator].isnull().sum())
                }
                for indicator in self.VALIDATION_RULES.keys()
            }
        }
        
        # Save the report
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Validation report saved to {report_file}")
        return str(report_file)

def main():
    """Test the data validator."""
    # Load the latest economic indicators data
    try:
        df = pd.read_csv("data/economic_indicators_latest.csv")
        validator = DataValidator()
        report_file = validator.generate_validation_report(df)
        
        # Print summary of validation results
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("\nValidation Summary:")
        print(f"Data Valid: {report['is_valid']}")
        print(f"Data Shape: {report['data_shape']}")
        print(f"Year Range: {report['year_range']}")
        
        if report['validation_messages']:
            print("\nValidation Issues Found:")
            for indicator, messages in report['validation_messages'].items():
                print(f"\n{indicator}:")
                for msg in messages:
                    print(f"  - {msg}")
        else:
            print("\nNo validation issues found!")
            
    except FileNotFoundError:
        print("Error: No economic indicators data found. Please run wb_data_downloader.py first.")

if __name__ == "__main__":
    main() 