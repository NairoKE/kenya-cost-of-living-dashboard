# 🇰🇪 Kenya Cost of Living Dashboard

**⚠️ Work in Progress ⚠️**

An interactive dashboard built with Streamlit and advanced machine learning models to provide **live cost of living analysis** for essential goods across major retailers in Kenya. This project is designed to track real-time price changes and their correlation with economic indicators, tax policies, and inflation rates.

## 🎯 Project Vision

This program aims to become a comprehensive tool that can:
- **Provide live cost of living data** for various staple goods across Kenya
- **Track and analyze** the impact of tax changes, inflation rates, and economic indicators on consumer prices
- **Predict price fluctuations** based on economic policy changes and market conditions
- **Monitor** the purchasing power of Kenyan consumers in real-time
- **Enable data-driven decision making** for consumers, policymakers, and businesses

The ultimate goal is to create a predictive system that can anticipate how changes in economic policy, taxation, and global economic conditions will affect the day-to-day cost of living for Kenyan families.

## ✨ Features

- **Real-time Price Tracking**
  - Automated daily price scraping from major retailers:
    - Carrefour
    - Naivas
    - Quickmart
    - Chandarana Foodplus
  - Track essential items (milk, bread, sugar, etc.)
  - Compare prices across different stores

- **Advanced Analytics**
  - Price trend visualization
  - Store-wise price comparison
  - Price volatility analysis
  - Interactive price forecasting with:
    - Inflation adjustments
    - Election period impacts
    - Tax change effects
    - Combined economic factors

- **Economic Indicators**
  - Consumer Price Index (CPI) tracking
  - Inflation rate impact analysis
  - Historical election period analysis (2017, 2022)
  - Tax change impact predictions
  - Combined economic factor modeling

- **Event Impact Analysis**
  - Election period price trends (2027 forecast)
  - Tax change impact tracking
  - Holiday season price variations
  - Historical shock detection
  - Confidence intervals for predictions

- **Interactive Visualizations**
  - Dynamic filtering by store and product
  - Time-series trend charts
  - Price comparison bar graphs
  - Future date prediction slider
  - Economic factor toggles
  - Impact breakdown cards

## 🚀 Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/kenya-cost-of-living
   cd kenya-cost-of-living
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install Chrome WebDriver**
   - The scrapers use Selenium with Chrome. Make sure you have Chrome installed.
   - WebDriver is automatically managed by webdriver_manager.

4. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

## 📊 Data Sources

- **Supermarket Websites**
  - Real-time price scraping from major retailer websites
  - Daily updates to track price changes

- **Historical Data**
  - Sourced from KNBS (Kenya National Bureau of Statistics)
  - OpenAfrica datasets
  - Historical election period data (2017, 2022)
  - CPI and inflation rate data

## 🔄 Using the Dashboard

1. **Price Tracking**
   - Select stores and products from the sidebar
   - View current prices and historical trends
   - Compare prices across different stores

2. **Price Predictions**
   - Use the date slider to select a future date
   - Toggle different economic factors:
     - Inflation impact
     - Election period effects
     - Tax changes
   - View predicted prices and impact breakdown

3. **Economic Analysis**
   - Monitor CPI trends
   - Track inflation impacts
   - Analyze election period effects
   - View tax change implications

4. **Data Updates**
   - Click "Update Prices Now" for fresh data
   - Data is automatically cached for 1 hour
   - Download filtered data as CSV

## 📈 Price Shock Prediction

The dashboard uses several indicators to predict potential price shocks:

1. **Historical Patterns**
   - Analysis of past price spikes
   - Seasonal variations
   - Event-related changes

2. **Event Calendar**
   - Election periods (e.g., 2027 General Election)
   - Major tax changes
   - Public holidays
   - Economic events

3. **Economic Indicators**
   - Inflation rates
   - CPI trends
   - Historical election impacts
   - Tax change effects

4. **Volatility Tracking**
   - Price variation monitoring
   - Store-wise comparison
   - Category-level analysis

## 🛠️ Technical Details

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Forecasting**: Facebook Prophet
- **Web Scraping**: Selenium, BeautifulSoup4
- **Visualization**: Plotly
- **Economic Analysis**: Custom indicators module

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ALX Data Science Program
- Kenya National Bureau of Statistics
- OpenAfrica
- Contributing retailers for their public price data

## 📧 Contact

For any queries or suggestions, please reach out to:
- LinkedIn: [Ian Njoroge](https://www.linkedin.com/in/ian-njoroge-0252aa70/)

---
Built with ❤️ for Kenya 🇰🇪
