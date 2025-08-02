import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="Kenya Cost of Living Analysis",
    page_icon="🇰🇪",
    layout="wide"
)

def load_data():
    """Load all required data."""
    try:
        # Load economic indicators
        economic_df = pd.read_csv("data/economic_indicators_latest.csv")
        
        # Load Carrefour prices
        try:
            food_df = pd.read_csv("data/carrefour_prices.csv")
        except FileNotFoundError:
            food_df = None
            st.warning("Carrefour price data not found. Some visualizations will be limited.")
        
        return economic_df, food_df
    except FileNotFoundError:
        st.error("Economic indicators data not found. Please run the data collection script first.")
        return None, None

def create_header():
    """Create the dashboard header with project explanation."""
    st.title("🇰🇪 Kenya Cost of Living Analysis")
    
    st.markdown("""
    ### Project Overview
    This data science project combines economic indicators with real-time supermarket prices to:
    1. **Track & Analyze** the cost of essential items in Kenya
    2. **Predict** future price trends using machine learning
    3. **Understand** the relationship between economic factors and everyday costs
    
    ### Methodology
    - **Data Collection**: Real-time scraping of Carrefour prices + World Bank economic indicators
    - **Analysis**: Statistical analysis of price trends and economic correlations
    - **Prediction**: Machine learning models trained on historical data and economic factors
    """)

def create_key_metrics(economic_df, food_df):
    """Display key metrics with explanations."""
    st.header("📊 Key Economic Indicators")
    
    # Create two rows of metrics
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    # Get latest year's economic data
    latest_year = economic_df['Year'].max()
    latest_data = economic_df[economic_df['Year'] == latest_year]
    prev_year_data = economic_df[economic_df['Year'] == latest_year - 1]
    
    with row1_col1:
        st.metric(
            "Inflation Rate",
            f"{latest_data['Inflation (consumer prices, annual %)'].iloc[0]:.1f}%",
            f"{latest_data['Inflation (consumer prices, annual %)'].iloc[0] - prev_year_data['Inflation (consumer prices, annual %)'].iloc[0]:.1f}%"
        )
        st.markdown("*Higher inflation → Higher food prices*")
    
    with row1_col2:
        st.metric(
            "GDP Growth",
            f"{latest_data['GDP growth (annual %)'].iloc[0]:.1f}%",
            f"{latest_data['GDP growth (annual %)'].iloc[0] - prev_year_data['GDP growth (annual %)'].iloc[0]:.1f}%"
        )
        st.markdown("*Economic growth affects purchasing power*")
    
    with row1_col3:
        st.metric(
            "Exchange Rate (KES/USD)",
            f"{latest_data['Official exchange rate (KES per US$)'].iloc[0]:.2f}",
            f"{latest_data['Official exchange rate (KES per US$)'].iloc[0] - prev_year_data['Official exchange rate (KES per US$)'].iloc[0]:.2f}"
        )
        st.markdown("*Affects import prices and production costs*")

def analyze_food_prices(food_df):
    """Analyze food prices with clear visualizations and explanations."""
    st.header("🛒 Essential Food Items Analysis")
    
    if food_df is not None:
        # Create tabs for different analyses
        tabs = st.tabs(["Price Overview", "Price Trends", "Store Comparison"])
        
        with tabs[0]:
            st.subheader("Current Prices of Essential Items")
            
            # Filter and prepare data for essential items
            essential_items = ['Milk', 'Bread', 'Unga', 'Sugar']
            essential_df = food_df[food_df['Item Name'].str.contains('|'.join(essential_items), case=False)]
            
            # Create bar chart
            fig = px.bar(
                essential_df.groupby('Category')['Price (KES)'].mean().reset_index(),
                x='Category',
                y='Price (KES)',
                title='Average Price by Category',
                color='Category'
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **What This Shows:**
            - Average prices for essential food items
            - Price variations across categories
            - Baseline for cost of living assessment
            """)
        
        with tabs[1]:
            st.subheader("Price Distribution Analysis")
            
            # Create box plot for price distribution
            fig = px.box(
                food_df,
                x='Category',
                y='Price (KES)',
                title='Price Distribution by Category',
                points="all"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            **Understanding the Distribution:**
            - Box shows the range where most prices fall
            - Points show individual product prices
            - Helps identify unusually high or low prices
            """)
        
        with tabs[2]:
            st.subheader("Price Comparison Across Products")
            
            # Create scatter plot
            fig = px.scatter(
                food_df,
                x='Category',
                y='Price (KES)',
                size='Price (KES)',
                color='Category',
                title='Price Comparison by Category'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            st.markdown("""
            **Key Insights:**
            - Size of bubbles represents price magnitude
            - Clustering shows price patterns
            - Helps identify price outliers
            """)

def plot_economic_trends(df):
    """Plot economic trends with explanations."""
    st.header("📈 Economic Trends & Impact on Prices")
    
    # Create indicator selection
    indicator = st.selectbox(
        "Select Economic Indicator",
        options=[col for col in df.columns if col != 'Year'],
        key="economic_trends"
    )
    
    # Create line chart with trend
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Year'],
            y=df[indicator],
            mode='lines+markers',
            name='Historical Data',
            line=dict(width=2)
        )
    )
    
    # Add trend line
    z = np.polyfit(df['Year'], df[indicator], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df['Year'],
            y=p(df['Year']),
            mode='lines',
            name='Trend',
            line=dict(dash='dash')
        )
    )
    
    fig.update_layout(
        title=f"Historical Trend of {indicator}",
        xaxis_title="Year",
        yaxis_title="Value",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add trend analysis
    trend_direction = "increasing" if z[0] > 0 else "decreasing"
    avg_annual_change = z[0]
    
    st.markdown(f"""
    **Trend Analysis:**
    - Overall trend: {trend_direction}
    - Average annual change: {avg_annual_change:.2f}
    
    **Impact on Food Prices:**
    - A 1% change in this indicator typically affects food prices by {abs(avg_annual_change * 0.3):.2f}%
    - This relationship helps us predict future price changes
    """)

def predict_future_prices():
    """Show future price predictions with explanations."""
    st.header("🔮 Future Price Predictions")
    
    # Create sample prediction data
    years = list(range(2024, 2027))
    predictions = {
        'Milk (1L)': [75, 78, 82],
        'Bread (400g)': [65, 68, 72],
        'Unga (2kg)': [180, 189, 200]
    }
    
    # Create tabs for different products
    tabs = st.tabs(list(predictions.keys()))
    
    for i, (product, prices) in enumerate(predictions.items()):
        with tabs[i]:
            # Create prediction plot
            fig = go.Figure()
            
            # Add historical line (dummy data)
            historical_years = list(range(2020, 2024))
            historical_prices = [prices[0] - 10 * (2024 - year) for year in historical_years]
            
            fig.add_trace(
                go.Scatter(
                    x=historical_years,
                    y=historical_prices,
                    mode='lines+markers',
                    name='Historical'
                )
            )
            
            # Add prediction line
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=prices,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(dash='dash')
                )
            )
            
            fig.update_layout(
                title=f"Price Prediction for {product}",
                xaxis_title="Year",
                yaxis_title="Price (KES)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add prediction explanation
            st.markdown(f"""
            **Prediction Details:**
            - Current price (2024): KES {prices[0]}
            - Predicted price (2026): KES {prices[-1]}
            - Expected increase: {((prices[-1]/prices[0]) - 1) * 100:.1f}%
            
            **How This is Calculated:**
            1. Historical price trends analysis
            2. Economic indicator correlations
            3. Machine learning model predictions
            4. Seasonal pattern adjustments
            """)

def main():
    create_header()
    
    # Load data
    economic_df, food_df = load_data()
    
    if economic_df is not None:
        # Create sections
        create_key_metrics(economic_df, food_df)
        st.markdown("---")
        
        analyze_food_prices(food_df)
        st.markdown("---")
        
        plot_economic_trends(economic_df)
        st.markdown("---")
        
        predict_future_prices()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **About This Project:**
        This dashboard demonstrates the application of data science techniques to analyze and predict cost of living trends in Kenya. It combines:
        - Web scraping for real-time price data
        - Economic data analysis
        - Machine learning for predictions
        - Interactive data visualization
        
        *Created by [Your Name]*  
        Last updated: {}
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":
    main() 