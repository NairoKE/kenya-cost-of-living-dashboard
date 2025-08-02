import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Union

from clean_data import load_and_clean_data
from forecast import forecast_prices, get_price_insights
from combine_scraped_data import run_all_scrapers
from economic_indicators import EconomicIndicators, load_kenya_cpi_data

# Page config
st.set_page_config(
    page_title="Kenya Cost of Living Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize economic indicators
@st.cache_resource
def get_economic_indicators():
    return EconomicIndicators()

# Load and clean data (with caching)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data():
    return load_and_clean_data()

# Load CPI data
@st.cache_data
def get_cpi_data():
    return load_kenya_cpi_data()

# Initialize
indicators = get_economic_indicators()
df = get_data()
cpi_data = get_cpi_data()

# Sidebar
with st.sidebar:
    st.title("🛠️ Controls")
    
    # Data Update Section
    st.subheader("🔄 Data Updates")
    if st.button("Update Prices Now"):
        with st.spinner("Scraping latest prices..."):
            try:
                run_all_scrapers()
                st.success("✅ Prices updated successfully!")
                st.cache_data.clear()  # Clear cache to load new data
                df = get_data()  # Reload data after update
            except Exception as e:
                st.error(f"Failed to update prices: {str(e)}")
    
    # Filters
    st.subheader("🔍 Filters")
    
    # Ensure we have the required columns
    if 'Store' not in df.columns:
        st.error("No store data available. Please update prices.")
        stores = []
    else:
        stores = sorted(df['Store'].unique())
    
    store = st.multiselect(
        "Select Stores",
        options=stores,
        default=stores[:3] if stores else []
    )
    
    # Handle Category filter
    if 'Category' not in df.columns:
        st.error("No category data available. Please update prices.")
        categories = ['All']
    else:
        categories = ['All'] + sorted(df['Category'].unique().tolist())
    
    category = st.selectbox(
        "Select Category",
        options=categories
    )
    
    # Handle Item filter
    if 'Item Name' not in df.columns:
        st.error("No item data available. Please update prices.")
        items = []
    else:
        if category != 'All':
            items = sorted(df[df['Category'] == category]['Item Name'].unique())
        else:
            items = sorted(df['Item Name'].unique())
    
    item_name = st.selectbox(
        "Select Item",
        options=items if items else ["No items available"]
    )

# Main content
st.title("📊 Kenya Cost of Living Dashboard")
st.markdown("""
Track and forecast essential goods prices across major retailers in Kenya. 
This dashboard helps consumers make informed decisions and analysts track price trends.
""")

# Filter data
if store and item_name != "No items available":
    mask = df['Store'].isin(store) & (df['Item Name'] == item_name)
    if category != 'All' and 'Category' in df.columns:
        mask &= df['Category'] == category
    df_filtered = df[mask].copy()
else:
    df_filtered = pd.DataFrame()

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    # Current Price Comparison
    st.subheader("📈 Current Price Comparison")
    
    fig_comparison = px.bar(
        df_filtered.groupby('Store')['Price (KES)'].mean().reset_index(),
        x='Store',
        y='Price (KES)',
        title=f"Average Price of {item_name} by Store",
        color='Store'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

# Price Trend Over Time
st.subheader("📉 Price Trends Analysis")

# Create tabs for different visualizations
trend_tab1, trend_tab2, trend_tab3 = st.tabs(["Price Trends", "Price Changes", "Economic Context"])

with trend_tab1:
    # Enhanced line graph with range selector
    df_trend = df_filtered.copy()
    df_trend['Date'] = pd.to_datetime(df_trend['Date'])
    df_trend = df_trend.sort_values(by=['Date'], ascending=[True], ignore_index=True)
    
    fig_trend = go.Figure()
    
    for store_name in df_trend['Store'].unique():
        store_data = df_trend[df_trend['Store'] == store_name]
        fig_trend.add_trace(
            go.Scatter(
                x=store_data['Date'],
                y=store_data['Price (KES)'],
                name=store_name,
                mode='lines+markers'
            )
        )
    
    fig_trend.update_layout(
        title=f"Price Trend for {item_name}",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with trend_tab2:
    # Price change analysis
    if not df_trend.empty:
        df_changes = df_trend.copy()
        df_changes['Previous Price'] = df_changes.groupby('Store')['Price (KES)'].shift(1)
        df_changes['Price Change'] = df_changes['Price (KES)'] - df_changes['Previous Price']
        df_changes['Price Change %'] = (df_changes['Price Change'] / df_changes['Previous Price']) * 100
        
        fig_changes = go.Figure()
        
        for store_name in df_changes['Store'].unique():
            store_data = df_changes[df_changes['Store'] == store_name]
            fig_changes.add_trace(
                go.Scatter(
                    x=store_data['Date'],
                    y=store_data['Price Change %'],
                    name=store_name,
                    mode='lines+markers'
                )
            )
        
        fig_changes.update_layout(
            title="Price Changes Over Time (%)",
            yaxis_title="Price Change (%)",
            showlegend=True
        )
        st.plotly_chart(fig_changes, use_container_width=True)
        
        # Summary statistics
        st.subheader("Price Change Summary")
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            max_increase = df_changes['Price Change %'].max()
            st.metric("Largest Price Increase", f"{max_increase:.1f}%")
            
        with summary_cols[1]:
            max_decrease = df_changes['Price Change %'].min()
            st.metric("Largest Price Decrease", f"{max_decrease:.1f}%")
            
        with summary_cols[2]:
            avg_change = df_changes['Price Change %'].mean()
            st.metric("Average Price Change", f"{avg_change:.1f}%")

with trend_tab3:
    # Economic context
    st.subheader("Economic Context")
    
    # Get economic indicators
    econ = get_economic_indicators()
    
    # Create metrics
    metric_cols = st.columns(3)
    
    with metric_cols[0]:
        st.metric("GDP Growth (2024)", f"{econ.gdp_growth['2024']}%")
        st.metric("Public Debt Ratio", f"{econ.economic_indicators['public_debt_ratio']}% of GDP")
        
    with metric_cols[1]:
        st.metric("Current Account Deficit", f"{econ.economic_indicators['current_account_deficit']}% of GDP")
        st.metric("Forex Reserves", f"${econ.economic_indicators['forex_reserves']}B")
        
    with metric_cols[2]:
        st.metric("Import Cover", f"{econ.economic_indicators['import_cover_months']} months")
        st.metric("Fiscal Deficit", f"{econ.economic_indicators['fiscal_deficit']}% of GDP")
    
    # Plot inflation trend
    inflation_df = pd.DataFrame(econ.inflation_data).reset_index()
    
    fig_inflation = go.Figure()
    fig_inflation.add_trace(
        go.Scatter(
            x=inflation_df['Date'],
            y=inflation_df['Inflation_Rate'],
            name='Inflation Rate',
            mode='lines+markers'
        )
    )
    
    fig_inflation.update_layout(
        title="Inflation Rate Trend",
        yaxis_title="Inflation Rate (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig_inflation, use_container_width=True)

# Economic Impact Analysis
st.subheader("🔮 Price Impact Prediction")

# Date slider for prediction
min_date = datetime.now()
max_date = datetime(2027, 12, 31)
prediction_date = st.slider(
    "Select future date for price prediction",
    min_value=min_date,
    max_value=max_date,
    value=min_date + timedelta(days=180),
    format="YYYY-MM-DD"
)

# Factor toggles
col3, col4, col5 = st.columns(3)
with col3:
    include_inflation = st.checkbox("Include Inflation Impact", value=True)
with col4:
    include_election = st.checkbox("Include Election Impact", value=True)
with col5:
    include_tax = st.checkbox("Include Tax Changes", value=True)

# Get current average price
current_price = float(df_filtered['Price (KES)'].mean())

# Get prediction
if current_price > 0:
    prediction = indicators.get_combined_forecast(
        current_price,
        prediction_date,
        include_inflation=include_inflation,
        include_election=include_election,
        include_tax=include_tax
    )
    
    if isinstance(prediction, dict):  # Type check for prediction
        # Display predictions
        col6, col7 = st.columns(2)
        
        with col6:
            st.markdown("### Price Prediction")
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current Price</h4>
                <h2>KES {current_price:.2f}</h2>
            </div>
            <div class="metric-card">
                <h4>Predicted Price ({prediction_date.strftime('%Y-%m-%d')})</h4>
                <h2>KES {prediction['adjusted_price']:.2f}</h2>
                <p>Expected change: {prediction['total_impact_percent']:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col7:
            st.markdown("### Impact Breakdown")
            impacts = prediction.get('impact_breakdown', [])
            if isinstance(impacts, list) and impacts:  # Type check for impacts
                for impact in impacts:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p>{impact}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No significant price impacts predicted for this period")
    else:
        st.error("Failed to generate price prediction")

# CPI Trend
st.subheader("📈 Consumer Price Index (CPI) Trend")
fig_cpi = px.line(
    cpi_data,
    x='Date',
    y='CPI',
    title="Kenya Consumer Price Index Trend"
)
fig_cpi.add_vline(
    x=datetime(2027, 8, 9),
    line_dash="dash",
    line_color="red",
    annotation_text="2027 Election"
)
st.plotly_chart(fig_cpi, use_container_width=True)

# Insights and Recommendations
st.subheader("💡 Insights")
col8, col9 = st.columns(2)

with col8:
    # Price Statistics
    stats_df = df_filtered.groupby('Store')['Price (KES)'].agg(['mean', 'min', 'max']).round(2)
    st.dataframe(stats_df, use_container_width=True)
    
    if not stats_df.empty:
        try:
            min_store = stats_df.index[stats_df['mean'].argmin()]
            min_price = stats_df.at[min_store, 'mean']
            st.info(f"💰 Best Value: {min_store} (Average: KES {min_price:.2f})")
        except (KeyError, IndexError):
            st.info("Unable to determine best value store")
    else:
        st.info("No price data available for the selected filters")

with col9:
    # Price insights
    insights = get_price_insights(df_filtered)
    if insights:
        for insight in insights:
            st.markdown(f"""
            <div class="metric-card">
                <p>{insight}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No insights available for the selected data")

# Download section
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Data")

# Download filtered data
if not df_filtered.empty:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Filtered Data",
        data=csv,
        file_name=f"kenya_prices_{item_name}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Built with ❤️ by [Your Name] | Data updated daily | 
    <a href="https://github.com/yourusername/kenya-cost-of-living" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
