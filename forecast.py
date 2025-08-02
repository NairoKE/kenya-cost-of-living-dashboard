from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

def detect_price_shocks(df):
    """
    Detect significant price changes that might indicate shocks.
    Returns DataFrame with shock events.
    """
    # Calculate daily price changes
    df_sorted = df.sort_values('Date')
    df_sorted['price_change'] = df_sorted.groupby('Store')['Price (KES)'].diff()
    df_sorted['price_change_pct'] = df_sorted.groupby('Store')['Price (KES)'].pct_change()
    
    # Calculate rolling mean and std of price changes
    window = 7  # 7-day window
    df_sorted['rolling_mean'] = df_sorted.groupby('Store')['price_change'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    df_sorted['rolling_std'] = df_sorted.groupby('Store')['price_change'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std()
    )
    
    # Define shocks as price changes > 2 standard deviations from mean
    df_sorted['is_shock'] = abs(df_sorted['price_change'] - df_sorted['rolling_mean']) > (2 * df_sorted['rolling_std'])
    
    # Filter shock events
    shocks = df_sorted[df_sorted['is_shock']].copy()
    shocks['shock_magnitude'] = abs(shocks['price_change_pct'] * 100)
    
    return shocks[['Date', 'Store', 'Price (KES)', 'price_change', 'shock_magnitude']]

def add_event_markers(model: Prophet, events_df: Optional[pd.DataFrame] = None) -> Prophet:
    """
    Add known events to the Prophet model.
    """
    # Add Kenyan elections (every 5 years in August)
    elections = pd.DataFrame({
        'holiday': 'Election',
        'ds': pd.to_datetime(['2027-08-09', '2032-08-09']),
        'lower_window': -30,
        'upper_window': 30
    })
    
    # Add major tax changes (example dates - update with actual dates)
    tax_changes = pd.DataFrame({
        'holiday': 'Tax Change',
        'ds': pd.to_datetime(['2024-07-01', '2025-07-01']),
        'lower_window': 0,
        'upper_window': 90
    })
    
    # Combine all events
    all_events = pd.concat([elections, tax_changes], ignore_index=True)
    model.add_country_holidays(country_name='KE')  # Add Kenyan holidays
    model.add_regressor('is_weekend', mode='multiplicative')
    
    # Add custom holidays
    if hasattr(model, 'holidays') and isinstance(model.holidays, pd.DataFrame):
        model.holidays = pd.concat([model.holidays, all_events], ignore_index=True)
    else:
        model.holidays = all_events
    
    return model

def forecast_prices(df, periods=180):
    """
    Generate price forecasts using Prophet.
    Includes event detection and confidence intervals.
    
    Args:
        df: DataFrame with columns Date, Price (KES)
        periods: Number of days to forecast (default 180 days)
    
    Returns:
        DataFrame with forecast results
    """
    # Prepare data for Prophet
    df_prophet = df.groupby('Date')['Price (KES)'].mean().reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Add weekend indicator
    df_prophet['is_weekend'] = df_prophet['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Initialize and configure Prophet model
    model = Prophet(
        changepoint_prior_scale=0.05,  # More flexible trend changes
        seasonality_prior_scale=10,    # Stronger seasonality
        holidays_prior_scale=10,       # Stronger holiday effects
        daily_seasonality='auto',      # Auto-detect daily seasonality
        weekly_seasonality='auto',     # Auto-detect weekly seasonality
        yearly_seasonality='auto'      # Auto-detect yearly seasonality
    )
    
    # Add events and holidays
    model = add_event_markers(model)
    
    # Detect historical shocks
    shocks = detect_price_shocks(df)
    if not shocks.empty:
        shock_dates = pd.DataFrame({
            'holiday': 'Historical Shock',
            'ds': pd.to_datetime(shocks['Date'].unique()),
            'lower_window': 0,
            'upper_window': 30
        })
        # Add shock dates to holidays
        if hasattr(model, 'holidays') and isinstance(model.holidays, pd.DataFrame):
            model.holidays = pd.concat([model.holidays, shock_dates], ignore_index=True)
        else:
            model.holidays = shock_dates
    
    # Fit model
    model.fit(df_prophet)
    
    # Create future dataframe
    future = model.make_future_dataframe(
        periods=periods,
        freq='D',
        include_history=True
    )
    future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Add shock probability
    forecast['shock_probability'] = 0.0
    for date in shocks['Date'].unique():
        # Higher probability around historical shock dates
        days_diff = abs((forecast['ds'] - pd.to_datetime(date)).dt.days)
        forecast.loc[days_diff <= 30, 'shock_probability'] += np.exp(-days_diff[days_diff <= 30] / 30)
    
    # Normalize shock probability
    forecast['shock_probability'] = forecast['shock_probability'].clip(0, 1)
    
    return forecast

def get_price_insights(df):
    """
    Generate insights about price trends and potential future shocks.
    """
    insights = []
    
    # Detect recent price trends
    recent_trend = df.sort_values('Date').tail(30)
    price_change = (recent_trend['Price (KES)'].iloc[-1] - recent_trend['Price (KES)'].iloc[0]) / recent_trend['Price (KES)'].iloc[0] * 100
    
    if abs(price_change) > 5:
        trend_direction = "increased" if price_change > 0 else "decreased"
        insights.append(f"Prices have {trend_direction} by {abs(price_change):.1f}% in the last 30 days")
    
    # Detect seasonality
    if len(df) >= 365:  # Need at least a year of data
        monthly_avg = df.groupby(df['Date'].dt.month)['Price (KES)'].mean()
        max_month = monthly_avg.idxmax()
        min_month = monthly_avg.idxmin()
        month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June',
                      7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        
        insights.append(f"Historically, prices tend to be highest in {month_names[max_month]} and lowest in {month_names[min_month]}")
    
    # Detect price volatility
    volatility = df['Price (KES)'].std() / df['Price (KES)'].mean() * 100
    if volatility > 15:
        insights.append(f"Prices show high volatility ({volatility:.1f}% coefficient of variation)")
    
    # Check for upcoming events
    today = pd.Timestamp.now()
    election_2027 = pd.Timestamp('2027-08-09')
    days_to_election = (election_2027 - today).days
    
    if 0 < days_to_election <= 180:
        insights.append(f"Upcoming election in {days_to_election} days may affect prices")
    
    return insights

if __name__ == "__main__":
    # Test the forecasting
    df = pd.read_csv("data/cost_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    forecast = forecast_prices(df)
    print("Forecast generated successfully")
