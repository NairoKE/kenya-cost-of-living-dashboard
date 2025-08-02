import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import logging
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PredictiveModel:
    """
    Implements various predictive models for economic indicators.
    """
    
    def __init__(self):
        self._setup_logging()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame, target_indicator: str) -> tuple:
        """
        Prepare features for the model.
        
        Args:
            df: DataFrame containing economic indicators
            target_indicator: The indicator to predict
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Create lag features (previous year values)
        df_features = df.copy()
        for col in df.columns:
            if col != 'Year':
                df_features[f"{col}_lag1"] = df_features[col].shift(1)
                df_features[f"{col}_lag2"] = df_features[col].shift(2)
        
        # Create year-over-year changes
        for col in df.columns:
            if col != 'Year':
                df_features[f"{col}_yoy"] = df_features[col].pct_change() * 100
        
        # Drop rows with NaN values (due to lag creation)
        df_features = df_features.dropna()
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns 
                       if col != 'Year' and col != target_indicator]
        X = df_features[feature_cols]
        y = df_features[target_indicator]
        
        return X, y, feature_cols
    
    def train_models(self, df: pd.DataFrame, target_indicator: str,
                    test_size: float = 0.2) -> dict:
        """
        Train multiple models for the target indicator.
        
        Args:
            df: DataFrame containing economic indicators
            target_indicator: The indicator to predict
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing model evaluation metrics
        """
        self.logger.info(f"Training models for {target_indicator}")
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df, target_indicator)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[target_indicator] = scaler
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = xgb_model.predict(X_test_scaled)
        
        # Train SARIMA
        try:
            sarima_model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_results = sarima_model.fit(disp=False)
            sarima_pred = sarima_results.forecast(len(y_test))
        except:
            self.logger.warning(f"SARIMA model failed for {target_indicator}")
            sarima_pred = None
        
        # Store models
        self.models[target_indicator] = {
            'random_forest': rf_model,
            'xgboost': xgb_model,
            'feature_names': feature_names
        }
        
        # Calculate feature importance
        self.feature_importance[target_indicator] = {
            'random_forest': dict(zip(feature_names, rf_model.feature_importances_)),
            'xgboost': dict(zip(feature_names, xgb_model.feature_importances_))
        }
        
        # Calculate metrics
        metrics = {
            'random_forest': {
                'mse': mean_squared_error(y_test, rf_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'r2': r2_score(y_test, rf_pred),
                'mape': mean_absolute_percentage_error(y_test, rf_pred)
            },
            'xgboost': {
                'mse': mean_squared_error(y_test, xgb_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'r2': r2_score(y_test, xgb_pred),
                'mape': mean_absolute_percentage_error(y_test, xgb_pred)
            }
        }
        
        if sarima_pred is not None:
            metrics['sarima'] = {
                'mse': mean_squared_error(y_test, sarima_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, sarima_pred)),
                'r2': r2_score(y_test, sarima_pred),
                'mape': mean_absolute_percentage_error(y_test, sarima_pred)
            }
        
        return metrics
    
    def predict_future(self, df: pd.DataFrame, target_indicator: str,
                       periods: int = 3, n_iterations: int = 100) -> pd.DataFrame:
        """
        Make predictions for future periods with confidence intervals.
        
        Args:
            df: DataFrame containing economic indicators
            target_indicator: The indicator to predict
            periods: Number of future periods to predict
            n_iterations: Number of bootstrap iterations for confidence intervals
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if target_indicator not in self.models:
            raise ValueError(f"No trained models found for {target_indicator}")
        
        # Prepare latest data
        X, _, _ = self.prepare_features(df, target_indicator)
        X_latest = X.iloc[-1:]
        
        # Scale features
        X_latest_scaled = self.scalers[target_indicator].transform(X_latest)
        
        # Initialize arrays for bootstrap predictions
        rf_predictions = np.zeros((n_iterations, periods))
        xgb_predictions = np.zeros((n_iterations, periods))
        
        # Generate bootstrap predictions
        for i in range(n_iterations):
            # Random Forest predictions with bootstrap
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[bootstrap_idx]
            y_bootstrap = df[target_indicator].iloc[bootstrap_idx]
            
            # Retrain models on bootstrap sample
            X_bootstrap_scaled = self.scalers[target_indicator].transform(X_bootstrap)
            rf_model = RandomForestRegressor(n_estimators=100, random_state=i)
            rf_model.fit(X_bootstrap_scaled, y_bootstrap)
            rf_predictions[i] = rf_model.predict(X_latest_scaled)[0]
            
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=i)
            xgb_model.fit(X_bootstrap_scaled, y_bootstrap)
            xgb_predictions[i] = xgb_model.predict(X_latest_scaled)[0]
        
        # Calculate confidence intervals
        rf_lower = np.percentile(rf_predictions, 2.5, axis=0)
        rf_upper = np.percentile(rf_predictions, 97.5, axis=0)
        rf_mean = np.mean(rf_predictions, axis=0)
        
        xgb_lower = np.percentile(xgb_predictions, 2.5, axis=0)
        xgb_upper = np.percentile(xgb_predictions, 97.5, axis=0)
        xgb_mean = np.mean(xgb_predictions, axis=0)
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'Year': [df['Year'].max() + i + 1 for i in range(periods)],
            'RandomForest_Mean': rf_mean,
            'RandomForest_Lower': rf_lower,
            'RandomForest_Upper': rf_upper,
            'XGBoost_Mean': xgb_mean,
            'XGBoost_Lower': xgb_lower,
            'XGBoost_Upper': xgb_upper
        })
        
        return predictions
    
    def visualize_predictions(self, df: pd.DataFrame, target_indicator: str,
                            predictions: pd.DataFrame,
                            output_dir: str = "visualizations") -> str:
        """
        Create enhanced visualization of predictions with confidence intervals.
        """
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=df['Year'],
                y=df[target_indicator],
                name='Historical',
                mode='lines+markers',
                line=dict(color='black')
            )
        )
        
        # Add Random Forest predictions with confidence interval
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['RandomForest_Upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['RandomForest_Lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(100, 100, 200, 0.2)',
                fill='tonexty',
                name='Random Forest 95% CI',
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['RandomForest_Mean'],
                name='Random Forest Prediction',
                mode='lines+markers',
                line=dict(color='blue', dash='dash')
            )
        )
        
        # Add XGBoost predictions with confidence interval
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['XGBoost_Upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['XGBoost_Lower'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(200, 100, 100, 0.2)',
                fill='tonexty',
                name='XGBoost 95% CI',
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=predictions['Year'],
                y=predictions['XGBoost_Mean'],
                name='XGBoost Prediction',
                mode='lines+markers',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"Predictions for {target_indicator}",
            xaxis_title="Year",
            yaxis_title="Value",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # Add annotation with prediction ranges
        latest_year = predictions['Year'].max()
        annotation_text = (
            f"Predictions for {latest_year}:<br>"
            f"Random Forest: {predictions['RandomForest_Mean'].iloc[-1]:.2f} "
            f"({predictions['RandomForest_Lower'].iloc[-1]:.2f} - {predictions['RandomForest_Upper'].iloc[-1]:.2f})<br>"
            f"XGBoost: {predictions['XGBoost_Mean'].iloc[-1]:.2f} "
            f"({predictions['XGBoost_Lower'].iloc[-1]:.2f} - {predictions['XGBoost_Upper'].iloc[-1]:.2f})"
        )
        
        fig.add_annotation(
            x=1.15,
            y=0.5,
            xref="paper",
            yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        # Save visualization
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"predictions_{target_indicator.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file)
        
        return str(output_file)
    
    def visualize_feature_importance(self, target_indicator: str,
                                   output_dir: str = "visualizations") -> str:
        """
        Create visualization of feature importance.
        
        Args:
            target_indicator: The indicator being analyzed
            output_dir: Directory to save visualization
            
        Returns:
            Path to the generated visualization
        """
        if target_indicator not in self.feature_importance:
            raise ValueError(f"No feature importance data found for {target_indicator}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Random Forest Feature Importance",
                          "XGBoost Feature Importance")
        )
        
        # Add Random Forest feature importance
        rf_importance = pd.DataFrame.from_dict(
            self.feature_importance[target_indicator]['random_forest'],
            orient='index',
            columns=['importance']
        ).sort_values('importance', ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=rf_importance['importance'],
                y=rf_importance.index,
                orientation='h',
                name='Random Forest'
            ),
            row=1, col=1
        )
        
        # Add XGBoost feature importance
        xgb_importance = pd.DataFrame.from_dict(
            self.feature_importance[target_indicator]['xgboost'],
            orient='index',
            columns=['importance']
        ).sort_values('importance', ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=xgb_importance['importance'],
                y=xgb_importance.index,
                orientation='h',
                name='XGBoost'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1000,
            title_text=f"Feature Importance for {target_indicator}",
            showlegend=False,
            template="plotly_white"
        )
        
        # Save visualization
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"feature_importance_{target_indicator.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file)
        
        return str(output_file)

def main():
    """Test the predictive models."""
    try:
        # Load the latest economic indicators data
        df = pd.read_csv("data/economic_indicators_latest.csv")
        
        # Create predictive model
        model = PredictiveModel()
        
        # List of indicators to predict
        indicators = [
            'GDP growth (annual %)',
            'Inflation (consumer prices, annual %)'
        ]
        
        # Generate predictions for each indicator
        for indicator in indicators:
            print(f"\n{'='*50}")
            print(f"Predictions for {indicator}")
            print('='*50)
            
            # Train models and get metrics
            metrics = model.train_models(df, indicator)
            print("\nModel Metrics:")
            print(json.dumps(metrics, indent=2))
            
            # Make predictions
            predictions = model.predict_future(df, indicator)
            print("\nPredictions with Confidence Intervals:")
            print(predictions)
            
            # Create visualizations
            pred_viz = model.visualize_predictions(df, indicator, predictions)
            print(f"\nPrediction visualization saved to: {pred_viz}")
            
            feat_viz = model.visualize_feature_importance(indicator)
            print(f"Feature importance visualization saved to: {feat_viz}")
        
    except FileNotFoundError:
        print("Error: No economic indicators data found. Please run wb_data_downloader.py first.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 