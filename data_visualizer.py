import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
import numpy as np
from datetime import datetime

class DataVisualizer:
    """
    Creates interactive visualizations for economic indicators data.
    """
    
    # Define indicator groups for better visualization
    INDICATOR_GROUPS = {
        'Price Stability': [
            'Inflation (consumer prices, annual %)',
            'Consumer Price Index (2010 = 100)',
            'Official exchange rate (KES per US$)',
        ],
        'Economic Growth': [
            'GDP growth (annual %)',
            'Agriculture, value added (annual % growth)',
            'Industry, value added (annual % growth)',
        ],
        'External Sector': [
            'Imports of goods and services (% of GDP)',
            'Exports of goods and services (% of GDP)',
            'Current account balance (% of GDP)',
        ],
        'Financial': [
            'Broad money (% of GDP)',
            'Lending interest rate (%)',
        ],
        'Social': [
            'Unemployment rate',
            'GNI per capita (current US$)',
            'Population growth (annual %)',
        ],
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
    
    def _add_trend_lines(self, df: pd.DataFrame, indicator: str) -> dict:
        """Add trend lines and calculate statistics for an indicator."""
        # Calculate trend line
        x = df['Year'].values
        y = df[indicator].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Calculate statistics
        trend_direction = "increasing" if z[0] > 0 else "decreasing"
        avg_annual_change = z[0]
        current_value = y[-1]
        historical_avg = y.mean()
        
        return {
            'trend_line': p(x),
            'direction': trend_direction,
            'avg_annual_change': avg_annual_change,
            'current_value': current_value,
            'historical_avg': historical_avg
        }
    
    def create_trend_dashboard(self, df: pd.DataFrame, output_dir: str = "visualizations") -> str:
        """
        Create an enhanced dashboard of economic indicator trends.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig = make_subplots(
            rows=len(self.INDICATOR_GROUPS),
            cols=1,
            subplot_titles=list(self.INDICATOR_GROUPS.keys()),
            vertical_spacing=0.05
        )
        
        # Add traces for each group
        row = 1
        annotations = []
        for group_name, indicators in self.INDICATOR_GROUPS.items():
            for indicator in indicators:
                # Add main data line
                fig.add_trace(
                    go.Scatter(
                        x=df['Year'],
                        y=df[indicator],
                        name=indicator,
                        mode='lines+markers',
                        line=dict(width=2),
                        hovertemplate='Year: %{x}<br>' + f'{indicator}: ' + '%{y:.2f}<extra></extra>'
                    ),
                    row=row,
                    col=1
                )
                
                # Calculate and add trend line
                trend_stats = self._add_trend_lines(df, indicator)
                fig.add_trace(
                    go.Scatter(
                        x=df['Year'],
                        y=trend_stats['trend_line'],
                        name=f'{indicator} (Trend)',
                        mode='lines',
                        line=dict(dash='dash', width=1),
                        showlegend=False,
                        hovertemplate='Trend<br>Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
                    ),
                    row=row,
                    col=1
                )
                
                # Add statistics annotation
                stats_text = (
                    f"{indicator}:<br>"
                    f"Trend: {trend_stats['direction']}<br>"
                    f"Avg Annual Change: {trend_stats['avg_annual_change']:.2f}<br>"
                    f"Current: {trend_stats['current_value']:.2f}<br>"
                    f"Historical Avg: {trend_stats['historical_avg']:.2f}"
                )
                annotations.append(
                    dict(
                        x=1.15,
                        y=0.5 - (0.1 * len(annotations)),
                        xref='paper',
                        yref='paper',
                        text=stats_text,
                        showarrow=False,
                        font=dict(size=10),
                        align='left'
                    )
                )
            
            # Update axes for each subplot
            fig.update_xaxes(title_text="Year", row=row, col=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="Value", row=row, col=1, gridcolor='lightgray')
            row += 1
        
        # Update layout
        fig.update_layout(
            title_text="Kenya Economic Indicators Dashboard",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            template="plotly_white",
            hovermode="x unified",
            height=1500,
            width=1200,
            annotations=annotations,
            plot_bgcolor='white'
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        # Save the dashboard
        output_file = output_dir / f"economic_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file)
        self.logger.info(f"Enhanced dashboard saved to {output_file}")
        
        return str(output_file)
    
    def create_correlation_heatmap(self, df: pd.DataFrame, output_dir: str = "visualizations") -> str:
        """
        Create an enhanced correlation heatmap with insights.
        """
        # Calculate correlations
        corr_matrix = df.drop('Year', axis=1).corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'indicator1': corr_matrix.columns[i],
                        'indicator2': corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>',
            colorscale="RdBu",
            zmid=0
        ))
        
        # Add annotations for strong correlations
        annotations = []
        for corr in strong_correlations:
            annotations.append(
                dict(
                    x=1.15,
                    y=-0.1 - (0.05 * len(annotations)),
                    xref='paper',
                    yref='paper',
                    text=f"Strong {corr['correlation']:.2f} correlation:<br>{corr['indicator1']}<br>vs<br>{corr['indicator2']}",
                    showarrow=False,
                    font=dict(size=10),
                    align='left'
                )
            )
        
        # Update layout
        fig.update_layout(
            title_text="Correlation Heatmap of Economic Indicators",
            template="plotly_white",
            height=900,
            width=1200,
            xaxis_tickangle=-45,
            annotations=annotations,
            margin=dict(r=300)  # Make room for annotations
        )
        
        # Save the heatmap
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"correlation_heatmap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file)
        self.logger.info(f"Enhanced heatmap saved to {output_file}")
        
        return str(output_file)
    
    def create_yoy_changes(self, df: pd.DataFrame, output_dir: str = "visualizations") -> str:
        """
        Create enhanced visualization of year-over-year changes.
        """
        # Calculate year-over-year changes
        yoy_df = df.copy()
        significant_changes = []
        
        for col in df.columns:
            if col != 'Year':
                yoy_df[f"{col} YoY Change"] = df[col].pct_change() * 100
                
                # Find significant changes (>= 2 standard deviations)
                std_dev = yoy_df[f"{col} YoY Change"].std()
                mean_change = yoy_df[f"{col} YoY Change"].mean()
                significant = yoy_df[abs(yoy_df[f"{col} YoY Change"] - mean_change) >= 2*std_dev]
                
                for _, row in significant.iterrows():
                    significant_changes.append({
                        'indicator': col,
                        'year': row['Year'],
                        'change': row[f"{col} YoY Change"]
                    })
        
        # Create subplots
        fig = make_subplots(
            rows=len(self.INDICATOR_GROUPS),
            cols=1,
            subplot_titles=[f"{group} - Year-over-Year Changes" for group in self.INDICATOR_GROUPS.keys()],
            vertical_spacing=0.05
        )
        
        # Add traces for each group
        row = 1
        for group_name, indicators in self.INDICATOR_GROUPS.items():
            for indicator in indicators:
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=yoy_df['Year'],
                        y=yoy_df[f"{indicator} YoY Change"],
                        name=indicator,
                        hovertemplate='Year: %{x}<br>' + f'{indicator} Change: ' + '%{y:.2f}%<extra></extra>'
                    ),
                    row=row,
                    col=1
                )
                
                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    row=row,
                    col=1
                )
            
            # Update axes
            fig.update_xaxes(title_text="Year", row=row, col=1, gridcolor='lightgray')
            fig.update_yaxes(title_text="% Change", row=row, col=1, gridcolor='lightgray')
            row += 1
        
        # Add annotations for significant changes
        annotations = []
        for i, change in enumerate(significant_changes):
            annotations.append(
                dict(
                    x=1.15,
                    y=-0.1 - (0.05 * i),
                    xref='paper',
                    yref='paper',
                    text=f"Significant change in {change['year']}:<br>{change['indicator']}: {change['change']:.2f}%",
                    showarrow=False,
                    font=dict(size=10),
                    align='left'
                )
            )
        
        # Update layout
        fig.update_layout(
            title_text="Year-over-Year Changes in Economic Indicators",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            template="plotly_white",
            hovermode="x unified",
            height=1500,
            width=1200,
            annotations=annotations,
            margin=dict(r=300),  # Make room for annotations
            plot_bgcolor='white'
        )
        
        # Save the visualization
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"yoy_changes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file)
        self.logger.info(f"Enhanced YoY changes visualization saved to {output_file}")
        
        return str(output_file)

def main():
    """Generate all visualizations."""
    try:
        # Load the latest economic indicators data
        df = pd.read_csv("data/economic_indicators_latest.csv")
        
        # Create visualizations
        visualizer = DataVisualizer()
        
        dashboard_file = visualizer.create_trend_dashboard(df)
        print(f"\nCreated enhanced trend dashboard: {dashboard_file}")
        
        heatmap_file = visualizer.create_correlation_heatmap(df)
        print(f"Created enhanced correlation heatmap: {heatmap_file}")
        
        yoy_file = visualizer.create_yoy_changes(df)
        print(f"Created enhanced YoY changes visualization: {yoy_file}")
        
    except FileNotFoundError:
        print("Error: No economic indicators data found. Please run wb_data_downloader.py first.")

if __name__ == "__main__":
    main() 