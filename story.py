import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple
import hashlib
from dataclasses import dataclass
from enum import Enum
import re
import base64

# Import from your existing system
from chart_utils import (
    create_dark_chart,
    get_business_relevant_columns,
    detect_value_type,
    format_value
)

# Story Types
class StoryType(Enum):
    SALES_PERFORMANCE = "sales_performance"
    MARKETING_ROI = "marketing_roi"
    CUSTOMER_JOURNEY = "customer_journey"
    OPPORTUNITY_HEALTH = "opportunity_health"
    CHURN_ANALYSIS = "churn_analysis"
    EXECUTIVE_DASHBOARD = "executive_dashboard"

# Data classes for story structure
@dataclass
class StoryChapter:
    """Individual chapter in a story"""
    id: str
    title: str
    subtitle: str
    narrative: str
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    actions: List[Dict[str, Any]]
    what_if_scenarios: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@dataclass
class DataStory:
    """Complete data story structure"""
    id: str
    title: str
    story_type: StoryType
    chapters: List[StoryChapter]
    executive_summary: str
    key_takeaways: List[str]
    recommended_actions: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Base Story Component - NO TYPE HINT for conversational_agent
class StoryComponent:
    """Base class for all story types"""
    
    def __init__(self, conversational_agent):  # No type hint to avoid circular import
        self.agent = conversational_agent
        self.story_type = None
        self.chapters = []
        
    def analyze_data(self, dataframe: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis for the story"""
        business_cols = get_business_relevant_columns(dataframe)
        
        analysis = {
            'shape': dataframe.shape,
            'columns': list(dataframe.columns),
            'business_cols': business_cols,
            'time_range': self._get_time_range(dataframe, business_cols),
            'key_metrics': self._calculate_key_metrics(dataframe, business_cols),
            'segments': self._analyze_segments(dataframe, business_cols),
            'trends': self._analyze_trends(dataframe, business_cols),
            'anomalies': self._detect_anomalies(dataframe, business_cols)
        }
        
        return analysis
    
    def _get_time_range(self, df: pd.DataFrame, business_cols: Dict) -> Dict:
        """Get time range from data"""
        if business_cols['temporal']:
            time_col = business_cols['temporal'][0]
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                return {
                    'start': df[time_col].min(),
                    'end': df[time_col].max(),
                    'duration': (df[time_col].max() - df[time_col].min()).days
                }
            except:
                pass
        return {}
    
    def _calculate_key_metrics(self, df: pd.DataFrame, business_cols: Dict) -> Dict:
        """Calculate key business metrics"""
        metrics = {}
        
        for metric_col in business_cols['numerical'][:5]:
            if metric_col in df.columns:
                value_type = detect_value_type(df[metric_col])
                metrics[metric_col] = {
                    'total': format_value(df[metric_col].sum(), value_type),
                    'average': format_value(df[metric_col].mean(), value_type),
                    'median': format_value(df[metric_col].median(), value_type),
                    'growth': self._calculate_growth(df, metric_col, business_cols),
                    'top_performer': self._get_top_performer(df, metric_col, business_cols)
                }
        
        return metrics
    
    def _calculate_growth(self, df: pd.DataFrame, metric_col: str, business_cols: Dict) -> str:
        """Calculate growth rate for a metric"""
        if business_cols['temporal']:
            try:
                time_col = business_cols['temporal'][0]
                df_sorted = df.sort_values(time_col)
                first_value = df_sorted[metric_col].iloc[:10].mean()
                last_value = df_sorted[metric_col].iloc[-10:].mean()
                
                if first_value > 0:
                    growth = ((last_value - first_value) / first_value) * 100
                    return f"{growth:+.1f}%"
            except:
                pass
        return "N/A"
    
    def _get_top_performer(self, df: pd.DataFrame, metric_col: str, business_cols: Dict) -> str:
        """Get top performer for a metric"""
        if business_cols['categorical']:
            cat_col = business_cols['categorical'][0]
            try:
                top = df.groupby(cat_col)[metric_col].sum().idxmax()
                value = df.groupby(cat_col)[metric_col].sum().max()
                value_type = detect_value_type(df[metric_col])
                return f"{top} ({format_value(value, value_type)})"
            except:
                pass
        return "N/A"
    
    def _analyze_segments(self, df: pd.DataFrame, business_cols: Dict) -> Dict:
        """Analyze performance by segments"""
        segments = {}
        
        for cat_col in business_cols['categorical'][:2]:
            if cat_col in df.columns:
                for metric_col in business_cols['numerical'][:2]:
                    if metric_col in df.columns:
                        grouped = df.groupby(cat_col)[metric_col].agg(['sum', 'mean', 'count'])
                        segments[f"{cat_col}_{metric_col}"] = {
                            'top_3': grouped.nlargest(3, 'sum').to_dict(),
                            'bottom_3': grouped.nsmallest(3, 'sum').to_dict(),
                            'variance': grouped['sum'].std() / grouped['sum'].mean() if grouped['sum'].mean() > 0 else 0
                        }
        
        return segments
    
    def _analyze_trends(self, df: pd.DataFrame, business_cols: Dict) -> Dict:
        """Analyze trends in the data"""
        trends = {}
        
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            for metric_col in business_cols['numerical'][:2]:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    time_series = df.groupby(df[time_col].dt.to_period('M'))[metric_col].sum()
                    
                    # Calculate trend direction
                    if len(time_series) > 1:
                        trend_direction = "increasing" if time_series.iloc[-1] > time_series.iloc[0] else "decreasing"
                        volatility = time_series.std() / time_series.mean() if time_series.mean() > 0 else 0
                        
                        trends[metric_col] = {
                            'direction': trend_direction,
                            'volatility': volatility,
                            'recent_change': ((time_series.iloc[-1] - time_series.iloc[-2]) / time_series.iloc[-2] * 100) if len(time_series) > 1 and time_series.iloc[-2] > 0 else 0
                        }
                except:
                    pass
        
        return trends
    
    def _detect_anomalies(self, df: pd.DataFrame, business_cols: Dict) -> List[Dict]:
        """Detect anomalies in the data"""
        anomalies = []
        
        for metric_col in business_cols['numerical'][:2]:
            if metric_col in df.columns:
                # Simple anomaly detection using z-score
                mean = df[metric_col].mean()
                std = df[metric_col].std()
                
                if std > 0:
                    df['z_score'] = (df[metric_col] - mean) / std
                    outliers = df[abs(df['z_score']) > 2.5]
                    
                    if len(outliers) > 0:
                        for cat_col in business_cols['categorical'][:1]:
                            if cat_col in df.columns:
                                top_anomaly = outliers.nlargest(1, metric_col)
                                if len(top_anomaly) > 0:
                                    anomalies.append({
                                        'metric': metric_col,
                                        'category': cat_col,
                                        'value': format_value(top_anomaly[metric_col].iloc[0], detect_value_type(df[metric_col])),
                                        'entity': top_anomaly[cat_col].iloc[0] if cat_col in top_anomaly.columns else 'Unknown',
                                        'severity': 'high' if abs(top_anomaly['z_score'].iloc[0]) > 3 else 'medium'
                                    })
        
        return anomalies[:3]  # Limit to top 3 anomalies

# Sales Performance Story
class SalesPerformanceStory(StoryComponent):
    """Generate sales performance narrative"""
    
    def __init__(self, conversational_agent):
        super().__init__(conversational_agent)
        self.story_type = StoryType.SALES_PERFORMANCE
    
    def generate(self, dataframe: pd.DataFrame, context: Dict[str, Any]) -> DataStory:
        """Generate complete sales performance story"""
        
        # Analyze data
        analysis = self.analyze_data(dataframe, context)
        
        # Create chapters
        chapters = [
            self._create_overview_chapter(dataframe, analysis),
            self._create_performance_trends_chapter(dataframe, analysis),
            self._create_segment_analysis_chapter(dataframe, analysis),
            self._create_opportunity_health_chapter(dataframe, analysis),
            self._create_forecast_chapter(dataframe, analysis),
            self._create_action_plan_chapter(dataframe, analysis)
        ]
        
        # Generate executive summary
        exec_summary = self._generate_executive_summary(analysis)
        
        # Extract key takeaways
        key_takeaways = self._extract_key_takeaways(analysis)
        
        # Compile recommended actions
        recommended_actions = self._compile_recommended_actions(chapters)
        
        return DataStory(
            id=f"sales_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Sales Performance Analysis",
            story_type=self.story_type,
            chapters=chapters,
            executive_summary=exec_summary,
            key_takeaways=key_takeaways,
            recommended_actions=recommended_actions,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'data_points': len(dataframe),
                'analysis': analysis
            }
        )
    
    def _create_overview_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create overview chapter with KPIs based on ACTUAL data"""
        
        # Use actual column names from Superstore data
        kpi_metrics = {}
        
        # Total Sales
        if 'Sales' in df.columns:
            total_sales = df['Sales'].sum()
            kpi_metrics['Total Sales'] = {
                'value': format_value(total_sales, 'currency'),
                'change': self._calculate_period_change(df, 'Sales', 'Order Date') if 'Order Date' in df.columns else 'N/A',
                'label': 'Total Sales'
            }
        
        # Total Profit
        if 'Profit' in df.columns:
            total_profit = df['Profit'].sum()
            profit_margin = (total_profit / df['Sales'].sum() * 100) if 'Sales' in df.columns else 0
            kpi_metrics['Total Profit'] = {
                'value': format_value(total_profit, 'currency'),
                'change': f"{profit_margin:.1f}% margin",
                'label': 'Total Profit'
            }
        
        # Order Count
        if 'Order ID' in df.columns:
            unique_orders = df['Order ID'].nunique()
            kpi_metrics['Total Orders'] = {
                'value': f"{unique_orders:,}",
                'change': 'N/A',
                'label': 'Unique Orders'
            }
        
        # Customer Count
        if 'Customer ID' in df.columns:
            unique_customers = df['Customer ID'].nunique()
            kpi_metrics['Total Customers'] = {
                'value': f"{unique_customers:,}",
                'change': 'N/A',
                'label': 'Unique Customers'
            }
        
        # Create visualizations based on actual data
        visualizations = []
        
        # Top products/categories by sales
        if 'Category' in df.columns and 'Sales' in df.columns:
            category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            
            fig_category = px.bar(
                x=category_sales.index,
                y=category_sales.values,
                title="Sales by Category",
                labels={'x': 'Category', 'y': 'Sales ($)'}
            )
            fig_category = create_dark_chart(fig_category)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_category,
                'title': 'Sales by Category'
            })
        
        # Generate narrative based on actual data
        narrative = f"""
        Welcome to your Sales Performance Analysis for Superstore data.
        
        Your dataset contains {len(df):,} transactions across {df['State'].nunique() if 'State' in df.columns else 'multiple'} states
        and {df['Category'].nunique() if 'Category' in df.columns else 'several'} product categories.
        
        Key highlights:
        - Total Revenue: {kpi_metrics.get('Total Sales', {}).get('value', 'N/A')}
        - Total Profit: {kpi_metrics.get('Total Profit', {}).get('value', 'N/A')} 
        - Serving {kpi_metrics.get('Total Customers', {}).get('value', 'N/A')} customers
        - Processing {kpi_metrics.get('Total Orders', {}).get('value', 'N/A')} orders
        """
        
        # Create main KPI dashboard
        fig_kpi = self._create_kpi_dashboard(kpi_metrics)
        visualizations.insert(0, {'type': 'plotly', 'figure': fig_kpi, 'title': 'Key Performance Indicators'})
        
        # Create summary chart based on business columns
        business_cols = analysis['business_cols']
        if business_cols['categorical'] and business_cols['numerical']:
            cat_col = business_cols['categorical'][0]
            metric_col = business_cols['numerical'][0]
            
            summary_data = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False).head(10)
            
            fig_summary = px.bar(
                x=summary_data.index,
                y=summary_data.values,
                title=f"Top 10 {cat_col} by {metric_col}",
                labels={'x': cat_col, 'y': metric_col}
            )
            fig_summary = create_dark_chart(fig_summary)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_summary,
                'title': 'Top Performers'
            })
        
        return StoryChapter(
            id="overview",
            title="Executive Overview",
            subtitle="Your sales performance at a glance",
            narrative=narrative,
            visualizations=visualizations,
            insights=self._generate_overview_insights(analysis),
            actions=[
                {
                    'id': 'download_summary',
                    'label': 'Download Executive Summary',
                    'type': 'download',
                    'data': analysis
                }
            ],
            what_if_scenarios=[],
            metrics=kpi_metrics
        )
    
    def _create_kpi_dashboard(self, kpi_metrics: Dict) -> go.Figure:
        """Create KPI dashboard visualization"""
        fig = go.Figure()
        
        # Create indicator for each KPI
        positions = [(0.2, 0.7), (0.5, 0.7), (0.8, 0.7), (0.35, 0.2), (0.65, 0.2)]
        
        for i, (metric, data) in enumerate(list(kpi_metrics.items())[:5]):
            x, y = positions[i] if i < len(positions) else (0.5, 0.5)
            
            # Determine color based on change
            change_value = data.get('change', 'N/A')
            if isinstance(change_value, str) and '+' in change_value:
                delta_color = "green"
            elif isinstance(change_value, str) and '-' in change_value:
                delta_color = "red"
            else:
                delta_color = "gray"
            
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=self._extract_numeric_value(data['value']),
                number={'prefix': "$" if "$" in str(data['value']) else "", 
                       'suffix': "%" if "%" in str(data['value']) else "",
                       'font': {'size': 40}},
                delta={'reference': self._extract_numeric_value(data['value']) * 0.9,
                      'relative': True,
                      'font': {'size': 20},
                      'increasing': {'color': 'green'},
                      'decreasing': {'color': 'red'}},
                title={'text': data['label'], 'font': {'size': 20}},
                domain={'x': [x-0.15, x+0.15], 'y': [y-0.15, y+0.15]}
            ))
        
        fig.update_layout(
            height=500,
            paper_bgcolor='#1f2a44',
            font={'color': 'white'},
            showlegend=False
        )
        
        return fig
    
    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from formatted string"""
        if isinstance(value_str, (int, float)):
            return round(float(value_str), 2)  # Round to 2 decimals
        
        # Remove currency symbols and commas
        cleaned = str(value_str).replace('$', '').replace(',', '').replace('%', '')
        try:
            return round(float(cleaned), 2)  # Round to 2 decimals
        except:
            return 0.0
    
    def _create_performance_trends_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create performance trends chapter"""
        business_cols = analysis['business_cols']
        visualizations = []
        
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            
            # Trend chart for primary metric
            metric_col = business_cols['numerical'][0]
            df[time_col] = pd.to_datetime(df[time_col])
            
            # Monthly aggregation
            monthly_data = df.groupby(df[time_col].dt.to_period('M'))[metric_col].agg(['sum', 'mean', 'count']).reset_index()
            monthly_data[time_col] = monthly_data[time_col].dt.to_timestamp()
            
            fig_trend = go.Figure()
            
            # Add main trend line
            fig_trend.add_trace(go.Scatter(
                x=monthly_data[time_col],
                y=monthly_data['sum'],
                mode='lines+markers',
                name='Total',
                line=dict(color='#4299e1', width=3),
                marker=dict(size=8)
            ))
            
            # Add moving average
            if len(monthly_data) > 3:
                monthly_data['ma3'] = monthly_data['sum'].rolling(window=3).mean()
                fig_trend.add_trace(go.Scatter(
                    x=monthly_data[time_col],
                    y=monthly_data['ma3'],
                    mode='lines',
                    name='3-Month MA',
                    line=dict(color='#48bb78', width=2, dash='dash')
                ))
            
            fig_trend.update_layout(
                title=f"{metric_col} Trend Analysis",
                xaxis_title="Month",
                yaxis_title=metric_col,
                hovermode='x unified'
            )
            fig_trend = create_dark_chart(fig_trend)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_trend,
                'title': 'Monthly Performance Trend'
            })
            
            # YoY comparison if enough data
            if analysis['time_range'].get('duration', 0) > 365:
                fig_yoy = self._create_yoy_comparison(df, time_col, metric_col)
                visualizations.append({
                    'type': 'plotly',
                    'figure': fig_yoy,
                    'title': 'Year-over-Year Comparison'
                })
        
        # Generate trend insights
        trend_narrative = self._generate_trend_narrative(analysis)
        
        return StoryChapter(
            id="trends",
            title="Performance Trends",
            subtitle="How your sales are evolving over time",
            narrative=trend_narrative,
            visualizations=visualizations,
            insights=self._generate_trend_insights(analysis),
            actions=[
                {
                    'id': 'export_trends',
                    'label': 'Export Trend Data',
                    'type': 'export',
                    'data': 'trend_data'
                }
            ],
            what_if_scenarios=[
                {
                    'id': 'growth_projection',
                    'label': 'Project Future Growth',
                    'type': 'slider',
                    'params': {
                        'min': -20,
                        'max': 50,
                        'default': 10,
                        'step': 5,
                        'unit': '%'
                    }
                }
            ],
            metrics={}
        )
    
    def _create_yoy_comparison(self, df: pd.DataFrame, time_col: str, metric_col: str) -> go.Figure:
        """Create year-over-year comparison chart"""
        df['year'] = df[time_col].dt.year
        df['month'] = df[time_col].dt.month
        
        # Get last two years
        years = sorted(df['year'].unique())[-2:]
        
        fig = go.Figure()
        
        for year in years:
            year_data = df[df['year'] == year].groupby('month')[metric_col].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=year_data['month'],
                y=year_data[metric_col],
                mode='lines+markers',
                name=str(year),
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title="Year-over-Year Performance Comparison",
            xaxis_title="Month",
            yaxis_title=metric_col,
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        )
        
        return create_dark_chart(fig)
    
    def _create_segment_analysis_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create segment analysis chapter"""
        business_cols = analysis['business_cols']
        visualizations = []
        
        if business_cols['categorical'] and business_cols['numerical']:
            # Segment performance comparison
            cat_col = business_cols['categorical'][0]
            metric_col = business_cols['numerical'][0]
            
            segment_data = df.groupby(cat_col)[metric_col].agg(['sum', 'mean', 'count']).reset_index()
            segment_data = segment_data.sort_values('sum', ascending=False).head(15)
            
            # Create treemap
            fig_treemap = px.treemap(
                segment_data,
                path=[cat_col],
                values='sum',
                title=f"{metric_col} Distribution by {cat_col}"
            )
            fig_treemap = create_dark_chart(fig_treemap)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_treemap,
                'title': 'Segment Distribution'
            })
            
            # Performance matrix
            if len(business_cols['numerical']) > 1:
                metric2_col = business_cols['numerical'][1]
                
                matrix_data = df.groupby(cat_col).agg({
                    metric_col: 'sum',
                    metric2_col: 'sum'
                }).reset_index()
                
                fig_scatter = px.scatter(
                    matrix_data,
                    x=metric_col,
                    y=metric2_col,
                    size='size' if 'size' in matrix_data.columns else metric_col,
                    text=cat_col,
                    title=f"Performance Matrix: {metric_col} vs {metric2_col}"
                )
                
                # Add quadrant lines
                x_mean = matrix_data[metric_col].mean()
                y_mean = matrix_data[metric2_col].mean()
                
                fig_scatter.add_hline(y=y_mean, line_dash="dash", line_color="gray")
                fig_scatter.add_vline(x=x_mean, line_dash="dash", line_color="gray")
                
                fig_scatter = create_dark_chart(fig_scatter)
                
                visualizations.append({
                    'type': 'plotly',
                    'figure': fig_scatter,
                    'title': 'Performance Matrix'
                })
        
        return StoryChapter(
            id="segments",
            title="Segment Deep Dive",
            subtitle="Understanding performance across segments",
            narrative=self._generate_segment_narrative(analysis),
            visualizations=visualizations,
            insights=self._generate_segment_insights(analysis),
            actions=[
                {
                    'id': 'focus_segments',
                    'label': 'Identify Focus Segments',
                    'type': 'analysis',
                    'params': {'top_n': 5}
                }
            ],
            what_if_scenarios=[
                {
                    'id': 'segment_reallocation',
                    'label': 'Simulate Resource Reallocation',
                    'type': 'multi_slider',
                    'params': {
                        'segments': segment_data[cat_col].tolist()[:5] if 'segment_data' in locals() else []
                    }
                }
            ],
            metrics={}
        )
    
    def _create_opportunity_health_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create opportunity health analysis chapter"""
        visualizations = []
        
        # Be transparent about simulated metrics
        narrative_intro = """
        Note: This analysis creates derived health metrics based on your sales data patterns. 
        The health score is calculated from factors like sales consistency, profit margins, and performance variance.
        """
        
        business_cols = analysis['business_cols']
        
        if business_cols['categorical'] and business_cols['numerical']:
            # For Superstore data, we'll use actual metrics
            cat_col = 'Segment' if 'Segment' in df.columns else business_cols['categorical'][0]
            
            # Calculate health scores based on ACTUAL data
            health_data = df.groupby(cat_col).agg({
                'Sales': ['sum', 'mean', 'std'],
                'Profit': ['sum', 'mean'],
                'Quantity': 'sum'
            }).reset_index()
            
            # Flatten column names
            health_data.columns = [cat_col, 'total_sales', 'avg_sales', 'sales_std', 
                                  'total_profit', 'avg_profit', 'total_quantity']
            
            # Calculate profit margin
            health_data['profit_margin'] = (health_data['total_profit'] / health_data['total_sales'] * 100)
            
            # Calculate health score based on real metrics
            # High profit margin = healthy
            # Low sales variance = healthy
            # Consistent performance = healthy
            
            # Normalize metrics
            margin_score = health_data['profit_margin'] / health_data['profit_margin'].max()
            
            # Lower variance is better, so invert
            variance_score = 1 - (health_data['sales_std'] / health_data['avg_sales']).fillna(0).clip(0, 1)
            
            # Volume score
            volume_score = health_data['total_sales'] / health_data['total_sales'].max()
            
            # Composite health score
            health_data['health_score'] = (
                margin_score * 0.4 +      # Profitability is most important
                variance_score * 0.3 +    # Consistency matters
                volume_score * 0.3        # Volume matters
            ) * 100
            
            health_data['health_category'] = pd.cut(
                health_data['health_score'],
                bins=[0, 33, 66, 100],
                labels=['At Risk', 'Needs Attention', 'Healthy']
            )
            
            # Create visualizations with ACTUAL data context
            avg_health = health_data['health_score'].mean()
            
            # Update narrative to be transparent
            narrative = f"""
            {narrative_intro}
            
            Based on your actual sales data, we've calculated a composite health score of {avg_health:.1f}%.
            This score reflects:
            - Profit margins across {cat_col}s
            - Sales consistency (lower variance = better)
            - Revenue volume
            
            Segments with high profit margins and consistent sales patterns score higher.
            """
            
            # Create gauge chart for overall health
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_health,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Pipeline Health"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            fig_gauge = create_dark_chart(fig_gauge)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_gauge,
                'title': 'Pipeline Health Score'
            })
            
            # Health distribution
            fig_dist = px.histogram(
                health_data,
                x='health_category',
                title="Pipeline Health Distribution"
            )
            fig_dist = create_dark_chart(fig_dist)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_dist,
                'title': 'Health Category Distribution'
            })
        else:
            avg_health = 0
            health_data = pd.DataFrame()
            narrative = narrative_intro + "\n\nInsufficient data to calculate health scores."
        
        return StoryChapter(
            id="health",
            title="Opportunity Health Check",
            subtitle="Assessing pipeline quality and risks",
            narrative=narrative,
            visualizations=visualizations,
            insights=self._generate_health_insights(health_data),
            actions=[
                {
                    'id': 'alert_at_risk',
                    'label': 'Alert Teams on At-Risk Deals',
                    'type': 'notification',
                    'urgency': 'high'
                },
                {
                    'id': 'schedule_reviews',
                    'label': 'Schedule Deal Reviews',
                    'type': 'calendar'
                }
            ],
            what_if_scenarios=[],
            metrics={
                'avg_health': avg_health,
                'at_risk_count': len(health_data[health_data['health_category'] == 'At Risk']) if len(health_data) > 0 and 'health_category' in health_data.columns else 0
            }
        )
    
    def _create_forecast_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create forecast chapter"""
        visualizations = []
        business_cols = analysis['business_cols']
        future_values_list = []  # Use a list from the start
        
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            metric_col = business_cols['numerical'][0]
            
            # Prepare time series data
            df[time_col] = pd.to_datetime(df[time_col])
            ts_data = df.groupby(df[time_col].dt.to_period('M'))[metric_col].sum().reset_index()
            ts_data[time_col] = ts_data[time_col].dt.to_timestamp()
            
            # Simple forecast (using trend line)
            if len(ts_data) > 3:
                # Calculate trend
                x = np.arange(len(ts_data))
                y = ts_data[metric_col].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Project forward 3 months
                future_x = np.arange(len(ts_data), len(ts_data) + 3)
                future_dates = pd.date_range(
                    start=ts_data[time_col].max() + pd.DateOffset(months=1),
                    periods=3,
                    freq='M'
                )
                future_values = p(future_x)
                
                # Convert to list immediately
                future_values_list = future_values.tolist() if hasattr(future_values, 'tolist') else list(future_values)
                
                # Create forecast visualization
                fig_forecast = go.Figure()
                
                # Historical data
                fig_forecast.add_trace(go.Scatter(
                    x=ts_data[time_col],
                    y=ts_data[metric_col],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#4299e1', width=3)
                ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#48bb78', width=3, dash='dash')
                ))
                
                # Confidence bands (simplified)
                std_dev = ts_data[metric_col].std()
                upper_bound = future_values + std_dev
                lower_bound = future_values - std_dev
                
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(72, 187, 120, 0.2)',
                    showlegend=False
                ))
                
                fig_forecast.update_layout(
                    title=f"{metric_col} Forecast - Next 3 Months",
                    xaxis_title="Month",
                    yaxis_title=metric_col
                )
                fig_forecast = create_dark_chart(fig_forecast)
                
                visualizations.append({
                    'type': 'plotly',
                    'figure': fig_forecast,
                    'title': 'Sales Forecast'
                })
        
        return StoryChapter(
            id="forecast",
            title="Predictive Insights",
            subtitle="What the future holds",
            narrative=self._generate_forecast_narrative(future_values_list),
            visualizations=visualizations,
            insights=self._generate_forecast_insights(analysis),
            actions=[
                {
                    'id': 'adjust_targets',
                    'label': 'Adjust Sales Targets',
                    'type': 'planning'
                }
            ],
            what_if_scenarios=[
                {
                    'id': 'scenario_planning',
                    'label': 'Scenario Planning',
                    'type': 'multi_scenario',
                    'scenarios': ['Conservative', 'Realistic', 'Optimistic']
                }
            ],
            metrics={
                'forecast_total': sum(future_values_list) if future_values_list else 0
            }
        )
    
    def _create_action_plan_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create action plan chapter"""
        # Compile all actions from previous chapters
        priority_actions = [
            {
                'priority': 1,
                'action': 'Focus on underperforming segments',
                'impact': 'High',
                'effort': 'Medium',
                'timeline': '2 weeks'
            },
            {
                'priority': 2,
                'action': 'Implement deal health monitoring',
                'impact': 'High',
                'effort': 'Low',
                'timeline': '1 week'
            },
            {
                'priority': 3,
                'action': 'Optimize resource allocation',
                'impact': 'Medium',
                'effort': 'High',
                'timeline': '1 month'
            }
        ]
        
        # Create action plan visualization
        action_df = pd.DataFrame(priority_actions)
        
        fig_actions = go.Figure(data=[
            go.Table(
                header=dict(
                    values=list(action_df.columns),
                    fill_color='#2d3748',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=[action_df[col] for col in action_df.columns],
                    fill_color=[['#1a202c' if i % 2 == 0 else '#2d3748' for i in range(len(action_df))]],
                    font=dict(color='white', size=12),
                    align='left',
                    height=30
                )
            )
        ])
        
        fig_actions.update_layout(
            title="Priority Action Plan",
            height=400
        )
        
        return StoryChapter(
            id="actions",
            title="Your Action Plan",
            subtitle="Turning insights into results",
            narrative="Based on our comprehensive analysis, here's your prioritized action plan to drive sales performance.",
            visualizations=[{
                'type': 'plotly',
                'figure': fig_actions,
                'title': 'Priority Actions'
            }],
            insights=[
                "Execute high-impact, low-effort actions first",
                "Monitor progress weekly",
                "Adjust strategies based on early results"
            ],
            actions=[
                {
                    'id': 'export_plan',
                    'label': 'Export Action Plan',
                    'type': 'export'
                },
                {
                    'id': 'create_tasks',
                    'label': 'Create Tasks in CRM',
                    'type': 'integration'
                }
            ],
            what_if_scenarios=[],
            metrics={}
        )
    
    # Helper methods for narratives and insights
    def _calculate_period_change(self, df: pd.DataFrame, metric_col: str, date_col: str) -> str:
        """Calculate period-over-period change for a metric"""
        try:
            if date_col not in df.columns or metric_col not in df.columns:
                return "N/A"
            
            # Convert to datetime
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Get the date range
            max_date = df[date_col].max()
            min_date = df[date_col].min()
            date_range = (max_date - min_date).days
            
            if date_range < 60:  # Less than 2 months of data
                return "N/A"
            
            # Calculate monthly aggregates
            monthly = df.groupby(df[date_col].dt.to_period('M'))[metric_col].sum()
            
            if len(monthly) < 2:
                return "N/A"
            
            # Compare last month to previous month
            last_month = monthly.iloc[-1]
            prev_month = monthly.iloc[-2]
            
            if prev_month == 0:
                return "N/A"
            
            change = ((last_month - prev_month) / prev_month) * 100
            return f"{change:+.1f}%"
            
        except Exception as e:
            print(f"Error calculating period change: {e}")
            return "N/A"

    def _generate_overview_insights(self, analysis: Dict) -> List[str]:
        """Generate overview insights"""
        insights = []
        
        # Extract insights from the analysis
        if analysis.get('key_metrics'):
            # Get the first few metrics
            for i, (metric, data) in enumerate(list(analysis['key_metrics'].items())[:3]):
                if 'total' in data:
                    insights.append(f"Total {metric}: {data['total']}")
                if 'growth' in data and data['growth'] != 'N/A':
                    insights.append(f"{metric} growth: {data['growth']}")
        
        # Add insights about data quality
        if analysis.get('shape'):
            insights.append(f"Analyzing {analysis['shape'][0]:,} transactions")
        
        # Add insights about segments
        if analysis.get('segments'):
            num_segments = len(analysis['segments'])
            if num_segments > 0:
                insights.append(f"Performance tracked across {num_segments} key dimensions")
        
        # Add anomaly insights
        if analysis.get('anomalies'):
            num_anomalies = len(analysis['anomalies'])
            if num_anomalies > 0:
                insights.append(f"Detected {num_anomalies} unusual patterns requiring attention")
        
        return insights[:5]  # Return top 5 insights

    def _generate_trend_narrative(self, analysis: Dict) -> str:
        """Generate trend narrative"""
        trend_insights = []
        
        # Check if we have trend data
        if analysis.get('trends'):
            for metric, trend_data in analysis['trends'].items():
                direction = trend_data.get('direction', 'stable')
                volatility = trend_data.get('volatility', 0)
                recent_change = trend_data.get('recent_change', 0)
                
                # Describe the trend
                if direction == 'increasing':
                    trend_desc = f"{metric} shows an upward trend"
                elif direction == 'decreasing':
                    trend_desc = f"{metric} shows a downward trend"
                else:
                    trend_desc = f"{metric} remains stable"
                
                # Add volatility context
                if volatility > 0.3:
                    trend_desc += " with high volatility"
                elif volatility > 0.1:
                    trend_desc += " with moderate fluctuations"
                else:
                    trend_desc += " with consistent performance"
                
                trend_insights.append(trend_desc)
                
                # Add recent change if significant
                if abs(recent_change) > 10:
                    change_desc = f"Recent period shows {abs(recent_change):.1f}% "
                    change_desc += "increase" if recent_change > 0 else "decrease"
                    trend_insights.append(change_desc)
        
        # Build narrative
        if trend_insights:
            narrative = f"""
            Your sales data reveals important patterns over time. {'. '.join(trend_insights[:2])}.
            
            Understanding these trends helps predict future performance and identify when to take action.
            Historical patterns suggest opportunities for optimization and growth.
            """
        else:
            narrative = """
            Trend analysis requires temporal data points. Your dataset provides the foundation
            for understanding performance patterns and seasonal variations.
            """
        
        return narrative

    def _generate_trend_insights(self, analysis: Dict) -> List[str]:
        """Generate specific trend insights"""
        insights = []
        
        # Extract trend insights
        if analysis.get('trends'):
            for metric, trend in analysis['trends'].items():
                # Direction insights
                direction = trend.get('direction', 'stable')
                if direction != 'stable':
                    insights.append(f"{metric} is {direction} over time")
                
                # Volatility insights
                volatility = trend.get('volatility', 0)
                if volatility > 0.5:
                    insights.append(f"High volatility in {metric} suggests inconsistent performance")
                elif volatility < 0.1:
                    insights.append(f"{metric} shows stable, predictable patterns")
                
                # Recent change insights
                recent_change = trend.get('recent_change', 0)
                if recent_change > 20:
                    insights.append(f"{metric} surged {recent_change:.1f}% in recent period")
                elif recent_change < -20:
                    insights.append(f"{metric} declined {abs(recent_change):.1f}% recently - investigate causes")
        
        # Time-based insights
        if analysis.get('time_range'):
            duration = analysis['time_range'].get('duration', 0)
            if duration > 365:
                insights.append(f"Multi-year data enables reliable trend analysis")
            elif duration > 90:
                insights.append(f"Quarterly patterns emerging from {duration} days of data")
        
        # Add generic insights if none found
        if not insights:
            insights = [
                "Monitor trends monthly for early warning signals",
                "Seasonal patterns may impact performance",
                "Historical data provides baseline for forecasting"
            ]
        
        return insights[:5]

    def _generate_segment_narrative(self, analysis: Dict) -> str:
        """Generate segment analysis narrative"""
        segment_insights = []
        
        # Check segment variance
        if analysis.get('segments'):
            high_variance_segments = []
            for segment_key, segment_data in analysis['segments'].items():
                if segment_data.get('variance', 0) > 0.5:
                    # Extract segment name from key (format: "Category_Sales")
                    segment_name = segment_key.split('_')[0]
                    high_variance_segments.append(segment_name)
            
            if high_variance_segments:
                segment_insights.append(
                    f"High performance variance detected in {', '.join(high_variance_segments[:2])} segments"
                )
        
        narrative = f"""
        Segment analysis reveals performance differences across your business dimensions.
        {'. '.join(segment_insights) if segment_insights else 'Each segment presents unique opportunities and challenges.'}
        
        Top performers demonstrate best practices that can be replicated, while underperformers 
        represent immediate improvement opportunities. Focus on standardizing success factors
        across all segments to maximize overall performance.
        """
        
        return narrative

    def _generate_segment_insights(self, analysis: Dict) -> List[str]:
        """Generate segment-specific insights"""
        insights = []
        
        if analysis.get('segments'):
            for segment_key, segment_data in analysis['segments'].items():
                # High variance insights
                variance = segment_data.get('variance', 0)
                if variance > 0.5:
                    segment_name = segment_key.replace('_', ' ')
                    insights.append(f"High variance in {segment_name} indicates optimization potential")
                
                # Top performer insights
                if 'top_3' in segment_data:
                    insights.append(f"Top performers identified in {segment_key.split('_')[0]}")
        
        # Add business impact insights
        if analysis.get('business_opportunities'):
            for opp in analysis['business_opportunities'][:2]:
                if opp['type'] == 'optimization':
                    insights.append(opp['description'])
        
        return insights[:5]

    def _generate_health_narrative(self, avg_health: float) -> str:
        """Generate health check narrative"""
        # Determine health status
        if avg_health >= 70:
            status = "strong"
            action = "Maintain current strategies while exploring growth opportunities"
        elif avg_health >= 50:
            status = "moderate"
            action = "Focus on improving underperforming areas"
        else:
            status = "concerning"
            action = "Immediate intervention required for at-risk segments"
        
        narrative = f"""
        Pipeline health analysis shows an overall score of {avg_health:.1f}%, indicating {status} performance.
        
        This score is calculated from:
        - Profit margins (40% weight) - higher margins indicate healthier operations
        - Performance consistency (30% weight) - lower variance suggests stability  
        - Revenue volume (30% weight) - scale matters for business impact
        
        {action}. Healthy segments can serve as models for improvement initiatives.
        """
        
        return narrative

    def _generate_health_insights(self, health_data: pd.DataFrame) -> List[str]:
        """Generate health check insights"""
        insights = []
        
        if len(health_data) > 0:
            # At-risk segments
            if 'health_category' in health_data.columns:
                at_risk = health_data[health_data['health_category'] == 'At Risk']
                if len(at_risk) > 0:
                    insights.append(f"{len(at_risk)} segments at risk - immediate attention required")
                    
                    # Identify specific at-risk segments
                    if health_data.columns[0]:  # First column is usually the category
                        worst = at_risk.nsmallest(1, 'health_score')
                        if len(worst) > 0:
                            insights.append(f"{worst.iloc[0][health_data.columns[0]]} has lowest health score")
            
            # Healthy segments
            if 'health_category' in health_data.columns:
                healthy = health_data[health_data['health_category'] == 'Healthy']
                if len(healthy) > 0:
                    insights.append(f"{len(healthy)} segments performing well - replicate their strategies")
            
            # Profit margin insights
            if 'profit_margin' in health_data.columns:
                avg_margin = health_data['profit_margin'].mean()
                insights.append(f"Average profit margin: {avg_margin:.1f}%")
                
                # Find best and worst margins
                if len(health_data) > 1:
                    best_margin = health_data.nlargest(1, 'profit_margin')
                    worst_margin = health_data.nsmallest(1, 'profit_margin')
                    if len(best_margin) > 0 and len(worst_margin) > 0:
                        margin_gap = best_margin.iloc[0]['profit_margin'] - worst_margin.iloc[0]['profit_margin']
                        insights.append(f"Profit margin gap: {margin_gap:.1f}% between best and worst")
        
        return insights[:5]

    def _generate_forecast_narrative(self, future_values: List[float]) -> str:
        """Generate forecast narrative"""
        if len(future_values) > 0:
            total_forecast = sum(future_values)
            avg_forecast = sum(future_values) / len(future_values)
            
            # Calculate growth from last historical point
            growth_rate = ((future_values[-1] - future_values[0]) / future_values[0] * 100) if future_values[0] > 0 else 0
            
            narrative = f"""
            Based on historical patterns and trend analysis, the forecast for the next quarter shows:
            
            - Projected total: {format_value(total_forecast, 'currency')}
            - Average monthly projection: {format_value(avg_forecast, 'currency')}
            - Expected growth rate: {growth_rate:+.1f}%
            
            This forecast assumes current market conditions continue and no major disruptions occur.
            Monitor actual performance against these projections to identify early variances.
            """
        else:
            narrative = """
            Forecast generation requires sufficient historical data with clear temporal patterns.
            As you accumulate more time-series data, predictive accuracy will improve.
            
            Focus on collecting consistent data points to enable reliable forecasting.
            """
        
        return narrative

    def _generate_forecast_insights(self, analysis: Dict) -> List[str]:
        """Generate forecast insights"""
        insights = [
            "Forecasts are based on historical trends and patterns",
            "Monitor leading indicators for early warning signals",
            "Regular forecast updates improve accuracy",
            "Consider seasonal factors in planning",
            "Build scenarios for different growth assumptions"
        ]
        
        # Add specific insights if available
        if analysis.get('trends'):
            trending_up = [m for m, t in analysis['trends'].items() if t.get('direction') == 'increasing']
            if trending_up:
                insights.insert(0, f"Positive trends in {', '.join(trending_up[:2])} support growth forecast")
        
        return insights[:5]

    def _generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary"""
        # Extract key facts
        total_records = analysis.get('shape', [0])[0]
        
        # Key metrics summary
        metrics_summary = []
        if analysis.get('key_metrics'):
            for metric, data in list(analysis['key_metrics'].items())[:2]:
                if 'total' in data:
                    metrics_summary.append(f"{metric}: {data['total']}")
        
        # Opportunities summary
        opportunities = len(analysis.get('business_opportunities', []))
        anomalies = len(analysis.get('anomalies', []))
        
        summary = f"""
        This comprehensive sales analysis examines {total_records:,} transactions to provide actionable insights
        for business growth. {' | '.join(metrics_summary) if metrics_summary else 'Key performance metrics analyzed.'} 
        
        The analysis identifies {opportunities} optimization opportunities and {anomalies} areas requiring attention.
        Immediate focus should be on high-impact, low-effort improvements to drive performance.
        
        Data-driven recommendations prioritize actions with the greatest potential for positive business impact.
        """
        
        return summary

    def _extract_key_takeaways(self, analysis: Dict) -> List[str]:
        """Extract key takeaways from analysis"""
        takeaways = []
        
        # Performance takeaways
        if analysis.get('key_metrics'):
            first_metric = list(analysis['key_metrics'].keys())[0]
            metric_data = analysis['key_metrics'][first_metric]
            takeaways.append(f"{first_metric} performance: {metric_data.get('total', 'N/A')}")
            
            if metric_data.get('growth', 'N/A') != 'N/A':
                takeaways.append(f"Growth trend: {metric_data['growth']}")
        
        # Segment takeaways
        if analysis.get('segments'):
            num_segments = len([s for s in analysis['segments'] if analysis['segments'][s]])
            takeaways.append(f"Analysis covers {num_segments} business dimensions")
        
        # Opportunity takeaways
        if analysis.get('business_opportunities'):
            high_impact = [o for o in analysis['business_opportunities'] if o.get('type') == 'optimization']
            if high_impact:
                takeaways.append(f"{len(high_impact)} high-impact optimization opportunities identified")
        
        # Health takeaways
        if analysis.get('anomalies'):
            takeaways.append(f"{len(analysis['anomalies'])} anomalies require investigation")
        
        return takeaways[:5]

    def _compile_recommended_actions(self, chapters: List[StoryChapter]) -> List[Dict[str, Any]]:
        """Compile recommended actions from all chapters"""
        all_actions = []
        
        # Extract high-priority actions from each chapter
        for chapter in chapters:
            for action in chapter.actions:
                # Prioritize certain action types
                if action.get('type') in ['notification', 'planning', 'optimization', 'analysis']:
                    action['priority_score'] = 1.0 if action.get('urgency') == 'high' else 0.5
                    all_actions.append(action)
        
        # Sort by priority and return top 5
        sorted_actions = sorted(all_actions, key=lambda x: x.get('priority_score', 0), reverse=True)
        return sorted_actions[:5]

# Marketing ROI Story
class MarketingROIStory(StoryComponent):
    """Generate marketing ROI narrative"""
    
    def __init__(self, conversational_agent):
        super().__init__(conversational_agent)
        self.story_type = StoryType.MARKETING_ROI
    
    def generate(self, dataframe: pd.DataFrame, context: Dict[str, Any]) -> DataStory:
        """Generate complete marketing ROI story"""
        
        # Analyze data
        analysis = self.analyze_data(dataframe, context)
        
        # Create chapters
        chapters = [
            self._create_campaign_overview_chapter(dataframe, analysis),
            self._create_channel_performance_chapter(dataframe, analysis),
            self._create_attribution_analysis_chapter(dataframe, analysis),
            self._create_optimization_chapter(dataframe, analysis)
        ]
        
        return DataStory(
            id=f"marketing_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="Marketing ROI Analysis",
            story_type=self.story_type,
            chapters=chapters,
            executive_summary=self._generate_executive_summary(analysis),
            key_takeaways=self._extract_key_takeaways(analysis),
            recommended_actions=self._compile_recommended_actions(chapters),
            metadata={
                'generated_at': datetime.now().isoformat(),
                'data_points': len(dataframe)
            }
        )
    
    def _create_campaign_overview_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create campaign overview chapter"""
        business_cols = analysis['business_cols']
        visualizations = []
        
        # Calculate marketing KPIs
        kpi_metrics = {}
        if business_cols['numerical']:
            # Simulate marketing metrics
            spend_col = None
            revenue_col = None
            
            for col in business_cols['numerical']:
                if 'spend' in col.lower() or 'cost' in col.lower():
                    spend_col = col
                elif 'revenue' in col.lower() or 'sales' in col.lower():
                    revenue_col = col
            
            if spend_col and revenue_col:
                total_spend = df[spend_col].sum()
                total_revenue = df[revenue_col].sum()
                roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
                
                kpi_metrics = {
                    'Total Spend': format_value(total_spend, 'currency'),
                    'Total Revenue': format_value(total_revenue, 'currency'),
                    'ROI': f"{roi:.1f}%",
                    'ROAS': f"{(total_revenue/total_spend):.2f}x" if total_spend > 0 else "N/A"
                }
                # Create KPI visualization
        if kpi_metrics:
            fig_kpi = self._create_marketing_kpi_dashboard(kpi_metrics)
            visualizations.append({
                'type': 'plotly',
                'figure': fig_kpi,
                'title': 'Marketing KPIs'
            })
        
        return StoryChapter(
            id="campaign_overview",
            title="Campaign Performance Overview",
            subtitle="Your marketing investment returns",
            narrative=self._generate_campaign_narrative(kpi_metrics),
            visualizations=visualizations,
            insights=self._generate_campaign_insights(analysis, kpi_metrics),
            actions=[
                {
                    'id': 'optimize_spend',
                    'label': 'Optimize Budget Allocation',
                    'type': 'optimization'
                }
            ],
            what_if_scenarios=[
                {
                    'id': 'budget_simulator',
                    'label': 'Budget Reallocation Simulator',
                    'type': 'budget_allocation'
                }
            ],
            metrics=kpi_metrics
        )
    
    def _create_marketing_kpi_dashboard(self, kpis: Dict) -> go.Figure:
        """Create marketing KPI dashboard"""
        fig = go.Figure()
        
        # Create cards layout
        positions = [(0.25, 0.7), (0.75, 0.7), (0.25, 0.3), (0.75, 0.3)]
        colors = ['#4299e1', '#48bb78', '#ed8936', '#9f7aea']
        
        for i, (metric, value) in enumerate(list(kpis.items())[:4]):
            x, y = positions[i]
            color = colors[i]
            
            fig.add_annotation(
                x=x, y=y,
                text=f"<b>{metric}</b><br><span style='font-size:24px'>{value}</span>",
                showarrow=False,
                font=dict(size=16, color='white'),
                bordercolor=color,
                borderwidth=2,
                borderpad=20,
                bgcolor=color,
                opacity=0.8
            )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='#1f2a44',
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1])
        )
        
        return fig
    
    def _create_channel_performance_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create channel performance chapter"""
        visualizations = []
        business_cols = analysis['business_cols']
        
        # Look for channel column
        channel_col = None
        for col in business_cols['categorical']:
            if 'channel' in col.lower() or 'source' in col.lower():
                channel_col = col
                break
        
        if not channel_col and business_cols['categorical']:
            channel_col = business_cols['categorical'][0]
        
        if channel_col and business_cols['numerical']:
            metric_col = business_cols['numerical'][0]
            
            # Channel performance comparison
            channel_data = df.groupby(channel_col)[metric_col].agg(['sum', 'mean', 'count']).reset_index()
            channel_data = channel_data.sort_values('sum', ascending=False)
            
            # Create channel comparison
            fig_channels = px.bar(
                channel_data,
                x=channel_col,
                y='sum',
                title=f"Performance by {channel_col}",
                color='sum',
                color_continuous_scale='viridis'
            )
            fig_channels = create_dark_chart(fig_channels)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_channels,
                'title': 'Channel Performance'
            })
            
            # Efficiency matrix
            if len(business_cols['numerical']) > 1:
                metric2_col = business_cols['numerical'][1]
                
                efficiency_data = df.groupby(channel_col).agg({
                    metric_col: 'sum',
                    metric2_col: 'mean'
                }).reset_index()
                
                fig_efficiency = px.scatter(
                    efficiency_data,
                    x=metric2_col,
                    y=metric_col,
                    size=metric_col,
                    text=channel_col,
                    title="Channel Efficiency Matrix"
                )
                fig_efficiency = create_dark_chart(fig_efficiency)
                
                visualizations.append({
                    'type': 'plotly',
                    'figure': fig_efficiency,
                    'title': 'Efficiency Analysis'
                })
        
        return StoryChapter(
            id="channels",
            title="Channel Performance Analysis",
            subtitle="Where your marketing dollars work hardest",
            narrative=self._generate_channel_narrative(channel_data if 'channel_data' in locals() else None),
            visualizations=visualizations,
            insights=self._generate_channel_insights(channel_data if 'channel_data' in locals() else None),
            actions=[
                {
                    'id': 'reallocate_budget',
                    'label': 'Reallocate to Top Channels',
                    'type': 'budget_action'
                }
            ],
            what_if_scenarios=[],
            metrics={}
        )
    
    def _create_attribution_analysis_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create attribution analysis chapter"""
        # Simplified attribution visualization
        attribution_models = ['First Touch', 'Last Touch', 'Linear', 'Time Decay']
        attribution_values = [30, 45, 35, 40]  # Simulated values
        
        fig_attribution = go.Figure(data=[
            go.Bar(
                x=attribution_models,
                y=attribution_values,
                marker_color=['#4299e1', '#48bb78', '#ed8936', '#9f7aea']
            )
        ])
        
        fig_attribution.update_layout(
            title="Attribution Model Comparison",
            xaxis_title="Attribution Model",
            yaxis_title="Attributed Revenue %"
        )
        fig_attribution = create_dark_chart(fig_attribution)
        
        return StoryChapter(
            id="attribution",
            title="Attribution Analysis",
            subtitle="Understanding the customer journey",
            narrative="Different attribution models tell different stories about your marketing effectiveness.",
            visualizations=[{
                'type': 'plotly',
                'figure': fig_attribution,
                'title': 'Attribution Models'
            }],
            insights=[
                "Last touch attribution shows highest impact",
                "Consider multi-touch attribution for complex journeys"
            ],
            actions=[],
            what_if_scenarios=[],
            metrics={}
        )
    
    def _create_optimization_chapter(self, df: pd.DataFrame, analysis: Dict) -> StoryChapter:
        """Create optimization recommendations chapter"""
        # Create optimization table
        optimizations = pd.DataFrame({
            'Opportunity': ['Shift budget to top channel', 'Pause underperforming campaigns', 'Increase spend on high ROI segments'],
            'Impact': ['$125K additional revenue', '$45K cost savings', '$80K profit increase'],
            'Effort': ['Low', 'Low', 'Medium'],
            'Timeline': ['1 week', 'Immediate', '2 weeks']
        })
        
        fig_opt = go.Figure(data=[
            go.Table(
                header=dict(
                    values=list(optimizations.columns),
                    fill_color='#2d3748',
                    font=dict(color='white', size=14)
                ),
                cells=dict(
                    values=[optimizations[col] for col in optimizations.columns],
                    fill_color='#1a202c',
                    font=dict(color='white', size=12),
                    height=30
                )
            )
        ])
        
        fig_opt.update_layout(title="Optimization Opportunities", height=300)
        
        return StoryChapter(
            id="optimization",
            title="Optimization Roadmap",
            subtitle="Your path to marketing excellence",
            narrative="Based on the analysis, here are your top optimization opportunities.",
            visualizations=[{
                'type': 'plotly',
                'figure': fig_opt,
                'title': 'Optimization Plan'
            }],
            insights=[
                "Quick wins available through budget reallocation",
                "Focus on efficiency before scaling spend"
            ],
            actions=[
                {
                    'id': 'implement_optimizations',
                    'label': 'Implement Optimizations',
                    'type': 'execution'
                }
            ],
            what_if_scenarios=[],
            metrics={}
        )
    
    # Helper methods
    def _generate_campaign_narrative(self, kpis: Dict) -> str:
        roi = kpis.get('ROI', 'N/A')
        return f"""
        Your marketing campaigns are generating an ROI of {roi}. This comprehensive analysis 
        examines performance across channels, attribution models, and optimization opportunities.
        """
    
    def _generate_campaign_insights(self, analysis: Dict, kpis: Dict) -> List[str]:
        insights = []
        if 'ROI' in kpis:
            insights.append(f"Overall marketing ROI: {kpis['ROI']}")
        if 'ROAS' in kpis:
            insights.append(f"Return on ad spend: {kpis['ROAS']}")
        return insights
    
    def _generate_channel_narrative(self, channel_data: pd.DataFrame) -> str:
        if channel_data is not None and len(channel_data) > 0:
            top_channel = channel_data.iloc[0].name if hasattr(channel_data.iloc[0], 'name') else 'Top channel'
            return f"""
            {top_channel} is your top performing channel. Channel mix optimization can significantly 
            improve overall marketing efficiency.
            """
        return "Analyze channel performance to optimize marketing spend allocation."
    
    def _generate_channel_insights(self, channel_data: pd.DataFrame) -> List[str]:
        insights = []
        if channel_data is not None and len(channel_data) > 0:
            insights.append(f"Top channel drives {(channel_data.iloc[0]['sum'] / channel_data['sum'].sum() * 100):.1f}% of results")
        return insights
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        return "Marketing ROI analysis reveals opportunities for budget optimization and channel reallocation."
    
    def _extract_key_takeaways(self, analysis: Dict) -> List[str]:
        return ["Focus on high-ROI channels", "Implement attribution modeling", "Optimize budget allocation"]
    
    def _compile_recommended_actions(self, chapters: List[StoryChapter]) -> List[Dict]:
        """Compile all recommended actions"""
        all_actions = []
        
        for chapter in chapters:
            for action in chapter.actions:
                if action.get('urgency') == 'high' or action.get('type') in ['notification', 'planning']:
                    all_actions.append(action)
        
        return all_actions[:5]  # Top 5 actions

# Story Engine
class StoryEngine:
    """Main engine for story generation and management"""
    
    def __init__(self, conversational_agent):
        self.agent = conversational_agent
        self.story_components = {
            StoryType.SALES_PERFORMANCE: SalesPerformanceStory(conversational_agent),
            StoryType.MARKETING_ROI: MarketingROIStory(conversational_agent),
            # Add other story types as implemented
        }
        self.current_story = None
        self.story_history = []
    
    def analyze_data_for_story_fit(self, dataframe: pd.DataFrame) -> Dict[StoryType, float]:
        """Analyze data to determine which story types are relevant"""
        business_cols = get_business_relevant_columns(dataframe)
        
        # Get all column names in lowercase for matching
        all_columns = [col.lower().replace(' ', '_').replace('-', '_') for col in dataframe.columns]
        all_columns_text = ' '.join(all_columns)
        
        story_fitness = {}
        
        # Sales Performance fitness
        sales_score = 0
        
        # Your Superstore data has: Sales, Profit, Quantity, Order Date, etc.
        # These are classic sales performance indicators
        
        # Check for sales-related columns
        sales_indicators = ['sales', 'revenue', 'profit', 'order', 'quantity', 'amount', 'price', 
                           'discount', 'margin', 'customer', 'product']
        
        for col in all_columns:
            for indicator in sales_indicators:
                if indicator in col:
                    sales_score += 0.2
                    break
        
        # Specific checks for your dataset
        actual_columns = dataframe.columns.tolist()
        
        # Check for key sales columns
        if 'Sales' in actual_columns:
            sales_score += 0.3
        if 'Profit' in actual_columns:
            sales_score += 0.2
        if 'Quantity' in actual_columns:
            sales_score += 0.1
        if any('Order' in col for col in actual_columns):
            sales_score += 0.1
        
        # Has temporal data (Order Date, Ship Date)
        if any('Date' in col for col in actual_columns):
            sales_score += 0.1
        
        # Has segmentation data (Region, State, City, Segment, Category)
        segmentation_cols = ['Region', 'State', 'City', 'Segment', 'Category', 'Sub-Category']
        if any(col in actual_columns for col in segmentation_cols):
            sales_score += 0.2
        
        story_fitness[StoryType.SALES_PERFORMANCE] = min(sales_score, 1.0)
        
        # Marketing ROI fitness - this dataset doesn't have marketing data
        marketing_score = 0
        marketing_indicators = ['campaign', 'marketing', 'channel', 'source', 'medium', 'click', 
                               'impression', 'conversion', 'ctr', 'cpc', 'roi', 'roas', 'spend', 
                               'budget', 'ad', 'email']
        
        for col in all_columns:
            for indicator in marketing_indicators:
                if indicator in col:
                    marketing_score += 0.2
                    break
        
        story_fitness[StoryType.MARKETING_ROI] = marketing_score
        
        # Customer Journey - has some customer data
        journey_score = 0
        if 'Customer ID' in actual_columns or 'Customer Name' in actual_columns:
            journey_score += 0.3
        if any('Order' in col for col in actual_columns):
            journey_score += 0.2
        
        story_fitness[StoryType.CUSTOMER_JOURNEY] = journey_score
        
        # Executive Dashboard - perfect for this data
        exec_score = 0
        if 'Sales' in actual_columns and 'Profit' in actual_columns:
            exec_score += 0.5
        if any(col in actual_columns for col in segmentation_cols):
            exec_score += 0.3
        if any('Date' in col for col in actual_columns):
            exec_score += 0.2
        
        story_fitness[StoryType.EXECUTIVE_DASHBOARD] = min(exec_score, 1.0)
        
        # Debug output
        print(f"Actual columns: {actual_columns[:10]}...")
        print(f"Story Fitness Scores: {story_fitness}")
        
        return story_fitness
    
    def detect_story_type_with_rag(self, prompt: str, dataframe: pd.DataFrame) -> Tuple[StoryType, float]:
        """Enhanced story type detection using RAG and data analysis"""
        
        # Get data fitness scores
        story_fitness = self.analyze_data_for_story_fit(dataframe)
        
        # Analyze prompt
        prompt_lower = prompt.lower()
        prompt_scores = {}
        
        # Sales keywords in prompt
        sales_keywords = ['sales', 'revenue', 'pipeline', 'opportunity', 'deal', 'quota', 'forecast', 
                         'performance', 'target', 'achievement']
        sales_prompt_score = sum(1 for keyword in sales_keywords if keyword in prompt_lower) / len(sales_keywords)
        prompt_scores[StoryType.SALES_PERFORMANCE] = sales_prompt_score
        
        # Marketing keywords in prompt
        marketing_keywords = ['marketing', 'campaign', 'roi', 'roas', 'channel', 'attribution', 
                            'conversion', 'advertising', 'spend', 'budget']
        marketing_prompt_score = sum(1 for keyword in marketing_keywords if keyword in prompt_lower) / len(marketing_keywords)
        prompt_scores[StoryType.MARKETING_ROI] = marketing_prompt_score
        
        # Use RAG if available
        rag_scores = {}
        if self.agent and hasattr(self.agent, 'rag_system') and self.agent.rag_system:
            try:
                # Create context for RAG
                data_context = {
                    'columns': list(dataframe.columns),
                    'sample_data': dataframe.head(5).to_dict('records'),
                    'data_types': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                    'user_query': prompt
                }
                
                # Get RAG enhancement
                rag_enhancement = self.agent.rag_system.enhance_query(
                    f"Determine the best story type for: {prompt}",
                    data_context,
                    ""
                )
                
                domain_info = rag_enhancement.get('domain_analysis', {})
                if domain_info.get('primary_domain'):
                    domain = domain_info['primary_domain'].lower()
                    confidence = domain_info.get('confidence', 0.5)
                    
                    if 'sales' in domain or 'revenue' in domain:
                        rag_scores[StoryType.SALES_PERFORMANCE] = confidence
                    elif 'marketing' in domain or 'advertising' in domain:
                        rag_scores[StoryType.MARKETING_ROI] = confidence
                    
            except Exception as e:
                print(f"RAG story detection failed: {e}")
        
        # Combine all scores
        final_scores = {}
        for story_type in StoryType:
            if story_type in [StoryType.SALES_PERFORMANCE, StoryType.MARKETING_ROI]:
                data_score = story_fitness.get(story_type, 0)
                prompt_score = prompt_scores.get(story_type, 0)
                rag_score = rag_scores.get(story_type, 0)
                
                # Weighted combination
                final_score = (
                    data_score * 0.5 +      # Data relevance is most important
                    prompt_score * 0.3 +    # User intent is important
                    rag_score * 0.2         # RAG provides additional context
                )
                final_scores[story_type] = final_score
        
        # Select best story type
        if final_scores:
            best_story_type = max(final_scores, key=final_scores.get)
            best_score = final_scores[best_story_type]
            
            # Only use the story if confidence is high enough
            if best_score >= 0.3:  # Threshold for story relevance
                return best_story_type, best_score
        
        # Default to sales performance if no clear match
        return StoryType.SALES_PERFORMANCE, 0.5
    
    def generate_story(self, prompt: str, dataframe: pd.DataFrame, context: Dict[str, Any] = None) -> DataStory:
        """Generate a complete data story with intelligent selection"""
        
        # Detect story type with confidence
        story_type, confidence = self.detect_story_type_with_rag(prompt, dataframe)
        
        print(f"📖 Selected story type: {story_type.value} (confidence: {confidence:.1%})")
        
        # Warn if low confidence
        if confidence < 0.3:
            st.warning(f"⚠️ The selected {story_type.value} story may not be ideal for your data. "
                      f"Consider providing more context or selecting a different story type.")
        
        # Get appropriate story component
        if story_type not in self.story_components:
            # Fallback to sales performance
            story_type = StoryType.SALES_PERFORMANCE
        
        story_component = self.story_components[story_type]
        
        # Generate story with confidence info
        story_context = context or {}
        story_context.update({
            'user_prompt': prompt,
            'business_context': st.session_state.get('business_context', ''),
            'story_confidence': confidence,
            'data_fitness': self.analyze_data_for_story_fit(dataframe)
        })
        
        story = story_component.generate(dataframe, story_context)
        
        # Add confidence to metadata
        story.metadata['confidence'] = confidence
        story.metadata['fitness_scores'] = story_context['data_fitness']
        
        # Store as current story
        self.current_story = story
        self.story_history.append(story)
        
        return story
    
    def get_current_story(self) -> Optional[DataStory]:
        """Get the current story"""
        return self.current_story
    
    def get_chapter(self, chapter_id: str) -> Optional[StoryChapter]:
        """Get specific chapter from current story"""
        if self.current_story:
            for chapter in self.current_story.chapters:
                if chapter.id == chapter_id:
                    return chapter
        return None

# Action Engine
class ActionEngine:
    """Handle actions from story insights"""
    
    def __init__(self):
        self.action_handlers = {
            'download': self._handle_download,
            'export': self._handle_export,
            'notification': self._handle_notification,
            'calendar': self._handle_calendar,
            'analysis': self._handle_analysis,
            'optimization': self._handle_optimization,
            'integration': self._handle_integration,
            'planning': self._handle_planning,
            'budget_action': self._handle_budget_action,
            'execution': self._handle_execution
        }
        self.action_log = []
    
    def execute_action(self, action: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute an action and return result"""
        action_type = action.get('type', 'unknown')
        
        if action_type in self.action_handlers:
            result = self.action_handlers[action_type](action, context)
        else:
            result = {'status': 'error', 'message': f'Unknown action type: {action_type}'}
        
        # Log action
        self.action_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'result': result
        })
        
        return result
    
    def _handle_download(self, action: Dict, context: Dict) -> Dict:
        """Handle download actions"""
        data = action.get('data', {})
        
        # Convert to JSON for download
        json_data = json.dumps(data, indent=2, default=str)
        b64 = base64.b64encode(json_data.encode()).decode()
        
        return {
            'status': 'success',
            'download_link': f'data:application/json;base64,{b64}',
            'filename': f"story_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        }
    
    def _handle_export(self, action: Dict, context: Dict) -> Dict:
        """Handle export actions"""
        return {
            'status': 'success',
            'message': 'Data exported successfully'
        }
    
    def _handle_notification(self, action: Dict, context: Dict) -> Dict:
        """Handle notification actions"""
        urgency = action.get('urgency', 'normal')
        
        return {
            'status': 'success',
            'message': f'Notification sent with {urgency} urgency',
            'recipients': ['sales_team@company.com']
        }
    
    def _handle_calendar(self, action: Dict, context: Dict) -> Dict:
        """Handle calendar actions"""
        return {
            'status': 'success',
            'message': 'Calendar events created',
            'events_created': 3
        }
    
    def _handle_analysis(self, action: Dict, context: Dict) -> Dict:
        """Handle analysis actions"""
        params = action.get('params', {})
        top_n = params.get('top_n', 5)
        
        return {
            'status': 'success',
            'message': f'Identified top {top_n} focus segments',
            'segments': ['Segment A', 'Segment B', 'Segment C'][:top_n]
        }
    
    def _handle_optimization(self, action: Dict, context: Dict) -> Dict:
        """Handle optimization actions"""
        return {
            'status': 'success',
            'message': 'Optimization recommendations generated',
            'recommendations': [
                'Reallocate 20% budget to top channel',
                'Pause campaigns with ROI < 1.5x',
                'Increase bid on high-converting keywords'
            ]
        }
    
    def _handle_integration(self, action: Dict, context: Dict) -> Dict:
        """Handle integration actions"""
        return {
            'status': 'success',
            'message': 'Tasks created in CRM',
            'tasks_created': 5,
            'integration': 'Salesforce'
        }
    
    def _handle_planning(self, action: Dict, context: Dict) -> Dict:
        """Handle planning actions"""
        return {
            'status': 'success',
            'message': 'Sales targets adjusted',
            'new_targets': {
                'Q1': '$2.5M',
                'Q2': '$2.8M',
                'Q3': '$3.1M'
            }
        }
    
    def _handle_budget_action(self, action: Dict, context: Dict) -> Dict:
        """Handle budget reallocation actions"""
        return {
            'status': 'success',
            'message': 'Budget reallocation plan created',
            'changes': {
                'Social Media': '+15%',
                'Display Ads': '-10%',
                'Search': '+5%'
            }
        }
    
    def _handle_execution(self, action: Dict, context: Dict) -> Dict:
        """Handle execution actions"""
        return {
            'status': 'success',
            'message': 'Optimizations queued for implementation',
            'timeline': '48 hours'
        }

# What-If Engine
class WhatIfEngine:
    """Handle interactive what-if scenarios"""
    
    def __init__(self):
        self.scenario_handlers = {
            'slider': self._handle_slider,
            'multi_slider': self._handle_multi_slider,
            'multi_scenario': self._handle_multi_scenario,
            'budget_allocation': self._handle_budget_allocation
        }
    
    def create_scenario_controls(self, scenario: Dict[str, Any]) -> Any:
        """Create Streamlit controls for what-if scenario"""
        scenario_type = scenario.get('type', 'slider')
        
        if scenario_type in self.scenario_handlers:
            return self.scenario_handlers[scenario_type](scenario)
        
        return None
    
    def _handle_slider(self, scenario: Dict) -> Any:
        """Create single slider control"""
        params = scenario.get('params', {})
        
        # Fix the format string issue
        unit = params.get('unit', '')
        if unit == '%':
            format_str = "%d%%"  # Double %% to escape in sprintf
        else:
            format_str = f"%d{unit}"
        
        value = st.slider(
            scenario.get('label', 'Adjust Value'),
            min_value=params.get('min', 0),
            max_value=params.get('max', 100),
            value=params.get('default', 50),
            step=params.get('step', 1),
            format=format_str
        )
        
        return value
    
    def _handle_multi_slider(self, scenario: Dict) -> Dict:
        """Create multiple slider controls"""
        params = scenario.get('params', {})
        segments = params.get('segments', [])
        
        values = {}
        if segments:
            st.write("Adjust allocation by segment:")
            
            # Ensure total equals 100%
            remaining = 100
            for i, segment in enumerate(segments[:-1]):
                max_val = min(remaining, 100)
                val = st.slider(
                    f"{segment} (%)",
                    min_value=0,
                    max_value=max_val,
                    value=min(20, max_val),
                    key=f"segment_{i}"
                )
                values[segment] = val
                remaining -= val
            
            # Last segment gets the remainder
            if segments:
                values[segments[-1]] = remaining
                st.write(f"{segments[-1]}: {remaining}%")
        
        return values
    
    def _handle_multi_scenario(self, scenario: Dict) -> str:
        """Create scenario selection"""
        scenarios = scenario.get('scenarios', ['Conservative', 'Realistic', 'Optimistic'])
        
        selected = st.select_slider(
            scenario.get('label', 'Select Scenario'),
            options=scenarios,
            value=scenarios[1] if len(scenarios) > 1 else scenarios[0]
        )
        
        return selected
    
    def _handle_budget_allocation(self, scenario: Dict) -> Dict:
        """Create budget allocation interface"""
        st.write("**Budget Reallocation Simulator**")
        
        # Sample channels
        channels = ['Search', 'Social', 'Display', 'Email', 'Direct']
        current_allocation = [30, 25, 20, 15, 10]
        
        new_allocation = {}
        remaining = 100
        
        cols = st.columns(len(channels))
        for i, (channel, current) in enumerate(zip(channels[:-1], current_allocation[:-1])):
            with cols[i]:
                max_val = min(remaining, 50)
                val = st.number_input(
                    f"{channel}",
                    min_value=0,
                    max_value=max_val,
                    value=current,
                    step=5,
                    key=f"budget_{channel}"
                )
                new_allocation[channel] = val
                remaining -= val
        
        # Last channel gets remainder
        with cols[-1]:
            new_allocation[channels[-1]] = remaining
            st.metric(channels[-1], f"{remaining}%")
        
        # Show impact
        if st.button("Calculate Impact"):
            impact = self._calculate_budget_impact(current_allocation, list(new_allocation.values()))
            st.info(f"Projected ROI Change: {impact:+.1f}%")
        
        return new_allocation
    
    def _calculate_budget_impact(self, current: List[float], new: List[float]) -> float:
        """Calculate impact of budget reallocation"""
        # Simplified impact calculation
        channel_efficiency = [2.5, 2.0, 1.5, 1.8, 1.2]  # ROI multipliers
        
        current_roi = sum(c * e for c, e in zip(current, channel_efficiency))
        new_roi = sum(n * e for n, e in zip(new, channel_efficiency))
        
        return ((new_roi - current_roi) / current_roi) * 100

# Story Mode UI
class StoryModeUI:
    """UI components for story mode"""
    
    @staticmethod
    def render_story_mode(story_engine: StoryEngine, dataframe: pd.DataFrame):
        """Render the complete story mode interface"""
        
        # Initialize session state
        if 'current_chapter_index' not in st.session_state:
            st.session_state.current_chapter_index = 0
        
        if 'story_mode_active' not in st.session_state:
            st.session_state.story_mode_active = False
       
        
        # Story generation interface
        if not st.session_state.story_mode_active:
            StoryModeUI._render_story_selector(story_engine, dataframe)
        else:
            StoryModeUI._render_active_story(story_engine)
    
    @staticmethod
    def _render_story_selector(story_engine: StoryEngine, dataframe: pd.DataFrame):
        """Render story type selection interface with relevance indicators"""
        
        st.markdown("### 🎯 Choose Your Story")
        
        # Analyze data fitness for each story type
        story_fitness = story_engine.analyze_data_for_story_fit(dataframe)
        
        # Predefined story templates
        story_templates = [
            {
                'title': '📈 Sales Performance Story',
                'description': 'Comprehensive analysis of sales metrics, pipeline health, and forecasts',
                'prompt': 'Generate a sales performance story with insights and recommendations',
                'icon': '💼',
                'story_type': StoryType.SALES_PERFORMANCE
            },
            {
                'title': '🎯 Marketing ROI Story',
                'description': 'Deep dive into campaign performance, channel attribution, and optimization',
                'prompt': 'Create a marketing ROI story with channel analysis',
                'icon': '📢',
                'story_type': StoryType.MARKETING_ROI
            },
            {
                'title': '🛤️ Customer Journey Story',
                'description': 'Map the customer experience from awareness to conversion',
                'prompt': 'Build a customer journey story with touchpoint analysis',
                'icon': '👥',
                'story_type': StoryType.CUSTOMER_JOURNEY
            },
            {
                'title': '⚡ Executive Dashboard Story',
                'description': 'High-level overview for C-suite decision making',
                'prompt': 'Generate an executive dashboard story with key metrics',
                'icon': '👔',
                'story_type': StoryType.EXECUTIVE_DASHBOARD
            }
        ]
        
        # Display story options with fitness indicators
        cols = st.columns(2)
        for i, template in enumerate(story_templates):
            with cols[i % 2]:
                # Get fitness score - use the enum value as key
                fitness = story_fitness.get(template['story_type'], 0)
                
                # Create relevance indicator
                if fitness >= 0.7:
                    relevance_indicator = "🟢 Highly Relevant"
                    button_type = "primary"
                elif fitness >= 0.4:
                    relevance_indicator = "🟡 Moderately Relevant"
                    button_type = "secondary"
                else:
                    relevance_indicator = "🔴 Low Relevance"
                    button_type = "secondary"
                
                with st.container():
                    # Show relevance
                    st.caption(f"{relevance_indicator} ({fitness:.0%} match)")
                    
                    # Always allow clicking, but warn on low relevance
                    if st.button(
                        f"{template['icon']} {template['title']}",
                        help=template['description'],
                        use_container_width=True,
                        key=f"story_template_{i}",
                        type=button_type if fitness >= 0.4 else "secondary"
                    ):
                        # Generate story based on relevance
                        if fitness < 0.3:
                            if st.session_state.get(f"confirm_low_relevance_{i}", False):
                                with st.spinner(f"🎬 Creating your {template['title']}..."):
                                    story = story_engine.generate_story(template['prompt'], dataframe)
                                    if story:
                                        st.session_state.current_story = story
                                        st.session_state.story_mode_active = True
                                        st.session_state.current_chapter_index = 0
                                        st.rerun()
                            else:
                                st.session_state[f"confirm_low_relevance_{i}"] = True
                                st.warning(
                                    f"⚠️ This story type has low relevance ({fitness:.0%}) for your data. "
                                    "Click again to continue anyway."
                                )
                        else:
                            with st.spinner(f"🎬 Creating your {template['title']}..."):
                                story = story_engine.generate_story(template['prompt'], dataframe)
                                if story:
                                    st.session_state.current_story = story
                                    st.session_state.story_mode_active = True
                                    st.session_state.current_chapter_index = 0
                                    st.rerun()
                    
                    st.caption(template['description'])
        
        # Custom story input
        st.markdown("---")
        st.markdown("### 🎨 Custom Story")
        
        custom_prompt = st.text_area(
            "Describe the story you want to create:",
            placeholder="e.g., Create a quarterly business review story focusing on growth opportunities and risks",
            height=100
        )
        
        if st.button("🎬 Generate Custom Story", type="primary", use_container_width=True):
            if custom_prompt:
                with st.spinner("🎬 Creating your custom story..."):
                    story = story_engine.generate_story(custom_prompt, dataframe)
                    
                    if story:
                        st.session_state.current_story = story
                        st.session_state.story_mode_active = True
                        st.session_state.current_chapter_index = 0
                        st.rerun()
            else:
                st.warning("Please describe the story you want to create")
        
        # Add data suitability info - NOT nested in expander
        st.markdown("---")
        with st.expander("📊 Data Suitability Analysis", expanded=False):
            st.markdown("**How well does your data match each story type?**")
            
            # Sort by fitness score
            sorted_fitness = sorted(story_fitness.items(), key=lambda x: x[1], reverse=True)
            
            for story_type, fitness in sorted_fitness:
                if fitness > 0:
                    # Create columns for better layout
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Color code the text
                        if fitness >= 0.7:
                            st.success(f"**{story_type.value.replace('_', ' ').title()}**")
                        elif fitness >= 0.4:
                            st.warning(f"**{story_type.value.replace('_', ' ').title()}**")
                        else:
                            st.info(f"**{story_type.value.replace('_', ' ').title()}**")
                    
                    with col2:
                        st.write(f"{fitness:.0%} match")
                    
                    st.progress(fitness)
                    st.markdown("")  # Add spacing
            
            # Show detected columns - OUTSIDE of nested expander
            st.markdown("---")
            st.markdown("**Detected Data Characteristics:**")
            
            business_cols = get_business_relevant_columns(dataframe)
            
            # Display in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**📊 Numerical ({len(business_cols['numerical'])})**")
                for col in business_cols['numerical'][:5]:
                    st.caption(f"• {col}")
                if len(business_cols['numerical']) > 5:
                    st.caption(f"... and {len(business_cols['numerical']) - 5} more")
            
            with col2:
                st.markdown(f"**🏷️ Categorical ({len(business_cols['categorical'])})**")
                for col in business_cols['categorical'][:5]:
                    st.caption(f"• {col}")
                if len(business_cols['categorical']) > 5:
                    st.caption(f"... and {len(business_cols['categorical']) - 5} more")
            
            with col3:
                st.markdown(f"**📅 Temporal ({len(business_cols['temporal'])})**")
                for col in business_cols['temporal'][:5]:
                    st.caption(f"• {col}")
                if len(business_cols['temporal']) > 5:
                    st.caption(f"... and {len(business_cols['temporal']) - 5} more")
    
    @staticmethod
    def _render_active_story(story_engine: StoryEngine):
        """Render the active story with navigation"""
        
        story = st.session_state.current_story
        if not story:
            st.error("No active story found")
            return
        
        # Story header
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("← Exit Story", key="exit_story"):
                st.session_state.story_mode_active = False
                st.session_state.current_chapter_index = 0
                st.rerun()
        
        with col2:
            st.markdown(f"<h2 style='text-align: center;'>{story.title}</h2>", unsafe_allow_html=True)
        
        with col3:
            # Download story button
            if st.button("💾 Save Story", key="save_story"):
                # Create download link
                story_data = {
                    'title': story.title,
                    'executive_summary': story.executive_summary,
                    'chapters': [
                        {
                            'title': ch.title,
                            'narrative': ch.narrative,
                            'insights': ch.insights
                        } for ch in story.chapters
                    ],
                    'key_takeaways': story.key_takeaways
                }
                
                json_str = json.dumps(story_data, indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="story_{story.id}.json">Download Story</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        # Chapter navigation
        chapter_index = st.session_state.current_chapter_index
        current_chapter = story.chapters[chapter_index]
        
        # Progress bar
        progress = (chapter_index + 1) / len(story.chapters)
        st.progress(progress)
        
        # Navigation controls
        nav_cols = st.columns([1, 3, 1])
        
        with nav_cols[0]:
            if chapter_index > 0:
                if st.button("← Previous", key="prev_chapter"):
                    st.session_state.current_chapter_index -= 1
                    st.rerun()
        
        with nav_cols[1]:
            st.markdown(
                f"<p style='text-align: center;'>Chapter {chapter_index + 1} of {len(story.chapters)}</p>",
                unsafe_allow_html=True
            )
        
        with nav_cols[2]:
            if chapter_index < len(story.chapters) - 1:
                if st.button("Next →", key="next_chapter"):
                    st.session_state.current_chapter_index += 1
                    st.rerun()
        
        # Render current chapter
        StoryModeUI._render_chapter(current_chapter, story_engine)
        
        # Chapter selector
        with st.expander("📑 Jump to Chapter", expanded=False):
            for i, chapter in enumerate(story.chapters):
                if st.button(
                    f"{i+1}. {chapter.title}",
                    key=f"jump_chapter_{i}",
                    disabled=(i == chapter_index)
                ):
                    st.session_state.current_chapter_index = i
                    st.rerun()
    
    @staticmethod
    def _render_chapter(chapter: StoryChapter, story_engine: StoryEngine):
        """Render individual chapter content with delete functionality"""
        
        # Chapter header
        st.markdown(f"## {chapter.title}")
        st.markdown(f"*{chapter.subtitle}*")
        
        # Narrative section
        with st.container():
            st.markdown(chapter.narrative)
        
        # Visualizations with delete buttons
        if chapter.visualizations:
            for viz_idx, viz in enumerate(chapter.visualizations):
                if viz and viz.get('type') == 'plotly' and viz.get('figure'):
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.plotly_chart(
                            viz['figure'],
                            use_container_width=True,
                            key=f"viz_{chapter.id}_{viz.get('title', 'chart')}_{viz_idx}"
                        )
                    with col2:
                        if st.button("🗑️", key=f"delete_viz_{chapter.id}_{viz_idx}", help="Delete this visualization"):
                            chapter.visualizations.pop(viz_idx)
                            st.rerun()
        
        # Key insights with delete functionality
        if chapter.insights:
            with st.expander("🔍 Key Insights", expanded=True):
                insights_to_show = chapter.insights.copy()  # Work with a copy
                for insight_idx, insight in enumerate(insights_to_show):
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.markdown(f"• {insight}")
                    with col2:
                        if st.button("❌", key=f"delete_insight_{chapter.id}_{insight_idx}", help="Delete this insight"):
                            if insight_idx < len(chapter.insights):
                                chapter.insights.pop(insight_idx)
                                st.rerun()
        
        # What-if scenarios
        if chapter.what_if_scenarios:
            with st.expander("🔮 What-If Analysis", expanded=False):
                what_if_engine = WhatIfEngine()
                
                for scenario in chapter.what_if_scenarios:
                    st.markdown(f"**{scenario.get('label', 'Scenario')}**")
                    
                    # Create controls
                    result = what_if_engine.create_scenario_controls(scenario)
                    
                    # Store result in session state
                    if result is not None:
                        scenario_key = f"scenario_{chapter.id}_{scenario['id']}"
                        st.session_state[scenario_key] = result
        
        # Actions with delete option
        if chapter.actions:
            st.markdown("### 🎯 Recommended Actions")
            
            action_engine = ActionEngine()
            cols = st.columns(len(chapter.actions) if len(chapter.actions) < 4 else 3)
            
            for i, action in enumerate(chapter.actions):
                with cols[i % len(cols)]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            action['label'],
                            key=f"action_{chapter.id}_{action['id']}",
                            use_container_width=True,
                            type="primary" if action.get('urgency') == 'high' else "secondary"
                        ):
                            # Execute action
                            result = action_engine.execute_action(action)
                            
                            if result['status'] == 'success':
                                st.success(f"✅ {result.get('message', 'Action completed')}")
                                
                                # Handle specific action types
                                if action['type'] == 'download' and result.get('download_link'):
                                    st.markdown(
                                        f'<a href="{result["download_link"]}" download="{result.get("filename", "data.json")}">Click to download</a>',
                                        unsafe_allow_html=True
                                    )
                            else:
                                st.error(f"❌ {result.get('message', 'Action failed')}")
                    
                    with col2:
                        if st.button("❌", key=f"delete_action_{chapter.id}_{i}", help="Delete this action"):
                            chapter.actions.pop(i)
                            st.rerun()
        
        # Metrics display (if any)
        if chapter.metrics:
            st.markdown("### 📊 Chapter Metrics")
            metric_cols = st.columns(len(chapter.metrics) if len(chapter.metrics) < 5 else 4)
            
            for i, (metric_name, metric_value) in enumerate(chapter.metrics.items()):
                with metric_cols[i % len(metric_cols)]:
                    if isinstance(metric_value, dict):
                        st.metric(
                            metric_value.get('label', metric_name),
                            metric_value.get('value', 'N/A'),
                            metric_value.get('change', None)
                        )
                    else:
                        st.metric(metric_name, metric_value)

# Integration function for main app
def render_story_mode(agent, dataframe: pd.DataFrame):
    """Main entry point for story mode from agentic_ai_new.py"""
    
    # Initialize story engine
    if 'story_engine' not in st.session_state:
        st.session_state.story_engine = StoryEngine(agent)
    
    story_engine = st.session_state.story_engine
    
    # Render story mode UI
    StoryModeUI.render_story_mode(story_engine, dataframe)

# Export all components
__all__ = [
    'StoryEngine',
    'StoryModeUI',
    'render_story_mode',
    'StoryType',
    'DataStory',
    'StoryChapter',
    'ActionEngine',
    'WhatIfEngine'
]