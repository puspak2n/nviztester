import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly  # Add this line
import pandas as pd
import json
import base64
import openai
from openai import OpenAI
import os
import numpy as np
from datetime import datetime
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import textwrap

# Import AGI_RAG components
try:
    from AGI_RAG import (
        AdvancedAgenticRAG, 
        AdvancedKnowledgeGraph,
        AgenticRAGIntegration
    )
    AGI_RAG_AVAILABLE = True
except ImportError:
    AGI_RAG_AVAILABLE = False
    print("AGI_RAG module not available. Running with limited functionality.")


# Configuration
def load_openai_key():
    """Load OpenAI API key from Streamlit secrets"""
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None

# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Enhanced Code Executor
class EnhancedCodeExecutor:
    def __init__(self):
        self.safe_globals = {
            'pd': pd, 'px': px, 'go': go, 'np': np,
            'plotly': plotly, 'datetime': datetime
        }
    
    def execute_chart_code(self, code: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Fixed code execution with proper error handling"""
        try:
            cleaned_code = self._clean_code(code)
            local_vars = {'df': dataframe.copy()}
            local_vars.update(self.safe_globals)
            
            exec(cleaned_code, {}, local_vars)
            
            fig = local_vars.get('fig')
            if fig is None:
                return {'success': False, 'error': 'Code did not produce a "fig" variable'}
            
            if hasattr(fig, 'update_layout'):
                fig = create_dark_chart(fig)
            
            # IMPORTANT: Also return the processed data that was used for the chart
            chart_data = None
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, pd.DataFrame) and var_name != 'df':
                    chart_data = var_value
                    break
            
            # If no processed dataframe found, try to extract data from the figure
            if chart_data is None and hasattr(fig, 'data'):
                try:
                    # Extract data from plotly figure
                    chart_data = self._extract_data_from_figure(fig)
                except:
                    chart_data = dataframe  # fallback to original
            
            return {
                'success': True, 
                'figure': fig, 
                'cleaned_code': cleaned_code,
                'chart_data': chart_data  # This is the key addition
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Code execution failed: {str(e)}"}
    
    def _extract_data_from_figure(self, fig):
        """Extract the actual data used in the chart"""
        try:
            if hasattr(fig, 'data') and len(fig.data) > 0:
                trace = fig.data[0]
                data_dict = {}
                
                # Extract x and y data
                if hasattr(trace, 'x') and trace.x is not None:
                    data_dict['x'] = list(trace.x)
                if hasattr(trace, 'y') and trace.y is not None:
                    data_dict['y'] = list(trace.y)
                
                # Create DataFrame from extracted data
                if data_dict:
                    return pd.DataFrame(data_dict)
        except Exception as e:
            print(f"Error extracting data from figure: {e}")
        
        return None
    
    def _clean_code(self, code: str) -> str:
        """Clean and validate code for safe execution"""
        # Remove file loading operations
        code = re.sub(r'pd\.read_[a-z]+\([^)]+\)', '', code, flags=re.IGNORECASE)
        code = re.sub(r'df\s*=\s*pd\.read_[a-z]+\([^)]+\)', '', code, flags=re.IGNORECASE)
        code = re.sub(r'fig\.show\(\)', '', code)
        
        if 'fig =' not in code and 'fig=' not in code:
            raise ValueError("Code must assign result to 'fig' variable")
        
        return code.strip()

# Utility Functions
def create_dark_chart(fig):
    """Apply dark theme only to charts, not the entire page"""
    if not hasattr(fig, 'update_layout'):
        return fig
    
    try:
        fig.update_layout(
            # Dark background for chart only
            paper_bgcolor='#1f2a44',
            plot_bgcolor='#1f2a44',
            
            # White text for chart elements
            font=dict(color='white', size=12),
            title_font=dict(size=16, color='white'),
            
            # Legend styling
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=1.01,
                bgcolor='rgba(30, 42, 68, 0.8)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1,
                font=dict(color='white', size=11)
            ),
            
            # Margins
            margin=dict(r=50, t=50, b=50, l=50),
            
            # Grid styling
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)', 
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='white')
            ),
        )
        
        # Update trace colors for visibility on dark background
        fig.update_traces(
            textfont=dict(color='white'),
            hoverlabel=dict(
                bgcolor='rgba(30, 42, 68, 0.9)',
                font_size=12,
                font_family="Arial",
                font_color='white'
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error applying dark theme to chart: {e}")
        return fig

def get_business_relevant_columns(df):
    """Filter out ID columns and other non-analytical fields"""
    avoid_patterns = [
        'id', 'row_id', 'order_id', 'customer_id', 'product_id', 
        'transaction_id', 'invoice_id', 'record_id', 'index', 'key',
        'guid', 'uuid', 'code', 'sku', '_id'
    ]
    business_columns = {
        'categorical': [],
        'numerical': [],
        'temporal': []
    }
    
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        
        # Skip ID columns
        if any(pattern in col_lower for pattern in avoid_patterns):
            continue
            
        # Skip high cardinality string columns
        if df[col].dtype == 'object' and df[col].nunique() > 0.8 * len(df):
            continue
            
        # Categorize columns
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            business_columns['categorical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col_lower:
            business_columns['temporal'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            business_columns['numerical'].append(col)
            
    return business_columns

def detect_value_type(series):
    """Detect if a numeric series contains currency, percentage, or plain numbers"""
    col_name = series.name.lower() if hasattr(series, 'name') else ''
    
    # Check column name for hints
    currency_hints = ['price', 'cost', 'revenue', 'sales', 'amount', 'value', 'salary', 'fee', 'payment']
    percentage_hints = ['rate', 'percent', 'ratio', 'margin', 'yield', 'growth']
    
    if any(hint in col_name for hint in currency_hints):
        return 'currency'
    elif any(hint in col_name for hint in percentage_hints):
        return 'percentage'
    elif series.dtype in ['int64', 'float64'] and series.max() <= 1.0 and series.min() >= 0:
        # Likely a percentage if between 0 and 1
        return 'percentage'
    else:
        return 'number'

def format_value(value, value_type='number'):
    """Format value based on its type"""
    try:
        if pd.isna(value):
            return "N/A"
        
        if value_type == 'currency':
            return f"${float(value):,.2f}"
        elif value_type == 'percentage':
            if value <= 1.0:
                return f"{float(value)*100:.1f}%"
            else:
                return f"{float(value):.1f}%"
        else:
            if isinstance(value, (int, np.integer)):
                return f"{int(value):,}"
            else:
                return f"{float(value):,.2f}"
    except:
        return str(value)

def generate_chart_insights(original_df, chart_data, chart_title, chart_type="bar", user_prompt=None, chart_context=None, code_used=None):
    """ENHANCED: Generate insights using RAG and Knowledge Graph in structured table format"""
    try:
        if not USE_OPENAI:
            return generate_fallback_insights(chart_data, chart_title, chart_type)
        
        # Use the actual chart data for insights, not the original dataframe
        data_for_insights = chart_data if chart_data is not None else original_df
        
        # RAG and Knowledge Graph Enhancement for Insights
        rag_insights = ""
        domain_context = ""
        kg_analysis = {}
        
        # Use RAG system for insights if available
        if AGI_RAG_AVAILABLE:
            try:
                # Import the agent's RAG system if available
                agent = st.session_state.get('conversational_agent')
                if agent and agent.rag_system:
                    # Get domain-specific insights context
                    insight_context = {
                        'chart_type': chart_type,
                        'chart_title': chart_title,
                        'user_query': user_prompt,
                        'data_columns': list(data_for_insights.columns) if hasattr(data_for_insights, 'columns') else [],
                        'business_context': st.session_state.get('business_context', '')
                    }
                    
                    # Enhance insights with RAG
                    rag_enhancement = agent.rag_system.enhance_query(
                        f"Generate insights for {chart_title}", 
                        insight_context, 
                        st.session_state.get('business_context', '')
                    )
                    
                    domain_info = rag_enhancement.get('domain_analysis', {})
                    if domain_info.get('confidence', 0) > 0.6:
                        domain_context = f"""
                        Domain: {domain_info.get('primary_domain', 'general')}
                        Business Function: {domain_info.get('business_function', 'analysis')}
                        Industry Best Practices: {rag_enhancement.get('knowledge_retrieval', {}).get('best_practices', [])}
                        Domain Insights: {rag_enhancement.get('knowledge_retrieval', {}).get('domain_insights', [])}
                        """
                
                # Use Knowledge Graph for additional context
                if agent and agent.knowledge_graph:
                    kg_analysis = agent.knowledge_graph.analyze_dataset(
                        data_for_insights, 
                        st.session_state.get('business_context', '')
                    )
                    
                    if kg_analysis:
                        rag_insights = f"""
                        Knowledge Graph Insights:
                        - Key Relationships: {kg_analysis.get('relationships', [])}
                        - Business Patterns: {kg_analysis.get('patterns', [])}
                        - Domain Recommendations: {kg_analysis.get('recommendations', [])}
                        """
                        
            except Exception as rag_error:
                print(f"⚠️ RAG enhancement for insights failed: {rag_error}")
        
        # Get detailed information about what's actually shown in the chart
        chart_summary = {
            'chart_title': chart_title,
            'chart_type': chart_type,
            'data_points': len(data_for_insights) if hasattr(data_for_insights, '__len__') else 'N/A',
            'columns_in_chart': list(data_for_insights.columns) if hasattr(data_for_insights, 'columns') else [],
            'sample_data': data_for_insights.head(10).to_dict('records') if hasattr(data_for_insights, 'head') else str(data_for_insights)[:500]
        }
        
        # Calculate key metrics for the table
        key_metrics = {}
        if hasattr(data_for_insights, 'describe'):
            try:
                numeric_cols = data_for_insights.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    col_stats = data_for_insights[col].describe()
                    value_type = detect_value_type(data_for_insights[col])
                    
                    # Calculate advanced metrics
                    total = data_for_insights[col].sum()
                    avg = data_for_insights[col].mean()
                    count = len(data_for_insights)
                    
                    key_metrics[col] = {
                        'total': format_value(total, value_type),
                        'average': format_value(avg, value_type),
                        'count': f"{count:,}",
                        'min': format_value(col_stats['min'], value_type),
                        'max': format_value(col_stats['max'], value_type),
                        'std': format_value(col_stats['std'], value_type),
                        'top_performer': None,
                        'bottom_performer': None
                    }
                    
                    # Get top and bottom performers if categorical data exists
                    if len(data_for_insights.columns) > 1:
                        categorical_cols = data_for_insights.select_dtypes(include=['object']).columns
                        if len(categorical_cols) > 0:
                            cat_col = categorical_cols[0]
                            grouped = data_for_insights.groupby(cat_col)[col].sum().sort_values(ascending=False)
                            if len(grouped) > 0:
                                key_metrics[col]['top_performer'] = f"{grouped.index[0]} ({format_value(grouped.iloc[0], value_type)})"
                                key_metrics[col]['bottom_performer'] = f"{grouped.index[-1]} ({format_value(grouped.iloc[-1], value_type)})"
                    
            except Exception as e:
                print(f"Error calculating key metrics: {e}")
        
        # Get statistics from the ACTUAL chart data with advanced analysis
        stats_info = f"Key Metrics: {json.dumps(key_metrics, indent=2)}"
        
        # Analyze the specific user prompt for context
        prompt_analysis = ""
        filter_context = ""
        if user_prompt:
            prompt_lower = user_prompt.lower()
            if 'profit margin' in prompt_lower:
                prompt_analysis = "FOCUS: User specifically asked about PROFIT MARGIN - insights must address margin analysis, not general profit"
            elif 'growth' in prompt_lower or 'trend' in prompt_lower:
                prompt_analysis = "FOCUS: User asked about growth/trends - insights should focus on temporal patterns and growth rates"
            elif 'top' in prompt_lower or 'best' in prompt_lower:
                prompt_analysis = "FOCUS: User asked about top performers - insights should rank and compare performance"
                filter_context = "Apply filter if mentioned: Focus on top performers only"
            elif 'compare' in prompt_lower or 'vs' in prompt_lower:
                prompt_analysis = "FOCUS: User asked for comparison - insights should highlight differences and relative performance"
        
        # Include the code that was used to generate the chart for context
        code_context = ""
        if code_used:
            code_context = f"\nCode Used for Chart:\n{code_used}"
        
        # Build context from user prompt and chart details
        context_info = ""
        if user_prompt:
            context_info += f"\nUser Request: {user_prompt}"
        if chart_context:
            context_info += f"\nChart Context: {chart_context}"
        
        # Enhanced insight prompt with structured table format
        insight_prompt = f"""
        Generate business insights in a STRUCTURED TABLE FORMAT like this example:

        | Apply filter if mentioned |
        |---------------------------|
        | **Key Metrics** |
        | Sales: $10M | Profit: $10K | Quantity: 100 | Revenue: $200K |
        |---------------------------|
        | **Top Insights** |
        | 1. Category is 60% of total |
        | 2. California is lagging on profit margin |
        | 3. Top 3 YoY Profit growth cities are: ... |
        | 4. Bottom 3 are ... |
        | 5. ... |
        |---------------------------|
        | **Recommendations** |
        | Texas is constantly lagging month-on-month or QoQ |
        | ... |

        Chart: {chart_title}
        Type: {chart_type}
        {context_info}
        {code_context}
        {prompt_analysis}
        {filter_context}
        
        DOMAIN EXPERTISE:
        {domain_context}
        
        RAG INSIGHTS:
        {rag_insights}
        
        ACTUAL Data Displayed in Chart:
        Columns: {chart_summary['columns_in_chart']}
        Data Points: {chart_summary['data_points']}
        Sample Values: {chart_summary['sample_data']}
        
        CALCULATED METRICS:
        {stats_info}
        
        CRITICAL REQUIREMENTS:
        1. MUST use the EXACT table format shown above with | borders
        2. Key Metrics row: Show 3-4 most important metrics with actual values from the data
        3. Top Insights: 4-5 numbered insights with specific data points and percentages
        4. Recommendations: 2-3 actionable recommendations
        5. Use REAL numbers from the chart data, not examples
        6. If user asked for filtering (top 10, etc.), mention in first row
        7. Focus on what user specifically asked for
        8. Include growth rates, percentages, and comparisons
        9. Format numbers properly: $1,234.56, 12.3%, 1,234
        10. Be specific about geographic/category performance
        
        EXAMPLES OF GOOD INSIGHTS:
        - "Technology segment represents 45.2% of total revenue ($2.1M)"
        - "California shows 15.3% profit margin, below company average of 18.7%"
        - "Q4 growth rate of 23% outpaced Q3 by 8 percentage points"
        
        Return ONLY the table format - no additional text.
        """
        
        client = OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": insight_prompt}],
            temperature=0.2,  # Lower temperature for more focused insights
            max_tokens=800
        )
        
        insights_text = response.choices[0].message.content.strip()
        
        # Return the structured table format as a single insight
        return [insights_text]
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        return generate_fallback_insights_table(chart_data or original_df, chart_title, chart_type)

def generate_fallback_insights_table(chart_data, chart_title, chart_type):
    """Generate fallback insights in table format"""
    try:
        if hasattr(chart_data, 'describe') and len(chart_data.columns) > 0:
            numeric_cols = chart_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                stats = chart_data[col].describe()
                value_type = detect_value_type(chart_data[col])
                
                total_val = format_value(stats['sum'] if 'sum' in stats else chart_data[col].sum(), value_type)
                avg_val = format_value(stats['mean'], value_type)
                count_val = f"{len(chart_data):,}"
                
                table_format = f"""
| **Key Metrics** |
|-----------------|
| Total: {total_val} \\| Average: {avg_val} \\| Count: {count_val} |
|-----------------|
| **Top Insights** |
| 1. Data shows range from {format_value(stats['min'], value_type)} to {format_value(stats['max'], value_type)} |
| 2. Standard deviation indicates {"high" if stats['std']/stats['mean'] > 0.5 else "moderate"} variability |
| 3. Analysis reveals optimization opportunities in performance gaps |
|-----------------|
| **Recommendations** |
| Focus on standardizing top performer practices |
| Investigate drivers of variation for improvement opportunities |
"""
                return [table_format]
    except Exception as e:
        print(f"Error in fallback table insights: {e}")
    
    return [f"""
| **Key Metrics** |
|-----------------|
| Analysis: {chart_title} |
|-----------------|
| **Top Insights** |
| 1. Chart reveals patterns requiring investigation |
| 2. Data variance suggests optimization potential |
|-----------------|
| **Recommendations** |
| Deep-dive into performance drivers |
"""]

def generate_fallback_insights(chart_data, chart_title, chart_type):
    """Generate fallback insights with proper type handling"""
    insights = []
    
    try:
        if hasattr(chart_data, 'describe') and len(chart_data.columns) > 0:
            # Get numeric columns
            numeric_cols = chart_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                stats = chart_data[col].describe()
                
                # Detect value type
                value_type = detect_value_type(chart_data[col])
                
                # Format values properly
                max_val = format_value(stats['max'], value_type)
                mean_val = format_value(stats['mean'], value_type)
                min_val = format_value(stats['min'], value_type)
                
                # Calculate variance ratio
                if stats['mean'] != 0:
                    variance_ratio = (stats['max'] - stats['min']) / stats['mean']
                    insights.append(f"Range Analysis: Values span from {min_val} to {max_val} with average {mean_val} — investigate drivers of variation.")
                    
                    if variance_ratio > 2:
                        insights.append(f"High Variance: Range is {variance_ratio:.1f}x the average — analyze outliers for insights.")
    except Exception as e:
        print(f"Error in fallback insights: {e}")
    
    # Always provide some insights
    if len(insights) == 0:
        insights = [
            f"Analysis: {chart_title} reveals patterns requiring further investigation.",
            "Opportunity: Data variance suggests optimization potential.",
            "Action: Deep-dive into top/bottom performers for strategic insights."
        ]
    
    return insights[:4]

# Main Conversational Agent
class ConversationalAgent:
    """Main conversational AI agent that handles all interactions"""
    
    def __init__(self):
        if AGI_RAG_AVAILABLE:
            self.rag_system = AdvancedAgenticRAG()
            self.knowledge_graph = AdvancedKnowledgeGraph()
            self.rag_integration = AgenticRAGIntegration()
        else:
            self.rag_system = None
            self.knowledge_graph = None
            self.rag_integration = None
            
        self.conversation_history = []
        self.context_memory = {}
        self.dataset_analysis = None
        self.code_executor = EnhancedCodeExecutor()
    
    def _get_conversation_context(self):
        """Get recent conversation context for better understanding"""
        context = []
        # Get last 3 interactions
        for conv in self.conversation_history[-3:]:
            if conv.get('user_message'):
                context.append(f"User: {conv['user_message']}")
            if conv.get('ai_response') and 'error' not in conv.get('ai_response', {}):
                response = conv['ai_response']
                context.append(f"Assistant created: {response.get('title', 'chart')} ({response.get('type', 'unknown')} chart)")
        return "\n".join(context)
    
    def _is_followup_request(self, prompt):
        """Detect if this is a follow-up request to modify previous chart"""
        followup_keywords = [
            'keep top', 'keep bottom', 'filter', 'only show', 'just show',
            'modify', 'change to', 'update', 'instead', 'but with',
            'same chart', 'that chart', 'previous', 'above'
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in followup_keywords)
    
    def _is_strategic_question(self, prompt):
        """ENHANCED: Detect if this is a strategic/analytical question rather than a visualization request"""
        strategic_keywords = [
            # Role-based questions
            'chief marketing officer', 'cmo', 'ceo', 'chief executive', 'vp of', 'head of',
            'marketing director', 'sales director', 'operations manager', 'strategy',
            
            # Strategic planning
            'what should i', 'what should we', 'how can i', 'how can we', 'what can i do',
            'plan next', 'next steps', 'planning', 'strategy', 'roadmap', 'priorities',
            
            # Analysis requests
            'analyze', 'insights', 'recommendations', 'suggest', 'advise', 'guidance',
            'opportunities', 'threats', 'strengths', 'weaknesses', 'swot',
            'trends', 'patterns', 'why is', 'what drives', 'factors', 'causes',
            'explain', 'understand', 'interpret', 'meaning',
            
            # Business questions
            'improve', 'optimize', 'increase', 'decrease', 'reduce', 'grow',
            'market', 'competition', 'competitive', 'advantage', 'positioning',
            'forecast', 'predict', 'future', 'outlook', 'projection',
            
            # Problem-solving
            'problem', 'issue', 'challenge', 'solution', 'fix', 'address',
            'concerning', 'worried', 'troubling', 'decline', 'drop'
        ]
        
        chart_keywords = [
            'show me', 'display', 'chart', 'graph', 'plot', 'visualize', 'create a',
            'draw', 'bar chart', 'line chart', 'pie chart', 'scatter plot'
        ]
        
        prompt_lower = prompt.lower()
        
        # Check for strategic keywords
        has_strategic = any(keyword in prompt_lower for keyword in strategic_keywords)
        
        # Check for explicit chart requests
        has_chart_request = any(keyword in prompt_lower for keyword in chart_keywords)
        
        # If there are strategic keywords and no explicit chart request, it's strategic
        return has_strategic and not has_chart_request
        
    def initialize_dataset_context(self, dataframe):
        """Initialize context understanding of the dataset"""
        try:
            self.dataset_analysis = self._analyze_dataset_comprehensively(dataframe)
            
            if self.knowledge_graph:
                kg_analysis = self.knowledge_graph.analyze_dataset(
                    dataframe, 
                    st.session_state.get('business_context', '')
                )
            else:
                kg_analysis = {}
            
            self.context_memory['dataset'] = {
                'shape': dataframe.shape,
                'columns': dataframe.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                'analysis': self.dataset_analysis,
                'knowledge_graph': kg_analysis,
                'business_columns': get_business_relevant_columns(dataframe)
            }
            
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize dataset context: {str(e)}")
            return False
    
    def _analyze_dataset_comprehensively(self, df):
        """Comprehensive dataset analysis"""
        analysis = {
            'data_quality': {},
            'column_types': {},
            'patterns': [],
            'relationships': [],
            'business_opportunities': []
        }
        
        business_cols = get_business_relevant_columns(df)
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                })
            
            analysis['column_types'][col] = col_info
        
        # Find patterns
        analysis['patterns'] = self._find_data_patterns(df, business_cols)
        analysis['relationships'] = self._find_relationships(df, business_cols)
        analysis['business_opportunities'] = self._identify_business_opportunities(df, business_cols)
        
        return analysis
    
    def _find_data_patterns(self, df, business_cols):
        """Find patterns in the data"""
        patterns = []
        
        # Growth patterns in temporal data
        for time_col in business_cols['temporal']:
            for metric_col in business_cols['numerical']:
                try:
                    df_temp = df.copy()
                    df_temp[time_col] = pd.to_datetime(df_temp[time_col])
                    df_temp = df_temp.sort_values(time_col)
                    
                    growth = df_temp[metric_col].pct_change().mean()
                    if abs(growth) > 0.01:
                        patterns.append({
                            'type': 'temporal_trend',
                            'description': f"{metric_col} shows {'growth' if growth > 0 else 'decline'} over {time_col}",
                            'strength': abs(growth),
                            'columns': [time_col, metric_col]
                        })
                except:
                    continue
        
        return patterns
    
    def _find_relationships(self, df, business_cols):
        """Find relationships between columns"""
        relationships = []
        
        # Correlations between metrics
        metric_cols = business_cols['numerical']
        if len(metric_cols) > 1:
            try:
                # Select only numeric columns that exist
                valid_metric_cols = [col for col in metric_cols if col in df.columns]
                if len(valid_metric_cols) > 1:
                    corr_matrix = df[valid_metric_cols].corr()
                    for i in range(len(valid_metric_cols)):
                        for j in range(i+1, len(valid_metric_cols)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.5:
                                relationships.append({
                                    'type': 'correlation',
                                    'col1': valid_metric_cols[i],
                                    'col2': valid_metric_cols[j],
                                    'strength': abs(corr_val),
                                    'direction': 'positive' if corr_val > 0 else 'negative'
                                })
            except:
                pass
        
        return relationships
    
    def _identify_business_opportunities(self, df, business_cols):
        """Identify potential business opportunities"""
        opportunities = []
        
        # High-variance metrics
        for metric_col in business_cols['numerical']:
            try:
                if metric_col in df.columns:
                    mean_val = df[metric_col].mean()
                    if mean_val != 0:
                        cv = df[metric_col].std() / mean_val
                        if cv > 0.5:
                            opportunities.append({
                                'type': 'optimization',
                                'description': f"High variance in {metric_col} suggests optimization potential",
                                'metric': metric_col,
                                'variance': cv
                            })
            except:
                continue
        
        return opportunities
    
    def generate_analysis_suggestions(self, user_query):
        """Generate contextual analysis suggestions based on user query"""
        try:
            if not self.dataset_analysis:
                return ["Please initialize dataset analysis first."]
            
            if not USE_OPENAI:
                return self._generate_fallback_suggestions(user_query)
            
            business_cols = self.context_memory['dataset']['business_columns']
            
            context_prompt = f"""
            User Query: {user_query}
            
            Dataset Context:
            - Shape: {self.context_memory['dataset']['shape']}
            - Business Metrics: {business_cols['numerical']}
            - Business Dimensions: {business_cols['categorical']}
            - Temporal Columns: {business_cols['temporal']}
            
            Generate 4-5 specific, actionable analysis suggestions that directly address the user's query.
            Each suggestion should be specific to the available data columns.
            
            Return as JSON array:
            [
                {{
                    "title": "Specific Analysis Title",
                    "description": "What this analysis will show",
                    "suggested_prompt": "Exact prompt to generate this",
                    "expected_insights": ["insight 1", "insight 2"],
                    "chart_type": "bar|line|scatter|pie",
                    "business_value": "Why this matters"
                }}
            ]
            """
            
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context_prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            suggestions = json.loads(response_text)
            return suggestions
            
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")
            return self._generate_fallback_suggestions(user_query)
    
    def _generate_fallback_suggestions(self, user_query):
        """Generate fallback suggestions when AI fails"""
        business_cols = self.context_memory['dataset']['business_columns']
        suggestions = []
        
        if business_cols['numerical'] and business_cols['categorical']:
            metric = business_cols['numerical'][0]
            dimension = business_cols['categorical'][0]
            
            suggestions.append({
                "title": f"Performance Analysis by {dimension}",
                "description": f"Analyze {metric} performance across different {dimension} categories",
                "suggested_prompt": f"Show me {metric} by {dimension} with insights on top and bottom performers",
                "expected_insights": ["Performance gaps identification", "Top performer characteristics"],
                "chart_type": "bar",
                "business_value": "Identify optimization opportunities"
            })
        
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            metric = business_cols['numerical'][0]
            
            suggestions.append({
                "title": f"{metric} Trend Analysis",
                "description": f"Track {metric} changes over {time_col}",
                "suggested_prompt": f"Show me {metric} trends over {time_col} with growth analysis",
                "expected_insights": ["Growth patterns", "Seasonal trends"],
                "chart_type": "line",
                "business_value": "Understand business trajectory"
            })
        
        return suggestions
    
    def create_custom_chart(self, dataframe, user_prompt, business_context=""):
        """Create a custom chart based on user request using OpenAI to parse the prompt"""
        try:
            # Initialize data context for RAG
            data_context = {
                'columns': list(dataframe.columns),
                'row_count': len(dataframe),
                'column_count': len(dataframe.columns),
                'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()}
            }
            
            # Use RAG system if available
            rag_context = ""
            rag_enhanced_prompt = None
            domain_info = {}
            rag_recommendations = []
            
            if self.rag_system:
                try:
                    # Get RAG enhancement
                    rag_enhancement = self.rag_system.enhance_query(
                        user_prompt, 
                        data_context, 
                        business_context
                    )
                    
                    # Extract enhanced prompt and other useful info
                    rag_enhanced_prompt = rag_enhancement.get('enhanced_prompt', '')
                    domain_info = rag_enhancement.get('domain_analysis', {})
                    rag_recommendations = rag_enhancement.get('recommendations', [])
                    
                    # Log domain detection for debugging
                    if domain_info:
                        print(f"🎯 RAG Domain: {domain_info.get('primary_domain', 'unknown')} "
                              f"(confidence: {domain_info.get('confidence', 0):.1%})")
                        print(f"📊 Business Function: {domain_info.get('business_function', 'unknown')}")
                    
                    # Only use domain-specific recommendations if confidence is high
                    if domain_info.get('confidence', 0) > 0.6:
                        # Extract key context from enhanced prompt for the parse prompt
                        rag_context = f"""
                        Domain: {domain_info.get('primary_domain', 'general')}
                        Business Function: {domain_info.get('business_function', 'analysis')}
                        Confidence: {domain_info.get('confidence', 0.5):.1%}
                        
                        Expert Knowledge Applied:
                        {self._extract_rag_insights(rag_enhancement)}
                        """
                    else:
                        # Low confidence - use minimal RAG context
                        rag_context = "Use standard visualization best practices."
                        rag_enhanced_prompt = None  # Don't use the enhanced prompt if low confidence
                        
                except Exception as rag_error:
                    print(f"⚠️ RAG enhancement failed: {rag_error}")
                    # Continue without RAG enhancement
            
            # Get conversation context
            recent_context = self._get_conversation_context()
            
            # Check if this is a follow-up request
            is_followup = self._is_followup_request(user_prompt)
            previous_chart_context = ""
            
            if is_followup and len(self.conversation_history) > 1:
                # Get the previous chart's context
                prev_response = self.conversation_history[-2].get('ai_response', {})
                if prev_response and 'error' not in prev_response:
                    previous_chart_context = f"""
                    Previous chart:
                    - Type: {prev_response.get('type')}
                    - Title: {prev_response.get('title')}
                    - X Column: {prev_response.get('x_col')}
                    - Y Column: {prev_response.get('y_col')}
                    - Code: {prev_response.get('code', '')}
                    """
            
            # Build comprehensive context
            business_cols = get_business_relevant_columns(dataframe)
            
            # Determine if we should use RAG-enhanced prompt or standard prompt
            if rag_enhanced_prompt and domain_info.get('confidence', 0) > 0.6:
                # Use enhanced prompt but add critical chart selection rules
                parse_prompt = rag_enhanced_prompt + f"""
                
                CRITICAL CHART SELECTION RULES (OVERRIDE ANY DOMAIN SUGGESTIONS IF NEEDED):
                1. Use BAR charts for: comparing categories, showing totals by group, rankings
                2. Use LINE charts for: trends over time, continuous data, growth patterns
                3. Use SCATTER charts for: correlations, relationships between two numeric variables
                4. Use PIE charts ONLY for: showing parts of a whole (and only if <6 categories)
                5. Use FUNNEL charts ONLY when user explicitly asks for "funnel", "pipeline", or "conversion stages"
                6. Remember the previous chart’s context and reuse it for follow-up requests:
                   • Chart type: {prev_context['chart_type']}
                   • Dimensions: {prev_context['dimensions']}
                   • Metrics: {prev_context['metrics']}
                   • Filters: {prev_context['filters']}
                
                User request: "{user_prompt}"
                
                DO NOT use funnel charts unless the words "funnel", "pipeline", waterfall or "conversion" appear in the user request.
                
                Generate JSON response:
                {{
                    "recommendations": [{{
                        "type": "bar|line|scatter|pie",
                        "x_col": "exact_column_name",
                        "y_col": "exact_column_name",
                        "title": "Descriptive title",
                        "reason": "Why this visualization answers the request",
                        "priority": "high",
                        "code": "Complete Python code using plotly",
                        "is_followup": {str(is_followup).lower()},
                        "context_used": "Domain: {domain_info.get('primary_domain', 'general')}"
                    }}]
                }}
                """
            else:
                # Standard prompt with explicit chart selection guidance
                parse_prompt = f"""
                You are a data visualization expert. Generate Python code for this request.
                
                Current Request: {user_prompt}
                
                Dataset Columns: {list(dataframe.columns)}
                Business Columns:
                - Numerical: {business_cols['numerical']}
                - Categorical: {business_cols['categorical']}
                - Temporal: {business_cols['temporal']}
                
                {previous_chart_context}
                
                Recent Conversation Context: {recent_context}
                
                RAG Context: {rag_context}
                
                Business Context: {business_context}
                
                CRITICAL CHART SELECTION RULES:
                1. For comparisons across categories → use BAR chart
                2. For time-based trends → use LINE chart
                3. For correlations between numbers → use SCATTER chart
                4. For parts of a whole → use PIE chart (only if ≤6 categories)
                5. NEVER use FUNNEL charts unless user explicitly mentions "funnel", "pipeline", or "conversion"
                6. Remember the previous chart’s context and reuse it for follow-up requests:
                   • Chart type: {prev_context['chart_type']}
                   • Dimensions: {prev_context['dimensions']}
                   • Metrics: {prev_context['metrics']}
                   • Filters: {prev_context['filters']}
                
                Examples of appropriate chart selection:
                - "show sales by region" → bar chart (comparing categories)
                - "display revenue trends over time" → line chart (time trend)
                - "analyze relationship between price and quantity" → scatter chart (correlation)
                - "compare performance across departments" → bar chart (category comparison)
                - "track monthly growth" → line chart (time trend)
                
                IMPORTANT INSTRUCTIONS:
                1. If this is a follow-up request (e.g., "keep top 10 only", "filter by X"), modify the PREVIOUS chart's code
                2. Understand the context - don't create a completely new chart unless explicitly asked
                3. For "top N" requests on existing charts, use .nlargest() or .head() on the existing grouping
                4. Use the exact column names from the dataset
                5. MUST use plotly (import plotly.express as px or plotly.graph_objects as go)
                6. NEVER use matplotlib, seaborn, or other libraries
                7. Create the chart and assign to 'fig' variable
                8. Do NOT use 'return fig' - just create fig
                9. Do NOT wrap code in a function
                10. Return ONLY valid JSON
                11. Store processed data in a variable with a clear name (like 'chart_data', 'grouped_data', etc.)
                
                Is this a follow-up? {is_followup}
                
                Generate JSON:
                {{
                    "recommendations": [{{
                        "type": "CHOOSE ONLY FROM: bar|line|scatter|pie",
                        "x_col": "exact_column_name",
                        "y_col": "exact_column_name",
                        "title": "Descriptive title based on the request",
                        "reason": "Why this specific chart type best answers the request",
                        "priority": "high",
                        "code": "Complete Python code using ONLY plotly that creates 'fig' AND stores processed data",
                        "is_followup": true/false,
                        "context_used": "Brief note on what context was considered"
                    }}]
                }}
                
                Example of CORRECT code:
                import plotly.express as px
                # Store the processed data for insights
                chart_data = df.groupby('City')['Sales'].sum().reset_index()
                chart_data = chart_data.sort_values('Sales', ascending=False)
                fig = px.bar(chart_data, x='City', y='Sales', title='Sales by City')
                
                Example of INCORRECT code (DO NOT DO THIS):
                - Using matplotlib
                - Using 'return fig'
                - Wrapping in a function
                - Using funnel chart when not explicitly requested
                - Not storing the processed data in a variable
                """
        
            # Continue with the rest of your existing code...
            try:
                client = OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": parse_prompt}],
                    temperature=0.2,
                    max_tokens=1000
                )
                
                response_text = response.choices[0].message.content.strip()
                print(f"DEBUG - AI Response: {response_text[:200]}...")
                
                # Clean up response
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                # Try to parse JSON
                try:
                    recs = json.loads(response_text).get('recommendations', [])
                except json.JSONDecodeError as json_err:
                    print(f"DEBUG - JSON parse failed: {json_err}")
                    print(f"DEBUG - Raw response: {response_text}")
                    
                    # Create a simple fallback recommendation
                    recs = self._create_fallback_recommendation(dataframe, user_prompt, business_cols)
                
                if not recs:
                    return {'error': 'AI did not return any chart recommendations.'}
                
            except Exception as api_err:
                print(f"DEBUG - API call failed: {api_err}")
                # Create emergency fallback
                recs = self._create_fallback_recommendation(dataframe, user_prompt, business_cols)
            
            # Get the actual code from the recommendation
            rec = recs[0]
            code = rec.get('code', '')
            
            # Execute the code and create the result
            result = self._execute_and_create_result(
                code, dataframe, rec, user_prompt, is_followup
            )
            
            # Add RAG-specific information to the result
            if domain_info:
                result['domain_analysis'] = domain_info
                result['rag_applied'] = True
                
                # Update confidence based on RAG
                if result.get('confidence'):
                    rag_confidence = domain_info.get('confidence', 0.5)
                    # Boost confidence if RAG was successful
                    result['confidence']['score'] = min(1.0, result['confidence']['score'] * 1.1)
                    result['confidence']['rag_boost'] = True
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def _extract_rag_insights(self, rag_enhancement):
        """Extract key insights from RAG enhancement for display"""
        insights = []
        
        knowledge = rag_enhancement.get('knowledge_retrieval', {})
        if knowledge.get('best_practices'):
            insights.extend([f"- {practice}" for practice in knowledge['best_practices'][:2]])
        
        if knowledge.get('domain_insights'):
            insights.extend([f"- {insight}" for insight in knowledge['domain_insights'][:2]])
        
        return '\n'.join(insights) if insights else "- Standard analysis patterns applied"

    def _create_fallback_recommendation(self, dataframe, user_prompt, business_cols):
        """Create fallback recommendation when AI fails"""
        prompt_lower = user_prompt.lower()
        
        # Determine best chart type based on request
        if any(word in prompt_lower for word in ['vs', 'versus', 'compare', 'correlation']):
            chart_type = "scatter"
            if len(business_cols['numerical']) >= 2:
                x_col = business_cols['numerical'][0]
                y_col = business_cols['numerical'][1]
            else:
                x_col = business_cols['categorical'][0] if business_cols['categorical'] else dataframe.columns[0]
                y_col = business_cols['numerical'][0] if business_cols['numerical'] else dataframe.columns[1]
        else:
            chart_type = "bar"
            x_col = business_cols['categorical'][0] if business_cols['categorical'] else dataframe.columns[0]
            y_col = business_cols['numerical'][0] if business_cols['numerical'] else dataframe.columns[1]
        
        return [{
            "type": chart_type,
            "x_col": x_col,
            "y_col": y_col, 
            "title": f"{y_col} by {x_col} Analysis",
            "reason": "Fallback visualization",
            "priority": "high",
            "code": f"""
import plotly.express as px
chart_data = df.groupby('{x_col}')['{y_col}'].sum().reset_index()
fig = px.{chart_type}(chart_data, x='{x_col}', y='{y_col}',
                    title='{y_col} by {x_col}')
"""
        }]

    def _execute_and_create_result(self, code, dataframe, rec, user_prompt, is_followup):
        """Execute code and create result object"""
        # Clean up the generated code
        code = re.sub(r'fig\.show\(\)', '', code)
        code = re.sub(r'\breturn\s+fig\b', '', code)
        code = re.sub(r'\breturn\s+.*', '', code)
        
        # Execute the code using the enhanced executor
        execution_result = self.code_executor.execute_chart_code(code, dataframe)
        
        if not execution_result['success']:
            return {'error': execution_result['error'], 'code': code}
        
        fig = execution_result['figure']
        chart_data = execution_result.get('chart_data')  # This is the processed data used for the chart
        
        if fig is None:
            return {'error': f'No result produced. Code: {code}', 'code': code}
        
        # Create result
        if isinstance(fig, pd.DataFrame):
            chart_type = 'table'
            display_fig = None
        else:
            chart_type = rec.get('type', 'custom')
            display_fig = create_dark_chart(fig)
        
        confidence_info = {
            'score': 0.85, 
            'level': 'High', 
            'method': 'ai_generated', 
            'color': '#28a745', 
            'icon': '✅'
        }
        
        return {
            'prompt': user_prompt,
            'figure': display_fig,
            'table_data': fig if isinstance(fig, pd.DataFrame) else None,
            'chart_data': chart_data,  # Store the processed chart data
            'x_col': rec.get('x_col'),
            'y_col': rec.get('y_col'),
            'code': code,
            'ai_analysis': rec.get('reason', ''),
            'type': chart_type,
            'title': rec.get('title', user_prompt),
            'confidence': confidence_info,
            'validation': {'passed': True, 'issues': []}
        }
    
    def generate_chart_with_insights(self, user_prompt, dataframe):
        """ENHANCED: Generate chart and insights based on user prompt with better strategic question detection"""
        try:
            self.conversation_history.append({
                'user_message': user_prompt,
                'timestamp': datetime.now().isoformat()
            })
            
            # ENHANCED: Better detection of strategic vs visualization requests
            is_strategic = self._is_strategic_question(user_prompt)
            
            # Check for explicit chart requests that override strategic detection
            chart_keywords = ['show', 'chart', 'plot', 'graph', 'visualize', 'display']
            prompt_lower = user_prompt.lower()
            explicitly_wants_chart = any(keyword in prompt_lower for keyword in chart_keywords)
            
            if is_strategic and not explicitly_wants_chart:
                # Generate strategic analytical insights without charts
                result = self.generate_strategic_insights(dataframe, user_prompt)
            else:
                # Generate chart with insights
                result = self.create_custom_chart(dataframe, user_prompt, 
                                                st.session_state.get('business_context', ''))
                
                if 'error' not in result:
                    # ENHANCED: Generate insights using the ACTUAL chart data, not original dataframe
                    chart_data = result.get('chart_data')  # This is the processed data from the chart
                    if chart_data is None:
                        # Fallback to trying to recreate the data from x_col and y_col
                        if result.get('x_col') and result.get('y_col'):
                            try:
                                chart_data = dataframe.groupby(result['x_col'])[result['y_col']].sum().reset_index()
                            except:
                                chart_data = dataframe
                        else:
                            chart_data = dataframe
                    
                    # Generate insights with the actual chart data and code context
                    chart_context = {
                        'user_prompt': user_prompt,
                        'x_col': result.get('x_col'),
                        'y_col': result.get('y_col'),
                        'chart_type': result.get('type'),
                        'is_followup': self._is_followup_request(user_prompt)
                    }
                    
                    insights = generate_chart_insights(
                        dataframe,  # original dataframe for context
                        chart_data,  # ACTUAL chart data for insights
                        result.get('title', user_prompt), 
                        result.get('type', 'custom'),
                        user_prompt=user_prompt,
                        chart_context=json.dumps(chart_context),
                        code_used=result.get('code')  # Pass the code for additional context
                    )
                    
                    result['insights'] = insights
            
            self.conversation_history[-1]['ai_response'] = result
            
            return result
            
        except Exception as e:
            error_result = {'error': f'Chart generation failed: {str(e)}'}
            self.conversation_history[-1]['ai_response'] = error_result
            return error_result
    
    def generate_strategic_insights(self, dataframe, user_prompt):
        """ENHANCED: Generate strategic business insights using RAG and Knowledge Graph in structured table format"""
        try:
            if not USE_OPENAI:
                return self._generate_fallback_strategic_insights(dataframe, user_prompt)
            
            # Analyze the data comprehensively for strategic insights
            business_cols = get_business_relevant_columns(dataframe)
            
            # Get comprehensive data analysis
            strategic_analysis = self._perform_strategic_analysis(dataframe, business_cols, user_prompt)
            
            # Extract user role and context
            role_context = self._extract_role_context(user_prompt)
            
            # RAG and Knowledge Graph Enhancement for Strategic Insights
            domain_expertise = ""
            industry_benchmarks = ""
            kg_strategic_insights = ""
            
            # Use RAG system for strategic domain knowledge
            if self.rag_system:
                try:
                    # Create strategic context for RAG
                    strategic_context = {
                        'role': role_context.get('role', 'Business Leader'),
                        'query_type': 'strategic_analysis',
                        'business_metrics': business_cols['numerical'][:5],
                        'business_dimensions': business_cols['categorical'][:3],
                        'data_size': len(dataframe),
                        'business_context': st.session_state.get('business_context', '')
                    }
                    
                    # Get RAG enhancement for strategic insights
                    rag_enhancement = self.rag_system.enhance_query(
                        f"Strategic analysis for {role_context.get('role')}: {user_prompt}", 
                        strategic_context, 
                        st.session_state.get('business_context', '')
                    )
                    
                    domain_info = rag_enhancement.get('domain_analysis', {})
                    if domain_info.get('confidence', 0) > 0.6:
                        knowledge_base = rag_enhancement.get('knowledge_retrieval', {})
                        
                        domain_expertise = f"""
                        Domain: {domain_info.get('primary_domain', 'general')}
                        Industry Best Practices: {knowledge_base.get('best_practices', [])}
                        Success Factors: {knowledge_base.get('success_factors', [])}
                        Common Challenges: {knowledge_base.get('challenges', [])}
                        """
                        
                        # Get industry benchmarks if available
                        if knowledge_base.get('benchmarks'):
                            industry_benchmarks = f"""
                            Industry Benchmarks:
                            {json.dumps(knowledge_base.get('benchmarks'), indent=2)}
                            """
                            
                except Exception as rag_error:
                    print(f"⚠️ RAG enhancement for strategic insights failed: {rag_error}")
            
            # Use Knowledge Graph for strategic relationships
            if self.knowledge_graph:
                try:
                    kg_analysis = self.knowledge_graph.analyze_dataset(
                        dataframe, 
                        st.session_state.get('business_context', '')
                    )
                    
                    if kg_analysis:
                        strategic_relationships = kg_analysis.get('strategic_relationships', [])
                        business_drivers = kg_analysis.get('business_drivers', [])
                        risk_factors = kg_analysis.get('risk_factors', [])
                        
                        kg_strategic_insights = f"""
                        Knowledge Graph Strategic Analysis:
                        Key Business Relationships: {strategic_relationships}
                        Business Performance Drivers: {business_drivers}
                        Risk Factors: {risk_factors}
                        Optimization Opportunities: {kg_analysis.get('optimization_opportunities', [])}
                        """
                        
                except Exception as kg_error:
                    print(f"⚠️ Knowledge Graph analysis failed: {kg_error}")
            
            # Enhanced strategic analysis with advanced metrics
            advanced_strategic_metrics = self._calculate_advanced_strategic_metrics(dataframe, business_cols, role_context)
            
            # Analyze user prompt for specific strategic focus
            strategic_focus = self._analyze_strategic_focus(user_prompt, role_context)
            
            # Calculate key strategic metrics for the table
            key_strategic_metrics = {}
            if business_cols['numerical']:
                for col in business_cols['numerical'][:4]:  # Top 4 metrics
                    if col in dataframe.columns:
                        total = dataframe[col].sum()
                        avg = dataframe[col].mean()
                        value_type = detect_value_type(dataframe[col])
                        
                        key_strategic_metrics[col] = {
                            'total': format_value(total, value_type),
                            'average': format_value(avg, value_type),
                            'type': value_type
                        }
            
            # Get top and bottom performers
            performance_insights = {}
            if business_cols['categorical'] and business_cols['numerical']:
                cat_col = business_cols['categorical'][0]
                metric_col = business_cols['numerical'][0]
                
                grouped = dataframe.groupby(cat_col)[metric_col].sum().sort_values(ascending=False)
                if len(grouped) > 0:
                    performance_insights = {
                        'top_performer': f"{grouped.index[0]} ({format_value(grouped.iloc[0], detect_value_type(dataframe[metric_col]))})",
                        'bottom_performer': f"{grouped.index[-1]} ({format_value(grouped.iloc[-1], detect_value_type(dataframe[metric_col]))})",
                        'performance_gap': f"{((grouped.iloc[0] - grouped.iloc[-1]) / grouped.iloc[-1] * 100):.1f}%" if grouped.iloc[-1] != 0 else "N/A"
                    }
            
            # Build enhanced strategic analysis prompt
            analysis_prompt = f"""
            Generate strategic business insights in a STRUCTURED TABLE FORMAT for a {role_context.get('role', 'business leader')}.
            
            Use this EXACT table format:

            | **Role Context & Strategic Focus** |
            |-----------------------------------|
            | Role: {role_context.get('role', 'Business Leader')} \\| Focus: {', '.join(strategic_focus)} \\| Industry: {domain_expertise.split('Domain:')[1].split('\\n')[0] if 'Domain:' in domain_expertise else 'General'} |
            |-----------------------------------|
            | **Key Strategic Metrics** |
            | [Show 4 most important business metrics with values] |
            |-----------------------------------|
            | **Strategic Situation Analysis** |
            | 1. [Current state with key numbers] |
            | 2. [Market position assessment] |
            | 3. [Performance vs industry benchmarks] |
            |-----------------------------------|
            | **Critical Strategic Insights** |
            | 1. [Top priority insight with specific data] |
            | 2. [Key opportunity or risk with metrics] |
            | 3. [Performance gap analysis] |
            | 4. [Growth/efficiency opportunity] |
            |-----------------------------------|
            | **Strategic Recommendations** |
            | Priority 1: [Immediate action with expected outcome] |
            | Priority 2: [Strategic initiative with timeline] |
            | Priority 3: [Risk mitigation or optimization] |
            |-----------------------------------|
            | **Success Metrics & KPIs** |
            | Track: [KPI 1] \\| Target: [KPI 2] \\| Monitor: [KPI 3] \\| Risk: [KPI 4] |

            User Question: {user_prompt}
            Role Context: {role_context.get('context', 'Executive seeking strategic guidance')}
            Strategic Focus: {strategic_focus}
            
            DOMAIN EXPERTISE:
            {domain_expertise}
            
            INDUSTRY CONTEXT:
            {industry_benchmarks}
            
            KNOWLEDGE GRAPH INSIGHTS:
            {kg_strategic_insights}
            
            Dataset Overview:
            - Total Records: {len(dataframe):,}
            - Business Metrics: {', '.join(business_cols['numerical'][:5])}
            - Key Dimensions: {', '.join(business_cols['categorical'][:3])}
            
            Strategic Analysis Data:
            {json.dumps(strategic_analysis, indent=2)}
            
            Key Strategic Metrics Calculated:
            {json.dumps(key_strategic_metrics, indent=2)}
            
            Performance Analysis:
            {json.dumps(performance_insights, indent=2)}
            
            Advanced Strategic Metrics:
            {json.dumps(advanced_strategic_metrics, indent=2)}
            
            Business Context: {st.session_state.get('business_context', 'General business analysis')}
            
            CRITICAL REQUIREMENTS:
            1. MUST use the EXACT table format shown above with | borders
            2. Role Context row: Show role, strategic focus, and industry in one row
            3. Key Metrics: 4 most important business metrics with actual values
            4. Strategic Situation: 3 key points about current state with numbers
            5. Critical Insights: 4 strategic insights with specific data and percentages
            6. Recommendations: 3 prioritized actions with clear outcomes
            7. Success Metrics: 4 KPIs to track with specific targets
            8. Use REAL numbers from the data analysis, not examples
            9. Focus on role-specific priorities (CMO = marketing, CEO = overall strategy)
            10. Include industry benchmarks and competitive positioning where available
            11. Address specific user question directly
            12. Format numbers properly: $1,234.56, 12.3%, 1,234
            13. Keep text concise to fit table format
            
            ROLE-SPECIFIC FOCUS:
            - CMO: Customer acquisition, brand strategy, marketing ROI, segment performance
            - CEO: Overall performance, market position, growth strategy, competitive advantage
            - Sales Director: Revenue optimization, sales efficiency, territory performance
            - Operations: Efficiency, cost optimization, process improvement
            
            Return ONLY the table format - no additional text.
            """
            
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2,  # Lower temperature for more focused strategic insights
                max_tokens=1500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Format the result for display
            return {
                'type': 'strategic_analysis',
                'role_context': role_context,
                'strategic_table': response_text,  # Store the full table
                'data_summary': strategic_analysis,
                'advanced_metrics': advanced_strategic_metrics,
                'prompt': user_prompt,
                'rag_applied': bool(domain_expertise),
                'kg_applied': bool(kg_strategic_insights)
            }
            
        except Exception as e:
            print(f"Error generating strategic insights: {e}")
            return self._generate_fallback_strategic_insights_table(dataframe, user_prompt, role_context)
    
    def _analyze_strategic_focus(self, user_prompt, role_context):
        """Analyze the user prompt to identify specific strategic focus areas"""
        prompt_lower = user_prompt.lower()
        role = role_context.get('role', '').lower()
        
        focus_areas = []
        
        # Role-specific focus detection
        if 'marketing' in role or 'cmo' in role:
            if any(word in prompt_lower for word in ['acquisition', 'customer', 'segment']):
                focus_areas.append('Customer Acquisition Strategy')
            if any(word in prompt_lower for word in ['roi', 'return', 'effectiveness']):
                focus_areas.append('Marketing ROI Optimization')
            if any(word in prompt_lower for word in ['brand', 'awareness', 'positioning']):
                focus_areas.append('Brand Strategy')
        
        # General strategic focus
        if any(word in prompt_lower for word in ['profit', 'margin', 'profitability']):
            focus_areas.append('Profitability Analysis')
        if any(word in prompt_lower for word in ['growth', 'expand', 'scale']):
            focus_areas.append('Growth Strategy')
        if any(word in prompt_lower for word in ['efficiency', 'optimize', 'improve']):
            focus_areas.append('Operational Excellence')
        if any(word in prompt_lower for word in ['risk', 'threat', 'challenge']):
            focus_areas.append('Risk Management')
        
        return focus_areas if focus_areas else ['General Strategic Analysis']
    
    def _calculate_advanced_strategic_metrics(self, dataframe, business_cols, role_context):
        """Calculate advanced strategic metrics based on role and data"""
        metrics = {}
        
        try:
            # Revenue and profitability metrics
            if business_cols['numerical']:
                revenue_cols = [col for col in business_cols['numerical'] if any(word in col.lower() for word in ['sales', 'revenue', 'amount'])]
                profit_cols = [col for col in business_cols['numerical'] if 'profit' in col.lower()]
                
                if revenue_cols:
                    revenue_col = revenue_cols[0]
                    total_revenue = dataframe[revenue_col].sum()
                    avg_revenue = dataframe[revenue_col].mean()
                    
                    metrics['revenue_metrics'] = {
                        'total_revenue': float(total_revenue),
                        'average_transaction': float(avg_revenue),
                        'revenue_concentration': float(dataframe[revenue_col].std() / avg_revenue) if avg_revenue > 0 else 0
                    }
                    
                    # Calculate profit margins if profit data exists
                    if profit_cols:
                        profit_col = profit_cols[0]
                        total_profit = dataframe[profit_col].sum()
                        metrics['profitability_metrics'] = {
                            'total_profit': float(total_profit),
                            'overall_margin': float(total_profit / total_revenue) if total_revenue > 0 else 0,
                            'profit_per_transaction': float(dataframe[profit_col].mean())
                        }
            
            # Customer/segment analysis
            if business_cols['categorical']:
                segment_col = business_cols['categorical'][0]
                segment_analysis = {}
                
                for metric_col in business_cols['numerical'][:2]:
                    if metric_col in dataframe.columns:
                        grouped = dataframe.groupby(segment_col)[metric_col].agg(['sum', 'mean', 'count'])
                        
                        # Calculate segment concentration
                        total = grouped['sum'].sum()
                        segment_shares = (grouped['sum'] / total * 100).round(2)
                        top_segment = segment_shares.idxmax()
                        
                        segment_analysis[metric_col] = {
                            'top_segment': top_segment,
                            'top_segment_share': float(segment_shares.max()),
                            'segment_count': len(grouped),
                            'herfindahl_index': float((segment_shares ** 2).sum() / 10000)  # Concentration index
                        }
                
                metrics['segment_metrics'] = segment_analysis
            
            # Performance variability (important for risk assessment)
            performance_metrics = {}
            for col in business_cols['numerical'][:3]:
                if col in dataframe.columns:
                    cv = dataframe[col].std() / dataframe[col].mean() if dataframe[col].mean() > 0 else 0
                    performance_metrics[f'{col}_variability'] = {
                        'coefficient_of_variation': float(cv),
                        'risk_level': 'High' if cv > 0.5 else 'Medium' if cv > 0.2 else 'Low'
                    }
            
            metrics['performance_variability'] = performance_metrics
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
        
    def _generate_fallback_strategic_insights_table(self, dataframe, user_prompt, role_context):
        """Generate fallback strategic insights in table format when AI is not available"""
        business_cols = get_business_relevant_columns(dataframe)
        
        # Calculate basic metrics
        key_metrics = []
        if business_cols['numerical']:
            for col in business_cols['numerical'][:4]:
                if col in dataframe.columns:
                    total = dataframe[col].sum()
                    value_type = detect_value_type(dataframe[col])
                    key_metrics.append(f"{col}: {format_value(total, value_type)}")
        
        # Basic performance analysis
        top_performer = "N/A"
        if business_cols['categorical'] and business_cols['numerical']:
            try:
                cat_col = business_cols['categorical'][0]
                metric_col = business_cols['numerical'][0]
                grouped = dataframe.groupby(cat_col)[metric_col].sum().sort_values(ascending=False)
                if len(grouped) > 0:
                    top_performer = grouped.index[0]
            except:
                pass
        
        table_format = f"""
| **Role Context & Strategic Focus** |
|-----------------------------------|
| Role: {role_context.get('role', 'Business Leader')} \\| Focus: Performance Analysis \\| Industry: General |
|-----------------------------------|
| **Key Strategic Metrics** |
| {' \\| '.join(key_metrics[:4]) if key_metrics else 'Metrics analysis in progress'} |
|-----------------------------------|
| **Strategic Situation Analysis** |
| 1. Current performance shows {len(dataframe):,} records across multiple dimensions |
| 2. Top performer identified: {top_performer} |
| 3. Data suggests optimization opportunities in key metrics |
|-----------------------------------|
| **Critical Strategic Insights** |
| 1. Performance variation indicates segmentation opportunities |
| 2. Market concentration suggests competitive advantages |
| 3. Resource allocation can be optimized based on performance gaps |
| 4. Growth potential exists in underperforming segments |
|-----------------------------------|
| **Strategic Recommendations** |
| Priority 1: Analyze top performer best practices for replication |
| Priority 2: Implement performance monitoring dashboard |
| Priority 3: Develop targeted improvement plans for underperformers |
|-----------------------------------|
| **Success Metrics & KPIs** |
| Track: Revenue Growth \\| Target: Cost Efficiency \\| Monitor: Market Share \\| Risk: Performance Variance |
"""
        
        return {
            'type': 'strategic_analysis',
            'role_context': role_context,
            'strategic_table': table_format,
            'prompt': user_prompt,
            'rag_applied': False,
            'kg_applied': False
        }
    
    def _extract_role_context(self, user_prompt):
        """Extract role and context from user prompt"""
        prompt_lower = user_prompt.lower()
        
        role_mapping = {
            'chief marketing officer': 'Chief Marketing Officer',
            'cmo': 'Chief Marketing Officer', 
            'marketing director': 'Marketing Director',
            'head of marketing': 'Head of Marketing',
            'vp marketing': 'VP of Marketing',
            'ceo': 'Chief Executive Officer',
            'chief executive': 'Chief Executive Officer',
            'sales director': 'Sales Director',
            'head of sales': 'Head of Sales',
            'vp sales': 'VP of Sales',
            'operations manager': 'Operations Manager',
            'head of operations': 'Head of Operations',
            'strategy': 'Strategy Executive'
        }
        
        detected_role = 'Business Leader'
        for key, role in role_mapping.items():
            if key in prompt_lower:
                detected_role = role
                break
        
        context_mapping = {
            'Chief Marketing Officer': 'Focused on customer acquisition, brand strategy, and marketing ROI',
            'Marketing Director': 'Responsible for marketing campaigns and lead generation',
            'Chief Executive Officer': 'Overall business strategy and performance',
            'Sales Director': 'Revenue generation and sales team performance',
            'Operations Manager': 'Operational efficiency and process optimization'
        }
        
        return {
            'role': detected_role,
            'context': context_mapping.get(detected_role, 'Executive leadership and strategic decision making')
        }
    
    def _perform_strategic_analysis(self, dataframe, business_cols, user_prompt):
        """Perform strategic-level analysis of the data"""
        analysis = {}
        
        # Key performance indicators
        kpis = {}
        for metric_col in business_cols['numerical'][:5]:
            if metric_col in dataframe.columns:
                kpis[metric_col] = {
                    'total': float(dataframe[metric_col].sum()),
                    'average': float(dataframe[metric_col].mean()),
                    'growth_potential': float(dataframe[metric_col].std() / dataframe[metric_col].mean()) if dataframe[metric_col].mean() != 0 else 0,
                    'top_contributors': self._get_top_contributors(dataframe, metric_col, business_cols['categorical'])
                }
        analysis['kpis'] = kpis
        
        # Market segmentation analysis
        segments = {}
        for cat_col in business_cols['categorical'][:3]:
            if cat_col in dataframe.columns:
                segment_performance = {}
                for metric_col in business_cols['numerical'][:2]:
                    if metric_col in dataframe.columns:
                        grouped = dataframe.groupby(cat_col)[metric_col].agg(['sum', 'mean', 'count'])
                        top_segment = grouped['sum'].idxmax()
                        segment_performance[metric_col] = {
                            'leader': top_segment,
                            'leader_value': float(grouped.loc[top_segment, 'sum']),
                            'segment_count': len(grouped),
                            'concentration': float(grouped['sum'].max() / grouped['sum'].sum())
                        }
                segments[cat_col] = segment_performance
        analysis['segments'] = segments
        
        # Growth opportunities
        opportunities = []
        for metric_col in business_cols['numerical']:
            if metric_col in dataframe.columns:
                # Find underperforming segments
                for cat_col in business_cols['categorical'][:2]:
                    if cat_col in dataframe.columns:
                        grouped = dataframe.groupby(cat_col)[metric_col].mean()
                        avg_performance = grouped.mean()
                        underperformers = grouped[grouped < avg_performance * 0.8]
                        if len(underperformers) > 0:
                            opportunities.append({
                                'type': 'underperforming_segment',
                                'metric': metric_col,
                                'dimension': cat_col,
                                'segments': underperformers.index.tolist(),
                                'potential_improvement': float((avg_performance - underperformers.min()) * len(underperformers))
                            })
        analysis['opportunities'] = opportunities
        
        return analysis
    
    def _get_top_contributors(self, dataframe, metric_col, categorical_cols):
        """Get top contributors for a metric"""
        contributors = {}
        for cat_col in categorical_cols[:2]:
            if cat_col in dataframe.columns:
                top_3 = dataframe.groupby(cat_col)[metric_col].sum().nlargest(3)
                contributors[cat_col] = {
                    'top_performer': top_3.index[0],
                    'top_value': float(top_3.iloc[0]),
                    'share_of_total': float(top_3.iloc[0] / dataframe[metric_col].sum())
                }
        return contributors
    
    def _generate_fallback_strategic_insights(self, dataframe, user_prompt):
        """Generate fallback strategic insights when AI is not available"""
        business_cols = get_business_relevant_columns(dataframe)
        
        # Basic strategic analysis
        key_insights = []
        recommendations = []
        
        if business_cols['numerical']:
            primary_metric = business_cols['numerical'][0]
            total_value = dataframe[primary_metric].sum()
            avg_value = dataframe[primary_metric].mean()
            
            key_insights.append(f"Total {primary_metric}: {format_value(total_value, detect_value_type(dataframe[primary_metric]))}")
            key_insights.append(f"Average {primary_metric}: {format_value(avg_value, detect_value_type(dataframe[primary_metric]))}")
            
            # Find top performers
            if business_cols['categorical']:
                cat_col = business_cols['categorical'][0]
                top_performer = dataframe.groupby(cat_col)[primary_metric].sum().idxmax()
                key_insights.append(f"Top performing {cat_col}: {top_performer}")
                recommendations.append(f"Investigate success factors in {top_performer} and replicate across other {cat_col.lower()}s")
        
        recommendations.extend([
            "Conduct deeper analysis on underperforming segments",
            "Implement performance monitoring dashboard",
            "Review resource allocation based on performance data"
        ])
        
        return {
            'type': 'strategic_analysis',
            'role_context': {'role': 'Business Leader', 'context': 'Strategic decision making'},
            'situation_summary': 'Strategic analysis based on available performance data',
            'key_insights': key_insights,
            'strategic_recommendations': recommendations,
            'immediate_next_steps': recommendations[:2],
            'kpis_to_watch': business_cols['numerical'][:3],
            'risk_assessment': 'Monitor performance trends and competitive positioning',
            'prompt': user_prompt
        }

    def generate_analytical_insights(self, dataframe, user_prompt):
        """Generate analytical insights without creating charts"""
        try:
            if not USE_OPENAI:
                return self._generate_fallback_analytical_insights(dataframe, user_prompt)
            
            # Analyze the data comprehensively
            business_cols = get_business_relevant_columns(dataframe)
            
            # Gather data statistics
            data_analysis = self._perform_comprehensive_analysis(dataframe, business_cols)
            
            # Build context for AI
            analysis_prompt = f"""
            You are an expert data analyst. The user is asking for analytical insights, not visualizations.
            
            User Question: {user_prompt}
            
            Dataset Overview:
            - Total Records: {len(dataframe):,}
            - Columns: {', '.join(business_cols['numerical'] + business_cols['categorical'])}
            
            Key Statistics:
            {json.dumps(data_analysis, indent=2)}
            
            Business Context: {st.session_state.get('business_context', 'General business analysis')}
            
            IMPORTANT INSTRUCTIONS:
            1. Provide specific, actionable insights based on the data
            2. Include actual numbers and percentages from the analysis
            3. Focus on answering the user's specific question
            4. Organize insights as bullet points
            5. Each insight should be 1-2 sentences with concrete data
            6. Suggest specific actions when appropriate
            7. Do NOT suggest creating charts or visualizations
            8. Format numbers properly:
               - For currency: use format like $1,234.56 (with dollar sign and commas)
               - For percentages: use format like 12.3%
               - NEVER use asterisks or markdown formatting
               - Avoid characters that might break display
            
            Return your response as a JSON object:
            {{
                "insights": [
                    "Specific insight with numbers...",
                    "Another actionable insight...",
                    "Pattern or trend discovered..."
                ],
                "key_findings": {{
                    "main_point": "The primary answer to the user's question",
                    "supporting_data": ["data point 1", "data point 2"]
                }},
                "recommendations": [
                    "Specific action to take...",
                    "Another recommendation..."
                ]
            }}
            """
            
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            analysis_result = json.loads(response_text)
            
            # Format the result for display
            return {
                'type': 'analysis',
                'insights': analysis_result.get('insights', []),
                'key_findings': analysis_result.get('key_findings', {}),
                'recommendations': analysis_result.get('recommendations', []),
                'data_summary': data_analysis,
                'prompt': user_prompt
            }
            
        except Exception as e:
            print(f"Error generating analytical insights: {e}")
            return self._generate_fallback_analytical_insights(dataframe, user_prompt)
    
    def _perform_comprehensive_analysis(self, dataframe, business_cols):
        """Perform comprehensive data analysis for insights"""
        analysis = {}
        
        # Analyze numerical columns
        for col in business_cols['numerical'][:5]:  # Limit to top 5
            if col in dataframe.columns:
                analysis[col] = {
                    'mean': float(dataframe[col].mean()),
                    'median': float(dataframe[col].median()),
                    'std': float(dataframe[col].std()),
                    'min': float(dataframe[col].min()),
                    'max': float(dataframe[col].max()),
                    'q25': float(dataframe[col].quantile(0.25)),
                    'q75': float(dataframe[col].quantile(0.75))
                }
                
                # Add CV for variability insight
                if analysis[col]['mean'] != 0:
                    analysis[col]['cv'] = analysis[col]['std'] / analysis[col]['mean']
        
        # Analyze categorical columns
        for col in business_cols['categorical'][:3]:  # Limit to top 3
            if col in dataframe.columns:
                value_counts = dataframe[col].value_counts()
                analysis[f"{col}_distribution"] = {
                    'unique_values': len(value_counts),
                    'top_3': value_counts.head(3).to_dict(),
                    'concentration': float(value_counts.iloc[0] / len(dataframe)) if len(value_counts) > 0 else 0
                }
        
        # Cross-analysis if profit/margin columns exist
        profit_cols = [col for col in business_cols['numerical'] if 'profit' in col.lower() or 'margin' in col.lower()]
        if profit_cols:
            profit_col = profit_cols[0]
            
            # Analyze by categories
            for cat_col in business_cols['categorical'][:2]:
                if cat_col in dataframe.columns:
                    grouped = dataframe.groupby(cat_col)[profit_col].agg(['mean', 'sum', 'count'])
                    top_performers = grouped.nlargest(3, 'mean')
                    bottom_performers = grouped.nsmallest(3, 'mean')
                    
                    analysis[f"{profit_col}_by_{cat_col}"] = {
                        'top_performers': top_performers.to_dict('index'),
                        'bottom_performers': bottom_performers.to_dict('index'),
                        'range': float(grouped['mean'].max() - grouped['mean'].min())
                    }
        
        # Correlation analysis for numerical columns
        if len(business_cols['numerical']) > 1:
            numeric_df = dataframe[business_cols['numerical']].select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                high_correlations = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.5:
                            high_correlations.append({
                                'var1': corr_matrix.columns[i],
                                'var2': corr_matrix.columns[j],
                                'correlation': float(corr_value)
                            })
                
                if high_correlations:
                    analysis['correlations'] = high_correlations
        
        return analysis
    
    def _generate_fallback_analytical_insights(self, dataframe, user_prompt):
        """Generate fallback insights when AI is not available"""
        business_cols = get_business_relevant_columns(dataframe)
        insights = []
        recommendations = []
        
        # Basic profit margin analysis
        profit_cols = [col for col in business_cols['numerical'] if 'profit' in col.lower() or 'margin' in col.lower()]
        
        if profit_cols:
            profit_col = profit_cols[0]
            mean_profit = dataframe[profit_col].mean()
            std_profit = dataframe[profit_col].std()
            
            insights.append(f"Average {profit_col} is {mean_profit:.2f} with standard deviation of {std_profit:.2f}")
            
            if std_profit / mean_profit > 0.5:
                insights.append(f"High variability in {profit_col} (CV: {std_profit/mean_profit:.2%}) indicates inconsistent performance")
                recommendations.append("Focus on standardizing processes to reduce profit variability")
            
            # Find underperformers
            below_avg = dataframe[dataframe[profit_col] < mean_profit]
            insights.append(f"{len(below_avg)/len(dataframe):.1%} of records are below average {profit_col}")
            
            # Analyze by categories
            for cat_col in business_cols['categorical'][:2]:
                if cat_col in dataframe.columns:
                    grouped = dataframe.groupby(cat_col)[profit_col].mean().sort_values(ascending=False)
                    insights.append(f"Top {cat_col} by {profit_col}: {grouped.index[0]} ({grouped.iloc[0]:.2f})")
                    insights.append(f"Bottom {cat_col} by {profit_col}: {grouped.index[-1]} ({grouped.iloc[-1]:.2f})")
                    recommendations.append(f"Investigate why {grouped.index[0]} outperforms {grouped.index[-1]}")
        
        return {
            'type': 'analysis',
            'insights': insights[:5],
            'key_findings': {
                'main_point': 'Analysis based on available data patterns',
                'supporting_data': insights[:2]
            },
            'recommendations': recommendations[:3],
            'prompt': user_prompt
        }
    
    def _generate_fallback_chart(self, user_prompt, dataframe):
        """Generate fallback chart when AI fails"""
        business_cols = self.context_memory['dataset']['business_columns']
        
        if business_cols['numerical'] and business_cols['categorical']:
            x_col = business_cols['categorical'][0]
            y_col = business_cols['numerical'][0]
            
            code = f"""
import plotly.express as px
chart_data = df.groupby('{x_col}')['{y_col}'].sum().reset_index()
fig = px.bar(chart_data, x='{x_col}', y='{y_col}', 
             title='{y_col} by {x_col}')
"""
            
            result = self.code_executor.execute_chart_code(code, dataframe)
            
            if result['success']:
                return {
                    'figure': result['figure'],
                    'chart_data': result.get('chart_data'),
                    'code': code,
                    'title': f"{y_col} by {x_col}",
                    'type': 'bar',
                    'insights': generate_fallback_insights(
                        result.get('chart_data', dataframe.groupby(x_col)[y_col].sum().reset_index()),
                        f"{y_col} by {x_col}",
                        'bar'
                    ),
                    'x_col': x_col,
                    'y_col': y_col,
                    'prompt': user_prompt
                }
        
        return {'error': 'Unable to generate chart'}

# UI Components
class ConversationUI:
    """UI components for conversational interface"""
    
    @staticmethod
    def render_mode_toggle():
        """Render toggle between Dashboard and Chat modes"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            mode = st.radio(
                "Choose Analysis Mode",
                ["📊 Dashboard Mode", "💬 Chat Mode"],
                horizontal=True,
                label_visibility="collapsed"
            )
        return "dashboard" if "Dashboard" in mode else "chat"
    
    @staticmethod
    def render_welcome_section(mode):
        """Render welcome section based on mode"""
        if mode == "dashboard":
            st.markdown("""
            <div class='welcome-section' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; padding: 2rem; margin-bottom: 2rem; 
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
                <div style='color: white; text-align: center;'>
                    <h1 style='margin: 0; font-size: 2.5em;'>📊 AI Dashboard Analysis</h1>
                    <p style='font-size: 1.2em; margin: 0.5rem 0; opacity: 0.9;'>
                        Get instant AI-powered visualizations and insights
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='welcome-section' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; padding: 2rem; margin-bottom: 2rem; 
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
                <div style='color: white; text-align: center;'>
                    <h1 style='margin: 0; font-size: 2.5em;'>💬 AI Chat Assistant</h1>
                    <p style='font-size: 1.2em; margin: 0.5rem 0; opacity: 0.9;'>
                        Have a conversation about your data
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_dashboard_mode(agent, dataframe):
        """Render dashboard mode interface"""
        st.markdown("### 🎯 Quick Analysis")
        
        # Get business columns to create dynamic quick actions
        business_cols = get_business_relevant_columns(dataframe)
        
        # Create dynamic quick actions based on available columns
        quick_actions = []
        
        # If we have numerical columns, create trend analysis
        if business_cols['numerical']:
            primary_metric = business_cols['numerical'][0]
            quick_actions.append(
                ("📈 Trend Analysis", f"Show me {primary_metric} trends and top performers")
            )
            
            # If we have a second metric, add comparison
            if len(business_cols['numerical']) > 1:
                secondary_metric = business_cols['numerical'][1]
                quick_actions.append(
                    ("💰 Metric Analysis", f"Analyze {primary_metric} vs {secondary_metric}")
                )
        
        # If we have categorical columns, add segmentation
        if business_cols['categorical'] and business_cols['numerical']:
            category = business_cols['categorical'][0]
            metric = business_cols['numerical'][0] if business_cols['numerical'] else 'values'
            quick_actions.append(
                ("🎯 Performance Metrics", f"Display {metric} by {category}")
            )
        
        # If we have multiple categories, add comparison
        if len(business_cols['categorical']) > 1 and business_cols['numerical']:
            cat1 = business_cols['categorical'][0]
            cat2 = business_cols['categorical'][1]
            quick_actions.append(
                ("📊 Comparative Analysis", f"Compare {cat1} and {cat2} performance")
            )
        
        # Fallback if no suitable columns found
        if not quick_actions:
            quick_actions = [
                ("📈 Data Overview", "Show me an overview of the data"),
                ("📊 Distribution Analysis", "Analyze data distributions"),
                ("🎯 Summary Stats", "Display summary statistics"),
                ("💡 Key Insights", "Find key insights in the data")
            ]
        
        # Ensure we have exactly 4 actions
        while len(quick_actions) < 4:
            quick_actions.append(
                ("🔍 Explore Data", "Explore patterns in the dataset")
            )
        quick_actions = quick_actions[:4]
        
        cols = st.columns(len(quick_actions))
        for i, (col, (label, prompt)) in enumerate(zip(cols, quick_actions)):
            with col:
                if st.button(label, key=f"quick_dash_{i}", use_container_width=True):
                    ConversationUI._handle_analysis_request(agent, prompt, dataframe)
        
        # Custom analysis input
        st.markdown("---")
        st.markdown("### 🔍 Custom Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_prompt = st.text_input(
                "What would you like to analyze?",
                placeholder="e.g., Show me monthly sales by region",
                key="dashboard_custom_prompt"
            )
        
        with col2:
            if st.button("Generate", type="primary", use_container_width=True, key="dashboard_generate_btn"):
                if custom_prompt:
                    ConversationUI._handle_analysis_request(agent, custom_prompt, dataframe)
                else:
                    st.warning("Please enter an analysis request")
        
        # Display results
        ConversationUI._render_dashboard_results()
    
    @staticmethod
    def render_chat_mode(agent, dataframe):
        """Enhanced chat mode with strategic insights and custom chart functionality"""
        # Initialize chat messages if not exists
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if message.get("type") == "analysis":
                        # Display analytical insights
                        st.markdown("**🧠 Analysis Results:**")
                        
                        # Key findings
                        if message.get("key_findings"):
                            findings = message["key_findings"]
                            st.info(f"**Main Finding:** {findings.get('main_point', '')}")
                        
                        # Insights
                        if message.get("insights"):
                            st.markdown("**📊 Key Insights:**")
                            for insight in message["insights"]:
                                st.markdown(f"• {insight}")
                        
                        # Recommendations
                        if message.get("recommendations"):
                            st.markdown("**💡 Recommendations:**")
                            for rec in message["recommendations"]:
                                st.success(f"→ {rec}")
                    
                    elif message.get("type") == "strategic_analysis":
                        # Display strategic analysis in table format
                        st.markdown("**🎯 Strategic Analysis:**")
                        
                        # Show RAG and KG indicators
                        if message.get('rag_applied') or message.get('kg_applied'):
                            col1, col2 = st.columns(2)
                            if message.get('rag_applied'):
                                with col1:
                                    st.success("🧠 RAG Enhanced")
                            if message.get('kg_applied'):
                                with col2:
                                    st.success("🕸️ Knowledge Graph Applied")
                        
                        # Display the strategic table
                        if message.get("strategic_table"):
                            st.markdown(message["strategic_table"])
                        
                    elif "chart" in message:
                        # Display the chart
                        st.plotly_chart(message["chart"], use_container_width=True, key=f"chat_history_chart_{i}")
                        
                        # Display insights with table format
                        if "insights" in message:
                            st.markdown("**🧠 Insights:**")
                            for insight in message["insights"]:
                                if '|' in insight and 'Key Metrics' in insight:
                                    # Display as markdown table
                                    st.markdown(insight)
                                else:
                                    # Display as bullet point
                                    st.markdown(f"• {insight}")
                        
                        # Display code in an expander
                        if "code" in message and message["code"]:
                            with st.expander("💻 View Code", expanded=False):
                                st.code(message["code"], language='python')
                        
                        # Display confidence if available
                        if "confidence" in message:
                            conf = message["confidence"]
                            st.caption(f"{conf.get('icon', '📊')} Confidence: {conf.get('score', 0.85)*100:.0f}% ({conf.get('level', 'High')})")
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your data..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your request..."):
                    result = agent.generate_chart_with_insights(prompt, dataframe)
                    
                    if 'error' not in result:
                        if result.get('type') == 'analysis':
                            # Display analytical insights
                            st.markdown("**🧠 Analysis Results:**")
                            
                            # Key findings
                            if result.get("key_findings"):
                                findings = result["key_findings"]
                                st.info(f"**Main Finding:** {findings.get('main_point', '')}")
                            
                            # Insights
                            if result.get("insights"):
                                st.markdown("**📊 Key Insights:**")
                                for insight in result["insights"]:
                                    st.markdown(f"• {insight}")
                            
                            # Recommendations
                            if result.get("recommendations"):
                                st.markdown("**💡 Recommendations:**")
                                for rec in result["recommendations"]:
                                    st.success(f"→ {rec}")
                            
                            # Add to chat history
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "type": "analysis",
                                "content": f"Analysis for: {prompt}",
                                "insights": result.get('insights', []),
                                "recommendations": result.get('recommendations', []),
                                "key_findings": result.get('key_findings', {})
                            })
                        
                        elif result.get('type') == 'strategic_analysis':
                            # Display strategic analysis in table format
                            st.markdown("**🎯 Strategic Analysis:**")
                            
                            # Show RAG and KG indicators
                            if result.get('rag_applied') or result.get('kg_applied'):
                                col1, col2 = st.columns(2)
                                if result.get('rag_applied'):
                                    with col1:
                                        st.success("🧠 RAG Enhanced")
                                if result.get('kg_applied'):
                                    with col2:
                                        st.success("🕸️ Knowledge Graph Applied")
                            
                            # Display the strategic table
                            if result.get("strategic_table"):
                                st.markdown(result["strategic_table"])
                            
                            # Add to chat history
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "type": "strategic_analysis",
                                "content": f"Strategic analysis for: {prompt}",
                                "strategic_table": result.get('strategic_table', ''),
                                "rag_applied": result.get('rag_applied', False),
                                "kg_applied": result.get('kg_applied', False)
                            })
                        
                        else:
                            # Display chart
                            if result.get('figure'):
                                st.plotly_chart(result['figure'], use_container_width=True, key=f"chat_new_chart_{len(st.session_state.chat_messages)}")

                            elif result.get('table_data') is not None:
                                st.markdown("**📋 Generated Table:**")
                                st.dataframe(result['table_data'], use_container_width=True)
                            
                            # Display insights with table format
                            if result.get('insights'):
                                st.markdown("**🧠 Insights:**")
                                for insight in result.get('insights', []):
                                    if '|' in insight and 'Key Metrics' in insight:
                                        # Display as markdown table
                                        st.markdown(insight)
                                    else:
                                        # Display as bullet point
                                        st.markdown(f"• {insight}")
                            
                            # Display code in expander
                            if result.get('code'):
                                with st.expander("💻 View Code", expanded=False):
                                    st.code(result['code'], language='python')
                            
                            # Display confidence
                            if result.get('confidence'):
                                conf = result['confidence']
                                st.caption(f"{conf.get('icon', '📊')} Confidence: {conf.get('score', 0.85)*100:.0f}% ({conf.get('level', 'High')})")
                            
                            # Add to chat history
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": f"Generated analysis for: {prompt}",
                                "chart": result.get('figure'),
                                "table": result.get('table_data'),
                                "insights": result.get('insights', []),
                                "code": result.get('code'),
                                "confidence": result.get('confidence')
                            })
                    else:
                        error_msg = f"I encountered an error: {result['error']}"
                        st.error(error_msg)
                        
                        # Provide helpful suggestions
                        st.info("💡 **Tips:**")
                        st.markdown("• Try rephrasing your request")
                        st.markdown("• Be specific about which columns to analyze")
                        st.markdown("• Examples: 'Show sales by region', 'As CMO, what should I focus on?'")
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    @staticmethod
    def _handle_analysis_request(agent, prompt, dataframe):
        """Handle analysis request and store results"""
        with st.spinner("🤖 Generating analysis..."):
            result = agent.generate_chart_with_insights(prompt, dataframe)
            
            if 'error' not in result:
                if 'dashboard_results' not in st.session_state:
                    st.session_state.dashboard_results = []
                st.session_state.dashboard_results.append(result)
                st.success("✅ Analysis completed!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
    
    @staticmethod
    def _render_dashboard_results():
        """Render dashboard results"""
        if 'dashboard_results' not in st.session_state:
            return
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        for i, result in enumerate(st.session_state.dashboard_results):
            with st.container():
                # Header with delete button
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"#### {result.get('title', 'Analysis')}")
                with col2:
                    if st.button("🗑️", key=f"delete_dash_{i}"):
                        st.session_state.dashboard_results.pop(i)
                        st.rerun()
                
                # Chart
                if result.get('figure'):
                    st.plotly_chart(result['figure'], use_container_width=True, key=f"dashboard_result_{i}")
                elif result.get('table_data') is not None:
                    st.dataframe(result['table_data'], use_container_width=True)
                
                # Insights with table format
                with st.expander("🧠 Insights", expanded=True):
                    for insight in result.get('insights', []):
                        if '|' in insight and 'Key Metrics' in insight:
                            # Display as markdown table
                            st.markdown(insight)
                        else:
                            # Display as bullet point
                            st.markdown(f"• {insight}")
                
                # Code
                with st.expander("💻 Code", expanded=False):
                    st.code(result.get('code', ''), language='python')
                
                st.markdown("---")
    
    @staticmethod
    def render_business_context():
        """Render business context section"""
        with st.expander("🏢 Business Context", expanded=False):
            business_context = st.text_area(
                "Provide context about your business:",
                value=st.session_state.get('business_context', ''),
                placeholder="e.g., We're a SaaS company focused on customer retention...",
                height=100
            )
            
            if st.button("💾 Save Context"):
                st.session_state.business_context = business_context
                st.success("✅ Context saved!")

# Custom CSS
def apply_custom_css():
    """Apply custom CSS styling - only to specific components, not the whole app"""
    st.markdown("""
    <style>
    /* Style only specific components */
    .suggestion-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #e6e9ef 100%);
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .suggestion-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        border-color: #4299e1;
    }
    
    .result-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(90deg, #4299e1 0%, #63b3ed 100%);
        color: white;
    }
    
    .assistant-message {
        background: linear-gradient(90deg, #38a169 0%, #48bb78 100%);
        color: white;
    }
    
    /* Dark theme only for charts */
    .js-plotly-plot .plotly .modebar {
        background-color: rgba(0,0,0,0.5) !important;
    }
    
    /* Welcome sections stay colorful */
    .welcome-section {
        color: white;
    }
    
    /* Keep the chart containers with subtle styling */
    .chart-container {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Fix text visibility and sizing */
    .stMarkdown, .stText {
        color: inherit !important;
    }
    
    /* Control markdown heading sizes */
    .stMarkdown h1 {
        font-size: 1.8rem !important;
        line-height: 1.2 !important;
    }
    
    .stMarkdown h2 {
        font-size: 1.5rem !important;
        line-height: 1.3 !important;
    }
    
    .stMarkdown h3 {
        font-size: 1.3rem !important;
        line-height: 1.4 !important;
    }
    
    .stMarkdown h4 {
        font-size: 1.1rem !important;
        line-height: 1.4 !important;
    }
    
    /* Control paragraph and list text size */
    .stMarkdown p {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    
    .stMarkdown li {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    
    /* Table styling for insights */
    .stMarkdown table {
        font-size: 0.9rem !important;
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 1rem 0 !important;
    }
    
    .stMarkdown table th,
    .stMarkdown table td {
        padding: 0.75rem !important;
        border: 1px solid #e2e8f0 !important;
        text-align: left !important;
        vertical-align: top !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    
    .stMarkdown table th {
        background-color: #f7fafc !important;
        font-weight: 600 !important;
    }
    
    .stMarkdown table tr:nth-child(even) {
        background-color: #f9fafb !important;
    }
    
    /* Compact spacing for tables */
    .stMarkdown table strong {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Ensure inputs are visible */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        font-size: 0.95rem !important;
    }
    
    .stTextArea > div > div > textarea {
        background-color: white !important;
        color: black !important;
        font-size: 0.95rem !important;
    }
    
    /* Fix button visibility and sizing */
    .stButton > button {
        background-color: #4299e1 !important;
        color: white !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #3182ce !important;
    }
    
    /* Fix metric visibility and sizing */
    [data-testid="metric-container"] {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 1.2rem !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 0.8rem !important;
    }
    
    /* Chat mode specific styles */
    .stChatMessage {
        background-color: transparent !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: #48bb78 !important;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #4299e1 !important;
    }
    
    /* Success/warning/info message sizing */
    .stSuccess, .stWarning, .stInfo, .stError {
        font-size: 0.9rem !important;
        padding: 0.75rem !important;
        line-height: 1.4 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    
    /* Caption styling */
    .stCaption {
        font-size: 0.8rem !important;
        color: #6b7280 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize the existing agentic_ai for backward compatibility
class AgenticAIAgent:
    """Wrapper for backward compatibility"""
    def __init__(self):
        self.agent = ConversationalAgent()
    
    def create_custom_chart(self, dataframe, user_prompt, business_context=""):
        st.session_state.business_context = business_context
        return self.agent.generate_chart_with_insights(user_prompt, dataframe)

# Global instance for backward compatibility
agentic_ai = AgenticAIAgent()

# Main function that includes everything
def agentic_ai_chart_tab():
    """Main function - includes all dashboard functionality + conversational mode"""
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    if 'conversational_agent' not in st.session_state:
        st.session_state.conversational_agent = ConversationalAgent()
    
    if 'dashboard_results' not in st.session_state:
        st.session_state.dashboard_results = []
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Initialize other required session state variables
    session_vars = [
        'agent_conversations', 'agent_status', 'agentic_charts', 
        'agent_recommendations', 'saved_dashboards', 'custom_charts',
        'data_analysis', 'user_feedback', 'agent_learning',
        'show_key_findings', 'show_next_steps'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == 'agent_status':
                st.session_state[var] = {
                    'data_analyst': 'idle',
                    'chart_creator': 'idle',
                    'insight_generator': 'idle'
                }
            elif var == 'agent_learning':
                st.session_state[var] = {
                    'preferred_charts': [],
                    'avoided_columns': [],
                    'business_context': ""
                }
            elif var in ['show_key_findings', 'show_next_steps']:
                st.session_state[var] = False
            else:
                st.session_state[var] = []
    
    # Check prerequisites
    if not st.session_state.get("agentic_ai_enabled", False):
        st.warning("🔒 Agentic AI is disabled. Enable it in the sidebar.")
        return
    
    if st.session_state.get("dataset") is None:
        st.info("📁 No dataset loaded. Please upload a dataset first.")
        return
    
    agent = st.session_state.conversational_agent
    dataframe = st.session_state.dataset
    
    # Initialize dataset context
    if not agent.dataset_analysis:
        with st.spinner("🧠 Analyzing your dataset..."):
            if not agent.initialize_dataset_context(dataframe):
                st.error("Failed to analyze dataset")
                return
    
    # Mode toggle
    mode = ConversationUI.render_mode_toggle()
    
    # Welcome section
    ConversationUI.render_welcome_section(mode)
    
    # Render based on mode
    if mode == "dashboard":
        ConversationUI.render_dashboard_mode(agent, dataframe)
        
        # Add the original dashboard features
        render_original_dashboard_features(dataframe)
    else:
        ConversationUI.render_chat_mode(agent, dataframe)
    
    # Business context (shown in both modes)
    ConversationUI.render_business_context()
    
    # RAG and Knowledge Graph Debug Panel
    with st.expander("🔍 RAG & Knowledge Graph Debug", expanded=False):
        if AGI_RAG_AVAILABLE and agent.rag_system:
            st.success("✅ RAG System Active")
            
            # Test RAG with current context
            if st.button("🧪 Test RAG Enhancement"):
                test_query = "Analyze profit margins by segment"
                test_context = {
                    'columns': list(dataframe.columns),
                    'row_count': len(dataframe),
                    'business_context': st.session_state.get('business_context', '')
                }
                
                try:
                    enhancement = agent.rag_system.enhance_query(test_query, test_context, "")
                    domain_info = enhancement.get('domain_analysis', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            'domain': domain_info.get('primary_domain', 'unknown'),
                            'confidence': f"{domain_info.get('confidence', 0):.1%}",
                            'business_function': domain_info.get('business_function', 'unknown')
                        })
                    
                    with col2:
                        knowledge = enhancement.get('knowledge_retrieval', {})
                        if knowledge:
                            st.markdown("**Knowledge Retrieved:**")
                            if knowledge.get('best_practices'):
                                st.markdown("• " + "\n• ".join(knowledge['best_practices'][:2]))
                            if knowledge.get('domain_insights'):
                                st.markdown("• " + "\n• ".join(knowledge['domain_insights'][:2]))
                        
                except Exception as e:
                    st.error(f"RAG test failed: {e}")
        else:
            st.warning("❌ RAG System not available")
        
        if AGI_RAG_AVAILABLE and agent.knowledge_graph:
            st.success("✅ Knowledge Graph Active")
            
            if st.button("🕸️ Test Knowledge Graph"):
                try:
                    kg_analysis = agent.knowledge_graph.analyze_dataset(
                        dataframe.head(100),  # Use sample for testing
                        st.session_state.get('business_context', '')
                    )
                    
                    if kg_analysis:
                        st.json({
                            'relationships_found': len(kg_analysis.get('relationships', [])),
                            'patterns_detected': len(kg_analysis.get('patterns', [])),
                            'recommendations': len(kg_analysis.get('recommendations', []))
                        })
                    else:
                        st.info("No knowledge graph analysis available")
                        
                except Exception as e:
                    st.error(f"Knowledge Graph test failed: {e}")
        else:
            st.warning("❌ Knowledge Graph not available")
        
        # Show insight generation capabilities
        st.markdown("### 🧠 Insight Generation Capabilities")
        insight_features = [
            ("✅ RAG-Enhanced Insights", AGI_RAG_AVAILABLE and agent.rag_system),
            ("✅ Knowledge Graph Integration", AGI_RAG_AVAILABLE and agent.knowledge_graph),
            ("✅ Domain-Specific Analysis", True),
            ("✅ Advanced Statistical Metrics", True),
            ("✅ Industry Benchmarking", AGI_RAG_AVAILABLE),
            ("✅ Role-Specific Recommendations", True)
        ]
        
        for feature, available in insight_features:
            if available:
                st.success(feature)
            else:
                st.warning(feature.replace("✅", "❌"))
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state.dashboard_results = []
            st.session_state.chat_messages = []
            st.session_state.conversational_agent = ConversationalAgent()
            st.success("✅ Cleared!")
            st.rerun()
    
    with col2:
        total_analyses = len(st.session_state.dashboard_results) + len([m for m in st.session_state.chat_messages if m.get("role") == "assistant" and "chart" in m])
        st.metric("Total Analyses", total_analyses)
    
    with col3:
        if agent.conversation_history:
            st.metric("Conversations", len(agent.conversation_history))

def render_original_dashboard_features(dataframe):
    """Render original dashboard features like custom charts"""
    st.markdown("---")
    st.markdown("### 🎨 Custom Chart Creation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_prompt = st.text_area(
            "Describe a specific chart you want:",
            placeholder="e.g., Show me profit margins by city below the average",
            height=100,
            key="custom_chart_prompt"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎯 Create Chart", type="primary", use_container_width=True):
            if custom_prompt:
                with st.spinner("Creating custom visualization..."):
                    result = st.session_state.conversational_agent.generate_chart_with_insights(
                        custom_prompt, 
                        dataframe
                    )
                    
                    if 'error' not in result:
                        if 'custom_charts' not in st.session_state:
                            st.session_state.custom_charts = []
                        st.session_state.custom_charts.append(result)
                        st.success("✅ Custom chart created!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['error']}")
            else:
                st.warning("Please describe what chart you want")
    
    # Display custom charts
    if st.session_state.get('custom_charts'):
        st.markdown("---")
        st.markdown("### 🎯 Custom Charts")
        
        for i, chart in enumerate(st.session_state.custom_charts):
            with st.container():
                # Header with delete button
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"#### Custom Chart: {chart.get('prompt', 'Analysis')[:50]}...")
                with col2:
                    if st.button("🗑️", key=f"delete_custom_{i}"):
                        st.session_state.custom_charts.pop(i)
                        st.rerun()
                
                # Chart
                if chart.get('figure'):
                    st.plotly_chart(chart['figure'], use_container_width=True, key=f"custom_chart_{i}")

                elif chart.get('table_data') is not None:
                    st.dataframe(chart['table_data'], use_container_width=True)
                
                # Insights
                with st.expander("🧠 Insights", expanded=True):
                    for insight in chart.get('insights', []):
                        st.markdown(f"• {insight}")
                
                # Code
                with st.expander("💻 Code", expanded=False):
                    st.code(chart.get('code', ''), language='python')

# Add any other required functions for backward compatibility
def render_analysis_planning_section(df):
    """Stub for analysis planning"""
    pass

def render_agentic_save_dashboard_section(supabase):
    """Stub for save dashboard"""
    pass

def render_agentic_executive_summary():
    """Stub for executive summary"""
    pass

# Export all necessary components
__all__ = [
    'agentic_ai_chart_tab',
    'ConversationalAgent',
    'AgenticAIAgent',
    'agentic_ai',
    'render_analysis_planning_section',
    'render_agentic_save_dashboard_section',
    'render_agentic_executive_summary'
]

# Main entry point
if __name__ == "__main__":
    agentic_ai_chart_tab()