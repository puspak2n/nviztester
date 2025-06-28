# main.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
import os
import openai
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from supabase import create_client
import re
from styles import load_custom_css
from chart_utils import render_chart, rule_based_parse, generate_insights, create_styled_chart
from calc_utils import evaluate_calculation, generate_formula_from_prompt, detect_outliers, PREDEFINED_CALCULATIONS, calculate_statistics
from prompt_utils import generate_sample_prompts, generate_prompts_with_llm, prioritize_fields
from utils import classify_columns, load_data, save_dashboard, load_dashboards, save_annotation, load_annotations, delete_dashboard, update_dashboard, load_openai_key, generate_gpt_insight_with_fallback, generate_unique_id, parse_prompt, setup_logging, fetch_dashboard_charts
import streamlit as st
from urllib.parse import urlparse, parse_qs
import hashlib
from agentic_ai_new import agentic_ai_chart_tab
import time
import json
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import io
import zipfile
import importlib.metadata
from ratelimit import limits, sleep_and_retry
from cryptography.fernet import Fernet
from typing import List, Dict, Optional, Any
import secrets

# Set up logging
logger = setup_logging()

def get_field_type(col, field_types):
    """Get the current field type for a column."""
    for field_type, columns in field_types.items():
        if col in columns:
            return field_type.title()
    return "Other"

def save_field_types(project_name, field_types):
    """Save field types to a JSON file in the project directory."""
    try:
        project_dir = f"projects/{project_name}"
        os.makedirs(project_dir, exist_ok=True)
        field_types_file = f"{project_dir}/field_types.json"
        with open(field_types_file, 'w') as f:
            json.dump(field_types, f)
        logger.info(f"Saved field types for project {project_name}: {field_types}")
    except Exception as e:
        logger.error(f"Failed to save field types for project {project_name}: {str(e)}")
        raise

def load_field_types(project_name):
    """Load field types from a JSON file in the project directory."""
    try:
        project_dir = f"projects/{project_name}"
        field_types_file = f"{project_dir}/field_types.json"
        if not os.path.exists(project_dir):
            logger.warning(f"Project directory does not exist: {project_dir}")
            return {}
        if not os.path.exists(field_types_file):
            logger.warning(f"Field types file does not exist: {field_types_file}")
            return {}
        with open(field_types_file, 'r') as f:
            field_types = json.load(f)
        logger.info(f"Loaded field types for project {project_name}: {field_types}")
        return field_types
    except Exception as e:
        logger.error(f"Failed to load field types for project {project_name}: {str(e)}")
        return {}



# Initialize Supabase (use your project URL and anon key)
supabase = create_client("https://fyyvfaqiohdxhnbdqoxu.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ5eXZmYXFpb2hkeGhuYmRxb3h1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc1NTA2MTYsImV4cCI6MjA2MzEyNjYxNn0.-h6sm3bgPzxDjxlmPhi5LNzsbhMJiz8-0HX80U7FiZc")

def handle_auth_callback():
    query_params = parse_qs(urlparse(st.query_params.get("url", [""])[0]).query)
    token_hash = query_params.get("token_hash", [None])[0]
    auth_type = query_params.get("type", [None])[0]
    if token_hash and auth_type == "email":
        try:
            response = supabase.auth.verify_otp({"token_hash": token_hash, "type": "email"})
            st.session_state.user_id = response.user.id
            st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")  # Ensure role is set
            st.success("Email confirmed!")
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Verification failed: {e}")

if "auth/callback" in st.query_params.get("url", [""])[0]:
    handle_auth_callback()




def get_analytics_type(chart):
    if isinstance(chart, dict) and "prompt" in chart:
        chart_prompt = chart["prompt"].lower()
    elif isinstance(chart, str):
        chart_prompt = chart.lower()
    else:
        return "Other"

    if any(word in chart_prompt for word in ["sales", "revenue", "income"]):
        return "Sales"
    elif any(word in chart_prompt for word in ["customer", "user", "client"]):
        return "Customer"
    elif any(word in chart_prompt for word in ["product", "item", "sku"]):
        return "Product"
    return "Other"

def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        "chart_history": [],
        "field_types": {},
        "dataset": None,
        "current_project": None,
        "sidebar_collapsed": False,
        "sort_order": {},
        "insights_cache": {},  # Cache for insights
        "chart_cache": {},    # Cache for chart render results
        "sample_prompts": [],
        "used_sample_prompts": [],
        "sample_prompt_pool": [],
        "last_used_pool_index": 0,
        "onboarding_seen": False,
        "classified": False,
        "last_manual_prompt": None,
        "chart_dimensions": {},
        "refresh_dashboards": False,
        "dashboard_order": [],
        "data_loaded": False,
        "loading_progress": 0,
        "last_data_update": None,
        "dataset_hash": None  # Store dataset hash for cache invalidation
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "current_project" not in st.session_state:
        st.session_state.current_project = "my_project"
# Rate limit configuration (e.g., 10 saves per minute)
SAVE_CALLS = 10
SAVE_PERIOD = 60

# Initialize encryption
ENCRYPTION_KEY = Fernet.generate_key()
cipher = Fernet(ENCRYPTION_KEY)

def handle_url_auth():
    """Check for auth tokens in URL parameters and set session state"""
    try:
        # Get URL parameters
        query_params = st.query_params
        
        if 'access_token' in query_params:
            access_token = query_params['access_token']
            refresh_token = query_params.get('refresh_token', '')
            user_id = query_params.get('user_id', '')
            user_email = query_params.get('user_email', '')
            
            # Set session with the tokens
            supabase.auth.set_session(access_token, refresh_token)
            
            # Verify the session is valid
            user = supabase.auth.get_user()
            if user and user.user:
                st.session_state.user_id = user.user.id
                st.session_state.user_email = user_email or user.user.email
                st.session_state.user_role = user.user.user_metadata.get("role", "Viewer")
                st.session_state.access_token = access_token
                st.session_state.refresh_token = refresh_token
                
                # Clear URL parameters for security
                st.query_params.clear()
                
                logger.info(f"User authenticated via URL: {st.session_state.user_id}")
                return True
    except Exception as e:
        logger.error(f"URL authentication failed: {str(e)}")
    
    return False

# Check URL auth before anything else
if 'user_id' not in st.session_state:
    handle_url_auth()
    
# Set seaborn style globally for consistent beautiful charts
def setup_seaborn_style():
    """Setup beautiful seaborn styling for all charts."""
    sns.set_style("whitegrid")  # Clean grid background
    sns.set_palette("husl")     # Beautiful color palette
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

# Call this once in your app initialization
setup_seaborn_style()

def create_seaborn_chart(df, chart_type, x_col, y_col=None, color_col=None, size_col=None, title="Chart"):
    """
    Universal seaborn chart creator - replaces all your pyplot charts.
    
    Args:
        df: DataFrame
        chart_type: 'bar', 'line', 'scatter', 'hist', 'box', 'violin', 'heatmap', 'count'
        x_col: X-axis column
        y_col: Y-axis column (optional for some charts)
        color_col: Column for color coding
        size_col: Column for size coding (scatter only)
        title: Chart title
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    try:
        if chart_type == 'bar':
            # Beautiful bar chart
            if y_col:
                # Aggregated bar chart
                plot_data = df.groupby(x_col)[y_col].sum().reset_index()
                sns.barplot(data=plot_data, x=x_col, y=y_col, hue=color_col, ax=ax)
            else:
                # Count bar chart
                sns.countplot(data=df, x=x_col, hue=color_col, ax=ax)
            
            # Rotate labels if needed
            if len(str(df[x_col].iloc[0])) > 8:
                plt.xticks(rotation=45, ha='right')
        
        elif chart_type == 'line':
            # Beautiful line chart
            sns.lineplot(data=df, x=x_col, y=y_col, hue=color_col, marker='o', ax=ax)
            
        elif chart_type == 'scatter':
            # Beautiful scatter plot
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=color_col, 
                          size=size_col, alpha=0.7, ax=ax)
            
        elif chart_type == 'hist':
            # Beautiful histogram
            if color_col:
                # Multiple histograms by category
                for category in df[color_col].unique():
                    subset = df[df[color_col] == category]
                    sns.histplot(subset[x_col], alpha=0.7, label=category, ax=ax)
                ax.legend()
            else:
                sns.histplot(df[x_col], kde=True, ax=ax)
            
        elif chart_type == 'box':
            # Beautiful box plot
            if y_col:
                sns.boxplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
            else:
                sns.boxplot(y=df[x_col], ax=ax)
                
        elif chart_type == 'violin':
            # Beautiful violin plot
            if y_col:
                sns.violinplot(data=df, x=x_col, y=y_col, hue=color_col, ax=ax)
            else:
                sns.violinplot(y=df[x_col], ax=ax)
                
        elif chart_type == 'heatmap':
            # Beautiful heatmap
            if y_col:
                # Pivot table heatmap
                pivot_data = df.pivot_table(values=y_col, index=x_col, columns=color_col, aggfunc='mean')
                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', ax=ax)
            else:
                # Correlation heatmap
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                
        elif chart_type == 'count':
            # Beautiful count plot
            sns.countplot(data=df, x=x_col, hue=color_col, ax=ax)
            plt.xticks(rotation=45, ha='right')
            
        elif chart_type == 'area':
            # Beautiful area chart
            if color_col:
                # Stacked area by category
                pivot_data = df.pivot_table(values=y_col, index=x_col, columns=color_col, aggfunc='sum').fillna(0)
                ax.stackplot(pivot_data.index, *[pivot_data[col] for col in pivot_data.columns], 
                           labels=pivot_data.columns, alpha=0.8)
                ax.legend(loc='upper left')
            else:
                ax.fill_between(df[x_col], df[y_col], alpha=0.7)
                
        # Universal styling
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        if y_col:
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        # Clean up the plot
        sns.despine()  # Remove top and right spines
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Fallback to simple plot
        ax.text(0.5, 0.5, f"Chart Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

def validate_session_security(supabase) -> bool:
    """Validate session authenticity and security"""
    try:
        session = supabase.auth.get_session()
        if not session or not session.access_token:
            return False
        
        # Verify token hasn't expired
        user = supabase.auth.get_user()
        if not user or not user.user:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Session validation failed: {str(e)}")
        return False

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not isinstance(input_str, str):
        return ""
    sanitized = ''.join(c for c in input_str if c.isalnum() or c in ' -_.')
    return sanitized[:100]

def convert_timestamps_to_strings(obj):
    """Recursively convert pandas Timestamps to ISO format strings for JSON serialization"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_timestamps_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_strings(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # Handle other datetime-like objects
        return obj.isoformat()
    else:
        return obj



def generate_pdf_report(charts, title, output_filename):
    """
    Generate a PDF report from charts using LaTeX.
    Args:
        charts (list): List of tuples (prompt, chart_type, chart_data, insights, fig)
        title (str): Report title
        output_filename (str): Output PDF filename
    Returns:
        bytes: PDF file content
    """
    logger.debug(f"Generating PDF report: {title}")

    # LaTeX preamble
    latex_content = r"""
\documentclass[a4paper,12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{DejaVuSans}
\geometry{margin=1in}
\definecolor{darkbg}{HTML}{1F2A44}
\definecolor{lighttext}{HTML}{FFFFFF}
\pagecolor{darkbg}
\color{lighttext}
\begin{document}
    """

    # Add title
    latex_content += f"""
\\textbf{{\\Large {title}}}\\\\
\\vspace{{0.5cm}}
Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\\\
\\vspace{{1cm}}
    """

    # Temporary directory for images
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (prompt, chart_type, chart_data, insights, fig) in enumerate(charts):
            # Save chart as PNG
            if fig is not None:
                image_path = os.path.join(tmpdir, f"chart_{idx}.png")
                try:
                    pio.write_image(fig, image_path, format="png")
                    logger.debug(f"Saved chart image: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to save chart image {idx}: {str(e)}")
                    continue

                # Add chart to LaTeX
                latex_content += f"""
\\section*{{Chart {idx + 1}: {prompt} ({chart_type})}}
\\includegraphics[width=\\textwidth]{{{image_path}}}
                """
            else:
                latex_content += f"""
\\section*{{Chart {idx + 1}: {prompt} ({chart_type})}}
                """

            # Add insights
            latex_content += r"""
\begin{itemize}
    """
            if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                for insight in insights:
                    latex_content += f"\\item {insight}\n"
            else:
                latex_content += "\\item No insights available.\n"
            latex_content += r"""
\end{itemize}
            """

            # Add basic statistics if available
            if not chart_data.empty and metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                stats = calculate_statistics(working_df, metric)
                if stats:
                    latex_content += r"""
\begin{tabular}{|l|c|}
\hline
\textbf{Statistic} & \textbf{Value} \\
\hline
Mean & """ + f"{stats['mean']:.2f}" + r""" \\
Median & """ + f"{stats['median']:.2f}" + r""" \\
Min & """ + f"{stats['min']:.2f}" + r""" \\
Max & """ + f"{stats['max']:.2f}" + r""" \\
\hline
\end{tabular}
                    """

            latex_content += r"""
\vspace{0.5cm}
            """

        # Close LaTeX document
        latex_content += r"""
\end{document}
        """

        # Write LaTeX file
        tex_path = os.path.join(tmpdir, "report.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)

        # Compile LaTeX to PDF
        try:
            subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True
            )
            #logger.info(f"PDF generated: {output_filename}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile LaTeX: {e.stderr}")
            st.error("Failed to generate PDF report. Please check logs.")
            return None

        # Read PDF
        pdf_path = os.path.join(tmpdir, "report.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            return pdf_content
        else:
            logger.error("PDF file not found after compilation")
            st.error("Failed to generate PDF report.")
            return None

def compute_dataset_hash(df):
    """Compute a hash of the DataFrame to detect changes."""
    if df is None:
        return None
    try:
        # Use columns and a sample of data to create a hash
        columns_str = ''.join(sorted(df.columns))
        sample_data = df.head(100).to_csv(index=False)  # Sample to avoid memory issues
        hash_input = columns_str + sample_data
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to compute dataset hash: {str(e)}")
        return None



# Load Custom CSS and Override
load_custom_css()
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    .stApp > div > div {
        min-height: 0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        color: white !important;
        width: 320px !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem !important;
        width: 100% !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #475569 !important;
    }
    [data-testid="stSidebar"] .stSelectbox, .stTextInput, .stExpander, .stInfo {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox>label, .stTextInput>label, .stExpander>label {
        color: white !important;
    }
    [data-testid="stSidebar"] .stExpander div[role="button"] p {
        color: white !important;
    }
    [data-testid="stSidebar"] .stInfo {
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
    }
    [data-testid="stSidebar"] .stInfo div {
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div {
        color: black !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-testid="stSelectbox"] div[role="option"] {
        color: black !important;
        background-color: white !important;
    }
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
        text-align: left;
    }
    .styled-table thead tr {
        background-color: #334155;
        color: white;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #334155;
    }
    [data-testid="stSidebar"] .saved-dashboard {
        color: black !important;
    }
    .saved-dashboard {
        color: black !important;
    }
    .sort-button {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        border: none !important;
        cursor: pointer !important;
        font-size: 0.9em !important;
    }
    .sort-button:hover {
        background-color: #475569 !important;
    }
    .main [data-testid="stExpander"] {
        background-color: #F5F7FA !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] {
        background-color: #334155 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        cursor: pointer !important;
    }
    .main [data-testid="stExpander"] > div[role="button"] p {
        color: white !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .main [data-testid="stExpander"] > div[role="button"]:hover {
        background-color: #475569 !important;
    }
    /* New styles for Saved Dashboards tab */
    .dashboard-controls {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
    }
    .reorder-button {
        background-color: #26A69A !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem !important;
        border: none !important;
        cursor: pointer !important;
    }
    .reorder-button:hover {
        background-color: #2E7D32 !important;
    }
    .annotation-input {
        background-color: #ECEFF1 !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        border: 1px solid #B0BEC5 !important;
    }
</style>
""", unsafe_allow_html=True)


# In main.py
def load_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None


# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Session State Init
def initialize_session_state():
    """Initialize all session state variables in one place."""
    defaults = {
        "chart_history": [],
        "field_types": {},
        "dataset": None,
        "current_project": None,
        "sidebar_collapsed": False,
        "sort_order": {},
        "insights_cache": {},
        "sample_prompts": [],
        "used_sample_prompts": [],
        "sample_prompt_pool": [],
        "last_used_pool_index": 0,
        "onboarding_seen": False,
        "classified": False,
        "last_manual_prompt": None,
        "chart_dimensions": {},
        "chart_cache": {},
        "refresh_dashboards": False,
        "dashboard_order": [],
        "data_loaded": False,
        "loading_progress": 0,
        "last_data_update": None,
        "proactive_insights": [],  # New: Store proactive insights
        "user_preferences": {}  # New: Store user preferences
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None

# Initialize session state at startup
initialize_session_state()





# Preprocess Dates (unchanged)
def preprocess_dates(df):
    """Forcefully preprocess date columns with format detection and consistent parsing."""
    # Common date formats to try
    date_formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
        '%m-%d-%Y', '%d-%m-%Y', '%b %d %Y', '%B %d %Y',
        '%d %b %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'
    ]
    
    for col in df.columns:
        if 'date' in col.lower() or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                # First try to detect the format from a sample of non-null values
                sample = df[col].dropna().head(50)  # Increased sample size for better format detection
                if len(sample) > 0:
                    detected_format = None
                    valid_count = 0
                    best_format = None
                    
                    # Try each format and count valid dates
                    for fmt in date_formats:
                        try:
                            parsed = pd.to_datetime(sample, format=fmt)
                            valid = parsed.notna().sum()
                            if valid > valid_count:
                                valid_count = valid
                                best_format = fmt
                        except:
                            continue
                    
                    if best_format:
                        # Use the best detected format for parsing
                        parsed_col = pd.to_datetime(df[col], format=best_format, errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        #logger.info(f"Parsed column '{col}' as datetime using format '{best_format}' with {valid_ratio:.1%} valid dates")
                        
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            #logger.info(f"Converted column '{col}' to datetime")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                    else:
                        # If no format detected, try parsing without format
                        parsed_col = pd.to_datetime(df[col], errors='coerce')
                        valid_ratio = parsed_col.notna().mean()
                        if valid_ratio > 0.5:  # Lowered threshold to 50% valid dates
                            df[col] = parsed_col
                            #logger.info(f"Converted column '{col}' to datetime using automatic format detection with {valid_ratio:.1%} valid dates")
                        else:
                            logger.warning(f"Column '{col}' has too many invalid dates ({valid_ratio:.1%} valid), skipping conversion")
                else:
                    logger.warning(f"Column '{col}' has no non-null values to detect date format")
            except Exception as e:
                logger.warning(f"Failed to parse date column '{col}': {str(e)}")
    return df

# Initialize session state
if 'current_project' not in st.session_state:
    st.session_state.current_project = "my_project"
    os.makedirs(f"projects/{st.session_state.current_project}", exist_ok=True)
if 'chart_history' not in st.session_state:
    st.session_state.chart_history = []
if 'current_dashboard' not in st.session_state:
    st.session_state.current_dashboard = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'classified' not in st.session_state:
    st.session_state.classified = False
if 'field_types' not in st.session_state:
    st.session_state.field_types = {}
if 'sample_prompts' not in st.session_state:
    st.session_state.sample_prompts = []
if 'used_sample_prompts' not in st.session_state:
    st.session_state.used_sample_prompts = []
if 'sample_prompt_pool' not in st.session_state:
    st.session_state.sample_prompt_pool = []
if 'last_used_pool_index' not in st.session_state:
    st.session_state.last_used_pool_index = 0
# Initialize session state variables if they don't exist
if 'agentic_ai_enabled' not in st.session_state:
    st.session_state.agentic_ai_enabled = False

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

if 'strict_agentic_mode' not in st.session_state:
    st.session_state.strict_agentic_mode = False

# Add the user role initialization if needed
if 'user_role' not in st.session_state:
    st.session_state.user_role = "Basic"  # Set a default role

# Check authentication state on page load
async def check_auth():
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.user_id = session.user.id
            st.session_state.user_email = session.user.email
            st.session_state.user_role = session.user.user_metadata.get("role", "Viewer")
            return True
    except:
        pass
    return False

# Run auth check if not already logged in
if 'user_id' not in st.session_state:
    import asyncio
    asyncio.run(check_auth())

# Sidebar
with st.sidebar:
    # Display logo at the top
    st.image("logo.png", width=300, use_container_width=False)

    # Custom CSS for yellow header and layout with updated button colors
    st.markdown("""
        <style>
        .sidebar .yellow-header {
            background-color: #FFFF00;
            padding: 10px;
            margin-bottom: 5px;
            text-align: center;
            font-weight: bold;
            color: black; /* Ensure header text is visible */
        }
        .sidebar .button-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .sidebar .input-field {
            width: 100%;
            margin-bottom: 5px;
        }
        .sidebar .login-button, .sidebar .signup-button {
            background-color: #3B82F6 !important; /* New blue color */
            color: white !important; /* Ensure button text is white */
            border-radius: 0.5rem !important;
            padding: 0.75rem !important;
            width: 100% !important;
            text-align: center !important;
        }
        .sidebar .login-button:hover, .sidebar .signup-button:hover {
            background-color: #2563EB !important; /* Darker blue on hover */
        }
        .sidebar .logged-in-text {
            color: black !important; /* Change Username text to black */
            margin-bottom: 5px;
        }
        .sidebar .project-item {
            padding: 5px 10px;
            margin-bottom: 2px;
            color: white; /* Ensure project items are visible */
        }
        .sidebar [data-testid="stMarkdownContainer"] strong {
            color: black !important; /* Change Login text to black */
        }
        .sidebar .yellow-header.projects {
            color: white !important; /* Change Projects text to white */
        }
        </style>
    """, unsafe_allow_html=True)

    # Replace the Analytics Mode section in the sidebar with this simplified version:

    st.markdown("### Analytics Mode")

    # For development/testing, bypass role check
    TESTING_MODE = True  # Set to False for production

    if TESTING_MODE or st.session_state.get("user_role") in ["Pro", "Enterprise"]:
        # Main AI toggle only
        st.session_state.agentic_ai_enabled = st.toggle(
            "Enable Agentic AI (Advanced Analytics)",
            value=st.session_state.get("agentic_ai_enabled", False),
            help="Toggle to enable advanced AI-powered analytics with OpenAI integration.",
            key="agentic_ai_toggle"
        )
        
        if st.session_state.agentic_ai_enabled:
            st.info("🤖 Agentic AI enabled! Advanced insights are now available.")
        else:
            st.info("📊 Using Smart Analytics (basic mode).")
    else:
        st.session_state.agentic_ai_enabled = False
        st.info("🔒 Agentic AI is a Pro/Enterprise feature. Upgrade to access.")
        
        # Add upgrade button for convenience
        if st.button("🚀 Upgrade to Pro", key="upgrade_btn"):
            st.info("Contact admin to upgrade your account.")

    # Keep the debug info for testing
    if st.sidebar.checkbox("🔧 Developer Mode", key="dev_mode"):
        test_role = st.sidebar.selectbox(
            "Test User Role",
            ["Viewer", "Pro", "Enterprise"],
            index=1  # Default to Pro
        )
        st.session_state.user_role = test_role
        st.sidebar.success(f"Testing as: {test_role}")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Debug: Current role = {st.session_state.get('user_role', 'Not set')}")
    st.sidebar.caption(f"Debug: User ID = {st.session_state.get('user_id', 'Not set')}")

    use_advanced_rag = st.checkbox(
        "🧠 Enable Advanced RAG Enhancement",
        value=True,
        help="Uses enterprise-grade knowledge base and graph analysis"
    )


    # Login Section
    st.markdown('<div class="yellow-header">Login</div>', unsafe_allow_html=True)
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("email", key="login_email", help="Enter your email address")
            password = st.text_input("Password", type="password", key="login_password", help="Enter your password")
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("Login")
            with col2:
                signup_button = st.form_submit_button("Sign up")
            
            if login_button:
                try:
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state.user_id = response.user.id
                    st.session_state.user_email = email  # Store email
                    st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                    supabase.auth.set_session(response.session.access_token, response.session.refresh_token)
                    st.session_state.access_token = response.session.access_token
                    st.session_state.refresh_token = response.session.refresh_token
                    session = supabase.auth.get_session()
                    current_user = supabase.auth.get_user()
                    logger.info(f"User logged in: {st.session_state.user_id}, Role: {st.session_state.user_role}")
                    st.success("Logged in!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Login failed: {str(e)}")
                    st.error(f"Login failed: {str(e)}")

            if signup_button:
                try:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                        "options": {"data": {"role": "Viewer"}}
                    })
                    st.session_state.user_id = response.user.id
                    st.session_state.user_email = email  # Store email
                    st.session_state.user_role = response.user.user_metadata.get("role", "Viewer")
                    logger.info(f"User signed up: {st.session_state.user_id}, Role: {st.session_state.user_role}")
                    st.success("Signed up! Check your email to confirm.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Sign-up failed: {e}")
                    st.error(f"Sign-up failed: {e}")
    else:
        # Display user profile
        st.markdown(f"""
        <div style="background-color: #334155; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            width: 35px; height: 35px; border-radius: 50%; 
                            display: flex; align-items: center; justify-content: center; 
                            font-weight: bold; color: white;">
                    {st.session_state.get('user_email', 'U')[0].upper()}
                </div>
                <div>
                    <div style="color: white; font-size: 0.9rem; font-weight: 500;">
                        {st.session_state.get('user_email', 'User')}
                    </div>
                    <div style="color: #a0aec0; font-size: 0.75rem;">
                        ID: {st.session_state.user_id[:8]}...
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            supabase.auth.sign_out()
            st.session_state.user_id = None
            st.session_state.user_email = None
            st.session_state.user_role = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            logger.info("User logged out")
            st.success("Logged out!")
            st.rerun()

    # Projects section
    st.markdown('<div class="yellow-header projects">\U0001F4C2 Projects</div>', unsafe_allow_html=True)
    
    # Demo projects (always visible, read-only)
    with st.expander("\U0001F4CA Demo Projects", expanded=True):  # 📊
        demo_projects = {}
        demo_dirs = ["marketing_demo", "sales_demo"]
        
        for project_dir in demo_dirs:
            project_path = os.path.join("projects", project_dir)
            if os.path.exists(project_path):
                # Get all CSV files in the project directory
                csv_files = [f for f in os.listdir(project_path) if f.endswith('.csv')]
                if csv_files:
                    demo_projects[project_dir] = {
                        "name": project_dir.replace('_', ' ').title(),
                        "dashboards": [f.replace('.csv', '') for f in csv_files]
                    }
        
        for project_id, project_info in demo_projects.items():
            if st.button(f"\U0001F4CA {project_info['name']}", key=f"demo_{project_id}", use_container_width=True):  # 📊
                st.session_state.current_project = project_id
                st.session_state.is_demo_project = True
                # Load the dataset when project is selected
                try:
                    if os.path.exists(f"projects/{project_id}/dataset.csv"):
                        df = pd.read_csv(f"projects/{project_id}/dataset.csv")
                        df = preprocess_dates(df)
                        st.session_state.dataset = df
                        # Load saved field types
                        saved_field_types = load_field_types(project_id)
                        if saved_field_types:
                            st.session_state.field_types = saved_field_types
                            st.session_state.classified = True
                        else:
                            st.session_state.classified = False
                        st.session_state.sample_prompts = []
                        st.session_state.used_sample_prompts = []
                        st.session_state.sample_prompt_pool = []
                        st.session_state.last_used_pool_index = 0
                        st.success(f"Opened: {project_id}")
                except Exception as e:
                    st.error(f"Failed to load dataset for project {project_id}: {e}")
                st.rerun()        
        
        # List saved dashboards
        dashboards_dir = f"projects/{st.session_state.current_project}/dashboards"
        if os.path.exists(dashboards_dir):
            for dashboard_file in os.listdir(dashboards_dir):
                if dashboard_file.endswith('.json'):
                    try:
                        with open(os.path.join(dashboards_dir, dashboard_file), 'r') as f:
                            dashboard_data = json.load(f)
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                if st.button(f"📈 {dashboard_data['name']}", key=f"dash_{dashboard_file}", use_container_width=True):
                                    st.session_state.current_dashboard = dashboard_data['id']
                                    st.session_state.chart_history = dashboard_data['charts']
                                    st.rerun()
                            with col2:
                                if st.button("🗑️", key=f"delete_{dashboard_file}"):
                                    try:
                                        os.remove(os.path.join(dashboards_dir, dashboard_file))
                                        if st.session_state.current_dashboard == dashboard_data['id']:
                                            st.session_state.current_dashboard = None
                                            st.session_state.chart_history = []
                                        st.success(f"Deleted dashboard: {dashboard_data['name']}")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to delete dashboard: {e}")
                    except Exception as e:
                        st.error(f"Failed to load dashboard: {e}")
    
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **NarraViz AI** is an AI-powered business intelligence platform that transforms data into actionable insights using natural language. Ask questions, visualize data, and uncover trends effortlessly.
        """)


# Main Content
if st.session_state.current_project:
    st.caption(f"---")

    st.caption(f"Active Project: **{st.session_state.current_project}**")
    if st.session_state.get('is_demo_project'):
        st.info("You are viewing a demo project. Create your own analytics in My Project.")
else:
    pass


# Onboarding Modal (unchanged)
if not st.session_state.onboarding_seen:
    with st.container():
        st.markdown("""
        <div style='background-color: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 0;'>
            <h2>Welcome to NarraVIZ AI! 🎉</h2>
            <p>Transform your data into insights with our AI-powered BI platform. Here's how to get started:</p>
            <ul>
                <li>📂 Create or open a project in the sidebar.</li>
                <li>📊 Upload a CSV or connect to a database.</li>
                <li>💬 Ask questions like "Top 5 Cities by Sales" in the prompt box.</li>
                <li>📈 Explore charts and AI-generated insights.</li>
            </ul>
            <p>Ready to dive in?</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Got it! Let's start.", key="onboarding_close"):
            st.session_state.onboarding_seen = True
            #logger.info("User completed onboarding")

# Save Dataset Changes (unchanged)
def save_dataset_changes():
    if st.session_state.current_project and st.session_state.dataset is not None:
        try:
            st.session_state.dataset.to_csv(f"projects/{st.session_state.current_project}/dataset.csv", index=False)
            # Also save field types
            save_field_types(st.session_state.current_project, st.session_state.field_types)
            #logger.info("Saved dataset and field types for project: %s", st.session_state.current_project)
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            logger.error("Failed to save dataset for project %s: %s", st.session_state.current_project, str(e))

def generate_pdf_summary(summary_points, overall_analysis):
    """
    Generate a PDF summary from summary points and overall analysis.
    Args:
        summary_points (list): List of summary points
        overall_analysis (list): List of overall analysis points
    Returns:
        bytes: PDF content as bytes
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Executive Summary Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Summary of Dashboard Analysis
        story.append(Paragraph("Summary of Dashboard Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in summary_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Overall Data Analysis
        story.append(PageBreak())
        story.append(Paragraph("Overall Data Analysis and Findings", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in overall_analysis:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        #logger.info("Generated PDF summary successfully")
        return pdf_content
    except Exception as e:
        logger.error(f"Failed to generate PDF summary: {str(e)}", exc_info=True)
        raise

# Generate GPT Insights (unchanged)
def generate_gpt_insights(stats, metric, prompt, chart_data, dimension=None, second_metric=None):
    """Generate insights using GPT-3.5-turbo."""
    if not USE_OPENAI:
        return []

    try:
        # Prepare the data summary
        data_summary = {
            "metric": metric,
            "dimension": dimension,
            "second_metric": second_metric,
            "stats": stats,
            "prompt": prompt,
            "data_points": len(chart_data)
        }

        # Create the prompt for GPT
        gpt_prompt = (
            f"Analyze this data visualization and provide 3 concise, insightful observations:\n"
            f"Metric: {data_summary['metric']}\n"
            f"Dimension: {data_summary['dimension']}\n"
            f"Statistics: Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, "
            f"Min={stats['min']:.2f}, Max={stats['max']:.2f}\n"
            f"Number of data points: {data_summary['data_points']}\n"
            f"Original prompt: {prompt}\n"
            f"Provide 3 specific, data-driven insights that would be valuable for business users."
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise, actionable insights from data visualizations."},
                {"role": "user", "content": gpt_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract insights from the response
        insights = [line.strip('- ').strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        return insights[:3]  # Return top 3 insights

    except Exception as e:
        logger.error(f"Failed to generate GPT insights: {str(e)}")
        return []
 

@sleep_and_retry
@limits(calls=SAVE_CALLS, period=SAVE_PERIOD)
def save_dashboard_secure(supabase, session_state, dashboard_name, project_id, tags=None):
    """Enterprise-grade dashboard saving with secure data embedding"""
    try:
        # Validate inputs
        if not dashboard_name or not str(dashboard_name).strip():
            raise ValueError("Dashboard name is required")
        if not project_id or not str(project_id).strip():
            raise ValueError("Project ID is required")
        if session_state.get("dataset") is None:
            raise ValueError("No dataset available to save with dashboard")
        if not session_state.get("user_id"):
            raise ValueError("User must be logged in to save dashboards")

        # Ensure string values
        dashboard_name = str(dashboard_name).strip()
        project_id = str(project_id).strip()

        # Create project directory
        project_dir = f"projects/{project_id}"
        os.makedirs(project_dir, exist_ok=True)

        # Get chart configurations
        chart_configs = []
        recommendations = session_state.get("recommendations", [])
        custom_charts = session_state.get("custom_charts", [])

        # Add active recommendation charts
        for idx, chart_tuple in enumerate(recommendations):
            if not session_state.get(f"delete_chart_{idx}", False):
                if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                    prompt, chart_type = chart_tuple[0], chart_tuple[1]
                    if prompt and isinstance(prompt, str) and prompt.strip():
                        chart_configs.append({
                            "prompt": prompt.strip(),
                            "chart_type": str(chart_type) if chart_type else "Bar"
                        })

        # Add active custom charts
        base_idx = len(recommendations)
        for custom_idx, chart_tuple in enumerate(custom_charts):
            idx = base_idx + custom_idx
            if not session_state.get(f"delete_chart_{idx}", False):
                if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                    prompt, chart_type = chart_tuple[0], chart_tuple[1]
                    if prompt and isinstance(prompt, str) and prompt.strip():
                        chart_configs.append({
                            "prompt": prompt.strip(),
                            "chart_type": str(chart_type) if chart_type else "Bar"
                        })

        if not chart_configs:
            raise ValueError("No valid charts available to save")

        df = session_state["dataset"]

        # Save dataset to project directory
        dataset_path = f"{project_dir}/dataset.csv"
        df.to_csv(dataset_path, index=False)
        logger.info(f"Saved dataset to {dataset_path}")

        # Create data snapshot with proper type conversion
        data_snapshot = {
            "version": "2.0",
            "saved_at": datetime.utcnow().isoformat(),
            "dataset_metadata": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": int(df.memory_usage(deep=True).sum()),
                "hash": compute_dataset_hash(df)
            }
        }

        # Prepare dataset for JSON serialization
        if len(df) <= 10000:
            df_serializable = df.copy()
            for col in df_serializable.columns:
                if pd.api.types.is_datetime64_any_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.api.types.is_categorical_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].astype(str)
                elif pd.api.types.is_integer_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].apply(lambda x: int(x) if pd.notna(x) else None)
                elif pd.api.types.is_float_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].apply(lambda x: float(x) if pd.notna(x) else None)
                elif pd.api.types.is_bool_dtype(df_serializable[col]):
                    df_serializable[col] = df_serializable[col].astype(bool)
                else:
                    df_serializable[col] = df_serializable[col].astype(str).replace('', None)
            
            records = convert_timestamps_to_strings(df_serializable.to_dict('records'))
            data_snapshot["full_dataset"] = records
            data_snapshot["is_sample"] = False
            logger.info(f"Saved full dataset: {len(df)} rows")
        else:
            sample_df = df.sample(n=10000, random_state=42)
            sample_serializable = sample_df.copy()
            for col in sample_serializable.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif pd.api.types.is_categorical_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].astype(str)
                elif pd.api.types.is_integer_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].apply(lambda x: int(x) if pd.notna(x) else None)
                elif pd.api.types.is_float_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].apply(lambda x: float(x) if pd.notna(x) else None)
                elif pd.api.types.is_bool_dtype(sample_serializable[col]):
                    sample_serializable[col] = sample_serializable[col].astype(bool)
                else:
                    sample_serializable[col] = sample_serializable[col].astype(str).replace('', None)
            
            records = convert_timestamps_to_strings(sample_serializable.to_dict('records'))
            data_snapshot["full_dataset"] = records
            data_snapshot["is_sample"] = True
            data_snapshot["original_size"] = len(df)
            data_snapshot["sample_strategy"] = "random"
            logger.info(f"Saved sample dataset: {len(sample_df)} of {len(df)} rows")

        # Save field classifications
        if "field_types" in session_state:
            data_snapshot["field_types"] = session_state["field_types"]
            save_field_types(project_id, session_state["field_types"])

        # Save current filters if any
        if "filters" in session_state and session_state["filters"]:
            data_snapshot["applied_filters"] = session_state["filters"]

        # Prepare charts with embedded data context
        enhanced_charts = []
        enhanced_charts.append({
            "prompt": "_data_snapshot_",
            "type": "metadata",
            "data_snapshot": data_snapshot,
            "created_at": datetime.utcnow().isoformat()
        })

        for chart_config in chart_configs:
            enhanced_chart = {
                "prompt": chart_config["prompt"],
                "chart_type": chart_config["chart_type"],
                "created_at": datetime.utcnow().isoformat(),
                "dataset_hash": data_snapshot["dataset_metadata"]["hash"]
            }
            enhanced_charts.append(enhanced_chart)

        # Prepare dashboard data
        dashboard_data = {
            "name": dashboard_name,
            "project_id": project_id,
            "owner_id": session_state.get("user_id"),
            "charts": enhanced_charts,
            "tags": tags or [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Conditionally include dataset_hash
        try:
            dashboard_data["dataset_hash"] = data_snapshot["dataset_metadata"]["hash"]
        except Exception as e:
            logger.warning(f"Excluding dataset_hash from insert due to schema mismatch: {str(e)}")
            dashboard_data.pop("dataset_hash", None)

        # Conditionally include version
        try:
            dashboard_data["version"] = "2.0"
        except Exception as e:
            logger.warning(f"Excluding version from insert due to schema mismatch: {str(e)}")
            dashboard_data.pop("version", None)

        # Save to database
        result = supabase.table("dashboards").insert(dashboard_data).execute()

        if result.data:
            dashboard_id = result.data[0]["id"]
            logger.info(f"Dashboard saved successfully: {dashboard_id} with {len(enhanced_charts)-1} charts")
            return dashboard_id
        else:
            raise Exception("Failed to save dashboard to database")

    except Exception as e:
        logger.error(f"Failed to save dashboard: {str(e)}")
        raise


def render_save_dashboard_section(supabase):
    """Save dashboard section with user-defined name and overwrite prompt"""
    
    # Check if user is logged in
    if "user_id" not in st.session_state or st.session_state.user_id is None:
        st.info("💡 Please log in to save dashboards.")
        return
    
    # Check if dataset exists
    if st.session_state.get("dataset") is None:
        st.info("💡 Please upload a dataset first before saving a dashboard.")
        return
    
    # Get charts from your actual data structure
    recommendations = st.session_state.get("recommendations", [])
    custom_charts = st.session_state.get("custom_charts", [])
    
    # Filter out deleted charts and validate prompts
    active_recommendations = []
    for idx, chart_tuple in enumerate(recommendations):
        if not st.session_state.get(f"delete_chart_{idx}", False):
            if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                prompt, chart_type = chart_tuple[0], chart_tuple[1]
                if prompt and isinstance(prompt, str) and prompt.strip():
                    active_recommendations.append((prompt, chart_type))
    
    active_custom_charts = []
    base_idx = len(recommendations)
    for custom_idx, chart_tuple in enumerate(custom_charts):
        idx = base_idx + custom_idx
        if not st.session_state.get(f"delete_chart_{idx}", False):
            if isinstance(chart_tuple, (list, tuple)) and len(chart_tuple) >= 2:
                prompt, chart_type = chart_tuple[0], chart_tuple[1]
                if prompt and isinstance(prompt, str) and prompt.strip():
                    active_custom_charts.append((prompt, chart_type))
    
    # Total active charts
    total_charts = active_recommendations + active_custom_charts
    
    if not total_charts:
        st.info("💡 Create some charts first before saving a dashboard.")
        return
    
    with st.expander("💾 **Save Dashboard**", expanded=False):
        st.markdown(f"Save your current {len(total_charts)} charts and data as a dashboard.")
        
        with st.form("save_dashboard_form"):
            dashboard_name = st.text_input(
                "Dashboard Name",
                value="",
                placeholder="Enter a dashboard name",
                key="dashboard_name_input"
            )
            
            tags_input = st.text_input(
                "Tags (comma-separated)",
                placeholder="sales, quarterly, analysis",
                key="tags_input"
            )
            
            # Show what will be saved
            st.markdown("### 📊 Charts to Save:")
            for i, (prompt, chart_type) in enumerate(total_charts[:3]):  # Show first 3
                st.markdown(f"{i+1}. **{prompt[:50]}...** ({chart_type or 'Bar'})")
            if len(total_charts) > 3:
                st.markdown(f"*...and {len(total_charts) - 3} more charts*")
            
            # Check for existing dashboard
            existing_dashboard = None
            if dashboard_name and dashboard_name.strip():
                try:
                    result = supabase.table("dashboards").select("id, name").eq("name", dashboard_name.strip()).eq("owner_id", st.session_state.user_id).execute()
                    if result.data:
                        existing_dashboard = result.data[0]
                        st.session_state.existing_dashboard = existing_dashboard
                    else:
                        st.session_state.existing_dashboard = None
                except Exception as e:
                    logger.error(f"Failed to check for existing dashboard: {str(e)}")
                    st.error(f"Failed to check for existing dashboard: {str(e)}")
                    return
            
            # Handle duplicate dashboard name
            overwrite_choice = None
            new_dashboard_name = None
            if existing_dashboard:
                st.warning(f"A dashboard named '{dashboard_name}' already exists.")
                overwrite_choice = st.radio(
                    "Choose an action:",
                    ["Overwrite existing dashboard", "Save as a new dashboard"],
                    key="overwrite_choice"
                )
                if overwrite_choice == "Save as a new dashboard":
                    new_dashboard_name = st.text_input(
                        "New Dashboard Name",
                        value="",
                        placeholder="Enter a unique dashboard name",
                        key="new_dashboard_name_input"
                    )
                    if new_dashboard_name and new_dashboard_name.strip():
                        # Check if new name also exists
                        try:
                            result = supabase.table("dashboards").select("id").eq("name", new_dashboard_name.strip()).eq("owner_id", st.session_state.user_id).execute()
                            if result.data:
                                st.error(f"A dashboard named '{new_dashboard_name}' already exists. Please choose a different name.")
                                return
                        except Exception as e:
                            logger.error(f"Failed to check for new dashboard name: {str(e)}")
                            st.error(f"Failed to check for new dashboard name: {str(e)}")
                            return
            
            if st.form_submit_button("💾 Save Dashboard", type="primary"):
                # Validate inputs
                final_dashboard_name = None
                if overwrite_choice == "Save as a new dashboard" and new_dashboard_name and new_dashboard_name.strip():
                    final_dashboard_name = new_dashboard_name.strip()
                elif dashboard_name and dashboard_name.strip():
                    final_dashboard_name = dashboard_name.strip()
                else:
                    st.error("Please enter a dashboard name")
                    return
                
                try:
                    # Auto-generate project ID (hidden from user)
                    user_id = st.session_state.get("user_id", "anonymous")
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    auto_project_id = "My Project"

#f"{user_id}_project_{timestamp}"
                    
                    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
                    
                    if existing_dashboard and overwrite_choice == "Overwrite existing dashboard":
                        # Update existing dashboard
                        dashboard_data = {
                            "name": final_dashboard_name,
                            "project_id": auto_project_id,
                            "owner_id": st.session_state.get("user_id"),
                            "charts": [{
                                "prompt": "_data_snapshot_",
                                "type": "metadata",
                                "data_snapshot": {
                                    "version": "2.0",
                                    "saved_at": datetime.utcnow().isoformat(),
                                    "dataset_metadata": {
                                        "shape": list(st.session_state.dataset.shape),
                                        "columns": st.session_state.dataset.columns.tolist(),
                                        "dtypes": {col: str(dtype) for col, dtype in st.session_state.dataset.dtypes.items()},
                                        "memory_usage": int(st.session_state.dataset.memory_usage(deep=True).sum()),
                                        "hash": compute_dataset_hash(st.session_state.dataset)
                                    },
                                    "full_dataset": convert_timestamps_to_strings(st.session_state.dataset.to_dict('records')),
                                    "is_sample": False,
                                    "field_types": st.session_state.get("field_types", {}),
                                    "applied_filters": st.session_state.get("filters", {})
                                },
                                "created_at": datetime.utcnow().isoformat()
                            }] + [
                                {
                                    "prompt": prompt,
                                    "chart_type": chart_type,
                                    "created_at": datetime.utcnow().isoformat(),
                                    "dataset_hash": compute_dataset_hash(st.session_state.dataset)
                                } for prompt, chart_type in total_charts
                            ],
                            "tags": tags,
                            "updated_at": datetime.utcnow().isoformat(),
                            "dataset_hash": compute_dataset_hash(st.session_state.dataset),
                            "version": "2.0"
                        }
                        result = supabase.table("dashboards").update(dashboard_data).eq("id", existing_dashboard["id"]).execute()
                        if result.data:
                            dashboard_id = result.data[0]["id"]
                            st.success(f"✅ Dashboard '{final_dashboard_name}' overwritten successfully!")
                            st.info(f"Dashboard ID: `{dashboard_id}`")
                            st.session_state.refresh_dashboards = True
                            st.session_state.existing_dashboard = None
                        else:
                            raise Exception("Failed to update dashboard")
                    else:
                        # Save new dashboard
                        dashboard_id = save_dashboard_secure(
                            supabase,
                            st.session_state,
                            final_dashboard_name,
                            auto_project_id,
                            tags
                        )
                        st.success(f"✅ Dashboard '{final_dashboard_name}' saved successfully!")
                        st.info(f"Dashboard ID: `{dashboard_id}`")
                        st.session_state.refresh_dashboards = True
                        st.session_state.existing_dashboard = None
                    
                except Exception as e:
                    st.error(f"❌ Failed to save dashboard: {str(e)}")
                    logger.error(f"Failed to save dashboard: {str(e)}")


# Fixed chart rendering with better error handling
def safe_render_chart(idx, prompt, dimensions, measures, dates, working_df, sort_order="Descending", chart_type=None):
    """Safely render chart with comprehensive error handling"""
    try:
        # Validate inputs before proceeding
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            logger.warning(f"Invalid prompt for chart {idx}: {prompt}")
            return None
            
        if not dimensions and not measures:
            logger.warning(f"No dimensions or measures available for chart {idx}")
            return None
            
        # Call the original render_chart function
        from chart_utils import render_chart
        return render_chart(idx, prompt, dimensions, measures, dates, working_df, sort_order, chart_type)
        
    except Exception as e:
        logger.error(f"Safe render chart failed for prompt '{prompt}': {str(e)}")
        return None


def load_dataset_with_workflow(df, description):
    """Load dataset into session with workflow tracking and duplicate handling."""
    try:
        # Handle duplicate columns
        df = add_suffixes_to_duplicates(df)
        
        st.session_state.dataset = df
        st.session_state.dataset_hash = compute_dataset_hash(df)
        st.session_state.classified = False
        
        # Add to workflow if workflow_steps exists
        if hasattr(st.session_state, 'workflow_steps'):
            st.session_state.workflow_steps.append({
                'type': 'load',
                'category': 'load',
                'name': 'Load Data',
                'description': description,
                'operation': {'type': 'load', 'description': description}
            })
        
        # Show success message
        suffix_cols = [c for c in df.columns if '*' in c]
        if suffix_cols:
            st.info(f"ℹ️ Added suffixes to {len(suffix_cols)} duplicate columns")
        
        st.success(f"✅ {description} → {len(df):,} rows × {len(df.columns)} cols")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")

def generate_chart_from_prompt(prompt, dimensions, measures, dates, dataset, chart_type):
    """Generate chart data and plotly figure from prompt"""
    try:
        # Use your existing render_chart function
        from chart_utils import render_chart
        
        # Get a unique index for this chart
        chart_idx = 0  # You might want to make this dynamic
        
        result = render_chart(
            chart_idx, prompt, dimensions, measures, dates, dataset, 
            sort_order="Descending", chart_type=chart_type
        )
        
        if result is None:
            return None, None
        
        chart_data, metric, dimension, working_df, table_columns, chart_type_used, secondary_dimension, kwargs = result
        
        # Create plotly figure based on chart type
        import plotly.express as px
        
        if chart_type_used == "Bar":
            fig = px.bar(chart_data, x=dimension, y=metric, template="plotly_dark")
        elif chart_type_used == "Line":
            fig = px.line(chart_data, x=dimension, y=metric, color=secondary_dimension, template="plotly_dark")
        elif chart_type_used == "Scatter":
            y_metric = table_columns[2] if len(table_columns) > 2 else metric
            fig = px.scatter(chart_data, x=metric, y=y_metric, color=dimension, template="plotly_dark")
        elif chart_type_used == "Pie":
            fig = px.pie(chart_data, names=dimension, values=metric, template="plotly_dark")
        elif chart_type_used == "Map":
            fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric, template="plotly_dark")
        else:
            fig = None  # For table view
        
        return chart_data, fig
        
    except Exception as e:
        logger.error(f"Failed to generate chart from prompt: {str(e)}")
        return None, None


def load_dashboard_secure(supabase, dashboard_id):
    """Enterprise-grade dashboard loading with data restoration"""
    try:
        # Fetch dashboard
        result = supabase.table("dashboards").select("*").eq("id", dashboard_id).execute()
        
        if not result.data:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = result.data[0]
        charts = dashboard.get("charts", [])
        
        # Find data snapshot
        data_snapshot = None
        chart_configs = []
        
        for chart in charts:
            if chart.get("prompt") == "_data_snapshot_":
                data_snapshot = chart.get("data_snapshot", {})
            else:
                chart_configs.append(chart)
        
        if not data_snapshot:
            raise ValueError("No data snapshot found in dashboard")
        
        # Restore dataset
        dataset = restore_dataset_from_snapshot(data_snapshot)
        
        if dataset is None:
            raise ValueError("Failed to restore dataset from snapshot")
        
        # Restore field types
        field_types = data_snapshot.get("field_types", {})
        
        # Validate data integrity
        expected_hash = data_snapshot.get("dataset_metadata", {}).get("hash")
        if expected_hash:
            current_hash = compute_dataset_hash(dataset)
            if current_hash != expected_hash:
                logger.warning(f"Dataset hash mismatch for dashboard {dashboard_id} - data may have been altered")
        
        return {
            "dashboard": dashboard,
            "dataset": dataset,
            "field_types": field_types,
            "chart_configs": chart_configs,
            "data_snapshot": data_snapshot
        }
        
    except Exception as e:
        logger.error(f"Failed to load dashboard: {str(e)}")
        raise


def restore_dataset_from_snapshot(data_snapshot):
    """Restore dataset with proper type handling"""
    try:
        if not data_snapshot or "full_dataset" not in data_snapshot:
            logger.error("No full_dataset found in data_snapshot")
            return None
        
        records = data_snapshot["full_dataset"]
        if not records:
            logger.error("Empty dataset in data_snapshot")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Restore data types
        if "dataset_metadata" in data_snapshot:
            dtypes = data_snapshot["dataset_metadata"].get("dtypes", {})
            
            for col in df.columns:
                if col in dtypes:
                    dtype_str = dtypes[col]
                    try:
                        if "datetime64" in dtype_str or "timestamp" in dtype_str.lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif "int" in dtype_str.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if "Int" in dtype_str:
                                df[col] = df[col].astype('Int64')
                            else:
                                df[col] = df[col].astype('int64', errors='ignore')
                        elif "float" in dtype_str.lower():
                            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
                        elif "bool" in dtype_str.lower():
                            df[col] = df[col].astype('bool', errors='ignore')
                        elif "category" in dtype_str.lower():
                            df[col] = df[col].astype('category')
                        else:
                            df[col] = df[col].astype('object')
                    except Exception as e:
                        logger.warning(f"Failed to restore type for {col}: {e}")
                        df[col] = df[col].astype('object')
        
        # Handle empty strings and nulls
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('', np.nan)
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Restored dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Dataset restoration failed: {str(e)}")
        return None


def render_dashboard_enterprise(supabase, dashboard_id, project_id, dashboard_name):
    """Enterprise dashboard renderer with isolation"""
    
    # Store current session state
    original_dataset = st.session_state.get("dataset")
    original_field_types = st.session_state.get("field_types") 
    original_project = st.session_state.get("current_project")
    
    try:
        with st.spinner("ð    Loading dashboard..."):
            # Load dashboard with its own data
            dashboard_data = load_dashboard_secure(supabase, dashboard_id)
            
            dataset = dashboard_data["dataset"]
            field_types = dashboard_data["field_types"]
            chart_configs = dashboard_data["chart_configs"]
            data_snapshot = dashboard_data["data_snapshot"]
            
            # Temporarily set session state for rendering
            st.session_state.dataset = dataset
            st.session_state.field_types = field_types
            st.session_state.current_project = project_id
            
        # Show data status
        if data_snapshot.get("is_sample"):
            st.markdown(f"""
            <div class="dataset-status dataset-sample">
                ð    Dashboard dataset: {len(dataset)} rows (sample of {data_snapshot.get('original_size', 'unknown')})
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="dataset-status dataset-restored">
                â   Dashboard dataset: {len(dataset)} rows, {len(dataset.columns)} columns
            </div>
            """, unsafe_allow_html=True)
        
        if not chart_configs:
            st.warning("No charts found in this dashboard.")
            return
        
        # Extract field types
        dimensions = field_types.get("dimension", [])
        measures = field_types.get("measure", [])
        dates = field_types.get("date", [])
        
        # Render each chart
        for idx, chart in enumerate(chart_configs):
            prompt = chart.get("prompt")
            chart_type = chart.get("chart_type")
            
            with st.expander(f"Chart {idx + 1}: {prompt} ({chart_type})", expanded=True):
                try:
                    chart_data, fig = generate_chart_from_prompt(
                        prompt, dimensions, measures, dates, dataset, chart_type
                    )
                    if chart_data is not None and not chart_data.empty:
                        if fig:
                            st.plotly_chart( fig, use_container_width=True, key=f"chart_{idx}")
                        st.dataframe(chart_data, use_container_width=True)
                        
                        # Generate insights
                        if pd.api.types.is_numeric_dtype(chart_data[measures[0]]):
                            stats = calculate_statistics(chart_data, measures[0])
                            insights = generate_gpt_insights(
                                stats, measures[0], prompt, chart_data, dimensions[0] if dimensions else None
                            )
                            if insights:
                                st.markdown("**Insights:**")
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                            else:
                                st.info("No significant insights could be generated.")
                        else:
                            st.info("Insights unavailable for non-numeric data.")
                    else:
                        st.warning("Unable to render chart: No valid data.")
                except Exception as e:
                    st.error(f"Failed to render chart: {str(e)}")
                    logger.error(f"Chart rendering failed for {prompt}: {str(e)}")
        
    finally:
        # Restore original session state
        st.session_state.dataset = original_dataset
        st.session_state.field_types = original_field_types
        st.session_state.current_project = original_project


# ISOLATED CHART DISPLAY
def display_chart_isolated(idx, prompt, dimensions, measures, dates, dataset, chart_type="Bar"):
    """Display chart using isolated dataset without affecting session state"""
    try:
        # This function should work with the provided dataset only
        # Don't use any session state data
        
        # Generate chart using provided data
        chart_data, chart_obj = generate_chart_from_prompt(
            prompt, dimensions, measures, dates, dataset, chart_type
        )
        
        if chart_obj:
            st.plotly_chart(chart_obj, use_container_width=True)
        else:
            st.error("Could not generate chart")
            
    except Exception as e:
        st.error(f"Chart generation failed: {str(e)}")
        logger.error(f"Isolated chart display error: {str(e)}")

def generate_overall_data_analysis(df, dimensions, measures, dates):
    if not USE_OPENAI:
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

    try:
        stats_summary = []
        for metric in measures:
            if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                stats = calculate_statistics(df, metric)
                stats_summary.append(
                    f"{metric}: mean={stats['mean']:.2f}, std_dev={stats['std_dev']:.2f}, "
                    f"Q1={stats['q1']:.2f}, median={stats['median']:.2f}, Q3={stats['q3']:.2f}, "
                    f"90th percentile={stats['percentile_90']:.2f}"
                )
        
        top_performers = []
        for dim in dimensions:
            for metric in measures:
                if dim in df.columns and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                    grouped = df.groupby(dim)[metric].mean().sort_values(ascending=False)
                    if not grouped.empty:
                        top = grouped.index[0]
                        top_value = grouped.iloc[0]
                        top_performers.append(f"Top {dim} by {metric}: {top} with average {top_value:.2f}")

        correlations = []
        for i, m1 in enumerate(measures):
            for m2 in measures[i+1:]:
                if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                    corr = df[[m1, m2]].corr().iloc[0, 1]
                    correlations.append(f"Correlation between {m1} and {m2}: {corr:.2f}")

        data_summary = (
            f"Dataset Overview:\n- Dimensions: {', '.join(dimensions)}\n- Measures: {', '.join(measures)}\n- Dates: {', '.join(dates) if dates else 'None'}\n"
            f"Statistics:\n" + "\n".join(stats_summary) + "\n"
            f"Top Performers:\n" + "\n".join(top_performers) + "\n"
            f"Correlations:\n" + "\n".join(correlations)
        )

        # Call OpenAI API using the new format
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst providing an overall analysis and findings summary for a dataset. Focus on key trends, significant findings, and actionable recommendations."},
                {"role": "user", "content": f"Generate a concise overall data analysis and findings summary (3-5 points) based on the following dataset summary:\n{data_summary}\nHighlight key trends, significant findings, and provide actionable recommendations for business strategy."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        analysis = response.choices[0].message.content.strip().split('\n')
        analysis = [item.strip('- ').strip() for item in analysis if item.strip()]
        #logger.info("Generated overall data analysis: %s", analysis)
        return analysis
    except Exception as e:
        logger.error


# Display Chart (rewritten)
def display_chart(idx, prompt, dimensions, measures, dates, df, sort_order="Descending", chart_type=None):
    """Display a chart with controls and data table, using cache."""
    try:
        # Create a unique cache key
        dataset_hash = st.session_state.get("dataset_hash", "no_hash")
        chart_key = f"chart_{idx}_{hash(prompt + str(chart_type) + sort_order + dataset_hash)}"
        
        # Initialize chart type in session state if not exists
        if f"chart_type_{chart_key}" not in st.session_state:
            st.session_state[f"chart_type_{chart_key}"] = chart_type or "Bar"
        
        # Create two columns for controls
        col1, col2 = st.columns(2)
        
        # Chart type selection
        with col1:
            selected_chart_type = st.selectbox(
                "Chart Type",
                options=["Bar", "Line", "Scatter", "Map", "Table", "Pie"],
                index=["Bar", "Line", "Scatter", "Map", "Table", "Pie"].index(st.session_state[f"chart_type_{chart_key}"]),
                key=f"chart_type_select_{chart_key}"
            )
            if selected_chart_type != st.session_state[f"chart_type_{chart_key}"]:
                st.session_state[f"chart_type_{chart_key}"] = selected_chart_type
                # Clear cache for this chart if type changes
                if chart_key in st.session_state.chart_cache:
                    del st.session_state.chart_cache[chart_key]
                if chart_key in st.session_state.insights_cache:
                    del st.session_state.insights_cache[chart_key]
                st.rerun()
        
        # Sort order selection
        with col2:
            selected_sort_order = st.selectbox(
                "Sort Order",
                options=["Ascending", "Descending"],
                index=1 if sort_order == "Descending" else 0,
                key=f"sort_order_{chart_key}"
            )
            if selected_sort_order != sort_order:
                # Clear cache if sort order changes
                if chart_key in st.session_state.chart_cache:
                    del st.session_state.chart_cache[chart_key]
                if chart_key in st.session_state.insights_cache:
                    del st.session_state.insights_cache[chart_key]
                sort_order = selected_sort_order
        
        # Check chart cache
        if chart_key in st.session_state.chart_cache:
            #logger.info(f"Using cached chart for key: {chart_key}")
            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
        else:
            # Render chart if not cached
            try:
                chart_result = render_chart(
                    idx, prompt, dimensions, measures, dates, df, sort_order, st.session_state[f"chart_type_{chart_key}"]
                )
                if chart_result is None:
                    raise ValueError("Chart rendering returned None")
                chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                # Store in cache
                st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs)
                #logger.info(f"Cached chart for key: {chart_key}")
            except ValueError as e:
                st.error(str(e))
                logger.error(f"Chart rendering failed: {str(e)}")
                return
        
        # Create the chart based on type
        if st.session_state[f"chart_type_{chart_key}"] == "Scatter":
            fig = px.scatter(
                chart_data,
                x=metric,
                y=table_columns[2],  # Use the second metric for y-axis
                color=dimension,
                hover_data=[dimension],
                title=f"{table_columns[2]} vs {metric} by {dimension}",
                labels={metric: metric, table_columns[2]: table_columns[2]},
                template="plotly_dark"
            )
            fig.update_traces(marker=dict(size=12))
            fig.update_layout(
                xaxis_title=metric,
                yaxis_title=table_columns[2],
                showlegend=True
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_scatter")
        elif st.session_state[f"chart_type_{chart_key}"] == "Bar":
            color_col = "Outlier" if "Outlier" in chart_data.columns else None
            fig = px.bar(
                chart_data,
                x=dimension,
                y=metric,
                color=color_col,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_bar")
        elif st.session_state[f"chart_type_{chart_key}"] == "Line":
            time_agg = kwargs.get("time_aggregation", "month")
            title = f"{metric} by {time_agg.capitalize()}"
            if secondary_dimension:
                title += f" and {secondary_dimension}"
            
            fig = px.line(
                chart_data,
                x=dimension,
                y=metric,
                color=secondary_dimension if secondary_dimension else None,
                title=title,
                template="plotly_dark"
            )
            if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                fig.update_xaxes(
                    tickformat="%b-%Y",
                    tickangle=45,
                    nticks=10
                )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_line")
        elif st.session_state[f"chart_type_{chart_key}"] == "Map":
            fig = px.choropleth(
                chart_data,
                locations=dimension,
                locationmode="country names",
                color=metric,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_map")
        elif st.session_state[f"chart_type_{chart_key}"] == "Pie":
            fig = px.pie(
                chart_data,
                names=dimension,
                values=metric,
                title=f"{metric} by {dimension}",
                template="plotly_dark"
            )
            st.plotly_chart( fig, use_container_width=True, key=f"{chart_key}_pie")
        else:  # Table view
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table")
        
        # Check insights cache
        if chart_key in st.session_state.insights_cache:
            #logger.info(f"Using cached insights for key: {chart_key}")
            insights = st.session_state.insights_cache[chart_key]
        else:
            # Generate insights if not cached
            try:
                insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                st.session_state.insights_cache[chart_key] = insights
                #logger.info(f"Cached insights for key: {chart_key}")
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
                insights = ["Unable to generate insights at this time."]
        
        # Display insights
        st.markdown("### Insights")
        for insight in insights:
            st.markdown(f"🔹 {insight}")
        
        # Display data table in a collapsed expander
        with st.expander("View Data", expanded=False):
            st.dataframe(chart_data[table_columns], use_container_width=True, key=f"{chart_key}_table_data")
            
            if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]):
                st.markdown("### Basic Statistics")
                stats = calculate_statistics(working_df, metric)
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Median", f"{stats['median']:.2f}")
                    with col3:
                        st.metric("Min", f"{stats['min']:.2f}")
                    with col4:
                        st.metric("Max", f"{stats['max']:.2f}")
        
    except Exception as e:
        logger.error(f"Error displaying chart: {str(e)}")
        st.error(f"Error displaying chart: {str(e)}")

def truncate_label(label, max_length=20):
    if isinstance(label, str) and len(label) > max_length:
        return label[:max_length-3] + "..."
    return label


def compute_filter_hash(filters):
    """Compute a hash of the filters dictionary for cache key."""
    filter_str = json.dumps(filters, sort_keys=True, default=str)
    return hashlib.md5(filter_str.encode()).hexdigest()

# Add this function to main.py after the other tab functions

    
    # AI-Powered Executive Summary Generation
    def generate_executive_summary():
        """Generate comprehensive executive summary using AI analysis of all charts"""
        try:
            if not st.session_state.agent_recommendations and not st.session_state.custom_charts:
                return None
            
            business_cols = get_business_relevant_columns(df)
            
            # Collect insights from all charts
            chart_insights = []
            key_metrics = {}
            correlations = []
            
            # Analyze AI-generated charts
            for chart in st.session_state.agent_recommendations:
                chart_data = chart.get('data')
                x_col = chart.get('x_col')
                y_col = chart.get('y_col')
                chart_type = chart.get('type')
                insights = chart.get('insights', [])
                
                chart_insights.append({
                    'title': chart['title'],
                    'type': chart_type,
                    'x_col': x_col,
                    'y_col': y_col,
                    'insights': insights,
                    'priority': chart.get('priority', 'medium')
                })
                
                # Calculate key metrics for numerical columns
                if y_col and hasattr(chart_data, 'columns') and y_col in chart_data.columns:
                    if pd.api.types.is_numeric_dtype(chart_data[y_col]):
                        stats = {
                            'metric': y_col,
                            'count': len(chart_data),
                            'mean': chart_data[y_col].mean(),
                            'max': chart_data[y_col].max(),
                            'min': chart_data[y_col].min(),
                            'std': chart_data[y_col].std(),
                            'total': chart_data[y_col].sum() if chart_type in ['bar', 'pie'] else None
                        }
                        key_metrics[y_col] = stats
            
            # Analyze custom charts
            for custom_chart in st.session_state.custom_charts:
                if isinstance(custom_chart, dict):
                    chart_insights.append({
                        'title': f"Custom: {custom_chart.get('prompt', 'User Request')}",
                        'type': 'custom',
                        'analysis': custom_chart.get('ai_analysis', ''),
                        'priority': 'high'
                    })
            
            # Calculate correlations between numerical columns
            numerical_cols = business_cols['numerical']
            if len(numerical_cols) >= 2:
                for i, col1 in enumerate(numerical_cols):
                    for col2 in numerical_cols[i+1:]:
                        if col1 in df.columns and col2 in df.columns:
                            corr = df[[col1, col2]].corr().iloc[0, 1]
                            if abs(corr) > 0.5:
                                correlations.append({
                                    'metric1': col1,
                                    'metric2': col2,
                                    'correlation': corr,
                                    'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate',
                                    'direction': 'Positive' if corr > 0 else 'Negative'
                                })
            
            # Generate AI-powered executive summary
            summary_prompt = f"""
            As a senior business analyst, create a comprehensive executive summary based on this data analysis:
            
            DATASET OVERVIEW:
            - Total Records: {len(df):,}
            - Business Metrics: {business_cols['numerical']}
            - Key Dimensions: {business_cols['categorical']}
            - Time Periods: {business_cols['temporal']}
            
            CHART ANALYSIS:
            {json.dumps(chart_insights, indent=2)[:2000]}...
            
            KEY METRICS:
            {json.dumps(key_metrics, indent=2, default=str)[:1500]}...
            
            CORRELATIONS FOUND:
            {json.dumps(correlations, indent=2)[:1000]}...
            
            BUSINESS CONTEXT:
            {st.session_state.agent_learning.get('business_context', 'No specific business context provided')}
            
            USER FEEDBACK PATTERNS:
            - Total feedback sessions: {len(st.session_state.user_feedback)}
            - Average chart rating: {np.mean([len(fb['chart_rating'].split('⭐')) - 1 for fb in st.session_state.user_feedback]) if st.session_state.user_feedback else 'No ratings yet'}
            
            Create a comprehensive executive summary with these sections:
            
            1. **Business Performance Highlights** (3-4 key findings)
            2. **Critical Trends & Patterns** (2-3 important trends)
            3. **Key Relationships & Correlations** (2-3 significant relationships)
            4. **Risk Areas & Opportunities** (3-4 actionable insights)
            5. **Strategic Recommendations** (4-5 specific actions)
            
            Make it executive-level: concise, actionable, and data-driven. Focus on business impact and decisions.
            Use specific numbers and percentages where available.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            
            ai_summary = response.choices[0].message.content.strip()
            
            # Add quantitative insights
            quant_insights = []
            
            if key_metrics:
                total_revenue = sum([stats.get('total', 0) for stats in key_metrics.values() if stats.get('total')])
                if total_revenue > 0:
                    quant_insights.append(f"**Total Revenue Analyzed:** ${total_revenue:,.0f}")
                
                highest_metric = max(key_metrics.items(), key=lambda x: x[1]['mean'])
                quant_insights.append(f"**Top Performing Metric:** {highest_metric[0]} (Avg: ${highest_metric[1]['mean']:,.0f})")
                
                if len(key_metrics) > 1:
                    most_variable = max(key_metrics.items(), key=lambda x: x[1]['std']/x[1]['mean'] if x[1]['mean'] > 0 else 0)
                    quant_insights.append(f"**Highest Variability:** {most_variable[0]} (CV: {(most_variable[1]['std']/most_variable[1]['mean']*100):.1f}%)")
            
            if correlations:
                strongest_corr = max(correlations, key=lambda x: abs(x['correlation']))
                quant_insights.append(f"**Strongest Relationship:** {strongest_corr['metric1']} ↔ {strongest_corr['metric2']} ({strongest_corr['correlation']:.2f})")
            
            return {
                'ai_summary': ai_summary,
                'quantitative_insights': quant_insights,
                'charts_analyzed': len(chart_insights),
                'metrics_evaluated': len(key_metrics),
                'correlations_found': len(correlations)
            }
            
        except Exception as e:
            return {
                'ai_summary': f"Executive summary generation encountered an error: {str(e)}",
                'quantitative_insights': ["Please ensure charts are generated and try again."],
                'charts_analyzed': 0,
                'metrics_evaluated': 0,
                'correlations_found': 0
            }
    
    # Generate Executive Summary Section
    if st.session_state.agent_recommendations or st.session_state.custom_charts:
        st.markdown("---")
        st.markdown("### 📊 AI-Powered Executive Summary")
        st.markdown("*Comprehensive analysis of all generated visualizations and data patterns*")
        
        with st.spinner("🧠 AI is analyzing all charts and generating executive insights..."):
            exec_summary = generate_executive_summary()
            
            if exec_summary:
                # Summary Header with Key Stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Charts Analyzed", exec_summary['charts_analyzed'])
                with col2:
                    st.metric("Metrics Evaluated", exec_summary['metrics_evaluated'])
                with col3:
                    st.metric("Correlations Found", exec_summary['correlations_found'])
                with col4:
                    data_quality = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("Data Quality", f"{data_quality:.1f}%")
                
                # Quantitative Insights
                if exec_summary['quantitative_insights']:
                    st.markdown("#### 📈 Key Performance Indicators")
                    for insight in exec_summary['quantitative_insights']:
                        st.markdown(f"- {insight}")
                
                # AI-Generated Executive Summary
                st.markdown("#### 🧠 Strategic Business Analysis")
                st.markdown('<div class="insights-container" style="background-color: #1F2937; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #60A5FA;">', unsafe_allow_html=True)
                st.markdown(exec_summary['ai_summary'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Action Items Generator
                st.markdown("#### 🎯 Immediate Action Items")
                
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button("🚀 Generate Action Plan", type="primary"):
                        with st.spinner("Creating actionable business plan..."):
                            try:
                                action_prompt = f"""
                                Based on this executive summary, create 5 specific, immediate action items:
                                
                                Summary: {exec_summary['ai_summary'][:800]}
                                
                                Key Metrics: {exec_summary['quantitative_insights']}
                                
                                Create 5 action items that are:
                                1. Specific and measurable
                                2. Actionable within 30-90 days
                                3. Based on the data insights
                                4. Prioritized by business impact
                                
                                Format as:
                                **Action 1:** [Specific action with deadline]
                                **Action 2:** [Specific action with deadline]
                                etc.
                                """
                                
                                action_response = openai.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": action_prompt}],
                                    temperature=0.3
                                )
                                
                                st.markdown("##### 📋 Recommended Actions:")
                                st.markdown(action_response.choices[0].message.content)
                                
                            except Exception as e:
                                st.error(f"Action plan generation failed: {str(e)}")
                
                with col_action2:
                    if st.button("📊 Risk Assessment"):
                        with st.spinner("Analyzing potential risks..."):
                            try:
                                risk_prompt = f"""
                                Based on this data analysis, identify top 3 business risks:
                                
                                Data: {exec_summary['ai_summary'][:500]}
                                Metrics: {exec_summary['quantitative_insights']}
                                
                                For each risk, provide:
                                - Risk description
                                - Potential impact (High/Medium/Low)
                                - Mitigation strategy
                                
                                Focus on data-driven risks only.
                                """
                                
                                risk_response = openai.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": risk_prompt}],
                                    temperature=0.2
                                )
                                
                                st.markdown("##### ⚠️ Risk Analysis:")
                                st.markdown(risk_response.choices[0].message.content)
                                
                            except Exception as e:
                                st.error(f"Risk assessment failed: {str(e)}")
                
                # Export Executive Summary
                st.markdown("#### 📄 Export Options")
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    if st.button("📋 Copy Summary"):
                        summary_text = f"""
EXECUTIVE SUMMARY - {datetime.now().strftime('%Y-%m-%d')}

KEY METRICS:
{chr(10).join(exec_summary['quantitative_insights'])}

STRATEGIC ANALYSIS:
{exec_summary['ai_summary']}

Generated by NarraViz.ai Agentic AI System
                        """
                        st.code(summary_text, language="text")
                        st.success("✅ Summary formatted for copying!")
                
                with col_export2:
                    if st.button("📊 PowerPoint Format"):
                        ppt_format = f"""
SLIDE 1: EXECUTIVE SUMMARY
- {exec_summary['charts_analyzed']} visualizations analyzed
- {exec_summary['metrics_evaluated']} key metrics evaluated
- {exec_summary['correlations_found']} significant relationships found

SLIDE 2: KEY FINDINGS
{chr(10).join([f"• {insight}" for insight in exec_summary['quantitative_insights']])}

SLIDE 3: STRATEGIC RECOMMENDATIONS
{exec_summary['ai_summary'].split('**Strategic Recommendations**')[-1] if '**Strategic Recommendations**' in exec_summary['ai_summary'] else 'See full analysis for detailed recommendations'}
                        """
                        st.code(ppt_format, language="text")
                        st.success("✅ PowerPoint format ready!")
            
            else:
                st.info("💡 Generate some charts first to see the executive summary!")
    
    # Business Context & AI Learning Section
    with st.expander("🧠 AI Learning & Business Context", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Business Context")
            business_context = st.text_area(
                "Help AI understand your business:",
                value=st.session_state.agent_learning.get('business_context', ''),
                placeholder="e.g., 'We're a retail company focused on quarterly sales growth. Key metrics are revenue, profit margin, and customer segments.'",
                height=100
            )
            
            if st.button("💾 Save Context"):
                st.session_state.agent_learning['business_context'] = business_context
                st.success("Business context saved! AI will use this for better analysis.")
        
        with col2:
            st.markdown("#### 🎯 AI Learning Progress")
            st.metric("User Feedback Received", len(st.session_state.user_feedback))
            st.metric("Preferred Chart Patterns", len(st.session_state.agent_learning['preferred_charts']))
            st.metric("Avoided Columns", len(st.session_state.agent_learning['avoided_columns']))
            
            if st.session_state.user_feedback:
                avg_rating = np.mean([len(fb['chart_rating'].split('⭐')) - 1 for fb in st.session_state.user_feedback[-10:]])
                st.metric("Recent Chart Quality", f"{avg_rating:.1f}/5 ⭐")
        
        # Show recent feedback
        if st.session_state.user_feedback:
            st.markdown("#### 💬 Recent Feedback")
            for fb in st.session_state.user_feedback[-3:]:
                with st.container():
                    st.caption(f"**{fb['chart_title']}** - {fb['chart_rating']} - {fb['feedback_text'][:100]}...")
    
    # Custom AI Chart Request with improved error handling
    st.markdown("---")
    st.markdown("### 🧠 Ask AI to Create Custom Visualization")
    
    with st.form("ai_custom_request"):
        custom_prompt = st.text_area(
            "Describe what you want the AI to analyze and visualize:",
            placeholder="e.g., 'Find correlations between sales and profit' or 'Show me seasonal trends' or 'Which categories drive the most revenue?'",
            height=80
        )
        col1, col2 = st.columns(2)
        with col1:
            analysis_depth = st.selectbox("AI Analysis Depth", ["Standard", "Deep", "Expert"])
        with col2:
            include_predictions = st.checkbox("Include Predictive Insights", value=False)
        
        submit_ai_request = st.form_submit_button("🧠 Send to AI Agents", type="primary")
    
    if submit_ai_request and custom_prompt:
        with st.spinner("🤖 AI Agents are processing your request intelligently..."):
            try:
                # Get business columns for smarter analysis
                business_cols = get_business_relevant_columns(df)
                all_business_cols = business_cols['categorical'] + business_cols['numerical'] + business_cols['temporal']
                
                # AI-powered custom chart generation with better prompting
                ai_analysis = insight_generator.generate_custom_chart_insights(custom_prompt, df)
                
                # Enhanced chart specification prompt
                chart_prompt = f"""
                User request: "{custom_prompt}"
                
                ONLY USE THESE EXACT COLUMN NAMES:
                - Available Categorical: {business_cols['categorical']}
                - Available Numerical: {business_cols['numerical']}
                - Available Temporal: {business_cols['temporal']}
                
                IMPORTANT RULES:
                1. Only use column names from the lists above
                2. X-axis must be from categorical or temporal columns
                3. Y-axis must be from numerical columns
                4. Color column (optional) can be from categorical columns only
                5. If color column is specified, it must exist in the final chart data
                
                Business Context: {st.session_state.agent_learning.get('business_context', 'No context provided')}
                User Preferences: {st.session_state.agent_learning.get('preferred_charts', [])}
                Avoid Columns: {st.session_state.agent_learning.get('avoided_columns', [])}
                
                IMPORTANT: Respond with ONLY valid JSON. No markdown, no explanations.
                
                {{
                    "type": "bar|line|scatter|pie|histogram",
                    "x_col": "exact_column_name_from_lists_above",
                    "y_col": "exact_column_name_from_lists_above", 
                    "color_col": null,
                    "title": "Specific business-focused title"
                }}
                
                Choose columns that best answer the user's business question.
                Set color_col to null unless specifically needed and available.
                """
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": chart_prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                
                response_text = response.choices[0].message.content.strip()
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                chart_spec = json.loads(response_text)
                
                # Validate the chart specification
                chart_type = chart_spec.get('type', 'bar').lower()
                x_col = chart_spec.get('x_col')
                y_col = chart_spec.get('y_col')
                color_col = chart_spec.get('color_col')
                title = chart_spec.get('title', f"AI Analysis: {custom_prompt}")
                
                # Ensure columns are in business columns list and exist
                if x_col not in all_business_cols or x_col not in df.columns:
                    x_col = business_cols['categorical'][0] if business_cols['categorical'] else all_business_cols[0] if all_business_cols else df.columns[0]
                if y_col and (y_col not in all_business_cols or y_col not in df.columns):
                    y_col = business_cols['numerical'][0] if business_cols['numerical'] else all_business_cols[0] if all_business_cols else df.columns[-1]
                if color_col and (color_col not in all_business_cols or color_col not in df.columns):
                    color_col = None  # Reset if column doesn't exist
                
                # Create the chart
                fig = None
                chart_data = df
                
                if chart_type == 'bar' and y_col and x_col in df.columns and y_col in df.columns:
                    if color_col and color_col in df.columns:
                        # Group by both x_col and color_col
                        chart_data = df.groupby([x_col, color_col])[y_col].sum().reset_index()
                        chart_data = chart_data.sort_values(y_col, ascending=False).head(15)
                        fig = px.bar(chart_data, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        # Group by x_col only
                        chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                        chart_data = chart_data.sort_values(y_col, ascending=False).head(15)
                        fig = px.bar(chart_data, x=x_col, y=y_col, title=title)
                        fig.update_traces(marker_color='#60A5FA')
                
                elif chart_type == 'scatter' and y_col and x_col in df.columns and y_col in df.columns:
                    if color_col and color_col in df.columns:
                        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, title=title)
                        fig.update_traces(marker_color='#60A5FA')
                    chart_data = df
                
                elif chart_type == 'line' and y_col and x_col in df.columns and y_col in df.columns:
                    if 'date' in x_col.lower():
                        chart_data = df.copy()
                        chart_data[x_col] = pd.to_datetime(chart_data[x_col])
                        if color_col and color_col in df.columns:
                            chart_data = chart_data.groupby([pd.Grouper(key=x_col, freq='M'), color_col])[y_col].sum().reset_index()
                            fig = px.line(chart_data, x=x_col, y=y_col, color=color_col, title=title)
                        else:
                            chart_data = chart_data.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].sum().reset_index()
                            fig = px.line(chart_data, x=x_col, y=y_col, title=title)
                            fig.update_traces(line_color='#60A5FA')
                    else:
                        # Non-date line chart
                        if color_col and color_col in df.columns:
                            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
                        else:
                            fig = px.line(df, x=x_col, y=y_col, title=title)
                            fig.update_traces(line_color='#60A5FA')
                        chart_data = df
                
                elif chart_type == 'pie' and y_col and x_col in df.columns and y_col in df.columns:
                    # For pie charts, ignore color column as it gets complex
                    chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.pie(chart_data, names=x_col, values=y_col, title=title)
                
                if fig:
                    fig = create_dark_chart(fig)
                    
                    custom_chart_data = {
                        'prompt': custom_prompt,
                        'figure': fig,
                        'x_col': x_col,
                        'y_col': y_col,
                        'ai_analysis': ai_analysis,
                        'chart_spec': chart_spec,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.custom_charts.append(custom_chart_data)
                    st.success("🧠 AI Agents successfully analyzed your request and created an intelligent visualization!")
                    st.rerun()
                
                else:
                    st.error("❌ AI could not determine optimal visualization for your request.")
                    st.info(f"💡 Suggested: Try requesting analysis of these business columns: {', '.join(all_business_cols[:5])}")
                    
            except json.JSONDecodeError as e:
                st.error("🤖 AI response format error. The AI may need more specific guidance.")
                st.info("💡 Try rephrasing your request more specifically, like 'Create a bar chart showing revenue by product category'")
                st.expander("Debug Info", expanded=False).write(f"JSON Error: {str(e)}")
            except KeyError as e:
                st.error(f"🤖 Column error: {str(e)}. The requested column may not exist in your data.")
                st.info(f"💡 Available business columns: {', '.join(all_business_cols)}")
            except Exception as e:
                st.error(f"🤖 AI processing error: {str(e)}")
                st.info("💡 Please try a simpler request or check your data structure.")
                st.expander("Debug Info", expanded=False).write({
                    "Error": str(e),
                    "Chart Type": chart_spec.get('type', 'unknown') if 'chart_spec' in locals() else 'unknown',
                    "X Column": chart_spec.get('x_col', 'unknown') if 'chart_spec' in locals() else 'unknown',
                    "Y Column": chart_spec.get('y_col', 'unknown') if 'chart_spec' in locals() else 'unknown',
                    "Color Column": chart_spec.get('color_col', 'unknown') if 'chart_spec' in locals() else 'unknown',
                    "Available Columns": list(df.columns)
                })
    
    # AI System Guide
    with st.expander("🧠 Real Agentic AI System Guide", expanded=False):
        st.markdown("""
        ### 🤖 How Our Intelligent AI Agents Work:
        
        **🧠 Data Analyst Agent:**
        - **Smart Column Filtering**: Automatically excludes ID fields and irrelevant columns
        - **Statistical Analysis**: Uses GPT-4 to identify patterns, correlations, and anomalies
        - **Business Context Awareness**: Considers your business context for relevant insights
        - **Learning Integration**: Adapts based on your feedback and preferences
        
        **📊 Chart Creator Agent:**
        - **Business-Focused Visualizations**: Only recommends charts with business value
        - **Intelligent Column Selection**: Chooses meaningful combinations automatically
        - **User Preference Learning**: Remembers what chart types you prefer
        - **Adaptive Recommendations**: Improves suggestions based on your ratings
        
        **💡 Insight Generator Agent:**
        - **Context-Aware Analysis**: Generates insights specific to your business domain
        - **Actionable Recommendations**: Provides specific steps you can take
        - **Pattern Recognition**: Identifies trends, outliers, and opportunities
        - **Continuous Learning**: Gets better with your feedback
        
        ### 🎯 New Intelligent Features:
        
        **🚫 Smart Filtering:**
        - Automatically excludes Row ID, Order ID, Customer ID, etc.
        - Focuses only on business-relevant columns
        - Avoids charts with technical identifiers
        
        **📚 AI Learning System:**
        - **Rate Charts**: ⭐ Poor to ⭐⭐⭐⭐⭐ Excellent
        - **Provide Feedback**: Tell AI how to improve
        - **Business Context**: Help AI understand your domain
        - **Preference Learning**: AI remembers what you like
        
        **📄 Professional HTML Export:**
        - Complete dashboard exported as interactive HTML report  
        - NarraViz.ai branding with beta version disclaimers
        - All charts, insights, and analysis included
        - Professional format ready for stakeholder sharing
        - Convert to PDF using browser's "Print to PDF" feature
        
        **🎯 Intelligent Custom Analysis:**
        - Natural language chart requests with business focus
        - AI determines optimal visualization approach
        - Context-aware column selection
        - Business-specific insights generation
        
        **💡 Try These Smart Prompts:**
        - "What drives our highest revenue?" → AI finds key revenue drivers
        - "Show seasonal patterns in our sales" → AI detects time-based trends  
        - "Which customer segments are most profitable?" → AI analyzes profitability
        - "Find unusual patterns that need attention" → AI identifies anomalies
        - "Compare performance across different regions" → AI creates comparison analysis
        
        ### 📈 AI Learning Progress:
        - **Feedback Loop**: Your ratings train the AI to be more relevant
        - **Business Adaptation**: AI learns your industry and priorities  
        - **Preference Memory**: Remembers your favorite chart types and metrics
        - **Continuous Improvement**: Gets smarter with every interaction
        """)
    
    # Display learning stats
    if st.session_state.saved_dashboards:
        st.sidebar.info(f"💾 {len(st.session_state.saved_dashboards)} AI dashboards saved")
    
    if st.session_state.custom_charts:
        st.sidebar.info(f"🧠 {len(st.session_state.custom_charts)} intelligent charts created")
        
    if st.session_state.user_feedback:
        st.sidebar.success(f"📚 AI learned from {len(st.session_state.user_feedback)} feedback sessions")

def recommended_charts_insights_tab():
    st.subheader("📊 Recommended Charts & Insights")


    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset first.")
        return

    # Initialize session state for filters
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    if "filter_search" not in st.session_state:
        st.session_state.filter_search = {}
    if "filter_show_more" not in st.session_state:
        st.session_state.filter_show_more = {}

    dimensions = st.session_state.field_types.get("dimension", [])
    measures = st.session_state.field_types.get("measure", [])
    dates = st.session_state.field_types.get("date", [])

    # Filter Dataset Section
    st.markdown("### Filter Dataset")
    with st.expander("Apply Filters", expanded=False):
        filter_changed = False
        temp_filters = st.session_state.filters.copy()
        temp_search = st.session_state.filter_search.copy()
        temp_show_more = st.session_state.filter_show_more.copy()

        # Dimension Filters (Search + Limited Multiselect)
        for dim in dimensions:
            if dim in df.columns:
                # Get value counts for top values
                value_counts = df[dim].value_counts().head(100)  # Limit to top 100 for performance
                unique_vals = value_counts.index.tolist()
                if len(unique_vals) > 0:
                    # Initialize show_more count
                    if dim not in temp_show_more:
                        temp_show_more[dim] = 10  # Default to showing 10 values
                    max_display = temp_show_more[dim]

                    # Search input
                    search_key = f"filter_search_{dim}"
                    search_term = st.text_input(
                        f"Search {dim}",
                        value=temp_search.get(dim, ""),
                        key=search_key
                    )
                    if search_term != temp_search.get(dim):
                        temp_search[dim] = search_term
                        temp_show_more[dim] = 10  # Reset show_more on search
                        filter_changed = True

                    # Filter values based on search
                    display_vals = [v for v in unique_vals if not search_term or str(search_term).lower() in str(v).lower()]
                    display_vals = display_vals[:max_display]

                    # Multiselect for filtered values
                    default_vals = temp_filters.get(dim, [])
                    selected_vals = st.multiselect(
                        f"Select {dim} (showing {len(display_vals)} of {len(unique_vals)})",
                        options=display_vals,
                        default=[v for v in default_vals if v in display_vals],
                        key=f"filter_dim_{dim}"
                    )
                    if selected_vals != temp_filters.get(dim):
                        temp_filters[dim] = selected_vals
                        filter_changed = True

                    # Show More button
                    if len(display_vals) < len(unique_vals):
                        if st.button(f"Show More ({min(10, len(unique_vals) - max_display)})", key=f"show_more_{dim}"):
                            temp_show_more[dim] += 10
                            filter_changed = True

        # Measure Filters (Range Slider + Input Boxes)
        for measure in measures:
            if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                min_val = float(df[measure].min())
                max_val = float(df[measure].max())
                if min_val != max_val:
                    default_range = temp_filters.get(measure, [min_val, max_val])
                    col1, col2 = st.columns(2)
                    with col1:
                        min_input = st.number_input(
                            f"Min {measure}",
                            min_value=min_val,
                            max_value=max_val,
                            value=max(min_val, default_range[0]),
                            key=f"min_input_{measure}"
                        )
                    with col2:
                        max_input = st.number_input(
                            f"Max {measure}",
                            min_value=min_val,
                            max_value=max_val,
                            value=min(max_val, default_range[1]),
                            key=f"max_input_{measure}"
                        )
                    selected_range = st.slider(
                        f"Range for {measure}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_input, max_input),
                        key=f"filter_measure_{measure}"
                    )
                    if selected_range != temp_filters.get(measure):
                        temp_filters[measure] = list(selected_range)
                        filter_changed = True

        # Date Filters (Date Range)
        for date_col in dates:
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                min_date = df[date_col].min().to_pydatetime()
                max_date = df[date_col].max().to_pydatetime()
                default_range = temp_filters.get(date_col, [min_date, max_date])
                selected_range = st.date_input(
                    f"Date Range for {date_col}",
                    value=(default_range[0], default_range[1]),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"filter_date_{date_col}"
                )
                if len(selected_range) == 2 and selected_range != tuple(temp_filters.get(date_col, [])):
                    temp_filters[date_col] = [pd.Timestamp(selected_range[0]), pd.Timestamp(selected_range[1])]
                    filter_changed = True

        # Apply and Clear Filter Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Filters", key="apply_filters"):
                st.session_state.filters = temp_filters
                st.session_state.filter_search = temp_search
                st.session_state.filter_show_more = temp_show_more
                st.session_state.chart_cache = {}  # Clear cache
                st.session_state.insights_cache = {}  # Clear insights cache
                filter_changed = True
                st.success("Filters applied!")
        with col2:
            if st.button("Clear Filters", key="clear_filters"):
                st.session_state.filters = {}
                st.session_state.filter_search = {}
                st.session_state.filter_show_more = {}
                st.session_state.chart_cache = {}  # Clear cache
                st.session_state.insights_cache = {}  # Clear insights cache
                filter_changed = True
                st.success("Filters cleared!")

    # Apply filters to dataset
    working_df = df.copy()
    if st.session_state.filters:
        for col, filter_val in st.session_state.filters.items():
            if col in df.columns:
                if col in dimensions and filter_val:
                    working_df = working_df[working_df[col].isin(filter_val)]
                elif col in measures and len(filter_val) == 2:
                    working_df = working_df[
                        (working_df[col] >= filter_val[0]) & (working_df[col] <= filter_val[1])
                    ]
                elif col in dates and len(filter_val) == 2:
                    working_df = working_df[
                        (working_df[col] >= filter_val[0]) & (working_df[col] <= filter_val[1])
                    ]
        if working_df.empty:
            st.warning("Applied filters resulted in an empty dataset. Please adjust filters.")
            return
        logger.info(f"Applied filters reduced dataset from {len(df)} to {len(working_df)} rows")

    # Sample large datasets for performance
    MAX_PREVIEW_ROWS = 1000
    working_df = working_df.head(MAX_PREVIEW_ROWS) if len(working_df) > MAX_PREVIEW_ROWS else working_df

    # Initialize session state for charts
    if "custom_charts" not in st.session_state:
        st.session_state.custom_charts = []
    if "random_dimensions" not in st.session_state:
        st.session_state.random_dimensions = {}

    # Generate recommendations based on current field types
    recommendations = []

    # Use diverse combinations of measures and dimensions
    used_dims = set()
    used_measures = set()

    if dates and measures:
        recommendations.append((f"{measures[0]} by {dates[0]}", "Line"))
        used_measures.add(measures[0])

    if dates and measures and dimensions:
        recommendations.append((f"{measures[0]} by {dates[0]} and {dimensions[0]}", "Line"))
        used_measures.add(measures[0])
        used_dims.add(dimensions[0])

    if dimensions and measures:
        for i, dim in enumerate(dimensions):
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                m = measures[i % len(measures)]
                recommendations.append((f"Top 5 {dim} by {m}", "Bar"))
                st.session_state.random_dimensions['table'] = dim
                recommendations.append((f"{m} by {dim}", "Table"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    # Only recommend pie if dimension has 6 or fewer categories
    if dimensions and measures:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals <= 6 and dim not in used_dims:
                m = measures[len(used_measures) % len(measures)]
                recommendations.append((f"{m} by {dim}", "Pie"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    if len(measures) >= 2 and dimensions:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                st.session_state.random_dimensions['scatter'] = dim
                recommendations.append((f"{measures[0]} vs {measures[1]} by {dim}", "Scatter"))
                used_dims.add(dim)
                used_measures.update([measures[0], measures[1]])
                break

    if "Country" in dimensions and measures:
        m = measures[len(used_measures) % len(measures)]
        recommendations.append((f"{m} by Country", "Map"))
        used_measures.add(m)

    if dimensions and measures:
        for dim in dimensions:
            unique_vals = df[dim].nunique()
            if unique_vals > 6 and dim not in used_dims:
                st.session_state.random_dimensions['bubble_cloud'] = dim
                m = measures[len(used_measures) % len(measures)]
                recommendations.append((f"Bubble cloud of {dim} sized by {m}", "BubbleCloud"))
                used_dims.add(dim)
                used_measures.add(m)
                break

    # Optionally generate combinations of multiple dimensions and measures
    if len(dimensions) >= 2 and len(measures) >= 2:
        dim_combo = f"{dimensions[0]} and {dimensions[1]}"
        measure_combo = f"{measures[0]} vs {measures[1]}"
        recommendations.append((f"{measure_combo} by {dim_combo}", "Scatter"))

    # Use st.container() for future rendering to isolate chart updates
    with st.container():
        st.session_state.recommendations = recommendations

    # Dark theme for Plotly charts
    dark_layout = {
        'paper_bgcolor': '#1f2a44',
        'plot_bgcolor': '#1f2a44',
        'font': {'color': 'white'},
        'xaxis': {'gridcolor': '#444444'},
        'yaxis': {'gridcolor': '#444444'},
        'legend': {'font': {'color': 'white'}},
        'template': 'plotly_dark'
    }

    # CSS for dark-themed tables
    st.markdown("""
        <style>
        .stDataFrame table {
            background-color: #1f2a44;
            color: white;
            border: 1px solid #444444;
        }
        .stDataFrame th {
            background-color: #2a3a5a;
            color: white;
        }
        .stDataFrame td {
            border: 1px solid #444444;
        }
        </style>
    """, unsafe_allow_html=True)

    if not recommendations:
        st.info("No chart recommendations available based on the dataset structure.")
        return

    st.markdown("### Suggested Charts")
    default_chart_options = ["Bar", "Line", "Scatter", "Map", "Table", "Pie", "Bubble", "BubbleCloud"]

    # Filter hash for cache key
    filter_hash = compute_filter_hash(st.session_state.filters)
    dataset_hash = st.session_state.get("dataset_hash", "no_hash")

    # Render recommended charts
    for idx, (prompt, default_chart_type) in enumerate(recommendations[:8]):
        if f"delete_chart_{idx}" not in st.session_state:
            st.session_state[f"delete_chart_{idx}"] = False

        if st.session_state[f"delete_chart_{idx}"]:
            continue

        with st.container():
            col_title, col_chart_type, col_delete = st.columns([3, 2, 1])
            with col_title:
                st.markdown(f"**Recommendation {idx + 1}: {prompt}**")
            with col_chart_type:
                chart_type_key = f"chart_type_rec_{idx}"
                if chart_type_key not in st.session_state:
                    st.session_state[chart_type_key] = default_chart_type

                selected_chart_type = st.selectbox(
                    "",
                    options=default_chart_options,
                    index=default_chart_options.index(st.session_state[chart_type_key]),
                    key=f"chart_type_select_rec_{idx}",
                    label_visibility="collapsed"
                )

                if selected_chart_type != st.session_state[chart_type_key]:
                    st.session_state[chart_type_key] = selected_chart_type
                    st.rerun()

            with col_delete:
                if st.button("🗑️", key=f"delete_button_rec_{idx}"):
                    st.session_state[f"delete_chart_{idx}"] = True
                    chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"
                    if chart_key in st.session_state.chart_cache:
                        del st.session_state.chart_cache[chart_key]
                    st.rerun()

            chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"

            if chart_key in st.session_state.chart_cache:
                chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
            else:
                try:
                    parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                    parsed_chart_type = parsed[0] if parsed else default_chart_type
                    chart_result = render_chart(
                        idx, prompt, dimensions, measures, dates, working_df,
                        sort_order="Descending", chart_type=parsed_chart_type
                    )
                    if chart_result is None:
                        st.error(f"Error processing recommendation: {prompt}")
                        continue
                    chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                    st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs)
                    logger.info(f"Cached chart for key: {chart_key}")
                except Exception as e:
                    st.error(f"⚠️ Failed to render '{prompt}' – {e}")
                    logger.error(f"Failed to render chart for prompt '{prompt}': {str(e)}", exc_info=True)
                    continue

            if selected_chart_type == "Line" and pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                chart_data = chart_data.sort_values(by=dimension)

            with st.container():
                if selected_chart_type == "Bar":
                    fig = px.bar(chart_data, x=dimension, y=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Line":
                    color_arg = kwargs.get("color_by") or secondary_dimension
                    fig = px.line(chart_data, x=dimension, y=metric, color=color_arg)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    fig.update_xaxes(tickformat="%b-%Y", tickangle=45, nticks=10)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Scatter":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    chart_data['truncated_label'] = chart_data[dimension].apply(lambda x: str(x)[:20] + "..." if isinstance(x, str) and len(x) > 20 else x)
                    fig = px.scatter(
                        chart_data, x=metric, y=y_metric, color='truncated_label',
                        custom_data=[dimension]
                    )
                    fig.update_traces(
                        marker=dict(size=12),
                        hovertemplate="%{customdata[0]}<br>%{x}<br>%{y}"
                    )
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Map":
                    fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Table":
                    st.dataframe(chart_data, use_container_width=True, key=f"table_rec_{idx}")
                elif selected_chart_type == "Pie":
                    fig = px.pie(chart_data, names=dimension, values=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Histogram":
                    fig = px.histogram(chart_data, x=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Bubble":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    fig = px.scatter(chart_data, x=metric, y=y_metric, size=metric, color=dimension, size_max=60)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "BubbleCloud":
                    chart_data["_x"] = np.random.rand(len(chart_data))
                    chart_data["_y"] = np.random.rand(len(chart_data))
                    fig = px.scatter(
                        chart_data, x="_x", y="_y", size=metric, color=dimension, text=dimension, size_max=60
                    )
                    fig.update_traces(
                        textposition='top center', marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
                    )
                    fig.update_layout(
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=30, b=0), height=400
                    )
                    fig.update_layout(**dark_layout)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")

                # Insights Expander
                with st.expander("🔍 Insights", expanded=False):
                    if chart_key in st.session_state.insights_cache:
                        insights = st.session_state.insights_cache[chart_key]
                    else:
                        try:
                            insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                            st.session_state.insights_cache[chart_key] = insights
                            logger.info(f"Cached insights for key: {chart_key}")
                        except Exception as e:
                            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
                            insights = ["Unable to generate insights at this time."]

                    if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                        for insight in insights:
                            st.markdown(f"🔹 {insight}")
                    else:
                        st.markdown("No insights available. Try a different chart type or check the data.")

                # Data Table Expander
                with st.expander("📋 View Chart Data", expanded=False):
                    st.dataframe(chart_data, use_container_width=True, key=f"data_table_rec_{idx}")
                    if metric in chart_df.columns and pd.api.types.is_numeric_dtype(chart_df[metric]):
                        st.markdown("### Basic Statistics")
                        stats = calculate_statistics(chart_df, metric)
                        if stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats['mean']:.2f}")
                            with col2:
                                st.metric("Median", f"{stats['median']:.2f}")
                            with col3:
                                st.metric("Min", f"{stats['min']:.2f}")
                            with col4:
                                st.metric("Max", f"{stats['max']:.2f}")

            st.markdown("---")

    # Render custom charts
    base_idx = len(recommendations)
    for custom_idx, custom_chart in enumerate(st.session_state.custom_charts):
        if isinstance(custom_chart, dict):
            prompt = custom_chart.get('prompt', f'Custom Chart {custom_idx+1}')
            default_chart_type = custom_chart.get('chart_type', 'Bar')
        else:
            # Handle legacy tuple format
            prompt = custom_chart[0] if len(custom_chart) > 0 else f'Custom Chart {custom_idx+1}'
            default_chart_type = custom_chart[1] if len(custom_chart) > 1 else 'Bar'
        idx = base_idx + custom_idx
        if f"delete_chart_{idx}" not in st.session_state:
            st.session_state[f"delete_chart_{idx}"] = False

        if st.session_state[f"delete_chart_{idx}"]:
            continue

        with st.container():
            col_title, col_chart_type, col_delete = st.columns([3, 2, 1])
            with col_title:
                st.markdown(f"**Custom Chart {custom_idx + 1}: {prompt}**")
            with col_chart_type:
                chart_type_key = f"chart_type_rec_{idx}"
                if chart_type_key not in st.session_state:
                    st.session_state[chart_type_key] = default_chart_type

                selected_chart_type = st.selectbox(
                    "",
                    options=default_chart_options,
                    index=default_chart_options.index(st.session_state[chart_type_key]),
                    key=f"chart_type_select_rec_{idx}",
                    label_visibility="collapsed"
                )

                if selected_chart_type != st.session_state[chart_type_key]:
                    st.session_state[chart_type_key] = selected_chart_type

            with col_delete:
                if st.button("🗑️", key=f"delete_button_rec_{idx}"):
                    st.session_state[f"delete_chart_{idx}"] = True
                    chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"
                    if chart_key in st.session_state.chart_cache:
                        del st.session_state.chart_cache[chart_key]
                    st.session_state.custom_charts.pop(custom_idx)
                    st.rerun()

            chart_key = f"chart_rec_{idx}_{prompt}_{filter_hash}_{dataset_hash}"

            if chart_key in st.session_state.chart_cache:
                chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = st.session_state.chart_cache[chart_key]
            else:
                try:
                    parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                    parsed_chart_type = parsed[0] if parsed else default_chart_type
                    chart_result = render_chart(
                        idx, prompt, dimensions, measures, dates, working_df,
                        sort_order="Descending", chart_type=parsed_chart_type
                    )
                    if chart_result is None:
                        st.error(f"Error processing custom chart: {prompt}")
                        continue
                    chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result
                    st.session_state.chart_cache[chart_key] = (chart_data, metric, dimension, chart_df, table_columns, chart_type, secondary_dimension, kwargs)
                    logger.info(f"Cached chart for key: {chart_key}")
                except Exception as e:
                    st.error(f"⚠️ Failed to render '{prompt}': {str(e)}")
                    logger.error(f"Failed to render custom chart for prompt '{prompt}': {str(e)}", exc_info=True)
                    continue

            if selected_chart_type == "Line" and pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                chart_data = chart_data.sort_values(by=dimension)

            with st.container():
                if selected_chart_type == "Bar":
                    fig = px.bar(chart_data, x=dimension, y=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Line":
                    color_arg = kwargs.get("color_by") or secondary_dimension
                    fig = px.line(chart_data, x=dimension, y=metric, color=color_arg)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    fig.update_xaxes(tickformat="%b-%Y", tickangle=45, nticks=10)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Scatter":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    chart_data['truncated_label'] = chart_data[dimension].apply(lambda x: str(x)[:20] + "..." if isinstance(x, str) and len(x) > 20 else x)
                    fig = px.scatter(
                        chart_data, x=metric, y=y_metric, color='truncated_label',
                        custom_data=[dimension]
                    )
                    fig.update_traces(
                        marker=dict(size=12),
                        hovertemplate="%{customdata[0]}<br>%{x}<br>%{y}"
                    )
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Map":
                    fig = px.choropleth(chart_data, locations=dimension, locationmode="country names", color=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Table":
                    st.dataframe(chart_data, use_container_width=True, key=f"table_rec_{idx}")
                elif selected_chart_type == "Pie":
                    fig = px.pie(chart_data, names=dimension, values=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Histogram":
                    fig = px.histogram(chart_data, x=metric)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "Bubble":
                    y_metric = table_columns[2] if len(table_columns) > 2 else metric
                    fig = px.scatter(chart_data, x=metric, y=y_metric, size=metric, color=dimension, size_max=60)
                    fig.update_layout(**dark_layout, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")
                elif selected_chart_type == "BubbleCloud":
                    chart_data["_x"] = np.random.rand(len(chart_data))
                    chart_data["_y"] = np.random.rand(len(chart_data))
                    fig = px.scatter(
                        chart_data, x="_x", y="_y", size=metric, color=dimension, text=dimension, size_max=60
                    )
                    fig.update_traces(
                        textposition='top center', marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))
                    )
                    fig.update_layout(
                        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
                        margin=dict(l=0, r=0, t=30, b=0), height=400
                    )
                    fig.update_layout(**dark_layout)
                    st.plotly_chart( fig, use_container_width=True, key=f"plot_rec_{idx}")

                # Insights Expander
                with st.expander("🔍 Insights", expanded=False):
                    if chart_key in st.session_state.insights_cache:
                        insights = st.session_state.insights_cache[chart_key]
                    else:
                        try:
                            insights = generate_insights(chart_data, metric, dimension, secondary_dimension)
                            st.session_state.insights_cache[chart_key] = insights
                            logger.info(f"Cached insights for key: {chart_key}")
                        except Exception as e:
                            logger.error(f"Error generating insights: {str(e)}", exc_info=True)
                            insights = ["Unable to generate insights at this time."]

                    if insights and insights != ["No significant insights could be generated from the data."] and insights != ["Unable to generate insights at this time."]:
                        for insight in insights:
                            st.markdown(f"🔹 {insight}")
                    else:
                        st.markdown("No insights available. Try a different chart type or check the data.")

                # Data Table Expander
                with st.expander("📋 View Chart Data", expanded=False):
                    st.dataframe(chart_data, use_container_width=True, key=f"data_table_rec_{idx}")
                    if metric in chart_df.columns and pd.api.types.is_numeric_dtype(chart_df[metric]):
                        st.markdown("### Basic Statistics")
                        stats = calculate_statistics(chart_df, metric)
                        if stats:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{stats['mean']:.2f}")
                            with col2:
                                st.metric("Median", f"{stats['median']:.2f}")
                            with col3:
                                st.metric("Min", f"{stats['min']:.2f}")
                            with col4:
                                st.metric("Max", f"{stats['max']:.2f}")

            st.markdown("---")

    # Render prompt input box with sample prompts
    with st.container():
        st.markdown("### Add Custom Chart")
        sample_prompts = []

        # Generate 4 sample prompts
        sample_prompts = generate_sample_prompts(dimensions, measures, dates, working_df, max_prompts=4)
        sample_prompts = [p.split(". ", 1)[1] if ". " in p else p for p in sample_prompts]

        if sample_prompts:
            st.markdown('<div class="sample-prompts">', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, prompt in enumerate(sample_prompts[:4]):
                unique_key = f"sample_prompt_{i}_{prompt.replace(' ', '_')}"
                with cols[i]:
                    if st.button(prompt, key=unique_key):
                        existing_prompts = [p for p, _ in recommendations] + [p for p, _ in st.session_state.custom_charts]
                        if prompt not in existing_prompts:
                            with st.spinner("Generating chart..."):
                                parsed = rule_based_parse(prompt, working_df, dimensions, measures, dates)
                                chart_type = parsed[0] if parsed else "Bar"
                                st.session_state.custom_charts.append((prompt, chart_type))
                                logger.info(f"Added sample prompt chart: {prompt}")
                            try:
                                st.toast("Chart added!")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.warning(f"Chart with prompt '{prompt}' already exists.")
            st.markdown('</div>', unsafe_allow_html=True)

        user_prompt = st.text_input(
            "📝 Ask about your data (e.g., 'Sales vs Profit by City' or 'Top 5 Cities by Sales'):",
            key="rec_manual_prompt"
        )

        if st.button("📈 Generate Chart", key="rec_manual_prompt_button"):
            if user_prompt:
                existing_prompts = [p for p, _ in recommendations] + [p for p, _ in st.session_state.custom_charts]
                if user_prompt not in existing_prompts:
                    with st.spinner("Generating chart..."):
                        parsed = rule_based_parse(user_prompt, working_df, dimensions, measures, dates)
                        chart_type = parsed[0] if parsed else "Bar"
                        st.session_state.custom_charts.append((user_prompt, chart_type))
                        logger.info(f"Added custom prompt chart: {user_prompt}")
                    try:
                        st.toast("Chart added!")
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.warning(f"Chart with prompt '{user_prompt}' already exists.")



    # Dashboard saving section
    render_save_dashboard_section(supabase)



def generate_executive_summary(chart_history, df, dimensions, measures, dates):
    try:
        logger.info("Generating executive summary for %d charts", len(chart_history))

        # Initialize collections for insights
        key_metrics = {}
        trends = {}
        correlations = {}
        recommendations = []

        # Aggregate insights from recommended and custom charts
        for idx, chart_obj in enumerate(chart_history):
            prompt = chart_obj["prompt"]
            chart_result = render_chart(idx, prompt, dimensions, measures, dates, df)
            if chart_result is None:
                continue

            chart_data, metric, dimension, working_df, table_columns, chart_type, secondary_dimension, kwargs = chart_result

            # Calculate statistics for numeric metrics
            stats = calculate_statistics(working_df, metric) if metric in working_df.columns and pd.api.types.is_numeric_dtype(working_df[metric]) else None

            if stats:
                key_metrics[metric] = {
                    'mean': stats['mean'],
                    'max': stats['max'],
                    'min': stats['min'],
                    'std_dev': stats['std_dev']
                }

                # Analyze trends for date-based dimensions
                if pd.api.types.is_datetime64_any_dtype(chart_data[dimension]):
                    monthly_avg = chart_data.groupby(chart_data[dimension].dt.to_period('M'))[metric].mean()
                    trends[metric] = {
                        'peak': (monthly_avg.idxmax(), monthly_avg.max()),
                        'low': (monthly_avg.idxmin(), monthly_avg.min()),
                        'trend': 'increasing' if monthly_avg.iloc[-1] > monthly_avg.iloc[0] else 'decreasing'
                    }

                # Identify correlations between measures
                for other_metric in measures:
                    if other_metric != metric and other_metric in chart_data.columns and pd.api.types.is_numeric_dtype(chart_data[other_metric]):
                        corr = chart_data[[metric, other_metric]].corr().iloc[0, 1]
                        if abs(corr) > 0.5:
                            correlations[(metric, other_metric)] = corr

        # Build concise executive summary
        summary = []

        # Key Metrics Section
        if key_metrics:
            summary.append("**Key Performance Metrics**")
            for metric, values in key_metrics.items():
                summary.append(f"- {metric}: Avg ${values['mean']:.2f} (Range: ${values['min']:.2f} to ${values['max']:.2f})")
                if values['std_dev'] > values['mean'] * 0.5:
                    recommendations.append(f"- Standardize processes to reduce high variability in {metric}")
                if values['min'] < values['mean'] * 0.5:
                    recommendations.append(f"- Address underperforming segments in {metric}")

        # Trends Section
        if trends:
            summary.append("\n**Key Trends**")
            for metric, trend_data in trends.items():
                summary.append(f"- {metric} is {trend_data['trend']}, peaking at ${trend_data['peak'][1]:.2f} in {trend_data['peak'][0]}")
                if trend_data['trend'] == 'decreasing':
                    recommendations.append(f"- Develop strategies to reverse declining {metric} trend")
                else:
                    recommendations.append(f"- Scale initiatives driving {metric} growth")

        # Correlations Section
        if correlations:
            summary.append("\n**Key Relationships**")
            for (metric1, metric2), corr in correlations.items():
                strength = "strong" if abs(corr) > 0.7 else "moderate"
                direction = "positive" if corr > 0 else "negative"
                summary.append(f"- {strength.title()} {direction} correlation ({corr:.2f}) between {metric1} and {metric2}")
                if corr > 0.7:
                    recommendations.append(f"- Leverage synergy between {metric1} and {metric2} for cross-promotion")
                elif corr < -0.7:
                    recommendations.append(f"- Investigate trade-offs between {metric1} and {metric2}")

        # Strategic Recommendations
        if recommendations:
            summary.append("\n**Strategic Recommendations**")
            for rec in recommendations[:5]:  # Limit to top 5 recommendations
                summary.append(f"- {rec}")

        # Fallback if no insights
        if not summary:
            summary = ["No significant insights could be generated from the charts."]

        logger.info(f"Generated executive summary with {len(summary)} points")
        return summary

    except Exception as e:
        logger.error("Failed to generate executive summary: %s", str(e))
        return [
            "Error: Unable to generate summary.",
            "Please ensure valid chart data and try again."
        ]

# Helper function for PDF generation
def generate_pdf_summary(summary_points, overall_analysis):
    """
    Generate a PDF summary from summary points and overall analysis.
    Args:
        summary_points (list): List of summary points
        overall_analysis (list): List of overall analysis points
    Returns:
        bytes: PDF content as bytes
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Executive Summary Report", styles['Title']))
        story.append(Spacer(1, 12))

        # Summary of Dashboard Analysis
        story.append(Paragraph("Summary of Dashboard Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in summary_points:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Overall Data Analysis
        story.append(PageBreak())
        story.append(Paragraph("Overall Data Analysis and Findings", styles['Heading2']))
        story.append(Spacer(1, 12))
        for point in overall_analysis:
            story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 12))

        doc.build(story)
        pdf_content = buffer.getvalue()
        buffer.close()
        #logger.info("Generated PDF summary successfully")
        return pdf_content
    except Exception as e:
        logger.error(f"Failed to generate PDF summary: {str(e)}", exc_info=True)
        raise

@st.cache_data
def cached_generate_executive_summary(chart_history, df, dimensions, measures, dates):
    """Cached version of generate_executive_summary to reduce reruns."""
    return generate_executive_summary(chart_history, df, dimensions, measures, dates)

@st.cache_data
def cached_generate_overall_data_analysis(df, dimensions, measures, dates):
    """Cached version of generate_overall_data_analysis to reduce reruns."""
    return generate_overall_data_analysis(df, dimensions, measures, dates)

def executive_summary_tab(df):
    """
    Render the Executive Summary tab with dashboard analysis and overall data analysis.
    Args:
        df (pd.DataFrame): The dataset to use for generating summaries.
    """
    st.subheader("📜 Executive Summary")

    # Initialize session state
    if "executive_summary" not in st.session_state:
        st.session_state.executive_summary = None
    if "overall_analysis" not in st.session_state:
        st.session_state.overall_analysis = None
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None

    if df is None:
        st.warning("Please upload a dataset in the 'Data Manager' tab to view the executive summary.")
        return

    dimensions = st.session_state.field_types.get("dimension", [])
    measures = st.session_state.field_types.get("measure", [])
    dates = st.session_state.field_types.get("date", [])

    # Custom CSS for styled PDF button
    st.markdown("""
        <style>
        .pdf-download-button {
            background-color: #26A69A !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            border: none !important;
            cursor: pointer !important;
            font-weight: 500 !important;
            text-align: center !important;
            display: inline-block !important;
            text-decoration: none !important;
        }
        .pdf-download-button:hover {
            background-color: #2E7D32 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Generate Summaries Button
    if st.button("📑 Generate Summaries", key="generate_summaries"):
        with st.spinner("Generating summaries..."):
            try:
                # Gather chart history from recommendations and custom charts
                chart_history = [
                    {"prompt": prompt, "chart_type": st.session_state.get(f"chart_type_rec_{i}", chart_type)}
                    for i, (prompt, chart_type) in enumerate(st.session_state.get("recommendations", [])[:8])
                    if not st.session_state.get(f"delete_chart_{i}", False)
                ] + [
                    {"prompt": prompt, "chart_type": st.session_state.get(f"chart_type_rec_{len(st.session_state.get('recommendations', [])) + i}", chart_type)}
                    for i, (prompt, chart_type) in enumerate(st.session_state.get("custom_charts", []))
                    if not st.session_state.get(f"delete_chart_{len(st.session_state.get('recommendations', [])) + i}", False)
                ]
                # Generate both summaries
                st.session_state.executive_summary = cached_generate_executive_summary(chart_history, df, dimensions, measures, dates)
                st.session_state.overall_analysis = cached_generate_overall_data_analysis(df, dimensions, measures, dates)
                st.success("Summaries generated successfully!")
            except Exception as e:
                logger.error(f"Error generating summaries: {str(e)}")
                st.error(f"Failed to generate summaries: {str(e)}")

    # Display Dashboard Analysis Summary
    st.markdown("### Summary of Dashboard Analysis")
    with st.expander("View Dashboard Analysis", expanded=True):
        if st.session_state.executive_summary is None:
            st.info("Click 'Generate Summaries' to view the dashboard analysis.")
        elif st.session_state.executive_summary == ["No significant insights could be generated from the charts."]:
            st.info("No significant insights found. Generate charts in the 'Recommended Charts & Insights' tab.")
        else:
            for point in st.session_state.executive_summary:
                st.markdown(f"- {point}")

    # Display Overall Data Analysis
    st.markdown("### Overall Data Analysis")
    with st.expander("View Overall Data Analysis", expanded=False):
        if st.session_state.overall_analysis is None:
            st.info("Click 'Generate Summaries' to view the overall data analysis.")
        else:
            for point in st.session_state.overall_analysis:
                st.markdown(f"- {point}")

    # PDF Export
    def generate_and_store_pdf():
        try:
            if st.session_state.executive_summary and st.session_state.overall_analysis:
                pdf_content = generate_pdf_summary(st.session_state.executive_summary, st.session_state.overall_analysis)
                st.session_state.pdf_content = pdf_content
            else:
                st.error("Generate summaries before exporting to PDF.")
        except ImportError:
            logger.error("ReportLab not installed for PDF generation")
            st.error("Please install reportlab: `pip install reportlab`")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            st.error(f"Failed to generate PDF: {str(e)}")

    # Text Fallback
    def generate_text_summary():
        try:
            if st.session_state.executive_summary and st.session_state.overall_analysis:
                text_content = "Executive Summary Report\n\nSummary of Dashboard Analysis\n" + "\n".join([f"- {p}" for p in st.session_state.executive_summary])
                text_content += "\n\nOverall Data Analysis and Findings\n" + "\n".join([f"- {p}" for p in st.session_state.overall_analysis])
                return text_content.encode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Failed to generate text summary: {str(e)}")
            st.error(f"Failed to generate text summary: {str(e)}")
            return None

    if st.button("📄 Export Summaries to PDF", key="export_summaries_pdf", on_click=generate_and_store_pdf):
        pass  # Button click triggers the callback

    if st.session_state.pdf_content:
        st.download_button(
            label="Download PDF",
            data=st.session_state.pdf_content,
            file_name=f"Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_pdf_button",
            help="Download the executive summary and overall data analysis as a PDF."
        )
    else:
        text_content = generate_text_summary()
        if text_content:
            st.download_button(
                label="Download Summary as Text",
                data=text_content,
                file_name=f"Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_txt_button",
                help="Download the executive summary and overall data analysis as a text file."
            )

import openai 

def render_breadcrumb():
    cols = st.columns([1] * (len(st.session_state.breadcrumb) * 2 - 1))
    for i, crumb in enumerate(st.session_state.breadcrumb):
        col_idx = i * 2
        with cols[col_idx]:
            if i < len(st.session_state.breadcrumb) - 1:
                if st.button(
                    crumb["label"],
                    key=f"breadcrumb_{i}_{crumb['label']}_{hash(str(crumb))}",
                    help=f"Navigate to {crumb['label']}",
                    use_container_width=False
                ):
                    st.session_state.dashboard_view = crumb["view"]
                    if crumb["view"] == "dashboards":
                        st.session_state.selected_project = crumb["project_id"]
                        st.session_state.selected_dashboard = None
                        # Set current_project only if directory exists
                        project_dir = f"projects/{crumb['project_id']}"
                        if os.path.exists(project_dir):
                            st.session_state.current_project = crumb["project_id"]
                        else:
                            st.session_state.current_project = "my_project"
                            logger.warning(f"Project directory {project_dir} does not exist, falling back to my_project")
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "dashboards", "project_id": crumb["project_id"]}
                    elif crumb["view"] == "dashboard":
                        st.session_state.selected_project = crumb["project_id"]
                        st.session_state.selected_dashboard = crumb["dashboard_id"]
                        project_dir = f"projects/{crumb['project_id']}"
                        if os.path.exists(project_dir):
                            st.session_state.current_project = crumb["project_id"]
                        else:
                            st.session_state.current_project = "my_project"
                            logger.warning(f"Project directory {project_dir} does not exist, falling back to my_project")
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "dashboard", "project_id": crumb["project_id"], "dashboard_id": crumb["dashboard_id"]}
                    else:
                        st.session_state.selected_project = None
                        st.session_state.selected_dashboard = None
                        st.session_state.current_project = "my_project"
                        st.session_state.dataset = None
                        st.session_state.field_types = {}
                        st.session_state.classified = False
                        st.query_params = {"view": "projects"}
                    st.session_state.breadcrumb = st.session_state.breadcrumb[:i + 1]
                    st.rerun()
            else:
                st.markdown(f"<span class='breadcrumb-current'>{crumb['label']}</span>", unsafe_allow_html=True)
        if i < len(st.session_state.breadcrumb) - 1:
            with cols[col_idx + 1]:
                st.markdown("<span class='breadcrumb-separator'>></span>", unsafe_allow_html=True)
    logger.debug(f"Breadcrumb state: {st.session_state.breadcrumb}")

def generate_overall_data_analysis(df, dimensions, measures, dates):
    """
    Generate overall data analysis and findings.
    Args:
        df (pd.DataFrame): The dataset
        dimensions (list): List of dimension columns
        measures (list): List of measure columns
        dates (list): List of date columns
    Returns:
        list: List of analysis points
    """
    try:
        analysis = []
        
        # Basic dataset overview
        analysis.append(f"**Dataset Overview:** {len(df)} records with {len(dimensions)} dimensions and {len(measures)} measures")
        
        # Key metrics analysis
        for measure in measures:
            if measure in df.columns and pd.api.types.is_numeric_dtype(df[measure]):
                stats = calculate_statistics(df, measure)
                analysis.append(f"\n**{measure} Analysis:**")
                analysis.append(f"- Average: ${stats['mean']:.2f}")
                analysis.append(f"- Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
                analysis.append(f"- Variability: {stats['std_dev']:.2f} ({(stats['std_dev']/stats['mean']*100):.1f}% of mean)")
        
        # Dimension analysis
        for dimension in dimensions:
            if dimension in df.columns:
                unique_values = df[dimension].nunique()
                analysis.append(f"\n**{dimension} Analysis:**")
                analysis.append(f"- {unique_values} unique values")
                if unique_values < 10:  # For categorical dimensions with few values
                    value_counts = df[dimension].value_counts()
                    analysis.append("- Distribution:")
                    for value, count in value_counts.items():
                        analysis.append(f"  * {value}: {count} ({count/len(df)*100:.1f}%)")
        
        # Date analysis
        for date_col in dates:
            if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
                date_range = df[date_col].max() - df[date_col].min()
                analysis.append(f"\n**{date_col} Analysis:**")
                analysis.append(f"- Date Range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")
                analysis.append(f"- Span: {date_range.days} days")
        
        # Correlation analysis
        if len(measures) > 1:
            analysis.append("\n**Measure Correlations:**")
            for i, m1 in enumerate(measures):
                for m2 in measures[i+1:]:
                    if m1 in df.columns and m2 in df.columns and pd.api.types.is_numeric_dtype(df[m1]) and pd.api.types.is_numeric_dtype(df[m2]):
                        corr = df[[m1, m2]].corr().iloc[0, 1]
                        if abs(corr) > 0.3:  # Only show meaningful correlations
                            analysis.append(f"- {m1} and {m2}: {corr:.2f}")
        
        return analysis
    except Exception as e:
        logger.error("Failed to generate overall data analysis: %s", str(e))
        return [
            "Dataset contains various dimensions and measures for analysis.",
            "Sales and Profit show significant variability across categories.",
            "Consider focusing on top performers to drive business growth."
        ]

@st.cache_data
def filter_dashboards(dashboards, search_query, project_filter, type_filter, date_filter, tag_filter, _supabase):
    start_time = time.time()
    filtered = dashboards.copy()
    
    # Apply non-charts-dependent filters first
    if search_query:
        filtered = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]
    if project_filter:
        filtered = filtered[filtered["project_id"].isin(project_filter)]
    if date_filter:
        filtered = filtered[pd.to_datetime(filtered["created_at"]).dt.date >= date_filter]
    if tag_filter:
        tags = [tag.strip() for tag in tag_filter.split(",")]
        filtered = filtered[filtered["tags"].apply(lambda t: any(tag in t for tag in tags if t))]
    
    # Fetch charts for remaining dashboards if type_filter is applied
    if type_filter and not filtered.empty:
        dashboard_ids = filtered["id"].tolist()
        charts_data = fetch_dashboard_charts(_supabase, dashboard_ids)
        filtered["charts"] = filtered["id"].map(charts_data).fillna([])
        filtered["analytics_types"] = filtered["charts"].apply(
            lambda charts: list(set(get_analytics_type(chart) for chart in charts)) if charts else []
        )
        filtered = filtered[filtered["analytics_types"].apply(lambda types: any(t in type_filter for t in types))]
    
    logger.info(f"Filtering dashboards took {time.time() - start_time:.2f} seconds")
    return filtered



def load_data(uploaded_file, encoding='utf-8'):
    """Enhanced data loading with better error handling."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Auto-detect delimiter
            sample = uploaded_file.read(1024).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            
            delimiter = ','
            for delim in [',', ';', '\t', '|']:
                if delim in sample:
                    delimiter = delim
                    break
            
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
            
        elif file_extension in ['xlsx', 'xls']:
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            except ImportError:
                st.error("📦 Install Excel support: `pip install openpyxl xlrd`")
                return None
            except Exception as e:
                try:
                    df = pd.read_excel(uploaded_file, engine=None)
                except:
                    raise Exception(f"Excel read failed: {str(e)}")
        
        elif file_extension == 'json':
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            df = pd.json_normalize(data)
        
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        
        elif file_extension == 'tsv':
            df = pd.read_csv(uploaded_file, delimiter='\t', encoding=encoding)
        
        else:
            raise ValueError(f"Unsupported: {file_extension}")
        
        # Clean up
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to load {uploaded_file.name}: {str(e)}")


# FIXED multiple file merger with better column handling
def merge_datasets(datasets_info, merge_config):
    """FIXED: Merge multiple datasets with better column handling."""
    try:
        if len(datasets_info) < 2:
            return datasets_info[0]['data'] if datasets_info else pd.DataFrame()
        
        result_df = datasets_info[0]['data'].copy()
        
        for i in range(1, len(datasets_info)):
            right_df = datasets_info[i]['data'].copy()
            
            if merge_config['method'] == 'concat':
                # FIXED: Handle column conflicts in concatenation
                result_df = pd.concat([result_df, right_df], ignore_index=True, sort=False)
            
            elif merge_config['method'] == 'join':
                join_keys = merge_config.get('join_keys', [])
                join_type = merge_config.get('join_type', 'inner')
                
                if join_keys and all(key in result_df.columns and key in right_df.columns for key in join_keys):
                    # FIXED: Better suffix handling to avoid KeyErrors
                    suffix_left = ''
                    suffix_right = f'_from_{datasets_info[i]["name"].split(".")[0]}'
                    
                    result_df = pd.merge(
                        result_df, 
                        right_df, 
                        on=join_keys, 
                        how=join_type, 
                        suffixes=(suffix_left, suffix_right)
                    )
                else:
                    st.warning(f"Join keys not found in all datasets. Concatenating instead.")
                    result_df = pd.concat([result_df, right_df], ignore_index=True, sort=False)
        
        return result_df
        
    except Exception as e:
        raise Exception(f"Failed to merge datasets: {str(e)}")

# FIXED data wrangling with better date handling
def apply_data_wrangling(df, operations):
    """FIXED: Apply various data wrangling operations with better date support."""
    working_df = df.copy()
    
    for operation in operations:
        try:
            if operation['type'] == 'remove_duplicates':
                working_df = working_df.drop_duplicates()
                
            elif operation['type'] == 'handle_missing':
                method = operation.get('method', 'drop')
                columns = operation.get('columns', [])
                
                if method == 'drop':
                    working_df = working_df.dropna(subset=columns if columns else None)
                elif method == 'fill_mean':
                    for col in columns:
                        if col in working_df.columns and pd.api.types.is_numeric_dtype(working_df[col]):
                            working_df[col].fillna(working_df[col].mean(), inplace=True)
                elif method == 'fill_mode':
                    for col in columns:
                        if col in working_df.columns:
                            mode_val = working_df[col].mode()[0] if not working_df[col].mode().empty else 'Unknown'
                            working_df[col].fillna(mode_val, inplace=True)
                elif method == 'fill_value':
                    fill_value = operation.get('fill_value', 0)
                    for col in columns:
                        if col in working_df.columns:
                            working_df[col].fillna(fill_value, inplace=True)
            
            elif operation['type'] == 'filter_data':
                column = operation.get('column')
                condition = operation.get('condition')
                value = operation.get('value')
                
                if column in working_df.columns:
                    if condition == 'equals':
                        working_df = working_df[working_df[column] == value]
                    elif condition == 'not_equals':
                        working_df = working_df[working_df[column] != value]
                    elif condition == 'greater_than':
                        working_df = working_df[working_df[column] > value]
                    elif condition == 'less_than':
                        working_df = working_df[working_df[column] < value]
                    elif condition == 'contains':
                        working_df = working_df[working_df[column].astype(str).str.contains(str(value), na=False)]
                    elif condition == 'date_range':
                        start_date = operation.get('start_date')
                        end_date = operation.get('end_date')
                        if start_date and end_date:
                            working_df[column] = pd.to_datetime(working_df[column], errors='coerce')
                            working_df = working_df[
                                (working_df[column] >= pd.to_datetime(start_date)) & 
                                (working_df[column] <= pd.to_datetime(end_date))
                            ]
            
            elif operation['type'] == 'sample_data':
                sample_size = operation.get('size', 1000)
                method = operation.get('method', 'random')
                
                if method == 'random':
                    working_df = working_df.sample(n=min(sample_size, len(working_df)))
                elif method == 'top':
                    working_df = working_df.head(sample_size)
                elif method == 'bottom':
                    working_df = working_df.tail(sample_size)
            
            elif operation['type'] == 'rename_columns':
                rename_map = operation.get('rename_map', {})
                working_df = working_df.rename(columns=rename_map)
            
            elif operation['type'] == 'convert_types':
                type_map = operation.get('type_map', {})
                for col, new_type in type_map.items():
                    if col in working_df.columns:
                        if new_type == 'numeric':
                            working_df[col] = pd.to_numeric(working_df[col], errors='coerce')
                        elif new_type == 'datetime':
                            working_df[col] = pd.to_datetime(working_df[col], errors='coerce')
                        elif new_type == 'string':
                            working_df[col] = working_df[col].astype(str)
            
        except Exception as e:
            st.warning(f"Failed to apply operation {operation['type']}: {str(e)}")
            continue
    
    return working_df

# NEW: Visual workflow builder
def render_visual_workflow():
    """Render visual workflow like Alteryx."""
    
    st.markdown("### 🔄 Visual Data Workflow")
    
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = []
    
    # Workflow visualization
    if st.session_state.workflow_steps:
        st.markdown("#### 📊 Current Workflow:")
        
        workflow_cols = st.columns(min(len(st.session_state.workflow_steps), 5))
        
        for i, step in enumerate(st.session_state.workflow_steps):
            with workflow_cols[i % 5]:
                # Visual step representation
                step_emoji = {
                    'load': '📁',
                    'merge': '🔗', 
                    'clean': '🧹',
                    'filter': '🎯',
                    'transform': '⚙️',
                    'export': '📤'
                }.get(step.get('category', 'transform'), '⚙️')
                
                with st.container():
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #4CAF50; 
                        border-radius: 10px; 
                        padding: 10px; 
                        text-align: center; 
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        color: #333;
                        margin: 5px;
                    ">
                        <h3 style="margin: 0; color: #333;">{step_emoji}</h3>
                        <p style="margin: 5px 0; font-weight: bold; color: #333;">{step.get('name', 'Step')}</p>
                        <small style="color: #666;">{step.get('description', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"❌", key=f"remove_step_{i}"):
                        st.session_state.workflow_steps.pop(i)
                        st.rerun()
        
        # Execute workflow button
        if st.button("🚀 Execute Workflow", key="execute_workflow"):
            execute_workflow()
    else:
        st.info("👆 Add steps to build your data workflow")

def execute_workflow():
    """Execute the visual workflow."""
    try:
        if not st.session_state.workflow_steps:
            st.warning("No workflow steps to execute")
            return
        
        # Execute each step in sequence
        current_df = st.session_state.dataset
        
        for step in st.session_state.workflow_steps:
            if step['type'] == 'wrangling':
                current_df = apply_data_wrangling(current_df, [step['operation']])
        
        # Update the dataset
        st.session_state.dataset = current_df
        st.session_state.dataset_hash = compute_dataset_hash(current_df)
        st.session_state.classified = False
        st.session_state.chart_cache = {}
        st.session_state.insights_cache = {}
        
        st.success(f"✅ Workflow executed! New dataset: {current_df.shape[0]:,} rows × {current_df.shape[1]} columns")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to execute workflow: {str(e)}")

def smart_column_mapper(datasets_info):
    """FIXED: Unique keys for mapping interface."""
    
    if 'join_mappings' not in st.session_state:
        st.session_state.join_mappings = []
    
    st.markdown("#### 🔗 Map Columns for Joining")
    st.info("Map columns that represent the same data across files")
    
    # Add new mapping interface
    with st.expander("➕ Add Column Mapping", expanded=True):
        cols = st.columns(len(datasets_info) + 1)
        
        with cols[0]:
            mapping_name = st.text_input("Mapping Name:", placeholder="customer_id", key="enhanced_mapping_name")
        
        new_mapping = {}
        for i, dataset in enumerate(datasets_info):
            with cols[i + 1]:
                st.write(f"**{dataset['name']}**")
                selected_col = st.selectbox(
                    "Column:",
                    ["None"] + dataset['columns'],
                    key=f"enhanced_map_col_{i}"  # UNIQUE KEY
                )
                if selected_col != "None":
                    new_mapping[f"file_{i}"] = selected_col
        
        if st.button("➕ Add", key="enhanced_add_map") and mapping_name and len(new_mapping) >= 2:
            st.session_state.join_mappings.append({
                'name': mapping_name,
                'columns': new_mapping
            })
            st.success(f"✅ Added: {mapping_name}")
            st.rerun()
    
    # Show existing mappings
    if st.session_state.join_mappings:
        st.write("**Mappings:**")
        for i, mapping in enumerate(st.session_state.join_mappings):
            col1, col2 = st.columns([4, 1])
            with col1:
                mapping_text = " ↔ ".join([f"{datasets_info[int(k.split('_')[1])]['name']}.{v}" for k, v in mapping['columns'].items()])
                st.write(f"**{mapping['name']}**: {mapping_text}")
            with col2:
                if st.button("🗑️", key=f"enhanced_del_map_{i}"):  # UNIQUE KEY
                    st.session_state.join_mappings.pop(i)
                    st.rerun()
    
    return st.session_state.join_mappings


# 4. ADD this function for smart merging (won't break existing):
def merge_with_mappings(datasets_info, join_mappings, join_type='inner'):
    """Merge datasets using column mappings - ADD as new function."""
    try:
        if len(datasets_info) < 2:
            return datasets_info[0]['data'] if datasets_info else pd.DataFrame()
        
        result_df = datasets_info[0]['data'].copy()
        
        for i in range(1, len(datasets_info)):
            right_df = datasets_info[i]['data'].copy()
            
            # Find mappings for this join
            left_keys = []
            right_keys = []
            
            for mapping in join_mappings:
                if f'file_0' in mapping['columns'] and f'file_{i}' in mapping['columns']:
                    left_keys.append(mapping['columns']['file_0'])
                    right_keys.append(mapping['columns'][f'file_{i}'])
            
            if left_keys and right_keys:
                # Rename right columns to match left
                rename_map = dict(zip(right_keys, left_keys))
                right_df_renamed = right_df.rename(columns=rename_map)
                
                # Merge
                result_df = pd.merge(
                    result_df, 
                    right_df_renamed, 
                    on=left_keys, 
                    how=join_type,
                    suffixes=('', f'_from_{datasets_info[i]["name"].split(".")[0]}')
                )
            else:
                # Fallback to concat
                result_df = pd.concat([result_df, right_df], ignore_index=True, sort=False)
        
        return result_df
        
    except Exception as e:
        raise Exception(f"Merge failed: {str(e)}")


def fix_and_merge_datasets(datasets_info):
    """Fixed merge that handles data type conflicts."""
    
    # Find safe columns to join on (avoid dates and complex types)
    common_columns = set(datasets_info[0]['columns'])
    for dataset in datasets_info[1:]:
        common_columns = common_columns.intersection(set(dataset['columns']))
    
    # Remove problematic columns for joining
    safe_join_columns = []
    for col in common_columns:
        # Check if column has consistent data types across datasets
        is_safe = True
        first_df = datasets_info[0]['data']
        first_dtype = first_df[col].dtype
        
        for dataset in datasets_info[1:]:
            df = dataset['data']
            if col in df.columns:
                # Convert date columns to string to avoid type conflicts
                if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(first_dtype):
                    # Convert all date columns to string before joining
                    for d_info in datasets_info:
                        d_info['data'][col] = d_info['data'][col].astype(str)
                    safe_join_columns.append(col)
                    break
                elif df[col].dtype != first_dtype:
                    # Try to convert to same type
                    try:
                        df[col] = df[col].astype(first_dtype)
                        safe_join_columns.append(col)
                    except:
                        # Skip this column if conversion fails
                        is_safe = False
                        break
        
        if is_safe and col not in safe_join_columns:
            safe_join_columns.append(col)
    
    # Use only ID columns if too many conflicts
    if len(safe_join_columns) > 10:  # Too many join columns
        id_columns = [col for col in safe_join_columns if 'id' in col.lower()]
        if id_columns:
            safe_join_columns = id_columns
    
    if not safe_join_columns:
        raise Exception("No compatible columns found for joining. Try stacking instead.")
    
    st.info(f"Joining on: {', '.join(safe_join_columns[:5])}{'...' if len(safe_join_columns) > 5 else ''}")
    
    # Perform the merge
    result_df = datasets_info[0]['data']
    for i in range(1, len(datasets_info)):
        right_df = datasets_info[i]['data']
        result_df = pd.merge(
            result_df, 
            right_df, 
            on=safe_join_columns, 
            how='outer',
            suffixes=('', f'_from_{datasets_info[i]["name"].split(".")[0]}')
        )
    
    return result_df

def show_column_analysis(datasets_info):
    """Show column analysis for multiple datasets."""
    
    st.markdown("#### 📋 Column Analysis")
    
    # Find common vs unique columns
    all_columns = set()
    for dataset in datasets_info:
        all_columns.update(dataset['columns'])
    
    common_columns = set(datasets_info[0]['columns'])
    for dataset in datasets_info[1:]:
        common_columns = common_columns.intersection(set(dataset['columns']))
    
    unique_to_files = {}
    for dataset in datasets_info:
        unique_cols = set(dataset['columns']) - common_columns
        if unique_cols:
            unique_to_files[dataset['name']] = list(unique_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**✅ Common columns ({len(common_columns)}):**")
        if common_columns:
            st.write(", ".join(sorted(common_columns)))
        else:
            st.write("None found")
    
    with col2:
        st.write(f"**⚠️ Unique to specific files:**")
        if unique_to_files:
            for filename, cols in unique_to_files.items():
                st.write(f"*{filename}*: {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")
        else:
            st.write("None")
def create_file_column_mappings(datasets_info):
    """Create mappings between columns from different files."""
    
    if 'file_column_mappings' not in st.session_state:
        st.session_state.file_column_mappings = []
    
    # Add new mapping interface
    with st.expander("➕ Add Column Mapping", expanded=True):
        st.write("**Map columns that represent the same data across files:**")
        
        # Create columns for interface
        mapping_cols = st.columns([2] + [2] * len(datasets_info) + [1])
        
        # Mapping name
        with mapping_cols[0]:
            mapping_name = st.text_input(
                "Mapping Name:",
                placeholder="e.g., customer_id",
                key="new_mapping_name",
                help="Descriptive name for this mapping"
            )
        
        # Individual file selectors
        file_selections = {}
        for i, dataset in enumerate(datasets_info):
            with mapping_cols[i + 1]:
                # Truncate long filenames for display
                display_name = dataset['name']
                if len(display_name) > 12:
                    display_name = display_name[:12] + "..."
                
                st.write(f"**{display_name}**")
                
                selected_col = st.selectbox(
                    "Column:",
                    ["None"] + dataset['columns'],
                    key=f"file_{i}_column",
                    help=f"Select column from {dataset['name']}"
                )
                
                if selected_col != "None":
                    file_selections[i] = {
                        'file_name': dataset['name'],
                        'column': selected_col
                    }
        
        # Add button
        with mapping_cols[-1]:
            st.write("**Action**")
            if st.button("➕", key="add_new_mapping", help="Add this mapping"):
                if mapping_name and len(file_selections) >= 2:
                    st.session_state.file_column_mappings.append({
                        'name': mapping_name,
                        'files': file_selections
                    })
                    st.success(f"✅ Added: {mapping_name}")
                    st.rerun()
                elif not mapping_name:
                    st.error("Please enter a mapping name")
                else:
                    st.error("Select columns from at least 2 files")
    
    # Show existing mappings
    if st.session_state.file_column_mappings:
        st.write("**Current Mappings:**")
        
        for i, mapping in enumerate(st.session_state.file_column_mappings):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Build display string
                file_mappings = []
                for file_idx, file_info in mapping['files'].items():
                    short_name = file_info['file_name'][:8] + "..." if len(file_info['file_name']) > 8 else file_info['file_name']
                    file_mappings.append(f"{short_name}.{file_info['column']}")
                
                st.write(f"**{mapping['name']}**: {' ↔ '.join(file_mappings)}")
            
            with col2:
                if st.button("🗑️", key=f"delete_mapping_{i}", help="Delete this mapping"):
                    st.session_state.file_column_mappings.pop(i)
                    st.rerun()
    
    return st.session_state.file_column_mappings

def show_file_info(datasets_info):
    """Show summary of uploaded files."""
    
    st.markdown("#### 📋 File Information")
    
    for dataset in datasets_info:
        with st.expander(f"📄 {dataset['name']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Rows:** {len(dataset['data']):,}")
                st.write(f"**Columns:** {len(dataset['columns'])}")
            
            with col2:
                st.write("**Sample Columns:**")
                for col in dataset['columns'][:5]:
                    st.write(f"• {col}")
                if len(dataset['columns']) > 5:
                    st.write(f"• ... and {len(dataset['columns']) - 5} more")


def perform_mapped_join(datasets_info, mappings, join_type='outer'):
    """Join files and rename duplicate columns with *1, *2 suffixes."""
    
    if not mappings:
        raise Exception("No column mappings provided")
    
    # Start with the first dataset
    result_df = datasets_info[0]['data'].copy()
    
    # Process each additional dataset
    for dataset_idx in range(1, len(datasets_info)):
        right_df = datasets_info[dataset_idx]['data'].copy()
        
        # Find applicable mappings for this dataset pair
        join_pairs = []
        
        for mapping in mappings:
            # Check if this mapping includes both the base dataset (0) and current dataset
            if 0 in mapping['files'] and dataset_idx in mapping['files']:
                left_col = mapping['files'][0]['column']
                right_col = mapping['files'][dataset_idx]['column']
                
                if left_col in result_df.columns and right_col in right_df.columns:
                    join_pairs.append((left_col, right_col, mapping['name']))
        
        if join_pairs:
            # Prepare for join
            left_join_cols = []
            
            for left_col, right_col, mapping_name in join_pairs:
                # Convert to string to avoid data type conflicts
                result_df[left_col] = result_df[left_col].astype(str)
                right_df[right_col] = right_df[right_col].astype(str)
                
                # If different column names, rename right to match left
                if left_col != right_col:
                    right_df = right_df.rename(columns={right_col: left_col})
                
                left_join_cols.append(left_col)
            
            # FIXED: Handle duplicate columns with *1, *2 suffixes
            right_df = handle_duplicate_columns(result_df, right_df, left_join_cols)
            
            # Perform the merge
            result_df = pd.merge(
                result_df,
                right_df,
                on=left_join_cols,
                how=join_type
            )
            
            # Show progress
            mapping_names = [pair[2] for pair in join_pairs]
            st.info(f"✅ Joined {datasets_info[dataset_idx]['name']} using: {', '.join(mapping_names)}")
        
        else:
            st.warning(f"⚠️ No mappings found for {datasets_info[dataset_idx]['name']} - skipping")
    
    # FINAL: Clean up any remaining duplicates with suffixes
    result_df = add_suffixes_to_duplicates(result_df)
    
    return result_df

def handle_duplicate_columns(left_df, right_df, join_cols):
    """Handle duplicate columns by adding suffixes."""
    
    # Find overlapping columns (excluding join columns)
    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)
    overlapping = (left_cols & right_cols) - set(join_cols)
    
    # Rename overlapping columns in right dataframe
    rename_map = {}
    for col in overlapping:
        rename_map[col] = f"{col}*2"
    
    if rename_map:
        right_df = right_df.rename(columns=rename_map)
        logger.info(f"Renamed overlapping columns: {rename_map}")
    
    return right_df

def add_suffixes_to_duplicates(df):
    """Add *1, *2, *3 suffixes to any duplicate column names."""
    
    new_columns = []
    column_counts = {}
    
    for col in df.columns:
        if col in column_counts:
            # This is a duplicate
            column_counts[col] += 1
            new_col_name = f"{col}*{column_counts[col]}"
        else:
            # First occurrence
            column_counts[col] = 1
            new_col_name = col
        
        new_columns.append(new_col_name)
    
    # Apply new column names
    df.columns = new_columns
    
    # Log changes
    renamed_count = len([c for c in new_columns if '*' in c])
    if renamed_count > 0:
        logger.info(f"Added suffixes to {renamed_count} duplicate columns")
    
    return df

# ENHANCED: Stack operation also handles duplicates
def stack_files_with_suffix_handling(datasets_info):
    """Stack files and handle duplicate columns with suffixes."""
    
    all_dataframes = []
    
    for dataset in datasets_info:
        df = dataset['data'].copy()
        # Clean each dataframe before stacking
        df = add_suffixes_to_duplicates(df)
        all_dataframes.append(df)
    
    # Stack all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Final cleanup
    combined_df = add_suffixes_to_duplicates(combined_df)
    
    return combined_df


def enhanced_file_upload_section():
    """Enhanced with duplicate column suffix handling."""
    
    st.markdown("### 📤 Upload Multiple Files")
    
    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, JSON files:",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
        accept_multiple_files=True,
        key="enhanced_multi_upload"
    )
    
    if uploaded_files and len(uploaded_files) > 1:
        encoding = st.selectbox("CSV Encoding:", ['utf-8', 'iso-8859-1', 'windows-1252'], key="enhanced_encoding")
        
        datasets_info = []
        for uploaded_file in uploaded_files:
            try:
                df = load_data(uploaded_file, encoding)
                datasets_info.append({
                    'name': uploaded_file.name,
                    'data': df,
                    'columns': list(df.columns)
                })
                st.success(f"✅ {uploaded_file.name}: {len(df):,} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"❌ {uploaded_file.name}: {str(e)}")
        
        if len(datasets_info) > 1:
            st.markdown("---")
            st.markdown("### 🔗 Combine Files")
            
            # Join method selection
            join_method = st.radio(
                "Combine Method:",
                ["Stack (Append Rows)", "Join Files (Map Columns)"],
                key="join_method_selection"
            )
            
            if join_method == "Stack (Append Rows)":
                if st.button("📚 Stack Files", key="stack_files_btn"):
                    try:
                        # UPDATED: Use suffix handling for stack
                        combined_df = stack_files_with_suffix_handling(datasets_info)
                        st.session_state.dataset = combined_df
                        st.session_state.dataset_hash = compute_dataset_hash(combined_df)
                        st.session_state.classified = False
                        clear_all_caches()
                        
                        # Show info about duplicate handling
                        suffix_cols = [c for c in combined_df.columns if '*' in c]
                        if suffix_cols:
                            st.info(f"ℹ️ Added suffixes to {len(suffix_cols)} duplicate columns: {', '.join(suffix_cols[:5])}{'...' if len(suffix_cols) > 5 else ''}")
                        
                        st.success(f"✅ Stacked {len(datasets_info)} files → {len(combined_df):,} rows × {len(combined_df.columns)} cols")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Stack failed: {str(e)}")
            
            else:  # Join Files
                st.markdown("#### 🎯 Map Columns for Joining")
                st.info("Select which columns from each file should be joined together")
                
                # Individual file column mapping interface
                join_mappings = create_file_column_mappings(datasets_info)
                
                if join_mappings:
                    # Join type selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        join_type = st.selectbox(
                            "Join Type:",
                            ["inner", "left", "right", "outer"],
                            index=3,  # Default to outer
                            help="Inner: only matching rows, Outer: all rows",
                            key="enhanced_join_type"
                        )
                    
                    with col2:
                        st.info(f"Ready to join with {len(join_mappings)} mapping(s)")
                    
                    if st.button("🔗 Join Files with Mappings", key="enhanced_join_btn"):
                        try:
                            result_df = perform_mapped_join(datasets_info, join_mappings, join_type)
                            st.session_state.dataset = result_df
                            st.session_state.dataset_hash = compute_dataset_hash(result_df)
                            st.session_state.classified = False
                            clear_all_caches()
                            
                            # Show info about duplicate handling
                            suffix_cols = [c for c in result_df.columns if '*' in c]
                            if suffix_cols:
                                st.info(f"ℹ️ Renamed {len(suffix_cols)} duplicate columns with suffixes: {', '.join(suffix_cols[:3])}{'...' if len(suffix_cols) > 3 else ''}")
                            
                            st.success(f"✅ Joined files → {len(result_df):,} rows × {len(result_df.columns)} cols")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Join failed: {str(e)}")
                else:
                    st.warning("⚠️ Create at least one column mapping to join files")
            
            # Show file summaries
            show_file_info(datasets_info)
        
        return True
    
    elif uploaded_files and len(uploaded_files) == 1:
        # Single file
        encoding = st.selectbox("CSV Encoding:", ['utf-8', 'iso-8859-1', 'windows-1252'], key="single_encoding")
        
        if st.button("📥 Load Single File", key="load_single_btn"):
            try:
                df = load_data(uploaded_files[0], encoding)
                # Handle any duplicate columns in single file
                df = add_suffixes_to_duplicates(df)
                
                st.session_state.dataset = df
                st.session_state.dataset_hash = compute_dataset_hash(df)
                st.session_state.classified = False
                clear_all_caches()
                st.success(f"✅ Loaded {uploaded_files[0].name}: {len(df):,} rows × {len(df.columns)} cols")
                st.rerun()
            except Exception as e:
                st.error(f"Load failed: {str(e)}")
        
        return True
    
    return False

# 3. NEW: Individual file column mapping interface
def create_individual_file_mappings(datasets_info):
    """Create individual file column mappings."""
    
    if 'file_mappings' not in st.session_state:
        st.session_state.file_mappings = []
    
    # Add new mapping
    with st.expander("➕ Add Column Mapping", expanded=True):
        st.write("**Map columns that represent the same data:**")
        
        # Create columns for each file
        file_cols = st.columns(len(datasets_info) + 2)
        
        # Mapping name
        with file_cols[0]:
            mapping_name = st.text_input(
                "Mapping Name:",
                placeholder="customer_id",
                key="individual_mapping_name",
                help="Name for this mapping (e.g., customer_id, order_id)"
            )
        
        # Individual file column selectors
        selected_columns = {}
        for i, dataset in enumerate(datasets_info):
            with file_cols[i + 1]:
                st.write(f"**{dataset['name'][:15]}...**" if len(dataset['name']) > 15 else f"**{dataset['name']}**")
                
                selected_col = st.selectbox(
                    "Column:",
                    ["None"] + dataset['columns'],
                    key=f"file_{i}_column_select",
                    help=f"Select column from {dataset['name']}"
                )
                
                if selected_col != "None":
                    selected_columns[f"file_{i}"] = {
                        'column': selected_col,
                        'file_name': dataset['name']
                    }
        
        # Add mapping button
        with file_cols[-1]:
            st.write("**Actions**")
            if st.button("➕ Add", key="add_individual_mapping"):
                if mapping_name and len(selected_columns) >= 2:
                    st.session_state.file_mappings.append({
                        'name': mapping_name,
                        'columns': selected_columns
                    })
                    st.success(f"✅ Added mapping: {mapping_name}")
                    st.rerun()
                else:
                    st.error("Need mapping name + at least 2 columns")
    
    # Show existing mappings
    if st.session_state.file_mappings:
        st.write("**Current Mappings:**")
        
        for i, mapping in enumerate(st.session_state.file_mappings):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Show mapping details
                mapping_display = []
                for file_key, col_info in mapping['columns'].items():
                    file_name = col_info['file_name'][:10] + "..." if len(col_info['file_name']) > 10 else col_info['file_name']
                    mapping_display.append(f"{file_name}.{col_info['column']}")
                
                st.write(f"**{mapping['name']}**: {' ↔ '.join(mapping_display)}")
            
            with col2:
                if st.button("🗑️", key=f"delete_individual_mapping_{i}"):
                    st.session_state.file_mappings.pop(i)
                    st.rerun()
    
    return st.session_state.file_mappings

# 4. NEW: Join files using individual mappings
def join_files_with_individual_mappings(datasets_info, mappings, join_type='outer'):
    """Join files using individual column mappings."""
    
    if not mappings:
        raise Exception("No mappings provided")
    
    # Start with first dataset
    result_df = datasets_info[0]['data'].copy()
    
    # Join each subsequent dataset
    for i in range(1, len(datasets_info)):
        right_df = datasets_info[i]['data'].copy()
        
        # Find mappings that involve this file
        applicable_mappings = []
        for mapping in mappings:
            if 'file_0' in mapping['columns'] and f'file_{i}' in mapping['columns']:
                left_col = mapping['columns']['file_0']['column']
                right_col = mapping['columns'][f'file_{i}']['column']
                applicable_mappings.append((left_col, right_col, mapping['name']))
        
        if applicable_mappings:
            # Prepare columns for joining
            left_keys = []
            right_keys = []
            
            for left_col, right_col, mapping_name in applicable_mappings:
                if left_col in result_df.columns and right_col in right_df.columns:
                    # Convert data types to strings to avoid conflicts
                    result_df[left_col] = result_df[left_col].astype(str)
                    right_df[right_col] = right_df[right_col].astype(str)
                    
                    # If columns have different names, rename right column to match left
                    if left_col != right_col:
                        right_df = right_df.rename(columns={right_col: left_col})
                    
                    left_keys.append(left_col)
                    right_keys.append(left_col)  # Now same as left after rename
            
            if left_keys:
                # Perform the merge
                result_df = pd.merge(
                    result_df,
                    right_df,
                    on=left_keys,
                    how=join_type,
                    suffixes=('', f'_from_{datasets_info[i]["name"].split(".")[0]}')
                )
                
                st.info(f"Joined {datasets_info[i]['name']} using: {', '.join(left_keys)}")
            else:
                st.warning(f"No valid mappings found for {datasets_info[i]['name']}")
        else:
            st.warning(f"No mappings defined for {datasets_info[i]['name']}")
    
    return result_df

# 5. Show file summaries
def show_file_summaries(datasets_info):
    """Show summary of each file."""
    
    st.markdown("#### 📋 File Summaries")
    
    summary_cols = st.columns(min(len(datasets_info), 3))
    
    for i, dataset in enumerate(datasets_info):
        with summary_cols[i % 3]:
            st.write(f"**{dataset['name']}**")
            st.write(f"• {len(dataset['data'])} rows")
            st.write(f"• {len(dataset['columns'])} columns")
            st.write(f"• Columns: {', '.join(dataset['columns'][:3])}{'...' if len(dataset['columns']) > 3 else ''}")

def show_data_quality_in_tab(df):
    """FIXED: No more Series comparison errors."""
        
    # Data Quality Visualization
    st.markdown("#### 📊 Data Quality Visualization")
    
    # FIXED: Quality issues analysis
    quality_issues = []
    try:
        for col in df.columns:
            if col in df.columns:  # Extra safety
                col_data = df[col]
                if hasattr(col_data, 'isnull'):
                    missing_count = col_data.isnull().sum()
                    total_count = len(col_data)
                    
                    if total_count > 0:
                        # FIXED: Ensure we get a scalar value
                        missing_pct_scalar = float(missing_count / total_count * 100)
                        
                        if missing_pct_scalar > 10:  # Now comparing scalar to scalar
                            quality_issues.append(f"⚠️ {col}: {missing_pct_scalar:.1f}% missing")
                        
                        # Check for high cardinality
                        if pd.api.types.is_object_dtype(col_data.dtype):
                            unique_count = col_data.nunique()
                            unique_pct_scalar = float(unique_count / total_count * 100)
                            
                            if unique_pct_scalar > 80:
                                quality_issues.append(f"ℹ️ {col}: Very high cardinality ({unique_pct_scalar:.1f}% unique)")
    
    except Exception as e:
        st.info("Could not analyze all quality metrics")
    
    if quality_issues:
        st.markdown("**Quality Notes:**")
        for issue in quality_issues[:5]:
            st.write(issue)
    
    # FIXED: Detailed analysis
    if st.checkbox("📊 Show Detailed Column Analysis", key="main_tab_detailed_analysis"):
        try:
            analysis_data = []
            
            for col in df.columns:
                try:
                    col_data = df[col]
                    
                    # FIXED: Safe calculations
                    missing_count = col_data.isnull().sum() if hasattr(col_data, 'isnull') else 0
                    total_rows = len(col_data)
                    missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
                    unique_count = col_data.nunique() if hasattr(col_data, 'nunique') else 0
                    
                    # Safe sample value
                    sample_val = "N/A"
                    try:
                        non_null_data = col_data.dropna()
                        if len(non_null_data) > 0:
                            sample_val = str(non_null_data.iloc[0])[:50]  # Limit length
                    except:
                        pass
                    
                    analysis_data.append({
                        'Column': col,
                        'Type': str(col_data.dtype),
                        'Missing': f"{missing_count:,}",
                        'Missing %': f"{missing_pct:.1f}%",
                        'Unique': f"{unique_count:,}",
                        'Sample': sample_val
                    })
                    
                except Exception as e:
                    # Add row with error info
                    analysis_data.append({
                        'Column': col,
                        'Type': 'Error',
                        'Missing': 'N/A',
                        'Missing %': 'N/A',
                        'Unique': 'N/A',
                        'Sample': f"Error: {str(e)[:30]}"
                    })
            
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True)
            else:
                st.info("No column analysis data available")
                
        except Exception as e:
            st.error(f"Could not generate detailed analysis: {str(e)}")
 
def load_dataset_into_session(df, description):
    """Helper to load dataset into session with workflow tracking."""
    try:
        # Handle duplicate columns
        df = add_suffixes_to_duplicates(df)
        
        st.session_state.dataset = df
        st.session_state.dataset_hash = compute_dataset_hash(df)
        st.session_state.classified = False
        clear_all_caches()
        
        # Add to workflow
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
            
        st.session_state.workflow_steps.append({
            'type': 'load',
            'category': 'load', 
            'name': 'Load Data',
            'description': description,
            'operation': {'type': 'load', 'description': description}
        })
        
        # Show success
        suffix_cols = [c for c in df.columns if '*' in c]
        if suffix_cols:
            st.info(f"ℹ️ Added suffixes to {len(suffix_cols)} duplicate columns")
        
        st.success(f"✅ {description} → {len(df):,} rows × {len(df.columns)} cols")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load dataset: {str(e)}")

# RESTORED: Correlation analysis
def show_correlation_analysis(df):
    """Show correlation analysis for numeric columns."""
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Correlation Analysis")
        
        with st.expander("📈 Correlation Matrix", expanded=False):
            try:
                # Calculate correlation
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, ax=ax)
                plt.title("Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show strongest correlations
                st.markdown("**Strongest Correlations:**")
                
                # Get correlation pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if not np.isnan(corr_val):
                            corr_pairs.append((abs(corr_val), corr_val, col1, col2))
                
                # Sort by absolute correlation
                corr_pairs.sort(reverse=True)
                
                for abs_corr, corr, col1, col2 in corr_pairs[:5]:
                    direction = "📈" if corr > 0 else "📉"
                    st.write(f"{direction} **{col1}** ↔ **{col2}**: {corr:.3f}")
                    
            except Exception as e:
                st.error(f"Could not generate correlation analysis: {str(e)}")
    else:
        st.info("ℹ️ Need at least 2 numeric columns for correlation analysis")

# RESTORED: Sample data
def show_sample_data(df):
    """Show sample data with options."""
    
    st.markdown("### 🔍 Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Sample size:", 5, min(100, len(df)), 10, key="sample_size")
    
    with col2:
        sample_type = st.selectbox("Sample type:", ["First rows", "Random sample", "Last rows"], key="sample_type")
    
    try:
        if sample_type == "First rows":
            sample_df = df.head(sample_size)
        elif sample_type == "Random sample":
            sample_df = df.sample(min(sample_size, len(df)))
        else:  # Last rows
            sample_df = df.tail(sample_size)
        
        st.dataframe(sample_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not display sample data: {str(e)}")


def show_dataset_overview_in_tab(df):
    """Show dataset overview in tab."""
    
    st.markdown("### 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    with col4:
        total_cells = len(df) * len(df.columns)
        if total_cells > 0:
            missing_pct = (df.isnull().sum().sum() / total_cells * 100)
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        else:
            st.metric("Missing Data", "0%")

# RESTORED: Correlation analysis
def show_correlation_analysis_in_tab(df):
    """Show correlation analysis in tab."""
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        st.markdown("### 🔗 Correlation Analysis")
        
        with st.expander("📈 View Correlation Matrix", expanded=False):
            try:
               
                
                # Calculate correlation
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5, ax=ax, fmt='.2f')
                plt.title("Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Show strongest correlations
                st.markdown("**Strongest Correlations:**")
                
                # Get correlation pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_val):
                            corr_pairs.append((abs(corr_val), corr_val, col1, col2))
                
                # Sort by absolute correlation
                corr_pairs.sort(reverse=True)
                
                for abs_corr, corr, col1, col2 in corr_pairs[:5]:
                    direction = "📈" if corr > 0 else "📉"
                    st.write(f"{direction} **{col1}** ↔ **{col2}**: {corr:.3f}")
                    
            except Exception as e:
                st.error(f"Could not generate correlation analysis: {str(e)}")
    else:
        st.info("ℹ️ Need at least 2 numeric columns for correlation analysis")

# RESTORED: Sample data
def show_sample_data_in_tab(df):
    """Show sample data in tab."""
    
    st.markdown("### 🔍 Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Sample size:", 5, min(100, len(df)), 10, key="tab_sample_size")
    
    with col2:
        sample_type = st.selectbox("Sample type:", ["First rows", "Random sample", "Last rows"], key="tab_sample_type")
    
    try:
        if sample_type == "First rows":
            sample_df = df.head(sample_size)
        elif sample_type == "Random sample":
            sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        else:  # Last rows
            sample_df = df.tail(sample_size)
        
        st.dataframe(sample_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not display sample data: {str(e)}")

# Simple join helper
def simple_join_datasets(datasets_info, join_keys, join_type='outer'):
    """Simple join using common column names."""
    
    result_df = datasets_info[0]['data'].copy()
    
    for i in range(1, len(datasets_info)):
        right_df = datasets_info[i]['data'].copy()
        
        # Handle duplicate columns
        right_df = handle_duplicate_columns(result_df, right_df, join_keys)
        
        # Perform merge
        result_df = pd.merge(result_df, right_df, on=join_keys, how=join_type)
        
        st.info(f"✅ Joined {datasets_info[i]['name']} on: {', '.join(join_keys)}")
    
    return result_df

def smart_auto_join(datasets_info):
    """Auto join with smart column detection."""
    
    # Find safe columns (prioritize ID columns)
    common_columns = set(datasets_info[0]['columns'])
    for dataset in datasets_info[1:]:
        common_columns = common_columns.intersection(set(dataset['columns']))
    
    # Prioritize ID columns
    id_columns = [col for col in common_columns if 'id' in col.lower()]
    if id_columns:
        join_columns = id_columns
    else:
        # Use all common columns but convert dates to strings
        join_columns = list(common_columns)
    
    # Fix data types before joining
    for dataset in datasets_info:
        df = dataset['data']
        for col in join_columns:
            if col in df.columns:
                # Convert dates to strings to avoid type conflicts
                if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                    df[col] = df[col].astype(str)
    
    st.info(f"Auto joining on: {', '.join(join_columns[:5])}{'...' if len(join_columns) > 5 else ''}")
    
    # Perform merge
    result_df = datasets_info[0]['data']
    for i in range(1, len(datasets_info)):
        right_df = datasets_info[i]['data']
        result_df = pd.merge(
            result_df, 
            right_df, 
            on=join_columns, 
            how='outer',
            suffixes=('', f'_from_{datasets_info[i]["name"].split(".")[0]}')
        )
    
    return result_df

# Manual join with user-selected columns
def manual_join_datasets(datasets_info, join_columns, join_type='outer'):
    """Manual join with user-selected columns."""
    
    # Fix data types for selected columns
    for dataset in datasets_info:
        df = dataset['data']
        for col in join_columns:
            if col in df.columns:
                # Convert dates to strings to avoid type conflicts
                if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                    df[col] = df[col].astype(str)
                # Convert all to string if mixed types
                elif df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
    
    # Perform merge
    result_df = datasets_info[0]['data']
    for i in range(1, len(datasets_info)):
        right_df = datasets_info[i]['data']
        result_df = pd.merge(
            result_df, 
            right_df, 
            on=join_columns, 
            how=join_type,
            suffixes=('', f'_from_{datasets_info[i]["name"].split(".")[0]}')
        )
    
    return result_df

# FIXED: Column analysis with unique keys
def show_column_analysis_multi(datasets_info):
    """Show column analysis with unique keys."""
    
    st.markdown("#### 📋 Column Analysis")
    
    # Find common vs unique columns
    common_columns = set(datasets_info[0]['columns'])
    for dataset in datasets_info[1:]:
        common_columns = common_columns.intersection(set(dataset['columns']))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**✅ Common columns ({len(common_columns)}):**")
        if common_columns:
            st.write(", ".join(sorted(list(common_columns))[:10]))
            if len(common_columns) > 10:
                st.write("... and more")
        else:
            st.write("None found")
    
    with col2:
        st.write("**📊 File Summaries:**")
        for dataset in datasets_info:
            unique_cols = set(dataset['columns']) - common_columns
            st.write(f"*{dataset['name']}*: {len(dataset['columns'])} cols ({len(unique_cols)} unique)")


# Helper function to clear caches
def clear_all_caches():
    """Clear all session state caches."""
    cache_keys = ['sample_prompts', 'used_sample_prompts', 'sample_prompt_pool', 
                  'chart_cache', 'insights_cache', 'join_mappings']
    
    for key in cache_keys:
        if key in st.session_state:
            if isinstance(st.session_state[key], dict):
                st.session_state[key] = {}
            elif isinstance(st.session_state[key], list):
                st.session_state[key] = []
    
    st.session_state.last_used_pool_index = 0


# ENHANCED data management UI with visual workflow
def render_enhanced_data_tab():
    """Enhanced data management tab with visual workflow."""
    
    st.subheader("📊 Enhanced Data Management")
    
    # Initialize session state for multiple files
    if 'uploaded_datasets' not in st.session_state:
        st.session_state.uploaded_datasets = []
    if 'merge_config' not in st.session_state:
        st.session_state.merge_config = {'method': 'concat', 'join_keys': [], 'join_type': 'inner'}
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = []
    
    # Tabs for different operations
    data_tab1, data_tab2, data_tab3 = st.tabs(["📁 Load & Merge", "🛠️ Visual Workflow", "📊 Data Quality"])
    
def render_enhanced_data_tab():
    """FIXED: Single upload + restored correlation/sample features."""
    
    # Data Tab content
    with st.container():
        # SINGLE upload section - no duplicates
        st.markdown("### 📤 Upload Dataset")
        
        uploaded_files = st.file_uploader(
            "Upload datasets (supports CSV, Excel, JSON, Parquet, TSV):",
            type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
            accept_multiple_files=True,
            key="unified_file_uploader"
        )
        
        if uploaded_files:
            # Show file count and encoding selector
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"📁 {len(uploaded_files)} files selected")
            
            with col2:
                encoding_options = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
                selected_encoding = st.selectbox("CSV Encoding:", encoding_options, key="encoding_selector")
            
            # Process files
            datasets_info = []
            
            for uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📄 {uploaded_file.name}", expanded=False):
                    try:
                        df = load_data(uploaded_file, encoding=selected_encoding)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                        
                        st.dataframe(df.head(), use_container_width=True)
                        
                        datasets_info.append({
                            'name': uploaded_file.name,
                            'data': df,
                            'columns': list(df.columns)
                        })
                        
                    except Exception as e:
                        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
            
            # Handle multiple files
            if len(datasets_info) > 1:
                st.markdown("### 🔗 Combine Multiple Files")
                
                # Choose combination method
                combine_method = st.radio(
                    "How to combine files:",
                    ["Stack (Append Rows)", "Join (Map Columns)", "Simple Join (Common Columns)"],
                    key="combine_method"
                )
                
                if combine_method == "Stack (Append Rows)":
                    if st.button("📚 Stack Files", key="stack_files"):
                        try:
                            combined_df = stack_files_with_suffix_handling(datasets_info)
                            load_dataset_into_session(combined_df, f"Stacked {len(datasets_info)} files")
                        except Exception as e:
                            st.error(f"Stack failed: {str(e)}")
                
                elif combine_method == "Join (Map Columns)":
                    # Individual column mapping
                    join_mappings = create_file_column_mappings(datasets_info)
                    
                    if join_mappings:
                        col1, col2 = st.columns(2)
                        with col1:
                            join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"], index=3)
                        with col2:
                            st.info(f"Ready with {len(join_mappings)} mapping(s)")
                        
                        if st.button("🔗 Join with Custom Mappings", key="custom_join"):
                            try:
                                result_df = perform_mapped_join(datasets_info, join_mappings, join_type)
                                load_dataset_into_session(result_df, f"Joined {len(datasets_info)} files")
                            except Exception as e:
                                st.error(f"Join failed: {str(e)}")
                    else:
                        st.warning("⚠️ Create at least one column mapping")
                
                else:  # Simple Join (Common Columns)
                    # Find common columns
                    common_columns = set(datasets_info[0]['columns'])
                    for dataset in datasets_info[1:]:
                        common_columns = common_columns.intersection(set(dataset['columns']))
                    
                    if common_columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            join_keys = st.multiselect("Join on columns:", list(common_columns))
                        with col2:
                            join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"], key="simple_join_type")
                        
                        if join_keys and st.button("🔗 Simple Join", key="simple_join"):
                            try:
                                result_df = simple_join_datasets(datasets_info, join_keys, join_type)
                                load_dataset_into_session(result_df, f"Joined {len(datasets_info)} files on {', '.join(join_keys)}")
                            except Exception as e:
                                st.error(f"Simple join failed: {str(e)}")
                    else:
                        st.error("❌ No common columns found for simple joining!")
                        st.info("💡 Try 'Join (Map Columns)' to map different column names")
            
            elif len(datasets_info) == 1:
                if st.button("📥 Load Single File", key="load_single"):
                    load_dataset_into_session(datasets_info[0]['data'], f"Loaded {datasets_info[0]['name']}")
        
        # RESTORED: Data analysis section (only show if dataset exists)
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
  
            
            st.markdown("---")
            
            # RESTORED: Missing features
            show_dataset_overview(df)
            
            # RESTORED: Correlation analysis
            show_correlation_analysis(df)
            
            # RESTORED: Sample data
            show_sample_data(df)



# Data quality report with visualization
def generate_data_quality_report(df):
 
    
    # Visual quality indicators
    st.markdown("#### 📊 Data Quality Visualization")
    
    # Missing data heatmap for first 20 columns
    display_cols = df.columns[:20] if len(df.columns) > 20 else df.columns
    missing_data = df[display_cols].isnull()
    
    if missing_data.any().any():
        import plotly.express as px
        
        # Create missing data pattern
        missing_df = missing_data.astype(int)
        fig = px.imshow(
            missing_df.T,
            color_continuous_scale=['lightblue', 'red'],
            title="Missing Data Pattern (Red = Missing)",
            labels={'x': 'Row Index', 'y': 'Columns'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Column-wise analysis
    if st.checkbox("Show Detailed Column Analysis", key="detailed_analysis"):
        analysis_data = []
        
        for col in df.columns:
            col_data = df[col]
            analysis_data.append({
                'Column': col,
                'Type': str(col_data.dtype),
                'Missing': col_data.isnull().sum(),
                'Missing %': f"{col_data.isnull().sum() / len(df) * 100:.1f}%",
                'Unique': col_data.nunique(),
                'Unique %': f"{col_data.nunique() / len(df) * 100:.1f}%",
                'Memory (KB)': f"{col_data.memory_usage(deep=True) / 1024:.1f}"
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        st.dataframe(analysis_df, use_container_width=True)


# Tabs

if "uploaded_data" not in st.session_state or st.session_state.uploaded_data is None:
 
   


#elif st.session_state.current_project or st.session_state.uploaded_data:
    tab1, tab2, tab6, tab7, tab4, tab5 = st.tabs(["📊 Data", "🛠️ Field Editor", "📊 Recommended Charts & Insights", "🤖 Agentic AI Charts", "📜 Executive Summary", "💾 Saved Dashboards"])

    with tab1:
        st.markdown("### 📤 Upload Dataset")
        
        # SINGLE file uploader (no duplicates)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload datasets (supports CSV, Excel, JSON, Parquet, TSV):",
                type=["csv", "xlsx", "xls", "json", "parquet", "tsv"],
                accept_multiple_files=True,
                key="unified_file_uploader"  # Single key
            )
        
        with col2:
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} files selected")
                
                # Encoding selector for CSV files
                encoding_options = ['utf-8', 'iso-8859-1', 'windows-1252', 'utf-16']
                selected_encoding = st.selectbox("CSV Encoding:", encoding_options, key="encoding_selector")
        
        # Process uploaded files
        if uploaded_files:
            datasets_info = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📄 {uploaded_file.name}", expanded=False):
                    try:
                        df = load_data(uploaded_file, encoding=selected_encoding)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rows", f"{len(df):,}")
                        with col2:
                            st.metric("Columns", len(df.columns))
                        with col3:
                            st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                        
                        st.dataframe(df.head(), use_container_width=True)
                        
                        datasets_info.append({
                            'name': uploaded_file.name,
                            'data': df,
                            'columns': list(df.columns)
                        })
                        
                    except Exception as e:
                        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
            
            # Multi-file operations
            if len(datasets_info) > 1:
                st.markdown("### 🔗 Combine Multiple Files")
                
                # ENHANCED: Three combination methods
                merge_method = st.radio(
                    "Combination Method:",
                    ["Stack (Append Rows)", "Join (Custom Mapping)", "Join (Common Columns)"],
                    key="enhanced_merge_method"
                )
                
                if merge_method == "Stack (Append Rows)":
                    if st.button("📚 Stack Files", key="enhanced_stack"):
                        try:
                            combined_df = stack_files_with_suffix_handling(datasets_info)
                            load_dataset_with_workflow(combined_df, f"Stacked {len(datasets_info)} files")
                        except Exception as e:
                            st.error(f"Stack failed: {str(e)}")
                
                elif merge_method == "Join (Custom Mapping)":
                    # Individual column mapping
                    st.info("💡 Map columns with different names across files")
                    join_mappings = create_file_column_mappings(datasets_info)
                    
                    if join_mappings:
                        col1, col2 = st.columns(2)
                        with col1:
                            join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"], index=3, key="custom_join_type")
                        with col2:
                            st.success(f"✅ Ready with {len(join_mappings)} mapping(s)")
                        
                        if st.button("🔗 Join with Custom Mappings", key="custom_join_btn"):
                            try:
                                result_df = perform_mapped_join(datasets_info, join_mappings, join_type)
                                load_dataset_with_workflow(result_df, f"Joined {len(datasets_info)} files with custom mappings")
                            except Exception as e:
                                st.error(f"Custom join failed: {str(e)}")
                    else:
                        st.warning("⚠️ Create at least one column mapping above")
                
                else:  # Join (Common Columns)
                    st.info("🔗 Quick join using columns with the same name")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Find common columns across all datasets
                        common_columns = set(datasets_info[0]['columns'])
                        for dataset in datasets_info[1:]:
                            common_columns = common_columns.intersection(set(dataset['columns']))
                        
                        if not common_columns:
                            st.error("❌ No common columns found for joining!")
                            st.info("💡 Try 'Join (Custom Mapping)' to map different column names")
                        else:
                            join_keys = st.multiselect(
                                "Join Columns (can select multiple):", 
                                list(common_columns), 
                                key="common_join_keys"
                            )
                    
                    with col2:
                        if common_columns:
                            join_type = st.selectbox("Join Type:", ["inner", "left", "right", "outer"], key="common_join_type")
                            if join_keys:
                                st.success(f"✅ Will join on: {', '.join(join_keys)}")
                    
                    if common_columns and join_keys and st.button("🔗 Join on Common Columns", key="common_join_btn"):
                        try:
                            # Use your existing merge_datasets function
                            merge_config = {
                                'method': 'join',
                                'join_keys': join_keys,
                                'join_type': join_type
                            }
                            combined_df = merge_datasets(datasets_info, merge_config)
                            load_dataset_with_workflow(combined_df, f"Joined {len(datasets_info)} files on {', '.join(join_keys)}")
                        except Exception as e:
                            st.error(f"Common join failed: {str(e)}")
            
            elif len(datasets_info) == 1:
                if st.button("📥 Load Single File", key="load_single"):
                    load_dataset_with_workflow(datasets_info[0]['data'], f"Loaded {datasets_info[0]['name']}")
        
        # Data analysis sections (only show if dataset exists)
        if st.session_state.dataset is not None:
            df = st.session_state.dataset
            
            # RESTORED: Dataset overview
            show_dataset_overview_in_tab(df)
            
            # RESTORED: Correlation analysis  
            show_correlation_analysis_in_tab(df)
            
            # RESTORED: Sample data
            show_sample_data_in_tab(df)


# Fields Tab
    with tab2:
        st.session_state.onboarding_seen = True
        if st.session_state.dataset is not None:
            st.subheader("🛠️ Field Editor")
            df = st.session_state.dataset

            if not st.session_state.classified:
                try:
                    dimensions, measures, dates, ids = classify_columns(df, st.session_state.field_types)
                    df = preprocess_dates(df)  # force parsing of any detected date columns
                    st.session_state.field_types = {
                        "dimension": dimensions,
                        "measure": measures,
                        "date": dates,
                        "id": ids,
                    }
                    st.session_state.classified = True
                    st.session_state.dataset = df
                    #logger.info("Classified columns for dataset in project %s: dimensions=%s, measures=%s, dates=%s, ids=%s",
                      #          st.session_state.current_project, dimensions, measures, dates, ids)
                except Exception as e:
                    st.error(f"Failed to classify columns: %s", str(e))
                    logger.error("Failed to classify columns: %s", str(e))
                    st.stop()

            st.markdown("### 🔧 Manage Fields and Types")
            with st.expander("Manage Fields and Types", expanded=False):
                for col in df.columns:
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{col}**")
                    with col2:
                        current_type = "Other"
                        if col in st.session_state.field_types.get("dimension", []):
                            current_type = "Dimension"
                        elif col in st.session_state.field_types.get("measure", []):
                            current_type = "Measure"
                        elif col in st.session_state.field_types.get("date", []):
                            current_type = "Date"
                        elif col in st.session_state.field_types.get("id", []):
                            current_type = "ID"
                        
                        new_type = st.selectbox(
                            f"Type for {col}",
                            ["Dimension", "Measure", "Date", "ID", "Other"],
                            index=["Dimension", "Measure", "Date", "ID", "Other"].index(current_type),
                            key=f"type_select_{col}",
                            label_visibility="collapsed"
                        )
                        
                        if new_type != current_type:
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            if new_type.lower() != "other":
                                if new_type.lower() in st.session_state.field_types:
                                    st.session_state.field_types[new_type.lower()].append(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.session_state.chart_cache = {}  # Clear chart cache
                            st.session_state.insights_cache = {}  # Clear insights cache
                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                            st.success(f"Field {col} type changed to {new_type}!")
                            #logger.info("Changed field type for %s to %s", col, new_type)
                    with col3:
                        new_name = st.text_input(
                            "Rename",
                            value=col,
                            key=f"rename_{col}",
                            label_visibility="collapsed",
                            placeholder="New name"
                        )
                        if new_name and new_name != col:
                            if new_name in df.columns:
                                st.error("Field name already exists!")
                                logger.warning("Attempted to rename field %s to %s, but name already exists", col, new_name)
                            else:
                                df.rename(columns={col: new_name}, inplace=True)
                                st.session_state.dataset = df
                                for t in ["dimension", "measure", "date", "id"]:
                                    if col in st.session_state.field_types.get(t, []):
                                        st.session_state.field_types[t].remove(col)
                                        st.session_state.field_types[t].append(new_name)
                                save_dataset_changes()
                                st.session_state.sample_prompts = []
                                st.session_state.used_sample_prompts = []
                                st.session_state.sample_prompt_pool = []
                                st.session_state.last_used_pool_index = 0
                                st.session_state.chart_cache = {}  # Clear chart cache
                                st.session_state.insights_cache = {}  # Clear insights cache
                                st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                                st.success(f"Field renamed to {new_name}!")
                                #logger.info("Renamed field %s to %s", col, new_name)
                    with col4:
                        if st.button("Delete", key=f"delete_btn_{col}"):
                            df.drop(columns=[col], inplace=True)
                            st.session_state.dataset = df
                            for t in ["dimension", "measure", "date", "id"]:
                                if col in st.session_state.field_types.get(t, []):
                                    st.session_state.field_types[t].remove(col)
                            save_dataset_changes()
                            st.session_state.sample_prompts = []
                            st.session_state.used_sample_prompts = []
                            st.session_state.sample_prompt_pool = []
                            st.session_state.last_used_pool_index = 0
                            st.session_state.chart_cache = {}  # Clear chart cache
                            st.session_state.insights_cache = {}  # Clear insights cache
                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                            st.success(f"Field {col} deleted!")
                            #logger.info("Deleted field: %s", col)

            st.markdown("### ➕ Create Calculated Fields")
            st.markdown("""
            Create a new calculated field by either describing it in plain English, selecting a predefined template, or directly entering a formula. Supported functions: SUM, AVG, COUNT, STDEV, MEDIAN, MIN, MAX, IF-THEN-ELSE-END.
            """)
            
            input_mode = st.radio(
                "Select Input Mode:",
                ["Prompt-based (Plain English)", "Direct Formula Input"],
                key="calc_input_mode"
            )
            
            st.markdown("#### Predefined Calculation Templates")
            template = st.selectbox(
                "Select a Template (Optional):",
                ["None"] + list(PREDEFINED_CALCULATIONS.keys()),
                key="calc_template"
            )
            
            calc_prompt = ""
            formula_input = ""
            
            if template != "None":
                calc_prompt = PREDEFINED_CALCULATIONS[template]["prompt"]
                formula_input = PREDEFINED_CALCULATIONS[template]["formula"]
            
            dimensions = st.session_state.field_types.get("dimension", [])
            group_by = st.selectbox(
                "Group By (Optional, for 'per' aggregations):",
                ["None"] + dimensions,
                key="calc_group_by"
            )
            group_by = None if group_by == "None" else group_by
            
            if input_mode == "Prompt-based (Plain English)":
                st.markdown("#### Describe Your Calculation")
                measures = st.session_state.field_types.get("measure", [])
                dimensions = st.session_state.field_types.get("dimension", [])
                sample_measure1 = measures[0] if measures else "Measure1"
                sample_measure2 = measures[1] if len(measures) > 1 else "Measure2"
                sample_dimension = dimensions[0] if dimensions else "Dimension1"
                
                examples = [
                    f"Mark {sample_measure1} as High if greater than 1000, otherwise Low",
                    f"Calculate the profit margin as {sample_measure1} divided by {sample_measure2}",
                    f"Flag outliers in {sample_measure1} where {sample_measure1} is more than 2 standard deviations above the average",
                    f"Calculate average {sample_measure1} per {sample_dimension} and flag if above overall average",
                    f"If {sample_measure1} is greater than 500 and {sample_measure2} is positive, then High Performer, else if {sample_measure1} is less than 200, then Low Performer, else Medium"
                ]
                
                st.markdown("Examples:")
                for example in examples:
                    st.markdown(f"- {example}")
                
                calc_prompt = st.text_area("Describe Your Calculation in Plain Text:", value=calc_prompt, key="calc_prompt")
            else:
                st.markdown("#### Enter Formula Directly")
                st.markdown("""
                Enter a formula using exact column names (e.g., Sales, not [Sales]). Examples:
                - IF Sales > 1000 THEN 'High' ELSE 'Low' END
                - Profit / Sales
                - IF Sales > AVG(Sales) + 2 * STDEV(Sales) THEN 'Outlier' ELSE 'Normal' END
                - IF AVG(Profit) PER Ship Mode > AVG(Profit) THEN 'Above Average' ELSE 'Below Average' END
                """)
                formula_input = st.text_area("Enter Formula:", value=formula_input, key="calc_formula_input")
            
            new_field_name = st.text_input("New Field Name:", key="calc_new_field")
            
            if st.button("Create Calculated Field", key="calc_create"):
                if new_field_name in df.columns:
                    st.error("Field name already exists!")
                    logger.warning("Attempted to create field %s, but name already exists", new_field_name)
                elif not new_field_name:
                    st.error("Please provide a new field name!")
                    logger.warning("User attempted to create a calculated field without a name")
                elif (input_mode == "Prompt-based (Plain English)" and not calc_prompt) or (input_mode == "Direct Formula Input" and not formula_input):
                    st.error("Please provide a calculation description or formula!")
                    logger.warning("User attempted to create a calculated field without a description or formula")
                else:
                    with st.spinner("Processing calculation..."):
                        proceed_with_evaluation = True

                        if input_mode == "Prompt-based (Plain English)":
                            formula = generate_formula_from_prompt(
                                calc_prompt,
                                st.session_state.field_types.get("dimension", []),
                                st.session_state.field_types.get("measure", []),
                                df
                            )
                        else:
                            formula = formula_input
                        
                        if not formula:
                            st.error("Could not generate a formula from the prompt.")
                            logger.warning("Failed to generate formula for prompt: %s", calc_prompt)
                            proceed_with_evaluation = False

                        if proceed_with_evaluation:
                            if '=' in formula:
                                parts = formula.split('=', 1)
                                if len(parts) == 2:
                                    formula = parts[1].strip()
                                else:
                                    st.error("Invalid formula format.")
                                    logger.warning("Invalid formula format: %s", formula)
                                    proceed_with_evaluation = False

                            if proceed_with_evaluation:
                                for col in df.columns:
                                    formula = formula.replace(f"[{col}]", col)
                                
                                working_df = df.copy()
                                formula_modified = formula
                                group_averages = {}
                                overall_avg = None
                                group_dim = None
                                
                                per_match = re.search(r'AVG\((\w+)\)\s+PER\s+(\w+(?:\s+\w+)*)', formula_modified, re.IGNORECASE)
                                if per_match:
                                    agg_col = per_match.group(1)
                                    group_dim = per_match.group(2)
                                    if agg_col in working_df.columns and group_dim in working_df.columns:
                                        overall_avg = working_df[agg_col].mean()
                                        group_averages = working_df.groupby(group_dim)[agg_col].mean().to_dict()
                                        formula_modified = formula_modified.replace(f"AVG({agg_col})", str(overall_avg))
                                        formula_modified = re.sub(r'\s+PER\s+\w+(?:\s+\w+)*', '', formula_modified)
                                    else:
                                        st.error("Invalid columns in PER expression.")
                                        logger.error("Invalid columns in PER expression: %s, %s", agg_col, group_dim)
                                        proceed_with_evaluation = False
                                else:
                                    for col in df.columns:
                                        if f"AVG({col})" in formula_modified:
                                            avg_value = working_df[col].mean()
                                            formula_modified = formula_modified.replace(f"AVG({col})", str(avg_value))
                                        if f"STDEV({col})" in formula_modified:
                                            std_value = working_df[col].std()
                                            formula_modified = formula_modified.replace(f"STDEV({col})", str(std_value))
                                
                                if proceed_with_evaluation:
                                    formula_modified = parse_if_statement(formula_modified)
                                    st.markdown(f"**Formula Used:** `{formula}`")
                                    st.markdown(f"**Processed Formula:** `{formula_modified}`")
                                    try:
                                        def evaluate_row(row):
                                            local_vars = row.to_dict()
                                            if group_averages and group_dim in local_vars:
                                                group_value = group_averages.get(local_vars[group_dim], overall_avg)
                                                condition_expr = formula_modified
                                                for col in df.columns:
                                                    condition_expr = condition_expr.replace(col, str(local_vars.get(col, 0)))
                                                condition_expr = condition_expr.replace(str(overall_avg), str(group_value))
                                                return eval(condition_expr, {"__builtins__": None}, {})
                                            else:
                                                return eval(formula_modified, {"__builtins__": None}, local_vars)

                                        result = working_df.apply(evaluate_row, axis=1)
                                        if result is not None:
                                            df[new_field_name] = result
                                            st.session_state.dataset = df
                                            st.session_state.dataset_hash = compute_dataset_hash(df)  # Update hash
                                            st.session_state.chart_cache = {}  # Clear chart cache
                                            st.session_state.insights_cache = {}  # Clear insights cache
                                            if pd.api.types.is_numeric_dtype(df[new_field_name]):
                                                if "measure" in st.session_state.field_types:
                                                    st.session_state.field_types["measure"].append(new_field_name)
                                            else:
                                                if "dimension" in st.session_state.field_types:
                                                    st.session_state.field_types["dimension"].append(new_field_name)
                                            save_dataset_changes()
                                            st.session_state.sample_prompts = []
                                            st.session_state.used_sample_prompts = []
                                            st.session_state.sample_prompt_pool = []
                                            st.session_state.last_used_pool_index = 0
                                            st.success(f"New field {new_field_name} created!")
                                            #logger.info("Created new calculated field %s with formula: %s", new_field_name, formula)
                                        else:
                                            st.error("Failed to evaluate the formula.")
                                            logger.error("Formula evaluation returned None for prompt: %s", calc_prompt)
                                    except Exception as e:
                                        st.error(f"Error evaluating formula: {str(e)}")
                                        logger.error("Failed to evaluate formula: %s", str(e))


    with tab4:
        st.session_state.onboarding_seen = True
        if st.session_state.dataset is not None:
            executive_summary_tab(st.session_state.dataset)
        else:
            st.info("No dataset loaded. Please upload a dataset in the 'Data' tab to view the executive summary.")

    
    with tab5:
            st.session_state.onboarding_seen = True
            st.subheader("Saved Dashboards")
    
            if "user_id" not in st.session_state or st.session_state.user_id is None:
                st.error("Please log in to view dashboards.")
            else:
                # Initialize session state for navigation
                if "dashboard_view" not in st.session_state:
                    st.session_state.dashboard_view = "projects"  # Options: "projects", "dashboards", "dashboard"
                if "selected_project" not in st.session_state:
                    st.session_state.selected_project = None
                if "selected_dashboard" not in st.session_state:
                    st.session_state.selected_dashboard = None
                if "breadcrumb" not in st.session_state:
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                if "dashboards_cache" not in st.session_state:
                    st.session_state.dashboards_cache = None
                if "refresh_dashboards" not in st.session_state:
                    st.session_state.refresh_dashboards = False
    
                # Load dashboards before query parameter handling
                page_size = 10
                if "dashboard_page" not in st.session_state:
                    st.session_state.dashboard_page = 0
    
                if st.session_state.dashboards_cache is None or st.session_state.refresh_dashboards:
                    st.session_state.dashboards_cache = load_dashboards(
                        supabase, st.session_state.user_id, st.session_state, limit=page_size, offset=st.session_state.dashboard_page * page_size
                    )
                    st.session_state.refresh_dashboards = False
    
                dashboards = st.session_state.dashboards_cache
    
                # Handle query parameters for navigation
                view_param = st.query_params.get("view", "projects")
                project_id_param = st.query_params.get("project_id")
                dashboard_id_param = st.query_params.get("dashboard_id")
    
                # Update session state based on query parameters
                if view_param != st.session_state.dashboard_view or project_id_param != st.session_state.selected_project or dashboard_id_param != st.session_state.selected_dashboard:
                    st.session_state.dashboard_view = view_param
                    st.session_state.selected_project = project_id_param
                    st.session_state.selected_dashboard = dashboard_id_param
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                    if view_param == "dashboards" and project_id_param:
                        st.session_state.breadcrumb.append({"label": project_id_param, "view": "dashboards", "project_id": project_id_param})
                    elif view_param == "dashboard" and project_id_param and dashboard_id_param:
                        # Check if dashboards is defined and not empty
                        if dashboards is not None and not dashboards.empty:
                            dashboard_name = dashboards[dashboards["id"] == dashboard_id_param]["name"].iloc[0] if not dashboards[dashboards["id"] == dashboard_id_param].empty else dashboard_id_param
                        else:
                            dashboard_name = dashboard_id_param
                        st.session_state.breadcrumb.append({"label": project_id_param, "view": "dashboards", "project_id": project_id_param})
                        st.session_state.breadcrumb.append({"label": dashboard_name, "view": "dashboard", "project_id": project_id_param, "dashboard_id": dashboard_id_param})
    
                # Custom CSS for Tableau-like styling and breadcrumb buttons
                st.markdown("""
                <style>
                .project-container, .dashboard-container, .chart-container {
                    border: 1px solid #475569;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    background-color: #1F2A44;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                .project-container:hover, .dashboard-container:hover {
                    background-color: #334155;
                }
                .project-title, .dashboard-title, .chart-title {
                    font-size: 1.2em;
                    font-weight: 600;
                    color: #FFFFFF;
                    margin-bottom: 0.5rem;
                }
                .project-info, .dashboard-info {
                    font-size: 0.9em;
                    color: #A0AEC0;
                }
                .breadcrumb-container {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 1rem;
                    font-size: 0.9em;
                }
                .breadcrumb-button {
                    background: none !important;
                    border: none !important;
                    color: #3B82F6 !important;
                    text-decoration: none !important;
                    cursor: pointer !important;
                    padding: 0 !important;
                    font-size: 0.9em !important;
                }
                .breadcrumb-button:hover {
                    text-decoration: underline !important;
                }
                .breadcrumb-separator {
                    color: #A0AEC0;
                }
                .breadcrumb-current {
                    color: #A0AEC0;
                    font-size: 0.9em;
                }
                .filter-container {
                    background-color: #1F2A44;
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                }
                .action-button {
                    background-color: #26A69A !important;
                    color: white !important;
                    border-radius: 0.5rem !important;
                    padding: 0.5rem !important;
                    border: none !important;
                    cursor: pointer !important;
                }
                .action-button:hover {
                    background-color: #2E7D32 !important;
                }
                .dataset-status {
                    padding: 0.5rem;
                    border-radius: 4px;
                    margin-bottom: 1rem;
                    font-size: 0.9em;
                }
                .dataset-restored {
                    background-color: #059669;
                    color: white;
                }
                .dataset-sample {
                    background-color: #0891b2;
                    color: white;
                }
                .dataset-missing {
                    background-color: #dc2626;
                    color: white;
                }
                </style>
                """, unsafe_allow_html=True)
   
    
                # Handle view switching
                if st.session_state.dashboard_view == "projects":
                    render_breadcrumb()
    
                    # Search and filters
                    with st.container():
                        st.markdown("### Filter Projects and Dashboards")
                        with st.container():
                            search_query = st.text_input("Search by name, prompt, or tag", key="dashboard_search")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                project_filter = st.multiselect("Project", options=sorted([p for p in dashboards["project_id"].unique() if p is not None]), key="project_filter")
                            with col2:
                                type_filter = st.multiselect("Analytics Type", options=["Sales", "Customer", "Product", "Other"], key="type_filter")
                            with col3:
                                date_filter = st.date_input("Created After", value=None, key="date_filter")
                            with col4:
                                tag_filter = st.text_input("Tags (comma-separated)", key="tag_filter")
    
                            filtered_dashboards = filter_dashboards(dashboards, search_query, project_filter, type_filter, date_filter, tag_filter, supabase)
    
                    # Pagination controls
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        if st.button("Previous", disabled=st.session_state.dashboard_page == 0):
                            st.session_state.dashboard_page -= 1
                            st.session_state.refresh_dashboards = True
                    with col2:
                        st.write(f"Page {st.session_state.dashboard_page + 1}")
                    with col3:
                        if st.button("Next", disabled=len(filtered_dashboards) < page_size):
                            st.session_state.dashboard_page += 1
                            st.session_state.refresh_dashboards = True
    
                    if st.button("Refresh Dashboards", key="refresh_dashboards_btn"):
                        st.session_state.refresh_dashboards = True
                        st.rerun()
    
                    # Display projects
                    if dashboards.empty:
                        st.markdown("No projects or dashboards found.")
                    else:
                        project_ids = sorted([p for p in dashboards["project_id"].unique() if p is not None])
                        for project_id in project_ids:
                            project_dashboards = filtered_dashboards[filtered_dashboards["project_id"] == project_id]
                            with st.container():
                                st.markdown(f"<div class='project-container'>", unsafe_allow_html=True)
                                st.markdown(f"<div class='project-title'>{project_id}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='project-info'>Dashboards: {len(project_dashboards)}</div>", unsafe_allow_html=True)
                                if st.button(f"View Dashboards", key=f"view_project_{project_id}"):
                                    st.session_state.dashboard_view = "dashboards"
                                    st.session_state.selected_project = project_id
                                    st.session_state.breadcrumb.append({"label": project_id, "view": "dashboards", "project_id": project_id})
                                    st.query_params = {"view": "dashboards", "project_id": project_id}
                                    st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
    
                elif st.session_state.dashboard_view == "dashboards":
                    render_breadcrumb()
                    project_id = st.session_state.selected_project
                    project_dashboards = dashboards[dashboards["project_id"] == project_id]
    
                    if project_dashboards.empty:
                        st.markdown(f"No dashboards found for project '{project_id}'.")
                        if st.button("Back to Projects", key="back_to_projects"):
                            st.session_state.dashboard_view = "projects"
                            st.session_state.selected_project = None
                            st.query_params = {"view": "projects"}
                            st.rerun()
                    else:
                        st.markdown(f"### Dashboards in {project_id}")
                        for _, dashboard in project_dashboards.iterrows():
                            dashboard_id = dashboard["id"]
                            dashboard_name = dashboard["name"]
                            created_at = dashboard["created_at"]
                            tags = dashboard.get("tags", [])
                            with st.container():
                                st.markdown(f"<div class='dashboard-container'>", unsafe_allow_html=True)
                                st.markdown(f"<div class='dashboard-title'>{dashboard_name}</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='dashboard-info'>Created: {created_at} | Tags: {', '.join(tags) if tags else 'None'}</div>", unsafe_allow_html=True)
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    if st.button(f"View Dashboard", key=f"view_dashboard_{dashboard_id}"):
                                        st.session_state.dashboard_view = "dashboard"
                                        st.session_state.selected_dashboard = dashboard_id
                                        st.session_state.breadcrumb.append({"label": dashboard_name, "view": "dashboard", "project_id": project_id, "dashboard_id": dashboard_id})
                                        st.query_params = {"view": "dashboard", "project_id": project_id, "dashboard_id": dashboard_id}
                                        st.rerun()
                                with col2:
                                    if st.button("Delete", key=f"delete_dashboard_{dashboard_id}"):
                                        supabase.table("dashboards").delete().eq("id", dashboard_id).execute()
                                        st.session_state.refresh_dashboards = True
                                        st.success(f"Deleted dashboard '{dashboard_name}'")
                                        st.rerun()
                                st.markdown("</div>", unsafe_allow_html=True)
    
                elif st.session_state.dashboard_view == "dashboard":
                    render_breadcrumb()
                    dashboard_id = st.session_state.selected_dashboard
                    project_id = st.session_state.selected_project
                    dashboard_data = dashboards[dashboards["id"] == dashboard_id]
                    
                    if dashboard_data.empty:
                        st.error(f"Dashboard with ID {dashboard_id} not found.")
                        if st.button("Back to Dashboards", key="back_to_dashboards"):
                            st.session_state.dashboard_view = "dashboards"
                            st.session_state.selected_dashboard = None
                            st.query_params = {"view": "dashboards", "project_id": project_id}
                            st.rerun()
                    else:
                        dashboard_name = dashboard_data["name"].iloc[0]
                        st.markdown(f"### Dashboard: {dashboard_name}")
                        
                        # Use enterprise dashboard renderer
                        render_dashboard_enterprise(supabase, dashboard_id, project_id, dashboard_name)
    
                        # Dashboard settings
                        with st.expander("⚙️ Dashboard Settings", expanded=False):
                            with st.form(key=f"settings_{dashboard_id}"):
                                new_name = st.text_input("Rename Dashboard", value=dashboard_name)
                                new_tags = st.text_input("Add Tags (comma-separated)")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.form_submit_button("💾 Apply Changes"):
                                        updates = {}
                                        if new_name != dashboard_name:
                                            updates["name"] = new_name
                                        if new_tags:
                                            tag_list = [tag.strip() for tag in new_tags.split(",")]
                                            existing_tags = dashboard_data["tags"].iloc[0] or []
                                            updates["tags"] = list(set(existing_tags + tag_list))
                                        
                                        if updates:
                                            updates["updated_at"] = datetime.utcnow().isoformat()
                                            supabase.table("dashboards").update(updates).eq("id", dashboard_id).execute()
                                            st.success("Dashboard updated!")
                                            st.rerun()
                                
                                with col2:
                                    if st.form_submit_button("🗑️ Delete Dashboard"):
                                        supabase.table("dashboards").delete().eq("id", dashboard_id).execute()
                                        st.success("Dashboard deleted!")
                                        st.session_state.dashboard_view = "dashboards"
                                        st.rerun()
    
                # Reset navigation button for debugging
                if st.button("Reset Navigation", key="reset_navigation"):
                    st.session_state.dashboard_view = "projects"
                    st.session_state.selected_project = None
                    st.session_state.selected_dashboard = None
                    st.session_state.breadcrumb = [{"label": "All Projects", "view": "projects"}]
                    st.query_params = {"view": "projects"}
                    st.rerun()



    with tab6:
        st.session_state.onboarding_seen = True

        recommended_charts_insights_tab()

    with tab7:

        st.session_state.onboarding_seen = True

        with st.spinner("Generating advanced insights..."):
            agentic_ai_chart_tab()