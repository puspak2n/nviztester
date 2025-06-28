# ai_validation_system.py
"""
AI Validation and Feedback System for NarraViz.ai
Handles chart validation, confidence scoring, and feedback learning
"""

import streamlit as st
import pandas as pd
import json
import ast
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from supabase import create_client, Client
import uuid
from datetime import datetime

class ChartValidator:
    """Validates AI-generated chart code before execution"""
    
    def __init__(self):
        self.unsafe_operations = [
            'eval', 'exec', 'compile', '__import__', 'open',
            'file', 'input', 'raw_input', 'execfile', 'reload',
            'import os', 'import sys', 'import subprocess'
        ]
        
    def validate_generated_code(self, code: str, df: pd.DataFrame, user_prompt: str) -> Dict[str, Any]:
        """Multi-step validation before execution"""
        validation_results = {
            'syntax_valid': self.check_syntax(code),
            'columns_exist': self.verify_columns(code, df),
            'safe_operations': self.check_safe_operations(code),
            'output_expected': self.verify_output_structure(code),
            'semantic_match': self.check_semantic_alignment(code, user_prompt),
            'passed': True,
            'issues': [],
            'confidence': 100
        }
        
        # Compile results
        for check, (passed, message) in validation_results.items():
            if check not in ['passed', 'issues', 'confidence'] and not passed:
                validation_results['passed'] = False
                validation_results['issues'].append(message)
                validation_results['confidence'] -= 20
                
        return validation_results
    
    def check_syntax(self, code: str) -> Tuple[bool, str]:
        """Verify Python syntax is valid"""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {str(e)}"
    
    def verify_columns(self, code: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Ensure all referenced columns exist"""
        column_patterns = [
            r"df\['([^']+)'\]",
            r'df\["([^"]+)"\]',
            r"data\['([^']+)'\]",
            r'data\["([^"]+)"\]',
            r"groupby\(\['([^']+)'\]\)",
            r'groupby\(\["([^"]+)"\]\)',
            r"groupby\('([^']+)'\)",
            r'groupby\("([^"]+)"\)',
            r"x='([^']+)'",
            r'x="([^"]+)"',
            r"y='([^']+)'",
            r'y="([^"]+)"',
            r"color='([^']+)'",
            r'color="([^"]+)"',
            r"key='([^']+)'",
            r'key="([^"]+)"'
        ]
        
        referenced_cols = set()
        for pattern in column_patterns:
            matches = re.findall(pattern, code)
            referenced_cols.update(matches)
        
        # Special handling for datetime attributes
        datetime_attrs = {'.dt.year', '.dt.month', '.dt.day', '.dt.hour', '.dt.quarter'}
        for attr in datetime_attrs:
            if attr in code:
                # Remove datetime attributes from column check
                referenced_cols = {col for col in referenced_cols if not any(attr in col for attr in ['year', 'month', 'day', 'hour', 'quarter'])}
        
        missing = referenced_cols - set(df.columns)
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        return True, "All columns exist"
    
    def check_safe_operations(self, code: str) -> Tuple[bool, str]:
        """Check for potentially unsafe operations"""
        for unsafe_op in self.unsafe_operations:
            if unsafe_op in code:
                return False, f"Unsafe operation detected: {unsafe_op}"
        
        # Check for file operations
        if any(pattern in code for pattern in ['read_csv', 'read_excel', 'to_csv', 'to_excel']):
            if 'df' not in code:  # Allow if it's working with the provided df
                return False, "File operations not allowed"
                
        return True, "No unsafe operations detected"
    
    def verify_output_structure(self, code: str) -> Tuple[bool, str]:
        """Verify code produces expected output (fig variable)"""
        # Check if code assigns to fig
        if 'fig =' not in code and 'fig=' not in code:
            return False, "Code doesn't create a 'fig' variable"
        
        # Check if it's using plotly
        if not any(lib in code for lib in ['px.', 'go.', 'plotly']):
            return False, "Code doesn't use Plotly for visualization"
            
        return True, "Output structure valid"
    
    def check_semantic_alignment(self, code: str, user_prompt: str) -> Tuple[bool, str]:
        """Check if generated code aligns with user request"""
        prompt_lower = user_prompt.lower()
        
        # Chart type alignment
        chart_types = {
            'line': ['trend', 'over time', 'timeline', 'monthly', 'yearly', 'daily'],
            'bar': ['compare', 'comparison', 'by category', 'top', 'bottom', 'ranking'],
            'scatter': ['correlation', 'relationship', 'vs', 'versus'],
            'pie': ['distribution', 'share', 'percentage of total', 'breakdown']
        }
        
        detected_chart = None
        for chart_type, keywords in chart_types.items():
            if any(keyword in prompt_lower for keyword in keywords):
                detected_chart = chart_type
                break
        
        if detected_chart and f'px.{detected_chart}' not in code and f'go.{detected_chart.capitalize()}' not in code:
            return False, f"Expected {detected_chart} chart based on prompt"
            
        return True, "Code aligns with user request"


class ConfidenceScorer:
    """Calculate confidence scores for AI-generated visualizations"""
    
    def calculate_confidence(self, validation_results: Dict, generation_method: str, 
                           error_corrections: int = 0, pattern_match: bool = False) -> Dict[str, Any]:
        """Calculate detailed confidence score"""
        
        base_scores = {
            'ai_generated': 85,
            'pattern_based': 95,
            'template_based': 90,
            'fallback': 70,
            'error_recovered': 75
        }
        
        score = base_scores.get(generation_method, 80)
        
        # Boost for positive signals
        if pattern_match:
            score += 10
        
        if validation_results.get('passed', False):
            score += 5
        
        # Deductions
        deductions = {
            'syntax_errors': 30,
            'missing_columns': 25,
            'unsafe_operations': 40,
            'no_output': 20,
            'semantic_mismatch': 15
        }
        
        issues = validation_results.get('issues', [])
        for issue_type, penalty in deductions.items():
            if any(issue_type in issue.lower() for issue in issues):
                score -= penalty
        
        # Error correction penalty
        score -= error_corrections * 5
        
        # Calculate confidence level
        if score >= 90:
            level = 'high'
            color = 'green'
            icon = '✅'
        elif score >= 75:
            level = 'medium'
            color = 'yellow'
            icon = '⚠️'
        else:
            level = 'low'
            color = 'red'
            icon = '⚡'
        
        return {
            'score': max(0, min(100, score)),
            'level': level,
            'color': color,
            'icon': icon,
            'method': generation_method,
            'issues': issues,
            'details': {
                'base_score': base_scores.get(generation_method, 80),
                'pattern_match_bonus': 10 if pattern_match else 0,
                'validation_bonus': 5 if validation_results.get('passed', False) else 0,
                'error_corrections_penalty': error_corrections * 5,
                'validation_issues_penalty': sum(penalty for issue_type, penalty in deductions.items() 
                                               if any(issue_type in issue.lower() for issue in issues))
            }
        }


class PatternLibrary:
    """Pre-validated chart patterns for common requests"""
    
    def __init__(self):
        self.patterns = {
            'monthly_trend': {
                'keywords': ['monthly', 'trend', 'over time', 'by month', 'each month', 'monthly sales', 'monthly revenue'],
                'required_elements': ['date', 'value'],
                'optional_elements': ['category'],
                'template': '''
# Prepare data
data = df.copy()
data['{date_col}'] = pd.to_datetime(data['{date_col}'])
data = data.sort_values('{date_col}')

# Group by month{category_grouping}
monthly = data.groupby([pd.Grouper(key='{date_col}', freq='M'){category_col}])['{value_col}'].sum().reset_index()

# Create line chart
fig = px.line(monthly, x='{date_col}', y='{value_col}'{color_param},
              title='{title}', markers=True)

# Format x-axis
fig.update_layout(
    xaxis=dict(
        tickformat="%b %Y",
        dtick="M1"
    ),
    hovermode='x unified'
)
                ''',
                'confidence': 95,
                'chart_type': 'line'
            },
            
            'top_n_items': {
                'keywords': ['top', 'best', 'highest', 'largest', 'biggest', 'leading'],
                'required_elements': ['category', 'value'],
                'optional_elements': ['n'],
                'template': '''
# Prepare data
data = df.copy()

# Get top {n} by {value_col}
top_items = data.groupby('{category_col}')['{value_col}'].sum().nlargest({n}).reset_index()

# Create bar chart
fig = px.bar(top_items, x='{category_col}', y='{value_col}',
             title='{title}',
             text='{value_col}')

# Format values
fig.update_traces(texttemplate='%{{text:,.0f}}', textposition='outside')
fig.update_layout(showlegend=False)
                ''',
                'confidence': 95,
                'chart_type': 'bar'
            },
            
            'year_over_year': {
                'keywords': ['yoy', 'year over year', 'yearly growth', 'annual growth', 'year-over-year'],
                'required_elements': ['date', 'value'],
                'optional_elements': ['category'],
                'template': '''
# Prepare data
data = df.copy()
data['{date_col}'] = pd.to_datetime(data['{date_col}'])

# Group by year{category_grouping}
yearly = data.groupby([data['{date_col}'].dt.year{category_col}])['{value_col}'].sum().reset_index()
yearly.columns = ['Year'{category_name}, '{value_col}']

# Calculate YoY growth
yearly['YoY Growth %'] = yearly.groupby({groupby_col})['{value_col}'].pct_change() * 100
yearly = yearly.dropna(subset=['YoY Growth %'])

# Create line chart
fig = px.line(yearly, x='Year', y='YoY Growth %'{color_param},
              title='{title}', markers=True)

# Add zero line
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_layout(hovermode='x unified')
                ''',
                'confidence': 90,
                'chart_type': 'line'
            },
            
            'distribution': {
                'keywords': ['distribution', 'share', 'breakdown', 'composition', 'mix', 'percentage'],
                'required_elements': ['category', 'value'],
                'optional_elements': [],
                'template': '''
# Prepare data
data = df.copy()

# Aggregate by category
dist_data = data.groupby('{category_col}')['{value_col}'].sum().reset_index()

# Calculate percentages
dist_data['Percentage'] = (dist_data['{value_col}'] / dist_data['{value_col}'].sum() * 100).round(1)

# Create pie chart if few categories, bar otherwise
if len(dist_data) <= 6:
    fig = px.pie(dist_data, names='{category_col}', values='{value_col}',
                 title='{title}')
    fig.update_traces(textposition='inside', textinfo='percent+label')
else:
    fig = px.bar(dist_data.sort_values('{value_col}', ascending=False), 
                 x='{category_col}', y='{value_col}',
                 title='{title}', text='Percentage')
    fig.update_traces(texttemplate='%{{text:.1f}}%', textposition='outside')
                ''',
                'confidence': 90,
                'chart_type': 'mixed'
            }
        }
    
    def match_pattern(self, user_prompt: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[Dict]]:
        """Find best matching pattern for user prompt"""
        prompt_lower = user_prompt.lower()
        best_match = None
        best_score = 0
        
        for pattern_name, pattern_info in self.patterns.items():
            # Calculate match score
            score = 0
            matched_keywords = []
            
            for keyword in pattern_info['keywords']:
                if keyword in prompt_lower:
                    score += len(keyword)  # Longer matches score higher
                    matched_keywords.append(keyword)
            
            if score > best_score:
                best_score = score
                best_match = (pattern_name, pattern_info, matched_keywords)
        
        if best_match and best_score > 5:  # Minimum threshold
            return best_match[0], best_match[1]
        
        return None, None
    
    def extract_elements(self, prompt: str, df: pd.DataFrame, pattern_info: Dict) -> Dict[str, str]:
        """Extract required elements from prompt and dataframe"""
        elements = {}
        
        # Smart column detection based on data types and names
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) 
                     or 'date' in col.lower() or 'time' in col.lower()]
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
                       and not any(id_term in col.lower() for id_term in ['id', 'key', 'code'])]
        
        category_cols = [col for col in df.columns if df[col].dtype == 'object' 
                        and df[col].nunique() < 50
                        and not any(id_term in col.lower() for id_term in ['id', 'key', 'code'])]
        
        # Match to required elements
        if 'date' in pattern_info['required_elements'] and date_cols:
            elements['date_col'] = date_cols[0]
        
        if 'value' in pattern_info['required_elements'] and numeric_cols:
            # Try to find mentioned column in prompt
            prompt_lower = prompt.lower()
            for col in numeric_cols:
                if col.lower() in prompt_lower:
                    elements['value_col'] = col
                    break
            else:
                elements['value_col'] = numeric_cols[0]
        
        if 'category' in pattern_info['required_elements'] and category_cols:
            # Try to find mentioned column in prompt
            for col in category_cols:
                if col.lower() in prompt.lower():
                    elements['category_col'] = col
                    break
            else:
                elements['category_col'] = category_cols[0]
        
        # Extract numbers (like top N)
        numbers = re.findall(r'\b(\d+)\b', prompt)
        if numbers and 'n' in pattern_info['optional_elements']:
            elements['n'] = numbers[0]
        else:
            elements['n'] = '10'  # Default
        
        # Generate title
        elements['title'] = self._generate_title(prompt, pattern_info['chart_type'])
        
        return elements


# Fix your FeedbackLearningSystem to actually save to Supabase

class FeedbackLearningSystem:
    def __init__(self, supabase_url=None, supabase_key=None):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase_client = None
        
        # Initialize Supabase client if credentials provided
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                self.supabase_client = create_client(supabase_url, supabase_key)
                print(f"✅ Supabase client initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Supabase: {e}")
                self.supabase_client = None
        
        # Fallback local storage
        self.local_feedback = []
    
    def store_feedback(self, feedback_data):
        """Store feedback in ai_feedback table (your existing table)"""
        try:
            # Prepare data for ai_feedback table structure
            ai_feedback_record = {
                'user_id': feedback_data.get('user_id', 'anonymous'),
                'feedback_type': feedback_data.get('type', 'chart_rating'),
                'rating': feedback_data.get('rating'),
                'content': {
                    'feedback_text': feedback_data.get('feedback', ''),
                    'chart_rating': feedback_data.get('chart_rating', ''),
                    'improvement': feedback_data.get('improvement', ''),
                    'preferred_type': feedback_data.get('preferred_type', ''),
                    'accuracy': feedback_data.get('accuracy', True),
                    'relevance': feedback_data.get('relevance', True),
                    'clarity': feedback_data.get('clarity', True),
                    'timestamp': feedback_data.get('timestamp', datetime.now().isoformat())
                },
                'dataset_info': {
                    'columns': feedback_data.get('columns', []),
                    'shape': feedback_data.get('dataset_shape'),
                    'data_types': feedback_data.get('data_types')
                },
                'chart_info': {
                    'chart_type': feedback_data.get('chart_type'),
                    'chart_title': feedback_data.get('chart_title'),
                    'prompt': feedback_data.get('prompt'),
                    'code': feedback_data.get('code'),
                    'chart_id': feedback_data.get('chart_id'),
                    'confidence': feedback_data.get('confidence'),
                    'generation_method': feedback_data.get('generation_method'),
                    'x_col': feedback_data.get('x_col'),
                    'y_col': feedback_data.get('y_col')
                }
            }
            
            # Try Supabase first
            if self.supabase_client:
                try:
                    # Insert into ai_feedback table (your existing table)
                    result = self.supabase_client.table('ai_feedback').insert(ai_feedback_record).execute()
                    
                    if result.data:
                        print(f"✅ Feedback saved to Supabase ai_feedback table: {feedback_data.get('rating', 'N/A')}/5")
                        return True
                    else:
                        print(f"❌ Supabase insert failed: {result}")
                        
                except Exception as supabase_error:
                    print(f"❌ Supabase error: {supabase_error}")
                    print(f"   Record being inserted: {ai_feedback_record}")
                    print(f"   Falling back to local storage")
            
            # Fallback to local storage
            self.local_feedback.append(feedback_data)
            print(f"📝 Feedback saved locally: {feedback_data.get('rating', 'N/A')}/5")
            return True
            
        except Exception as e:
            print(f"❌ Error storing feedback: {e}")
            print(f"   Original feedback data: {feedback_data}")
            return False
    
    def submit_feedback(self, feedback_data):
        """Alias for store_feedback"""
        return self.store_feedback(feedback_data)
    
    def get_statistics(self):
        """Get feedback statistics from ai_feedback table + local"""
        try:
            all_feedback = []
            
            # Get from Supabase ai_feedback table
            if self.supabase_client:
                try:
                    result = self.supabase_client.table('ai_feedback').select('*').execute()
                    if result.data:
                        # Convert ai_feedback records to standard format
                        for record in result.data:
                            standard_feedback = {
                                'rating': record.get('rating'),
                                'chart_type': record.get('chart_info', {}).get('chart_type'),
                                'feedback': record.get('content', {}).get('feedback_text', ''),
                                'timestamp': record.get('created_at')
                            }
                            all_feedback.append(standard_feedback)
                except Exception as e:
                    print(f"Error fetching from Supabase ai_feedback: {e}")
            
            # Add local feedback
            all_feedback.extend(self.local_feedback)
            
            if not all_feedback:
                return {
                    'total_feedback': 0,
                    'avg_rating': 0,
                    'success_rate': 0,
                    'patterns_learned': 0,
                    'recent_improvements': []
                }
            
            # Calculate statistics
            ratings = [f.get('rating', 0) for f in all_feedback if f.get('rating')]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            success_rate = (len([r for r in ratings if r >= 4]) / len(ratings) * 100) if ratings else 0
            
            return {
                'total_feedback': len(all_feedback),
                'avg_rating': avg_rating,
                'success_rate': success_rate,
                'patterns_learned': len(set(f.get('chart_type', '') for f in all_feedback)),
                'recent_improvements': [
                    f"Collected {len(all_feedback)} feedback entries",
                    f"Average rating: {avg_rating:.1f}/5",
                    f"Success rate: {success_rate:.1f}%"
                ]
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {'total_feedback': 0, 'avg_rating': 0, 'success_rate': 0, 'patterns_learned': 0, 'recent_improvements': []}

    def debug_feedback_flow(self, feedback_data):
        """Debug function to trace feedback flow"""
        print(f"🔍 DEBUG - Feedback received:")
        print(f"   Data: {feedback_data}")
        print(f"   Supabase client: {self.supabase_client is not None}")
        print(f"   Supabase URL: {self.supabase_url}")
        print(f"   Local feedback count: {len(self.local_feedback)}")
        
        if self.supabase_client:
            try:
                # Test Supabase connection with ai_feedback table
                test_result = self.supabase_client.table('ai_feedback').select('count').execute()
                print(f"   Supabase ai_feedback connection test: ✅ {len(test_result.data) if test_result.data else 0} records")
            except Exception as e:
                print(f"   Supabase ai_feedback connection test: ❌ {e}")





class SmartErrorRecovery:
    """Intelligent error recovery system"""
    
    def __init__(self, feedback_system: FeedbackLearningSystem = None):  # Fixed: Added missing underscore
        self.feedback_system = feedback_system
        self.error_fixes = {
            'KeyError': self._fix_key_error,
            'AttributeError': self._fix_attribute_error,
            'TypeError': self._fix_type_error,
            'ValueError': self._fix_value_error,
            'Period': self._fix_period_error,
            'JSON': self._fix_json_error
        }
    
    # Fixed: Added all missing method implementations
    def _fix_key_error(self, error, context=None):
        """Fix KeyError by suggesting alternative keys or providing defaults"""
        try:
            error_msg = str(error)
            missing_key = error_msg.split("'")[1] if "'" in error_msg else "unknown"
            
            suggestions = {
                'suggestion': f"Key '{missing_key}' not found. Check available keys or use .get() method with default value.",
                'fix': f"Use dict.get('{missing_key}', default_value) instead of dict['{missing_key}']",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('KeyError', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'KeyError encountered. Check dictionary keys.', 'context': context}
    
    def _fix_attribute_error(self, error, context=None):
        """Fix AttributeError by suggesting correct attribute names"""
        try:
            error_msg = str(error)
            suggestions = {
                'suggestion': f"Attribute error: {error_msg}. Check object type and available attributes.",
                'fix': "Use hasattr() to check if attribute exists before accessing",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('AttributeError', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'AttributeError encountered. Check object attributes.', 'context': context}
    
    def _fix_type_error(self, error, context=None):
        """Fix TypeError by suggesting correct types"""
        try:
            error_msg = str(error)
            suggestions = {
                'suggestion': f"Type error: {error_msg}. Check data types and conversions.",
                'fix': "Ensure correct data types are used or add type conversion",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('TypeError', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'TypeError encountered. Check data types.', 'context': context}
    
    def _fix_value_error(self, error, context=None):
        """Fix ValueError by suggesting valid values"""
        try:
            error_msg = str(error)
            suggestions = {
                'suggestion': f"Value error: {error_msg}. Check input values and ranges.",
                'fix': "Validate input values before processing",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('ValueError', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'ValueError encountered. Check input values.', 'context': context}
    
    def _fix_period_error(self, error, context=None):
        """Fix period-related errors"""
        try:
            error_msg = str(error)
            suggestions = {
                'suggestion': f"Period error: {error_msg}. Check date/time formatting and periods.",
                'fix': "Ensure proper datetime formatting and period specifications",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('Period', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'Period error encountered. Check datetime formatting.', 'context': context}
    
    def _fix_json_error(self, error, context=None):
        """Fix JSON-related errors"""
        try:
            error_msg = str(error)
            suggestions = {
                'suggestion': f"JSON error: {error_msg}. Check JSON formatting and structure.",
                'fix': "Validate JSON structure and handle parsing errors",
                'context': context
            }
            
            if self.feedback_system:
                self.feedback_system.log_error_fix('JSON', suggestions)
            
            return suggestions
        except Exception:
            return {'suggestion': 'JSON error encountered. Check JSON formatting.', 'context': context}
    
    def recover_from_error(self, error_type, error, context=None):
        """Main method to recover from errors"""
        if error_type in self.error_fixes:
            return self.error_fixes[error_type](error, context)
        else:
            return {
                'suggestion': f"Unknown error type: {error_type}. {str(error)}",
                'fix': "Check error type and implement appropriate handling",
                'context': context
            }


# Export the main components
__all__ = [
    'ChartValidator',
    'ConfidenceScorer',
    'PatternLibrary',
    'FeedbackLearningSystem',
    'SmartErrorRecovery'
]