# ai_feedback_ui.py
"""
UI components for AI feedback and validation display
"""

import streamlit as st
from typing import Dict, Any, Optional
import json
from datetime import datetime

class AIFeedbackUI:
    """UI components for collecting and displaying AI feedback"""
    
    @staticmethod
    def display_confidence_badge(confidence_info):
        """Display confidence badge with proper error handling"""
        try:
            # Use .get() method with defaults to avoid KeyError
            icon = confidence_info.get('icon', '📊')  # Default icon
            score = confidence_info.get('score', 0.0)
            level = confidence_info.get('level', 'Unknown')
            color = confidence_info.get('color', 'gray')
            
            # Create the confidence badge
            st.markdown(f"""
            <div style="
                display: inline-flex; 
                align-items: center; 
                background-color: {color}; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold; 
                color: white;
                margin: 2px;
            ">
                {icon} {level} ({score:.1%})
            </div>
            """, unsafe_allow_html=True)
            
            # Handle optional details section safely
            details = confidence_info.get('details', {})
            if details:
                # Display additional details if available
                base_score = details.get('base_score', score * 100)
                factors = details.get('factors', [])
                
                with st.expander("📋 Confidence Details", expanded=False):
                    st.markdown(f"**Base Score:** {base_score:.1f}%")
                    
                    if factors:
                        st.markdown("**Contributing Factors:**")
                        for factor in factors:
                            factor_name = factor.get('name', 'Unknown Factor')
                            factor_impact = factor.get('impact', 0)
                            impact_icon = "📈" if factor_impact > 0 else "📉" if factor_impact < 0 else "➖"
                            st.markdown(f"- {impact_icon} {factor_name}: {factor_impact:+.1f}%")
            
        except Exception as e:
            # Fallback display if anything goes wrong
            st.markdown(f"""
            <div style="
                display: inline-flex; 
                align-items: center; 
                background-color: gray; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold; 
                color: white;
                margin: 2px;
            ">
                📊 Confidence: {confidence_info.get('score', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def display_validation_results(validation_results: Dict[str, Any]):
        """Display validation results in a clear format"""
        if validation_results.get('passed', True):
            st.success("✅ All validations passed!")
        else:
            issues = validation_results.get('issues', [])
            st.error(f"❌ Validation failed: {len(issues)} issues found")
            
            with st.expander("🔍 Validation Details", expanded=True):
                checks = ['syntax_valid', 'columns_exist', 'safe_operations', 
                         'output_expected', 'semantic_match']
                
                for check in checks:
                    if check in validation_results:
                        passed, message = validation_results[check]
                        if passed:
                            st.markdown(f"✅ **{check.replace('_', ' ').title()}**: {message}")
                        else:
                            st.markdown(f"❌ **{check.replace('_', ' ').title()}**: {message}")
    
    @staticmethod
    def display_rating_widget_fixed(chart_id: str, feedback_system=None):
        """Fixed rating widget in expandable container with no auto-save"""
        
        # Use unique keys for session state
        rating_key = f"rating_{chart_id}"
        feedback_key = f"feedback_{chart_id}"
        submitted_key = f"submitted_{chart_id}"
        
        # Initialize session state if not exists
        if rating_key not in st.session_state:
            st.session_state[rating_key] = 3
        if feedback_key not in st.session_state:
            st.session_state[feedback_key] = ""
        if submitted_key not in st.session_state:
            st.session_state[submitted_key] = False
        
        # Check if already submitted
        if st.session_state[submitted_key]:
            # Show compact success message
            st.success("✅ Thank you! Your feedback has been submitted.")
            with st.expander("📝 Submit New Rating", expanded=False):
                if st.button("Rate Again", key=f"rate_again_{chart_id}"):
                    st.session_state[submitted_key] = False
                    st.rerun()
            return None
        
        # Expandable feedback container
        with st.expander("📝 Rate This Visualization", expanded=False):
            st.markdown("Help us improve by rating this chart!")
            
            # Use form to prevent auto-saving on widget changes
            with st.form(key=f"rating_form_{chart_id}"):
                # Create two columns for better layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Rating slider - only updates on form submit, not on change
                    rating = st.slider(
                        "Rating (1-5 stars)",
                        min_value=1,
                        max_value=5,
                        value=st.session_state[rating_key],
                        help="Rate the quality and usefulness of this visualization"
                    )
                    
                    # Feedback text area
                    feedback_text = st.text_area(
                        "Optional Feedback:",
                        value=st.session_state[feedback_key],
                        placeholder="What did you like or dislike about this chart? Any suggestions for improvement?",
                        height=80
                    )
                
                with col2:
                    # Display star rating visually
                    stars = "⭐" * rating + "☆" * (5 - rating)
                    st.markdown(f"**{stars}**")
                    st.markdown(f"**{rating}/5**")
                    
                    st.markdown("---")
                    st.caption("Your feedback helps improve our AI chart generation!")
                
                # Form submit buttons
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    save_clicked = st.form_submit_button("💾 Submit Feedback", type="primary")
                
                with col2:
                    skip_clicked = st.form_submit_button("Skip")
            
            # Handle form submission OUTSIDE the form
            if save_clicked:
                try:
                    # Only update session state when form is submitted
                    st.session_state[rating_key] = rating
                    st.session_state[feedback_key] = feedback_text
                    
                    # Prepare feedback data
                    feedback_data = {
                        'type': 'chart_rating',
                        'chart_id': chart_id,
                        'rating': rating,
                        'feedback': feedback_text,
                        'user_id': st.session_state.get('user_id', 'anonymous'),
                        'timestamp': datetime.now().isoformat(),
                        'chart_rating': '⭐' * rating
                    }
                    
                    if feedback_system:
                        # Use store_feedback method
                        if hasattr(feedback_system, 'store_feedback'):
                            result = feedback_system.store_feedback(feedback_data)
                        elif hasattr(feedback_system, 'submit_feedback'):
                            result = feedback_system.submit_feedback(feedback_data)
                        else:
                            st.error("❌ Feedback system method not found")
                            return None
                        
                        if result:
                            st.success("✅ Thank you for your feedback!")
                            st.balloons()  # Celebration animation
                            st.session_state[submitted_key] = True
                            st.rerun()
                        else:
                            st.error("❌ Failed to save feedback. Please try again.")
                    else:
                        # Fallback - save to session state
                        if 'saved_feedback' not in st.session_state:
                            st.session_state['saved_feedback'] = []
                        st.session_state['saved_feedback'].append(feedback_data)
                        
                        st.success("✅ Rating saved locally!")
                        st.session_state[submitted_key] = True
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error saving feedback: {str(e)}")
                    print(f"Rating submission error: {e}")
            
            elif skip_clicked:
                st.session_state[submitted_key] = True
                st.rerun()
        
        return None
    
    @staticmethod
    def collect_chart_feedback(chart_info: Dict[str, Any], chart_index: int, 
                             feedback_system: Any) -> Dict[str, Any]:
        """Collect user feedback on a chart - LEGACY VERSION (use display_rating_widget_fixed instead)"""
        feedback_key = f"feedback_{chart_index}_{chart_info.get('type', 'unknown')}"
        
        st.markdown("---")
        st.markdown("### 📝 Rate This Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall rating
            rating = st.select_slider(
                "Overall Quality",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key=f"{feedback_key}_rating",
                help="Rate the overall quality of this visualization"
            )
            
            # Specific aspects
            accuracy = st.checkbox("✅ Data is accurate", key=f"{feedback_key}_accurate", value=True)
            relevant = st.checkbox("✅ Answers my question", key=f"{feedback_key}_relevant", value=True)
            clear = st.checkbox("✅ Easy to understand", key=f"{feedback_key}_clear", value=True)
        
        with col2:
            # Improvement suggestions
            improvement = st.text_area(
                "How can we improve this chart?",
                placeholder="e.g., 'Show monthly totals instead of daily' or 'Add trend line'",
                key=f"{feedback_key}_improvement",
                height=100
            )
            
            # Chart type preference
            preferred_type = st.selectbox(
                "Preferred chart type for this data:",
                options=['Current is good', 'Line', 'Bar', 'Scatter', 'Pie', 'Heatmap', 'Other'],
                key=f"{feedback_key}_preferred_type"
            )
        
        # Submit feedback
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("👍 Save Feedback", key=f"{feedback_key}_save"):
                feedback_data = {
                    'type': 'chart_rating',
                    'chart_type': chart_info.get('type'),
                    'chart_title': chart_info.get('title'),
                    'rating': rating,
                    'accuracy': accuracy,
                    'relevance': relevant,
                    'clarity': clear,
                    'improvement': improvement,
                    'preferred_type': preferred_type,
                    'code': chart_info.get('code'),
                    'prompt': chart_info.get('prompt'),
                    'columns': [chart_info.get('x_col'), chart_info.get('y_col')],
                    'confidence': chart_info.get('confidence', {}).get('score', 0),
                    'generation_method': chart_info.get('confidence', {}).get('method', 'unknown')
                }
                
                if feedback_system and feedback_system.store_feedback(feedback_data):
                    st.success("Thank you! Your feedback helps improve our AI.")
                    return feedback_data
                    
        with col2:
            if st.button("📌 Pin Chart", key=f"{feedback_key}_pin"):
                if 'pinned_charts' not in st.session_state:
                    st.session_state.pinned_charts = []
                st.session_state.pinned_charts.append(chart_info)
                st.success("Chart pinned to dashboard!")
        
        with col3:
            if st.button("🔄 Regenerate", key=f"{feedback_key}_regen"):
                st.session_state[f'regenerate_chart_{chart_index}'] = True
                st.rerun()
        
        return None
    
    @staticmethod
    def display_learning_progress(feedback_system: Any):
        """Display AI learning progress and statistics"""
        st.markdown("### 📊 AI Learning Progress")
        
        if not feedback_system:
            st.info("Feedback system not initialized")
            return
        
        try:
            # Get statistics safely
            stats = feedback_system.get_statistics() if hasattr(feedback_system, 'get_statistics') else {}
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Feedback",
                    stats.get('total_feedback', 0),
                    help="Total number of feedback entries collected"
                )
            
            with col2:
                st.metric(
                    "Avg Rating",
                    f"{stats.get('avg_rating', 0):.1f} ⭐",
                    help="Average rating across all visualizations"
                )
            
            with col3:
                st.metric(
                    "Success Rate",
                    f"{stats.get('success_rate', 0):.1f}%",
                    help="Percentage of charts rated 4+ stars"
                )
            
            with col4:
                st.metric(
                    "Patterns Learned",
                    stats.get('patterns_learned', 0),
                    help="Number of successful patterns identified"
                )
            
            # Show recent improvements
            with st.expander("🚀 Recent Improvements"):
                improvements = stats.get('recent_improvements', [])
                if improvements:
                    for improvement in improvements[:5]:
                        st.markdown(f"- {improvement}")
                else:
                    st.markdown("No recent improvements to show")
        
        except Exception as e:
            st.error(f"Error displaying learning progress: {str(e)}")
    
    @staticmethod
    def display_error_message(error: Exception, recovery_attempted: bool = False, 
                            recovery_message: str = ""):
        """Display user-friendly error message"""
        st.error("😕 Something went wrong")
        
        with st.expander("🔍 Error Details", expanded=True):
            st.markdown(f"**Error Type:** {type(error).__name__}")
            st.markdown(f"**Message:** {str(error)}")
            
            if recovery_attempted:
                if recovery_message.startswith("Fixed"):
                    st.info(f"🔧 {recovery_message}")
                else:
                    st.warning(f"🔧 Recovery attempted: {recovery_message}")
            
            st.markdown("**What you can do:**")
            st.markdown("1. Try rephrasing your request")
            st.markdown("2. Check if column names are spelled correctly")
            st.markdown("3. Ensure date columns are properly formatted")
            st.markdown("4. Use the feedback button to report this issue")


class ChartDisplayUI:
    """Enhanced chart display with feedback integration"""
    
    @staticmethod
    def display_chart_with_feedback(chart, chart_index, confidence, validation, feedback_system):
        """Display chart with all feedback and validation components - FIXED VERSION"""
        try:
            # Handle different input formats for chart
            if isinstance(chart, dict):
                chart_info = chart
                figure = chart.get('figure') or chart.get('chart')
                title = chart.get('title', f'Chart {chart_index}')
            else:
                # If chart is just the figure
                figure = chart
                chart_info = {'figure': chart, 'title': f'Chart {chart_index}'}
                title = f'Chart {chart_index}'
            
            # Handle different input formats for confidence
            if isinstance(confidence, (int, float)):
                # If confidence is just a number, convert to proper format
                confidence_info = {
                    'score': confidence,
                    'level': 'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'Low',
                    'color': '#28a745' if confidence >= 0.8 else '#ffc107' if confidence >= 0.6 else '#fd7e14',
                    'icon': '✅' if confidence >= 0.8 else '⚠️' if confidence >= 0.6 else '📉'
                }
            elif isinstance(confidence, dict):
                # If it's already a dict, ensure all keys are present
                confidence_info = {
                    'score': confidence.get('score', 0.0),
                    'level': confidence.get('level', 'Unknown'),
                    'color': confidence.get('color', 'gray'),
                    'icon': confidence.get('icon', '📊')
                }
            else:
                # Fallback for any other type
                confidence_info = {
                    'score': 0.0,
                    'level': 'Unknown',
                    'color': 'gray',
                    'icon': '📊'
                }
            
            # Handle validation
            if not isinstance(validation, dict):
                validation = {'passed': True, 'issues': []}
            
            # Chart container
            with st.container():
                # Header with title
                st.markdown(f"### {title}")
                
                # AI reasoning if available
                if chart_info.get('ai_analysis'):
                    st.caption(f"🤖 AI Reasoning: {chart_info.get('ai_analysis')}")
                
                # Confidence badge
                AIFeedbackUI.display_confidence_badge(confidence_info)
                
                # Display the chart
                if figure:
                    st.plotly_chart(figure, use_container_width=True, key=f"chart_{chart_index}")
                else:
                    st.error("No chart figure available to display")
                
                # Code expander with validation status
                if chart_info.get('code'):
                    with st.expander("💻 Generated Code", expanded=False):
                        if validation.get('passed', True):
                            st.success("✅ Code validated successfully")
                        else:
                            st.warning("⚠️ Code has validation warnings")
                            
                        st.code(chart_info.get('code', ''), language="python")
                
                # Display insights
                if chart_info.get('insights'):
                    st.markdown("### 🧠 AI-Generated Insights")
                    for insight in chart_info.get('insights', [])[:4]:
                        st.markdown(f"• **{insight}**")
                
                # Fixed rating system
                chart_id = f"chart_{chart_index}_{hash(str(chart_info.get('title', '')))}"
                AIFeedbackUI.display_rating_widget_fixed(chart_id, feedback_system)
                
                return None
            
        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
            print(f"Chart display error: {e}")
            return None


# Export UI components
__all__ = [
    'AIFeedbackUI',
    'ChartDisplayUI'
]