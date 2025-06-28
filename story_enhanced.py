# story_enhanced.py
"""
Enhanced Story System with optional RAG integration, Knowledge Graph support,
and dynamic LLM-based narrative generation. Works with or without these components.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import base64

# Import existing utilities
from chart_utils import (
    create_dark_chart,
    get_business_relevant_columns,
    detect_value_type,
    format_value
)

# Optional imports for enhanced capabilities
try:
    from rag_system import RAGSystem
    HAS_RAG = True
except ImportError:
    RAGSystem = None
    HAS_RAG = False

try:
    from knowledge_graph import KnowledgeGraph
    HAS_KG = True
except ImportError:
    KnowledgeGraph = None
    HAS_KG = False

try:
    from llm_client import LLMClient
    HAS_LLM = True
except ImportError:
    LLMClient = None
    HAS_LLM = False

# Import the original story components to maintain compatibility
from story import (
    StoryType,
    StoryChapter as OriginalStoryChapter,
    DataStory as OriginalDataStory,
    StoryComponent,
    SalesPerformanceStory,
    MarketingROIStory,
    StoryEngine as OriginalStoryEngine,
    ActionEngine,
    WhatIfEngine,
    StoryModeUI
)

# Enhanced versions that add optional features
@dataclass
class StoryChapter(OriginalStoryChapter):
    """Enhanced chapter with optional dynamic content"""
    metadata: Dict[str, Any] = None

@dataclass  
class DataStory(OriginalDataStory):
    """Enhanced story structure with optional RAG/KG context"""
    rag_context: Dict[str, Any] = None
    kg_entities: List[Dict[str, Any]] = None

# Insights Ranker
class InsightsRanker:
    """Rank insights by business importance"""
    
    def __init__(self, kg=None):
        self.kg = kg
        self.importance_weights = {
            'anomaly': 0.9,
            'trend_change': 0.8,
            'performance_gap': 0.7,
            'opportunity': 0.6,
            'benchmark': 0.5,
            'general': 0.3
        }
    
    def rank_insights(self, raw_insights: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[str]:
        """Rank and format insights by importance"""
        scored_insights = []
        
        for insight in raw_insights:
            # Calculate base score
            insight_type = insight.get('type', 'general')
            base_score = self.importance_weights.get(insight_type, 0.3)
            
            # Adjust score based on magnitude
            magnitude = insight.get('magnitude', 0)
            if magnitude > 0.5:
                base_score *= 1.5
            elif magnitude > 0.3:
                base_score *= 1.2
            
            # Boost score if entities are in KG (only if KG available)
            if HAS_KG and self.kg and insight.get('entities'):
                kg_boost = 0
                for entity in insight['entities']:
                    try:
                        if self.kg.entity_exists(entity):
                            kg_boost += 0.1
                    except:
                        pass
                base_score += kg_boost
            
            # Add business context scoring
            if context and context.get('user_focus'):
                if any(focus in insight.get('text', '').lower() for focus in context['user_focus']):
                    base_score *= 1.3
            
            scored_insights.append({
                'score': base_score,
                'text': insight.get('text', ''),
                'metadata': insight
            })
        
        # Sort by score and return formatted text
        scored_insights.sort(key=lambda x: x['score'], reverse=True)
        return [si['text'] for si in scored_insights[:10]]  # Top 10 insights

# Enhanced Story Component that extends the original
class EnhancedStoryComponent(StoryComponent):
    """Enhanced base class with optional RAG, KG, and LLM integration"""
    
    def __init__(self, conversational_agent):
        super().__init__(conversational_agent)
        
        # Check for optional components
        self.rag = getattr(conversational_agent, 'rag_system', None) if HAS_RAG else None
        self.kg = getattr(conversational_agent, 'knowledge_graph', None) if HAS_KG else None
        self.llm = getattr(conversational_agent, 'llm_client', None) if HAS_LLM else None
        self.insights_ranker = InsightsRanker(self.kg)
        
        # Report available features
        self._report_features()
    
    def _report_features(self):
        """Report which enhanced features are available"""
        features = []
        if self.rag:
            features.append("RAG")
        if self.kg:
            features.append("Knowledge Graph")
        if self.llm:
            features.append("LLM")
        
        if features:
            print(f"✅ Enhanced story mode initialized with: {', '.join(features)}")
            st.sidebar.success(f"Story Mode: Enhanced ({', '.join(features)})")
        else:
            print("📚 Story mode using basic features (RAG/KG/LLM not available)")
            st.sidebar.info("Story Mode: Basic (No RAG/KG/LLM)")
            st.sidebar.caption("Enable diagnostics in sidebar to add test components")
    
    def analyze_data_enhanced(self, dataframe: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis with optional RAG and KG enrichment"""
        # Start with basic analysis
        analysis = self.analyze_data(dataframe, context)
        
        # Only enhance if components are available
        if self.rag and context.get('user_prompt'):
            try:
                # Prepare metadata for RAG
                data_metadata = {
                    'columns': list(dataframe.columns),
                    'shape': dataframe.shape,
                    'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                    'sample': dataframe.head(3).to_dict('records')
                }
                
                # Get RAG enhancement
                rag_enhancement = self.rag.enhance_query(
                    context['user_prompt'],
                    data_metadata,
                    context.get('business_context', '')
                )
                
                analysis['rag_context'] = rag_enhancement
                analysis['relevant_docs'] = rag_enhancement.get('documents', [])
                
            except Exception as e:
                print(f"RAG enhancement failed: {e}")
        
        # Knowledge Graph enhancement (if available)
        if self.kg:
            try:
                entities = self._extract_entities_from_data(dataframe, analysis)
                analysis['kg_entities'] = entities
                
            except Exception as e:
                print(f"KG enhancement failed: {e}")
        
        return analysis
    
    def _extract_entities_from_data(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Extract relevant entities from data and analysis"""
        entities = []
        
        # Extract from categorical columns
        business_cols = analysis.get('business_cols', {})
        for cat_col in business_cols.get('categorical', [])[:3]:
            if cat_col in df.columns:
                top_values = df[cat_col].value_counts().head(5).index.tolist()
                entities.extend([str(v) for v in top_values])
        
        # Extract from anomalies
        for anomaly in analysis.get('anomalies', []):
            if anomaly.get('entity'):
                entities.append(str(anomaly['entity']))
        
        return list(set(entities))  # Unique entities
    
    def generate_dynamic_narrative(self, section: str, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate narrative - use LLM if available, otherwise use templates"""
        # If LLM is available, try dynamic generation
        if self.llm:
            try:
                return self._generate_llm_narrative(section, facts, context)
            except Exception as e:
                print(f"LLM narrative generation failed: {e}")
        
        # Fallback to enhanced templates
        return self._generate_enhanced_template_narrative(section, facts, context)
    
    def _generate_llm_narrative(self, section: str, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate narrative using LLM"""
        # Build context for LLM
        llm_context = {
            'section': section,
            'facts': facts,
            'user_query': context.get('user_prompt', ''),
            'business_context': context.get('business_context', '')
        }
        
        # Add any available enhancements
        if facts.get('relevant_docs'):
            llm_context['external_context'] = facts['relevant_docs'][:3]
        
        # Simple prompt
        prompt = f"""
        Generate a {section} narrative for a data story.
        
        Key facts: {json.dumps(facts.get('key_metrics', {}), default=str)}
        User focus: {context.get('user_prompt', 'General analysis')}
        
        Write 2-3 paragraphs in a professional, engaging style.
        """
        
        return self.llm.generate(prompt, temperature=0.3)
    
    def _generate_enhanced_template_narrative(self, section: str, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Enhanced template-based narrative with dynamic elements"""
        # Get base narrative from templates
        narratives = {
            'overview': self._build_overview_narrative(facts, context),
            'trends': self._build_trends_narrative(facts, context),
            'segments': self._build_segments_narrative(facts, context),
            'health': self._build_health_narrative(facts, context),
            'forecast': self._build_forecast_narrative(facts, context),
            'actions': self._build_actions_narrative(facts, context)
        }
        
        return narratives.get(section, self._build_default_narrative(facts, context))
    
    def _build_overview_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build overview narrative with available enhancements"""
        metrics_summary = []
        for metric, data in list(facts.get('key_metrics', {}).items())[:3]:
            if 'total' in data:
                metrics_summary.append(f"{metric}: {data['total']}")
        
        anomaly_count = len(facts.get('anomalies', []))
        
        narrative = f"""
        Welcome to your comprehensive data analysis. This report examines {facts.get('shape', [0])[0]:,} data points
        to deliver actionable insights tailored to your needs.
        
        Key performance indicators show {' | '.join(metrics_summary) if metrics_summary else 'multiple metrics analyzed'}.
        """
        
        # Add RAG context if available
        if facts.get('relevant_docs'):
            narrative += f"""
        
        External market intelligence has been incorporated to provide additional context and benchmarks
        for your performance metrics.
        """
        
        # Add anomaly information
        if anomaly_count > 0:
            narrative += f"""
        
        Notably, {anomaly_count} significant anomalies have been detected that warrant immediate attention.
        These outliers may represent both risks and opportunities for your business.
        """
        
        return narrative
    
    def _build_trends_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build trends narrative"""
        trends = facts.get('trends', {})
        
        if not trends:
            return "Trend analysis requires time-series data. Additional historical data would enable more comprehensive trend insights."
        
        # Analyze trend directions
        increasing = sum(1 for t in trends.values() if t.get('direction') == 'increasing')
        decreasing = sum(1 for t in trends.values() if t.get('direction') == 'decreasing')
        
        narrative = f"""
        Performance trend analysis reveals important patterns in your data over time.
        """
        
        if increasing > decreasing:
            narrative += f" The overall trajectory is positive, with {increasing} metrics showing upward trends."
        elif decreasing > increasing:
            narrative += f" Attention is needed as {decreasing} metrics show declining performance."
        else:
            narrative += " Performance shows mixed results with both growth and decline across different metrics."
        
        # Add volatility insight
        high_volatility = [m for m, t in trends.items() if t.get('volatility', 0) > 0.3]
        if high_volatility:
            narrative += f"""
        
        High volatility detected in {', '.join(high_volatility[:2])}, suggesting unstable performance
        that may benefit from root cause analysis and stabilization efforts.
        """
        
        return narrative
    
    def _build_segments_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build segments narrative"""
        segments = facts.get('segments', {})
        
        if not segments:
            return "Segment analysis provides insights into performance variations across different business dimensions."
        
        # Find high variance segments
        high_variance = sum(1 for s in segments.values() if s.get('variance', 0) > 0.5)
        
        narrative = f"""
        Segment performance analysis uncovers significant variations across your business dimensions.
        """
        
        if high_variance > 0:
            narrative += f"""
        
        {high_variance} segments show high performance variance, indicating substantial opportunities
        for optimization. Best practices from top performers can be replicated to improve lagging segments.
        """
        
        return narrative
    
    def _build_health_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build health check narrative"""
        anomalies = facts.get('anomalies', [])
        
        narrative = """
        Business health assessment evaluates the overall state of your operations, identifying
        both strengths to leverage and weaknesses to address.
        """
        
        if anomalies:
            high_severity = sum(1 for a in anomalies if a.get('severity') == 'high')
            if high_severity > 0:
                narrative += f"""
        
        {high_severity} high-severity anomalies require immediate investigation. These outliers
        significantly deviate from expected patterns and may indicate critical issues or opportunities.
        """
        
        return narrative
    
    def _build_forecast_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build forecast narrative"""
        return """
        Forward-looking analysis projects future performance based on historical patterns and current trends.
        These projections assume continuation of current market conditions and should be adjusted
        for known future changes or strategic initiatives.
        
        Regular monitoring against these forecasts will help identify early deviations and enable
        timely course corrections.
        """
    
    def _build_actions_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build actions narrative"""
        opportunities = facts.get('business_opportunities', [])
        
        narrative = """
        Based on comprehensive analysis, here is your prioritized action plan designed to maximize
        impact while optimizing resource allocation.
        """
        
        if opportunities:
            narrative += f"""
        
        {len(opportunities)} specific opportunities have been identified, ranging from quick wins
        requiring minimal effort to strategic initiatives that could transform performance.
        """
        
        return narrative
    
    def _build_default_narrative(self, facts: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Default narrative for unknown sections"""
        return "This section provides detailed analysis and insights based on your data patterns."
    
    def _extract_raw_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract raw insights with metadata for ranking"""
        raw_insights = []
        
        # Extract from anomalies
        for anomaly in analysis.get('anomalies', []):
            raw_insights.append({
                'type': 'anomaly',
                'text': f"{anomaly['entity']} shows unusual {anomaly['metric']} of {anomaly['value']}",
                'magnitude': abs(anomaly.get('z_score', 0)) / 3 if 'z_score' in anomaly else 0.5,
                'entities': [anomaly['entity']],
                'metadata': anomaly
            })
        
        # Extract from trends
        for metric, trend in analysis.get('trends', {}).items():
            if abs(trend.get('recent_change', 0)) > 20:
                raw_insights.append({
                    'type': 'trend_change',
                    'text': f"{metric} {'surged' if trend['recent_change'] > 0 else 'dropped'} {abs(trend['recent_change']):.1f}% recently",
                    'magnitude': abs(trend['recent_change']) / 100,
                    'entities': [metric],
                    'metadata': trend
                })
        
        # Extract from segments
        for segment_key, segment_data in analysis.get('segments', {}).items():
            if segment_data.get('variance', 0) > 0.5:
                raw_insights.append({
                    'type': 'performance_gap',
                    'text': f"High variance in {segment_key} indicates optimization opportunity",
                    'magnitude': segment_data['variance'],
                    'entities': [segment_key.split('_')[0]],
                    'metadata': segment_data
                })
        
        # Add general insights if few specific ones
        if len(raw_insights) < 5:
            # Add some general insights based on data
            if analysis.get('key_metrics'):
                for metric, data in list(analysis['key_metrics'].items())[:3]:
                    if data.get('growth') and data['growth'] != 'N/A':
                        raw_insights.append({
                            'type': 'general',
                            'text': f"{metric} growth: {data['growth']}",
                            'magnitude': 0.3,
                            'entities': [metric],
                            'metadata': data
                        })
        
        return raw_insights
    
    def generate_dynamic_actions(self, insights: List[str], analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions - use LLM if available, otherwise use smart templates"""
        # Base actions from insights
        actions = []
        
        # Anomaly-based actions
        if any('unusual' in insight or 'anomaly' in insight for insight in insights):
            actions.append({
                'id': 'investigate_anomalies',
                'label': 'Investigate Detected Anomalies',
                'type': 'analysis',
                'urgency': 'high',
                'description': 'Deep dive into unusual patterns to identify root causes'
            })
        
        # Trend-based actions
        if any('surged' in insight or 'dropped' in insight for insight in insights):
            actions.append({
                'id': 'trend_response',
                'label': 'Respond to Trend Changes',
                'type': 'planning',
                'urgency': 'high',
                'description': 'Develop strategies to capitalize on positive trends or mitigate negative ones'
            })
        
        # Optimization actions
        if any('optimization' in insight or 'variance' in insight for insight in insights):
            actions.append({
                'id': 'optimize_segments',
                'label': 'Optimize Underperforming Segments',
                'type': 'optimization',
                'urgency': 'medium',
                'description': 'Implement best practices from top performers'
            })
        
        # Always include monitoring
        actions.append({
            'id': 'setup_monitoring',
            'label': 'Establish KPI Monitoring',
            'type': 'monitoring',
            'urgency': 'medium',
            'description': 'Set up dashboards and alerts for continuous tracking'
        })
        
        return actions[:5]  # Return top 5 actions

# Enhanced Sales Performance Story
class EnhancedSalesPerformanceStory(EnhancedStoryComponent, SalesPerformanceStory):
    """Sales story with optional dynamic narrative generation"""
    
    def __init__(self, conversational_agent):
        EnhancedStoryComponent.__init__(self, conversational_agent)
        self.story_type = StoryType.SALES_PERFORMANCE
    
    def generate(self, dataframe: pd.DataFrame, context: Dict[str, Any]) -> DataStory:
        """Generate complete sales performance story with optional enhancements"""
        
        # Use enhanced analysis if available
        if self.rag or self.kg:
            analysis = self.analyze_data_enhanced(dataframe, context)
        else:
            analysis = self.analyze_data(dataframe, context)
        
        # Create chapters
        chapters = []
        
        # Overview chapter
        overview_chapter = self._create_enhanced_overview_chapter(dataframe, analysis, context)
        chapters.append(overview_chapter)
        
        # Trends chapter
        trends_chapter = self._create_enhanced_trends_chapter(dataframe, analysis, context)
        chapters.append(trends_chapter)
        
        # Continue with other chapters using original logic but with enhancements
        chapters.extend([
            self._create_segment_analysis_chapter(dataframe, analysis),
            self._create_opportunity_health_chapter(dataframe, analysis),
            self._create_forecast_chapter(dataframe, analysis),
            self._create_enhanced_action_plan_chapter(dataframe, analysis, context)
        ])
        
        # Generate executive summary
        exec_summary = self._generate_enhanced_executive_summary(analysis, context)
        
        # Extract key takeaways with ranking
        key_takeaways = self._extract_enhanced_takeaways(analysis, chapters)
        
        # Compile recommended actions
        recommended_actions = self._compile_enhanced_actions(chapters)
        
        # Build story with optional enhancements
        story_dict = {
            'id': f"sales_story_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'title': "Sales Performance Analysis",
            'story_type': self.story_type,
            'chapters': chapters,
            'executive_summary': exec_summary,
            'key_takeaways': key_takeaways,
            'recommended_actions': recommended_actions,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_points': len(dataframe),
                'analysis': analysis,
                'enhanced_features': {
                    'rag_enabled': bool(self.rag),
                    'kg_enabled': bool(self.kg),
                    'llm_enabled': bool(self.llm)
                }
            }
        }
        
        # Add optional fields if available
        if analysis.get('rag_context'):
            story_dict['rag_context'] = analysis['rag_context']
        if analysis.get('kg_entities'):
            story_dict['kg_entities'] = analysis['kg_entities']
        
        return DataStory(**story_dict)
    
    def _create_enhanced_overview_chapter(self, df: pd.DataFrame, analysis: Dict[str, Any], 
                                        context: Dict[str, Any]) -> StoryChapter:
        """Create overview chapter with enhancements"""
        # Extract and rank insights
        raw_insights = self._extract_raw_insights(analysis)
        ranked_insights = self.insights_ranker.rank_insights(raw_insights, context)
        
        # Generate narrative
        chapter_facts = {
            'key_metrics': analysis.get('key_metrics', {}),
            'shape': analysis.get('shape', (0, 0)),
            'anomalies': analysis.get('anomalies', []),
            'relevant_docs': analysis.get('relevant_docs', [])
        }
        
        narrative = self.generate_dynamic_narrative('overview', chapter_facts, context)
        
        # Create visualizations using original logic
        visualizations = []
        
        # KPI Dashboard
        kpi_metrics = self._extract_kpi_metrics(df, analysis)
        if kpi_metrics:
            fig_kpi = self._create_kpi_dashboard(kpi_metrics)
            visualizations.append({
                'type': 'plotly',
                'figure': fig_kpi,
                'title': 'Key Performance Indicators'
            })
        
        # Top performers chart
        business_cols = analysis['business_cols']
        if business_cols['categorical'] and business_cols['numerical']:
            cat_col = business_cols['categorical'][0]
            metric_col = business_cols['numerical'][0]
            
            summary_data = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False).head(10)
            
            fig_summary = px.bar(
                x=summary_data.index,
                y=summary_data.values,
                title=f"Top 10 {cat_col} by {metric_col}"
            )
            fig_summary = create_dark_chart(fig_summary)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_summary,
                'title': 'Top Performers'
            })
        
        # Generate actions
        actions = self.generate_dynamic_actions(ranked_insights[:3], analysis, context)
        
        return StoryChapter(
            id="overview",
            title="Executive Overview",
            subtitle="Your sales performance at a glance",
            narrative=narrative,
            visualizations=visualizations,
            insights=ranked_insights[:5],
            actions=actions[:3],
            what_if_scenarios=[],
            metrics=kpi_metrics,
            metadata={
                'enhancement_level': 'full' if self.llm else 'basic',
                'insights_ranked': True
            }
        )
    
    def _create_enhanced_trends_chapter(self, df: pd.DataFrame, analysis: Dict[str, Any],
                                      context: Dict[str, Any]) -> StoryChapter:
        """Create trends chapter with enhancements"""
        # Extract trend-specific insights
        trend_insights_raw = []
        for metric, trend in analysis.get('trends', {}).items():
            trend_insights_raw.append({
                'type': 'trend_change' if abs(trend.get('recent_change', 0)) > 10 else 'general',
                'text': f"{metric} is {trend.get('direction', 'stable')} with {trend.get('volatility', 0):.2f} volatility",
                'magnitude': abs(trend.get('recent_change', 0)) / 100,
                'entities': [metric],
                'metadata': trend
            })
        
        ranked_insights = self.insights_ranker.rank_insights(trend_insights_raw, context)
        
        # Generate narrative
        narrative = self.generate_dynamic_narrative('trends', analysis, context)
        
        # Create visualizations (use original logic)
        visualizations = []
        business_cols = analysis['business_cols']
        
        if business_cols['temporal'] and business_cols['numerical']:
            time_col = business_cols['temporal'][0]
            metric_col = business_cols['numerical'][0]
            
            # Create trend visualization
            df[time_col] = pd.to_datetime(df[time_col])
            monthly_data = df.groupby(df[time_col].dt.to_period('M'))[metric_col].sum().reset_index()
            monthly_data[time_col] = monthly_data[time_col].dt.to_timestamp()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=monthly_data[time_col],
                y=monthly_data[metric_col],
                mode='lines+markers',
                name='Actual'
            ))
            
            fig_trend.update_layout(title=f"{metric_col} Trend Over Time")
            fig_trend = create_dark_chart(fig_trend)
            
            visualizations.append({
                'type': 'plotly',
                'figure': fig_trend,
                'title': 'Performance Trend'
            })
        
        return StoryChapter(
            id="trends",
            title="Performance Trends",
            subtitle="How your sales are evolving over time",
            narrative=narrative,
            visualizations=visualizations,
            insights=ranked_insights[:5],
            actions=[
                {
                    'id': 'monitor_trends',
                    'label': 'Set Up Trend Monitoring',
                    'type': 'monitoring',
                    'urgency': 'medium'
                }
            ],
            what_if_scenarios=[
                {
                    'id': 'growth_projection',
                    'label': 'Project Future Growth',
                    'type': 'slider',
                    'params': {'min': -20, 'max': 50, 'default': 10, 'step': 5, 'unit': '%'}
                }
            ],
            metrics={}
        )
    
    def _create_enhanced_action_plan_chapter(self, df: pd.DataFrame, analysis: Dict[str, Any],
                                           context: Dict[str, Any]) -> StoryChapter:
        """Create action plan chapter with all insights integrated"""
        # Collect all insights from analysis
        all_insights_raw = self._extract_raw_insights(analysis)
        
        # Get top insights
        top_insights = self.insights_ranker.rank_insights(all_insights_raw, context)[:10]
        
        # Generate comprehensive actions
        all_actions = self.generate_dynamic_actions(top_insights, analysis, context)
        
        # Create action visualization
        action_df = pd.DataFrame(all_actions[:5])
        if 'urgency' in action_df.columns:
            action_df['Priority'] = action_df['urgency'].map({'high': 1, 'medium': 2, 'low': 3})
            action_df = action_df.sort_values('Priority')
        
        fig_actions = go.Figure(data=[
            go.Table(
                header=dict(
                    values=['Action', 'Type', 'Urgency'],
                    fill_color='#2d3748',
                    font=dict(color='white', size=14)
                ),
                cells=dict(
                    values=[
                        action_df['label'].tolist() if 'label' in action_df else [],
                        action_df['type'].tolist() if 'type' in action_df else [],
                        action_df['urgency'].tolist() if 'urgency' in action_df else []
                    ],
                    fill_color='#1a202c',
                    font=dict(color='white', size=12),
                    height=30
                )
            )
        ])
        
        fig_actions.update_layout(title="Priority Action Plan", height=400)
        
        narrative = self.generate_dynamic_narrative('actions', {'actions': all_actions}, context)
        
        return StoryChapter(
            id="actions",
            title="Your Action Plan",
            subtitle="Turning insights into results",
            narrative=narrative,
            visualizations=[{
                'type': 'plotly',
                'figure': fig_actions,
                'title': 'Priority Actions'
            }],
            insights=["Execute high-impact actions first", "Monitor progress weekly"],
            actions=all_actions[:3],
            what_if_scenarios=[],
            metrics={'total_actions': len(all_actions)}
        )
    
    def _extract_kpi_metrics(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract KPI metrics for dashboard"""
        kpi_metrics = {}
        
        # From key metrics
        for metric, data in list(analysis.get('key_metrics', {}).items())[:4]:
            kpi_metrics[metric] = {
                'value': data.get('total', 'N/A'),
                'change': data.get('growth', 'N/A'),
                'label': metric
            }
        
        return kpi_metrics
    
    def _generate_enhanced_executive_summary(self, analysis: Dict[str, Any], 
                                           context: Dict[str, Any]) -> str:
        """Generate executive summary with available enhancements"""
        summary_facts = {
            'shape': analysis.get('shape', (0, 0)),
            'key_metrics': analysis.get('key_metrics', {}),
            'anomalies': analysis.get('anomalies', []),
            'opportunities': analysis.get('business_opportunities', [])
        }
        
        return self.generate_dynamic_narrative('executive_summary', summary_facts, context)
    
    def _extract_enhanced_takeaways(self, analysis: Dict[str, Any], 
                                  chapters: List[StoryChapter]) -> List[str]:
        """Extract key takeaways using insight ranking"""
        all_takeaways = []
        
        # Collect top insights from each chapter
        for chapter in chapters:
            if chapter.insights:
                all_takeaways.extend(chapter.insights[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_takeaways = []
        for takeaway in all_takeaways:
            if takeaway not in seen:
                seen.add(takeaway)
                unique_takeaways.append(takeaway)
        
        return unique_takeaways[:5]
    
    def _compile_enhanced_actions(self, chapters: List[StoryChapter]) -> List[Dict[str, Any]]:
        """Compile all actions with priority"""
        all_actions = []
        
        for chapter in chapters:
            for action in chapter.actions:
                action['source_chapter'] = chapter.id
                all_actions.append(action)
        
        # Sort by urgency
        urgency_order = {'high': 0, 'medium': 1, 'low': 2}
        all_actions.sort(key=lambda x: urgency_order.get(x.get('urgency', 'medium'), 1))
        
        return all_actions[:5]

# Enhanced Story Engine
class EnhancedStoryEngine(OriginalStoryEngine):
    """Enhanced engine with optional RAG and KG integration"""
    
    def __init__(self, conversational_agent):
        # Initialize base engine
        super().__init__(conversational_agent)
        
        # Check for enhanced components
        self.rag = getattr(conversational_agent, 'rag_system', None) if HAS_RAG else None
        self.kg = getattr(conversational_agent, 'knowledge_graph', None) if HAS_KG else None
        self.llm = getattr(conversational_agent, 'llm_client', None) if HAS_LLM else None
        
        # Update story components with enhanced versions if features available
        if any([self.rag, self.kg, self.llm]):
            self.story_components[StoryType.SALES_PERFORMANCE] = EnhancedSalesPerformanceStory(conversational_agent)
            # Add other enhanced story types as implemented
            print("Story Engine initialized with enhanced features")
    
    def generate_story(self, prompt: str, dataframe: pd.DataFrame, context: Dict[str, Any] = None) -> DataStory:
        """Generate story with optional enhancements"""
        # Prepare context
        full_context = context or {}
        full_context['user_prompt'] = prompt
        full_context['business_context'] = st.session_state.get('business_context', '')
        
        # Detect story type (with or without RAG)
        if self.rag:
            story_type, confidence = self._detect_story_type_enhanced(prompt, dataframe)
        else:
            story_type, confidence = self.detect_story_type_with_rag(prompt, dataframe)
        
        print(f"📖 Selected story type: {story_type.value} (confidence: {confidence:.1%})")
        
        # Add confidence to context
        full_context['story_confidence'] = confidence
        
        # Generate story
        story_component = self.story_components.get(story_type, self.story_components[StoryType.SALES_PERFORMANCE])
        story = story_component.generate(dataframe, full_context)
        
        # Store story
        self.current_story = story
        self.story_history.append(story)
        
        return story
    
    def _detect_story_type_enhanced(self, prompt: str, dataframe: pd.DataFrame) -> Tuple[StoryType, float]:
        """Enhanced story type detection using RAG"""
        try:
            # Prepare metadata
            metadata = {
                'columns': list(dataframe.columns),
                'shape': dataframe.shape,
                'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
                'sample': dataframe.head(3).to_dict('records')
            }
            
            # Get RAG enhancement
            rag_result = self.rag.enhance_query(prompt, metadata, "")
            
            # Map domain to story type
            domain = rag_result.get('primary_domain', '').lower()
            confidence = rag_result.get('confidence', 0.5)
            
            if 'sales' in domain or 'revenue' in domain:
                return StoryType.SALES_PERFORMANCE, confidence
            elif 'marketing' in domain:
                return StoryType.MARKETING_ROI, confidence
            else:
                return StoryType.SALES_PERFORMANCE, 0.5
                
        except Exception as e:
            print(f"Enhanced detection failed: {e}")
            return StoryType.SALES_PERFORMANCE, 0.5

# Main entry point - works with or without enhancements
def render_story_mode(agent, dataframe: pd.DataFrame):
    """Main entry point for story mode from agentic_ai_new.py"""
    
    # Add diagnostic sidebar option
    with st.sidebar:
        if st.checkbox("🔧 Show Story Mode Diagnostics", key="show_story_diagnostics"):
            st.markdown("---")
            st.markdown("### Story Mode Components")
            
            # Check what's available
            has_rag = hasattr(agent, 'rag_system')
            has_kg = hasattr(agent, 'knowledge_graph')
            has_llm = hasattr(agent, 'llm_client')
            
            # Display status
            st.write("**Component Status:**")
            st.write(f"RAG System: {'✅' if has_rag else '❌'}")
            st.write(f"Knowledge Graph: {'✅' if has_kg else '❌'}")
            st.write(f"LLM Client: {'✅' if has_llm else '❌'}")
            
            # Option to add mock components for testing
            if not all([has_rag, has_kg, has_llm]):
                if st.button("🧪 Add Mock Components (Testing)", key="add_mock_components"):
                    # Add mock components
                    if not has_rag:
                        class MockRAG:
                            def enhance_query(self, q, m, c):
                                return {'primary_domain': 'sales', 'confidence': 0.8, 'documents': []}
                        agent.rag_system = MockRAG()
                    
                    if not has_kg:
                        class MockKG:
                            def entity_exists(self, e): return True
                            def get_entity_context(self, e, d=2): return {'entity': e}
                        agent.knowledge_graph = MockKG()
                    
                    if not has_llm:
                        # Import the LLM client from the wrapper
                        try:
                            from llm_client_wrapper import create_llm_client
                            agent.llm_client = create_llm_client()
                            st.info("Added sophisticated LLM client (local mode)")
                        except:
                            class MockLLM:
                                def generate(self, p, temperature=0.3, max_tokens=500):
                                    return "Dynamic narrative: Your data shows interesting patterns worth exploring."
                            agent.llm_client = MockLLM()
                    
                    st.success("Mock components added! Refresh to use enhanced features.")
                    # Clear the story engine to force reinitialization
                    if 'story_engine' in st.session_state:
                        del st.session_state.story_engine
                    st.rerun()
    
    # Initialize appropriate story engine
    if 'story_engine' not in st.session_state:
        # Check if agent has enhanced features
        has_enhancements = any([
            hasattr(agent, 'rag_system'),
            hasattr(agent, 'knowledge_graph'),
            hasattr(agent, 'llm_client')
        ])
        
        print(f"Story Mode Initialization:")
        print(f"  - RAG System: {'Available' if hasattr(agent, 'rag_system') else 'Not found'}")
        print(f"  - Knowledge Graph: {'Available' if hasattr(agent, 'knowledge_graph') else 'Not found'}")
        print(f"  - LLM Client: {'Available' if hasattr(agent, 'llm_client') else 'Not found'}")
        print(f"  - Using: {'Enhanced' if has_enhancements else 'Basic'} Story Engine")
        
        if has_enhancements:
            st.session_state.story_engine = EnhancedStoryEngine(agent)
        else:
            st.session_state.story_engine = OriginalStoryEngine(agent)
    
    story_engine = st.session_state.story_engine
    
    # Render story mode UI (works with both engines)
    StoryModeUI.render_story_mode(story_engine, dataframe)

# Export all components
__all__ = [
    'EnhancedStoryEngine',
    'EnhancedSalesPerformanceStory', 
    'EnhancedStoryComponent',
    'InsightsRanker',
    'render_story_mode',
    # Keep original exports for compatibility
    'StoryEngine',
    'StoryModeUI',
    'StoryType',
    'DataStory',
    'StoryChapter',
    'ActionEngine',
    'WhatIfEngine'
]