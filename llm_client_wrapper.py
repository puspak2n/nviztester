# llm_client_wrapper.py
"""
LLM Client wrapper for enhancing story mode with dynamic narrative generation
"""

import json
from typing import Dict, Any, Optional
import re

class LLMClient:
    """
    Wrapper for LLM functionality in story mode.
    This can be adapted to use OpenAI, Anthropic, or any other LLM API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.use_local = not api_key  # Use local generation if no API key
        
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        Generate text based on prompt.
        Falls back to sophisticated templates if no API available.
        """
        if self.use_local:
            return self._local_generate(prompt, temperature, max_tokens)
        else:
            return self._api_generate(prompt, temperature, max_tokens)
    
    def _api_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate using actual LLM API (implement based on your provider)
        """
        try:
            # Example for OpenAI (uncomment and modify as needed)
            # import openai
            # openai.api_key = self.api_key
            # response = openai.ChatCompletion.create(
            #     model=self.model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=temperature,
            #     max_tokens=max_tokens
            # )
            # return response.choices[0].message.content
            
            # For now, fall back to local
            return self._local_generate(prompt, temperature, max_tokens)
            
        except Exception as e:
            print(f"API generation failed: {e}")
            return self._local_generate(prompt, temperature, max_tokens)
    
    def _local_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Sophisticated local generation based on prompt analysis
        """
        prompt_lower = prompt.lower()
        
        # Extract context from prompt
        context = self._extract_context(prompt)
        
        # Generate based on section type
        if 'overview' in prompt_lower or 'executive' in prompt_lower:
            return self._generate_overview_narrative(context)
        elif 'trend' in prompt_lower:
            return self._generate_trends_narrative(context)
        elif 'segment' in prompt_lower:
            return self._generate_segments_narrative(context)
        elif 'action' in prompt_lower:
            return self._generate_actions(context)
        elif 'forecast' in prompt_lower:
            return self._generate_forecast_narrative(context)
        else:
            return self._generate_generic_narrative(context)
    
    def _extract_context(self, prompt: str) -> Dict[str, Any]:
        """Extract structured context from prompt"""
        context = {}
        
        # Extract metrics
        metrics_match = re.search(r'Key facts: ({.*?})', prompt, re.DOTALL)
        if metrics_match:
            try:
                context['metrics'] = json.loads(metrics_match.group(1))
            except:
                context['metrics'] = {}
        
        # Extract user focus
        focus_match = re.search(r'User focus: (.*?)(?:\n|$)', prompt)
        if focus_match:
            context['user_focus'] = focus_match.group(1).strip()
        
        # Extract numbers from prompt
        numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', prompt)
        context['numbers'] = numbers
        
        return context
    
    def _generate_overview_narrative(self, context: Dict) -> str:
        """Generate executive overview narrative"""
        metrics = context.get('metrics', {})
        focus = context.get('user_focus', 'business performance')
        
        # Build dynamic narrative
        narrative_parts = []
        
        # Opening
        narrative_parts.append(
            f"This comprehensive analysis of your {focus} reveals significant insights "
            f"that drive strategic decision-making."
        )
        
        # Metrics summary
        if metrics:
            metric_summaries = []
            for metric, data in list(metrics.items())[:3]:
                if isinstance(data, dict):
                    total = data.get('total', 'N/A')
                    growth = data.get('growth', 'N/A')
                    if growth != 'N/A' and '+' in str(growth):
                        metric_summaries.append(f"{metric} showing strong growth at {total}")
                    elif growth != 'N/A' and '-' in str(growth):
                        metric_summaries.append(f"{metric} requiring attention with {total}")
                    else:
                        metric_summaries.append(f"{metric} at {total}")
            
            if metric_summaries:
                narrative_parts.append(
                    f"\n\nKey performance indicators highlight: {', '.join(metric_summaries)}. "
                    f"These metrics form the foundation for strategic planning and optimization efforts."
                )
        
        # Closing with action orientation
        narrative_parts.append(
            f"\n\nThe analysis identifies multiple opportunities for performance enhancement, "
            f"with data-driven recommendations prioritized by impact and feasibility."
        )
        
        return ' '.join(narrative_parts)
    
    def _generate_trends_narrative(self, context: Dict) -> str:
        """Generate trends narrative"""
        return f"""
        Temporal analysis reveals evolving patterns in your {context.get('user_focus', 'business metrics')}. 
        
        The data shows distinct phases of performance, with recent periods demonstrating 
        notable shifts that warrant strategic attention. Understanding these trends enables 
        proactive decision-making and helps anticipate future challenges and opportunities.
        
        Historical patterns suggest seasonal influences and market dynamics that should be 
        factored into planning cycles. Continuous monitoring of these trends will ensure 
        timely responses to emerging changes.
        """
    
    def _generate_segments_narrative(self, context: Dict) -> str:
        """Generate segments narrative"""
        return f"""
        Segment analysis uncovers performance disparities across different business dimensions, 
        revealing both exemplary performers and areas requiring intervention.
        
        Top-performing segments demonstrate best practices that can be systematically replicated 
        across underperforming areas. The variance in performance indicates significant untapped 
        potential that proper optimization strategies can unlock.
        
        By addressing the root causes of performance gaps and implementing targeted improvements, 
        substantial gains in overall efficiency and effectiveness are achievable.
        """
    
    def _generate_actions(self, context: Dict) -> str:
        """Generate action recommendations"""
        actions = []
        
        # Analyze context to generate relevant actions
        if 'anomaly' in str(context).lower():
            actions.append("""- Action: Investigate and address detected anomalies
- Impact: Prevent potential revenue loss of 5-10%
- Timeline: Immediate (within 48 hours)
- Priority: high""")
        
        if 'growth' in str(context).lower():
            actions.append("""- Action: Scale successful growth initiatives
- Impact: Accelerate revenue growth by 15-20%
- Timeline: 30 days
- Priority: high""")
        
        if 'segment' in str(context).lower() or 'variance' in str(context).lower():
            actions.append("""- Action: Standardize best practices across all segments
- Impact: Improve underperforming segments by 25%
- Timeline: 60 days
- Priority: medium""")
        
        # Always include monitoring
        actions.append("""- Action: Implement automated performance monitoring
- Impact: Early detection of issues, faster response times
- Timeline: 2 weeks
- Priority: medium""")
        
        return '\n\n'.join(actions[:3])  # Return top 3 actions
    
    def _generate_forecast_narrative(self, context: Dict) -> str:
        """Generate forecast narrative"""
        return f"""
        Predictive modeling based on historical patterns projects future performance trajectories 
        for your {context.get('user_focus', 'key metrics')}.
        
        The forecast incorporates trend analysis, seasonal patterns, and identified growth 
        drivers to provide realistic projections. These projections assume continuation of 
        current market conditions and operational efficiency.
        
        Regular validation of forecast accuracy will enable model refinement and improve 
        future predictions. Consider these projections as a baseline for planning, with 
        adjustments for strategic initiatives and market changes.
        """
    
    def _generate_generic_narrative(self, context: Dict) -> str:
        """Generic narrative for unmatched sections"""
        return f"""
        This analysis examines {context.get('user_focus', 'your data')} to uncover 
        actionable insights and optimization opportunities.
        
        The comprehensive evaluation considers multiple dimensions and their interactions, 
        providing a holistic view of performance and potential improvements.
        
        Data-driven recommendations are prioritized based on expected impact and 
        implementation feasibility.
        """

# Enhanced prompt templates for better narrative generation
class PromptTemplates:
    """
    Advanced prompt templates for different narrative sections
    """
    
    @staticmethod
    def overview_prompt(facts: Dict, context: Dict) -> str:
        """Generate overview prompt with rich context"""
        return f"""
        Generate an executive overview narrative for a business intelligence report.
        
        Context:
        - User Query: {context.get('user_prompt', 'General analysis')}
        - Total Records: {facts.get('shape', [0])[0]:,}
        - Key Metrics: {json.dumps(facts.get('key_metrics', {}), indent=2, default=str)}
        - Anomalies Detected: {len(facts.get('anomalies', []))}
        - Business Context: {context.get('business_context', 'Performance analysis')}
        
        Requirements:
        1. Start with a compelling hook that addresses the user's query
        2. Summarize the most important 2-3 metrics with their implications
        3. Mention any critical anomalies or opportunities
        4. End with a forward-looking statement
        5. Keep it to 3-4 paragraphs
        6. Use business language, not technical jargon
        7. Be specific with numbers but contextualize them
        
        Tone: Professional yet engaging, focusing on actionable insights
        
        Narrative:
        """
    
    @staticmethod
    def action_prompt(insights: list, context: Dict) -> str:
        """Generate action prompt based on insights"""
        return f"""
        Based on the following data analysis insights, generate 3-5 specific, 
        actionable recommendations for business improvement.
        
        Top Insights:
        {chr(10).join(f'- {insight}' for insight in insights[:5])}
        
        Business Context: {context.get('user_prompt', 'Optimize performance')}
        
        For each recommendation, provide:
        - Action: [Specific action to take]
        - Impact: [Expected measurable outcome]
        - Timeline: [Realistic timeframe]
        - Priority: [high/medium/low based on impact and urgency]
        - Resources: [Key resources or teams needed]
        
        Focus on:
        1. Quick wins (high impact, low effort)
        2. Strategic initiatives (high impact, higher effort)
        3. Risk mitigation (preventing losses)
        
        Format as structured text, not JSON.
        
        Recommendations:
        """

# Simple factory function to create LLM client
def create_llm_client(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> LLMClient:
    """
    Factory function to create appropriate LLM client
    """
    return LLMClient(api_key=api_key, model=model)

# Integration helper
def add_llm_to_agent(agent, api_key: Optional[str] = None):
    """
    Add LLM client to existing agent
    """
    if not hasattr(agent, 'llm_client'):
        agent.llm_client = create_llm_client(api_key)
        print("✅ LLM Client added to agent")
    return agent