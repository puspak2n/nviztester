# story_diagnostic.py
"""
Diagnostic helper to check what components are available for enhanced story mode
"""

import streamlit as st

def diagnose_story_components(agent):
    """Check which components are available on the agent"""
    
    st.write("### 🔍 Story Mode Component Diagnostics")
    
    # Check for RAG system
    has_rag = hasattr(agent, 'rag_system')
    if has_rag:
        st.success("✅ RAG System: Available")
        st.write(f"   - Type: {type(agent.rag_system)}")
        st.write(f"   - Has enhance_query: {hasattr(agent.rag_system, 'enhance_query')}")
    else:
        st.warning("❌ RAG System: Not found")
        st.write("   - The agent needs a 'rag_system' attribute")
    
    # Check for Knowledge Graph
    has_kg = hasattr(agent, 'knowledge_graph')
    if has_kg:
        st.success("✅ Knowledge Graph: Available")
        st.write(f"   - Type: {type(agent.knowledge_graph)}")
    else:
        st.warning("❌ Knowledge Graph: Not found")
        st.write("   - The agent needs a 'knowledge_graph' attribute")
    
    # Check for LLM client
    has_llm = hasattr(agent, 'llm_client')
    if has_llm:
        st.success("✅ LLM Client: Available")
        st.write(f"   - Type: {type(agent.llm_client)}")
        st.write(f"   - Has generate: {hasattr(agent.llm_client, 'generate')}")
    else:
        st.warning("❌ LLM Client: Not found")
        st.write("   - The agent needs an 'llm_client' attribute")
    
    # Check what attributes the agent actually has
    st.write("\n### 📋 Agent Attributes:")
    agent_attrs = [attr for attr in dir(agent) if not attr.startswith('_')]
    
    # Group attributes
    method_attrs = []
    property_attrs = []
    
    for attr in agent_attrs:
        try:
            attr_value = getattr(agent, attr)
            if callable(attr_value):
                method_attrs.append(attr)
            else:
                property_attrs.append(f"{attr}: {type(attr_value).__name__}")
        except:
            pass
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Methods:**")
        for method in method_attrs[:10]:  # Show first 10
            st.write(f"- {method}()")
        if len(method_attrs) > 10:
            st.write(f"... and {len(method_attrs) - 10} more")
    
    with col2:
        st.write("**Properties:**")
        for prop in property_attrs[:10]:  # Show first 10
            st.write(f"- {prop}")
        if len(property_attrs) > 10:
            st.write(f"... and {len(property_attrs) - 10} more")
    
    # Return summary
    return {
        'has_rag': has_rag,
        'has_kg': has_kg,
        'has_llm': has_llm,
        'can_use_enhanced': has_rag or has_kg or has_llm
    }

def add_mock_components(agent):
    """Add mock components for testing enhanced features"""
    
    class MockRAGSystem:
        def enhance_query(self, query, metadata, context):
            return {
                'primary_domain': 'sales',
                'confidence': 0.85,
                'documents': [
                    {'title': 'Sales Best Practices', 'content': 'Focus on customer value...', 'score': 0.9}
                ]
            }
    
    class MockKnowledgeGraph:
        def entity_exists(self, entity):
            return True
        
        def get_entity_context(self, entity, depth=2):
            return {
                'entity': entity,
                'properties': {'type': 'product', 'category': 'technology'},
                'relationships': [{'type': 'sold_by', 'target': 'Sales Team'}]
            }
    
    class MockLLMClient:
        def generate(self, prompt, temperature=0.3, max_tokens=500):
            # Simple template-based response
            if 'overview' in prompt.lower():
                return """
                Your sales performance demonstrates strong momentum with notable achievements 
                across key segments. The data reveals emerging opportunities in underserved 
                markets while highlighting areas requiring strategic attention.
                """
            elif 'action' in prompt.lower():
                return """
                - Action: Focus on top-performing segments
                - Impact: 15-20% revenue increase
                - Timeline: 30 days
                - Priority: high
                
                - Action: Address declining product categories  
                - Impact: Prevent 10% revenue loss
                - Timeline: 2 weeks
                - Priority: high
                """
            else:
                return "Analysis complete. Multiple optimization opportunities identified."
    
    # Add mock components
    if not hasattr(agent, 'rag_system'):
        agent.rag_system = MockRAGSystem()
        st.info("Added mock RAG system for testing")
    
    if not hasattr(agent, 'knowledge_graph'):
        agent.knowledge_graph = MockKnowledgeGraph()
        st.info("Added mock Knowledge Graph for testing")
    
    if not hasattr(agent, 'llm_client'):
        agent.llm_client = MockLLMClient()
        st.info("Added mock LLM client for testing")
    
    return agent