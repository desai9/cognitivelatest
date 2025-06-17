# ==============================================================================
# conversational_agent.py - LLM-Powered Conversational Financial Advisor
# ==============================================================================

import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage

from config_manager import llm_manager, session_manager
from data_models import ConversationEntry, FinancialDataSummary, EnhancedTransaction, FinancialHealthMetrics


class ConversationalAgent:
    """Advanced LLM-powered conversational financial advisor"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        self.conversation_memory = []
        self.user_profile = {}
        self.max_memory_entries = 20
    
    def generate_response(self, query: str, context: Dict = None) -> ConversationEntry:
        """Generate intelligent response with full context awareness"""
        if not self.llm:
            return self._fallback_response(query, context)
        
        try:
            # Prepare comprehensive context
            full_context = self._prepare_context(query, context)
            
            # Analyze user intent and emotion
            intent_analysis = self._analyze_user_intent(query)
            
            # Generate contextual response
            response = self._generate_llm_response(query, full_context, intent_analysis)
            
            # Create conversation entry
            conversation_entry = ConversationEntry(
                user_query=query,
                ai_response=response,
                emotion=intent_analysis.get('emotion', 'neutral'),
                intent=intent_analysis.get('intent', 'general_inquiry'),
                topics=intent_analysis.get('topics', ['general']),
                confidence=intent_analysis.get('confidence', 'medium'),
                follow_up_suggestions=intent_analysis.get('follow_up_suggestions', [])
            )
            
            # Update conversation memory
            self._update_conversation_memory(conversation_entry)
            
            return conversation_entry
            
        except Exception as e:
            st.warning(f"LLM conversation failed: {e}")
            return self._fallback_response(query, context)
    
    def _prepare_context(self, query: str, context: Dict = None) -> Dict:
        """Prepare comprehensive context for LLM"""
        context = context or {}
        
        # Financial context
        financial_context = {}
        
        # Health metrics context
        if 'health_metrics' in context and context['health_metrics']:
            metrics = context['health_metrics']
            financial_context.update({
                'health_score': metrics.get('overall_score', 0),
                'risk_level': metrics.get('risk_level', 'Unknown'),
                'strengths': metrics.get('llm_insights', {}).get('strengths', []),
                'weaknesses': metrics.get('llm_insights', {}).get('weaknesses', []),
                'analysis_quality': metrics.get('analysis_quality', 'basic')
            })
        
        # Transaction context
        if 'transactions' in context and context['transactions']:
            transactions = context['transactions']
            financial_context.update({
                'total_transactions': len(transactions),
                'date_range': self._get_date_range(transactions),
                'top_spending_categories': self._get_top_spending_categories(transactions)
            })
        
        # Conversation history context
        recent_conversations = self.conversation_memory[-5:] if self.conversation_memory else []
        
        return {
            'current_query': query,
            'financial_context': financial_context,
            'conversation_history': recent_conversations,
            'user_profile': self.user_profile,
            'session_context': context
        }
    
    def _analyze_user_intent(self, query: str) -> Dict:
        """Analyze user intent and emotional state"""
        if not self.llm:
            return self._basic_intent_analysis(query)
        
        try:
            prompt = f"""
            Analyze this user query for financial coaching context:
            
            Query: "{query}"
            
            Determine:
            1. Emotional state (stressed, anxious, confident, excited, confused, neutral, frustrated, motivated)
            2. Primary intent (seeking_advice, asking_question, sharing_concern, requesting_analysis, goal_setting, celebrating_progress, venting_frustration)
            3. Financial topics mentioned (budgeting, saving, debt, investing, spending, income, emergency_fund, retirement, taxes, insurance, financial_goals)
            4. Urgency level (urgent, high, medium, low)
            5. Confidence level in financial knowledge (high, medium, low)
            6. Suggested follow-up questions (3 relevant, engaging questions)
            
            Consider:
            - Tone and language used
            - Specific financial terms mentioned
            - Implied concerns or goals
            - Level of financial sophistication
            
            Respond in JSON format:
            {{
                "emotion": "primary_emotion_detected",
                "intent": "primary_intent",
                "topics": ["topic1", "topic2", "topic3"],
                "urgency": "urgency_level",
                "confidence": "user_confidence_level",
                "financial_sophistication": "beginner|intermediate|advanced",
                "follow_up_suggestions": [
                    "engaging_followup_question_1",
                    "engaging_followup_question_2", 
                    "engaging_followup_question_3"}}