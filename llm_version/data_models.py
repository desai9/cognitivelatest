# ==============================================================================
# data_models.py - Core Data Models and Schemas
# ==============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


@dataclass
class EnhancedTransaction:
    """Enhanced transaction with LLM analysis"""
    date: str
    description: str
    amount: float
    category: str = "other"
    subcategory: str = "uncategorized"
    spending_type: str = "regular"
    confidence: float = 0.0
    llm_reasoning: str = ""
    
    def to_dict(self):
        return {
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'subcategory': self.subcategory,
            'spending_type': self.spending_type,
            'confidence': self.confidence,
            'llm_reasoning': self.llm_reasoning
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create transaction from dictionary"""
        return cls(
            date=data.get('date', ''),
            description=data.get('description', ''),
            amount=float(data.get('amount', 0.0)),
            category=data.get('category', 'other'),
            subcategory=data.get('subcategory', 'uncategorized'),
            spending_type=data.get('spending_type', 'regular'),
            confidence=float(data.get('confidence', 0.0)),
            llm_reasoning=data.get('llm_reasoning', '')
        )


@dataclass
class FinancialHealthMetrics:
    """Comprehensive financial health metrics"""
    cashflow_score: float = 0.0
    savings_ratio_score: float = 0.0
    spending_stability_score: float = 0.0
    emergency_fund_score: float = 0.0
    overall_score: float = 0.0
    risk_level: str = "Unknown"
    trend: str = "Stable"
    improvement_suggestions: List[str] = field(default_factory=list)
    llm_insights: Dict = field(default_factory=dict)
    llm_recommendations: Dict = field(default_factory=dict)
    analysis_quality: str = "basic"
    
    def to_dict(self):
        return {
            'cashflow_score': self.cashflow_score,
            'savings_ratio_score': self.savings_ratio_score,
            'spending_stability_score': self.spending_stability_score,
            'emergency_fund_score': self.emergency_fund_score,
            'overall_score': self.overall_score,
            'risk_level': self.risk_level,
            'trend': self.trend,
            'improvement_suggestions': self.improvement_suggestions,
            'llm_insights': self.llm_insights,
            'llm_recommendations': self.llm_recommendations,
            'analysis_quality': self.analysis_quality
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create metrics from dictionary"""
        return cls(
            cashflow_score=data.get('cashflow_score', 0.0),
            savings_ratio_score=data.get('savings_ratio_score', 0.0),
            spending_stability_score=data.get('spending_stability_score', 0.0),
            emergency_fund_score=data.get('emergency_fund_score', 0.0),
            overall_score=data.get('overall_score', 0.0),
            risk_level=data.get('risk_level', 'Unknown'),
            trend=data.get('trend', 'Stable'),
            improvement_suggestions=data.get('improvement_suggestions', []),
            llm_insights=data.get('llm_insights', {}),
            llm_recommendations=data.get('llm_recommendations', {}),
            analysis_quality=data.get('analysis_quality', 'basic')
        )


@dataclass
class ConversationEntry:
    """Conversation history entry"""
    user_query: str
    ai_response: str
    emotion: str = "neutral"
    intent: str = "general_inquiry"
    topics: List[str] = field(default_factory=list)
    confidence: str = "medium"
    follow_up_suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'user_query': self.user_query,
            'ai_response': self.ai_response,
            'emotion': self.emotion,
            'intent': self.intent,
            'topics': self.topics,
            'confidence': self.confidence,
            'follow_up_suggestions': self.follow_up_suggestions,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create conversation entry from dictionary"""
        return cls(
            user_query=data.get('user_query', ''),
            ai_response=data.get('ai_response', ''),
            emotion=data.get('emotion', 'neutral'),
            intent=data.get('intent', 'general_inquiry'),
            topics=data.get('topics', []),
            confidence=data.get('confidence', 'medium'),
            follow_up_suggestions=data.get('follow_up_suggestions', []),
            timestamp=data.get('timestamp', datetime.now().isoformat())
        )


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis operations"""
    success: bool
    data: Dict = field(default_factory=dict)
    confidence: str = "medium"
    reasoning: str = ""
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'success': self.success,
            'data': self.data,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }


@dataclass 
class UserProfile:
    """User financial profile"""
    user_id: str = ""
    financial_goals: List[str] = field(default_factory=list)
    risk_tolerance: str = "medium"
    income_bracket: str = "unknown"
    age_group: str = "unknown"
    conversation_preferences: Dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'financial_goals': self.financial_goals,
            'risk_tolerance': self.risk_tolerance,
            'income_bracket': self.income_bracket,
            'age_group': self.age_group,
            'conversation_preferences': self.conversation_preferences,
            'created_at': self.created_at
        }


class FinancialDataSummary:
    """Comprehensive financial data summary for LLM analysis"""
    
    def __init__(self, transactions: List[EnhancedTransaction], metrics: FinancialHealthMetrics):
        self.transactions = transactions
        self.metrics = metrics
        self._calculate_summary()
    
    def _calculate_summary(self):
        """Calculate comprehensive financial summary"""
        if not self.transactions:
            self.summary = self._empty_summary()
            return
        
        # Basic calculations
        self.total_income = sum(tx.amount for tx in self.transactions if tx.amount > 0)
        self.total_expenses = sum(abs(tx.amount) for tx in self.transactions if tx.amount < 0)
        self.net_flow = self.total_income - self.total_expenses
        self.savings_rate = (self.net_flow / self.total_income) if self.total_income > 0 else 0
        
        # Category breakdown
        self.category_spending = {}
        self.subcategory_spending = {}
        
        for tx in self.transactions:
            if tx.amount < 0:
                # Category spending
                category = tx.category
                self.category_spending[category] = self.category_spending.get(category, 0) + abs(tx.amount)
                
                # Subcategory spending
                subcat_key = f"{tx.category}_{tx.subcategory}"
                self.subcategory_spending[subcat_key] = self.subcategory_spending.get(subcat_key, 0) + abs(tx.amount)
        
        # Monthly patterns
        self.monthly_data = {}
        for tx in self.transactions:
            month = tx.date[:7]  # YYYY-MM format
            if month not in self.monthly_data:
                self.monthly_data[month] = {'income': 0, 'expenses': 0, 'transactions': 0}
            
            if tx.amount > 0:
                self.monthly_data[month]['income'] += tx.amount
            else:
                self.monthly_data[month]['expenses'] += abs(tx.amount)
            self.monthly_data[month]['transactions'] += 1
        
        # Spending behaviors
        self.spending_behaviors = {}
        for tx in self.transactions:
            if tx.amount < 0:  # Only expenses
                behavior = tx.spending_type
                self.spending_behaviors[behavior] = self.spending_behaviors.get(behavior, 0) + 1
        
        # Date range
        dates = [tx.date for tx in self.transactions]
        self.date_range = {
            'start': min(dates),
            'end': max(dates),
            'span_days': (datetime.fromisoformat(max(dates)) - datetime.fromisoformat(min(dates))).days
        }
        
        # Create summary dict
        self.summary = {
            'financial_overview': {
                'total_income': self.total_income,
                'total_expenses': self.total_expenses,
                'net_flow': self.net_flow,
                'savings_rate': self.savings_rate
            },
            'spending_breakdown': {
                'by_category': self.category_spending,
                'by_subcategory': self.subcategory_spending,
                'by_behavior': self.spending_behaviors
            },
            'temporal_patterns': {
                'monthly_data': self.monthly_data,
                'date_range': self.date_range
            },
            'health_metrics': self.metrics.to_dict(),
            'transaction_count': len(self.transactions)
        }
    
    def _empty_summary(self):
        """Return empty summary for no transactions"""
        return {
            'financial_overview': {
                'total_income': 0,
                'total_expenses': 0,
                'net_flow': 0,
                'savings_rate': 0
            },
            'spending_breakdown': {
                'by_category': {},
                'by_subcategory': {},
                'by_behavior': {}
            },
            'temporal_patterns': {
                'monthly_data': {},
                'date_range': {'start': '', 'end': '', 'span_days': 0}
            },
            'health_metrics': {},
            'transaction_count': 0
        }
    
    def to_llm_context(self) -> str:
        """Convert summary to LLM-friendly context string"""
        return json.dumps(self.summary, indent=2, default=str)
    
    def get_top_categories(self, limit: int = 5) -> List[tuple]:
        """Get top spending categories"""
        return sorted(self.category_spending.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_average_monthly_spending(self) -> float:
        """Calculate average monthly spending"""
        if not self.monthly_data:
            return 0
        
        monthly_expenses = [data['expenses'] for data in self.monthly_data.values()]
        return sum(monthly_expenses) / len(monthly_expenses) if monthly_expenses else 0
    
    def get_spending_trend(self) -> str:
        """Analyze spending trend"""
        if len(self.monthly_data) < 2:
            return "insufficient_data"
        
        months = sorted(self.monthly_data.keys())
        expenses = [self.monthly_data[month]['expenses'] for month in months]
        
        # Simple trend analysis
        if len(expenses) >= 3:
            recent_avg = sum(expenses[-3:]) / 3
            earlier_avg = sum(expenses[:-3]) / len(expenses[:-3]) if len(expenses[:-3]) > 0 else recent_avg
            
            if recent_avg > earlier_avg * 1.1:
                return "increasing"
            elif recent_avg < earlier_avg * 0.9:
                return "decreasing"
        
        return "stable"