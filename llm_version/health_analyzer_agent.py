# ==============================================================================
# health_analyzer_agent.py - LLM-Enhanced Financial Health Analysis Agent
# ==============================================================================

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage

from config_manager import llm_manager
from data_models import EnhancedTransaction, FinancialHealthMetrics, FinancialDataSummary, LLMAnalysisResult


class HealthCalculatorEngine:
    """Core financial health calculation engine"""
    
    @staticmethod
    def calculate_basic_metrics(transactions: List[EnhancedTransaction]) -> FinancialHealthMetrics:
        """Calculate basic financial health metrics"""
        metrics = FinancialHealthMetrics()
        
        if not transactions:
            return metrics
        
        # Basic financial calculations
        total_income = sum(tx.amount for tx in transactions if tx.amount > 0)
        total_expenses = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
        current_balance = sum(tx.amount for tx in transactions)
        
        # Time period calculations
        dates = [datetime.fromisoformat(tx.date) for tx in transactions]
        date_range = max(dates) - min(dates)
        months_span = max(1, date_range.days / 30.44)  # Average days per month
        
        # 1. Savings Ratio Score (30% weight)
        if total_income > 0:
            savings_rate = (total_income - total_expenses) / total_income
            # Score: 0% savings = 0, 20%+ savings = 100
            metrics.savings_ratio_score = min(100, max(0, savings_rate * 500))
        
        # 2. Emergency Fund Score (20% weight)
        avg_monthly_expenses = total_expenses / months_span if months_span > 0 else 0
        if avg_monthly_expenses > 0:
            months_covered = current_balance / avg_monthly_expenses
            # Score: 0 months = 0, 6+ months = 100
            metrics.emergency_fund_score = min(100, (months_covered / 6) * 100)
        
        # 3. Spending Stability Score (25% weight)
        monthly_expenses = HealthCalculatorEngine._calculate_monthly_expenses(transactions)
        if len(monthly_expenses) > 1:
            cv = np.std(monthly_expenses) / (np.mean(monthly_expenses) + 1)
            # Lower coefficient of variation = higher stability
            metrics.spending_stability_score = max(0, min(100, 100 - (cv * 50)))
        else:
            metrics.spending_stability_score = 50
        
        # 4. Cashflow Score (25% weight)
        recent_cashflow = HealthCalculatorEngine._calculate_recent_cashflow(transactions)
        metrics.cashflow_score = min(100, max(0, 50 + (recent_cashflow / 100)))
        
        # Overall Score (weighted average)
        metrics.overall_score = (
            metrics.savings_ratio_score * 0.30 +
            metrics.emergency_fund_score * 0.20 +
            metrics.spending_stability_score * 0.25 +
            metrics.cashflow_score * 0.25
        )
        
        # Risk Level Assessment
        if metrics.overall_score >= 80:
            metrics.risk_level = "Low Risk"
        elif metrics.overall_score >= 60:
            metrics.risk_level = "Medium Risk"
        elif metrics.overall_score >= 40:
            metrics.risk_level = "High Risk"
        else:
            metrics.risk_level = "Critical Risk"
        
        # Basic improvement suggestions
        suggestions = []
        if metrics.savings_ratio_score < 50:
            suggestions.append("Increase your savings rate to at least 10% of income")
        if metrics.emergency_fund_score < 50:
            suggestions.append("Build an emergency fund covering 3-6 months of expenses")
        if metrics.spending_stability_score < 50:
            suggestions.append("Create a budget to make spending more predictable")
        if metrics.cashflow_score < 50:
            suggestions.append("Monitor and improve your monthly cash flow")
        
        metrics.improvement_suggestions = suggestions
        
        return metrics
    
    @staticmethod
    def _calculate_monthly_expenses(transactions: List[EnhancedTransaction]) -> List[float]:
        """Calculate monthly expense totals"""
        monthly_data = {}
        
        for tx in transactions:
            if tx.amount < 0:  # Only expenses
                month_key = tx.date[:7]  # YYYY-MM
                monthly_data[month_key] = monthly_data.get(month_key, 0) + abs(tx.amount)
        
        return list(monthly_data.values())
    
    @staticmethod
    def _calculate_recent_cashflow(transactions: List[EnhancedTransaction]) -> float:
        """Calculate recent cashflow trend"""
        if len(transactions) < 5:
            return 0
        
        # Use last 30% of transactions as "recent"
        recent_count = max(5, len(transactions) // 3)
        recent_transactions = transactions[-recent_count:]
        
        return sum(tx.amount for tx in recent_transactions)


class LLMHealthAnalyzer:
    """LLM-enhanced financial health analysis with intelligent insights"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
        self.calculator = HealthCalculatorEngine()
    
    def analyze_financial_health(self, transactions: List[EnhancedTransaction]) -> FinancialHealthMetrics:
        """Comprehensive financial health analysis with LLM enhancement"""
        # Start with basic metrics calculation
        basic_metrics = self.calculator.calculate_basic_metrics(transactions)
        
        if not self.llm or not transactions:
            basic_metrics.analysis_quality = 'basic'
            return basic_metrics
        
        try:
            # Prepare comprehensive data summary
            data_summary = FinancialDataSummary(transactions, basic_metrics)
            
            # Get LLM-enhanced insights
            llm_insights = self._generate_llm_insights(data_summary)
            
            # Generate LLM recommendations
            llm_recommendations = self._generate_llm_recommendations(data_summary, llm_insights)
            
            # Enhance basic metrics with LLM analysis
            enhanced_metrics = basic_metrics
            enhanced_metrics.llm_insights = llm_insights
            enhanced_metrics.llm_recommendations = llm_recommendations
            enhanced_metrics.analysis_quality = 'llm_enhanced'
            
            return enhanced_metrics
            
        except Exception as e:
            st.warning(f"LLM health analysis failed, using basic analysis: {e}")
            basic_metrics.analysis_quality = 'basic'
            return basic_metrics
    
    def _generate_llm_insights(self, data_summary: FinancialDataSummary) -> Dict:
        """Generate comprehensive financial insights using LLM"""
        try:
            prompt = self._create_insights_prompt(data_summary)
            
            messages = [
                SystemMessage(content="You are a senior certified financial planner with expertise in personal finance analysis. Provide comprehensive, actionable insights based on financial data. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse and validate response
            insights = json.loads(response.content)
            return self._validate_insights(insights)
            
        except Exception as e:
            return self._default_insights()
    
    def _create_insights_prompt(self, data_summary: FinancialDataSummary) -> str:
        """Create comprehensive prompt for financial insights"""
        summary_data = data_summary.summary
        
        return f"""
        Analyze this comprehensive financial data and provide expert insights:
        
        FINANCIAL OVERVIEW:
        - Total Income: ${summary_data['financial_overview']['total_income']:,.2f}
        - Total Expenses: ${summary_data['financial_overview']['total_expenses']:,.2f}
        - Net Cash Flow: ${summary_data['financial_overview']['net_flow']:,.2f}
        - Savings Rate: {summary_data['financial_overview']['savings_rate']:.1%}
        
        HEALTH METRICS:
        - Overall Score: {summary_data['health_metrics']['overall_score']:.1f}/100
        - Risk Level: {summary_data['health_metrics']['risk_level']}
        - Cashflow Score: {summary_data['health_metrics']['cashflow_score']:.1f}/100
        - Savings Score: {summary_data['health_metrics']['savings_ratio_score']:.1f}/100
        - Stability Score: {summary_data['health_metrics']['spending_stability_score']:.1f}/100
        - Emergency Fund Score: {summary_data['health_metrics']['emergency_fund_score']:.1f}/100
        
        SPENDING ANALYSIS:
        Category Breakdown: {json.dumps(summary_data['spending_breakdown']['by_category'], indent=2)}
        Spending Behaviors: {json.dumps(summary_data['spending_breakdown']['by_behavior'], indent=2)}
        
        TEMPORAL PATTERNS:
        Monthly Data: {json.dumps(summary_data['temporal_patterns']['monthly_data'], indent=2)}
        Date Range: {summary_data['temporal_patterns']['date_range']['start']} to {summary_data['temporal_patterns']['date_range']['end']}
        
        Provide expert analysis covering:
        1. Overall financial health assessment (2-3 sentences)
        2. Key financial strengths (3-5 specific points)
        3. Critical weaknesses and vulnerabilities (3-5 specific points)
        4. Risk factors that could impact financial stability (3-5 points)
        5. Behavioral patterns and insights (3-5 observations)
        6. Trend analysis and trajectory (2-3 sentences)
        7. Interpretation of key metrics and what they mean (2-3 sentences)
        
        Focus on:
        - Specific, actionable insights based on the actual data
        - Professional financial planning perspective
        - Personalized analysis, not generic advice
        - Clear explanations of what the numbers mean
        
        Respond in this exact JSON format:
        {{
            "overall_assessment": "comprehensive 2-3 sentence assessment",
            "strengths": ["strength1", "strength2", "strength3", "strength4", "strength5"],
            "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4", "weakness5"],
            "risk_factors": ["risk1", "risk2", "risk3", "risk4", "risk5"],
            "behavioral_insights": ["insight1", "insight2", "insight3", "insight4", "insight5"],
            "trend_analysis": "2-3 sentence trend analysis",
            "key_metrics_interpretation": "2-3 sentence explanation of what the key metrics mean"
        }}
        """
    
    def _generate_llm_recommendations(self, data_summary: FinancialDataSummary, insights: Dict) -> Dict:
        """Generate personalized recommendations using LLM"""
        try:
            prompt = self._create_recommendations_prompt(data_summary, insights)
            
            messages = [
                SystemMessage(content="You are a certified financial planner providing personalized financial advice. Focus on practical, achievable recommendations tailored to the individual's specific situation. Always respond with valid JSON."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse and validate response
            recommendations = json.loads(response.content)
            return self._validate_recommendations(recommendations)
            
        except Exception as e:
            return self._default_recommendations()
    
    def _create_recommendations_prompt(self, data_summary: FinancialDataSummary, insights: Dict) -> str:
        """Create comprehensive prompt for personalized recommendations"""
        summary_data = data_summary.summary
        
        return f"""
        Based on this financial analysis and insights, provide specific, personalized recommendations:
        
        CURRENT FINANCIAL STATE:
        {json.dumps(summary_data['financial_overview'], indent=2)}
        
        HEALTH SCORES:
        {json.dumps(summary_data['health_metrics'], indent=2)}
        
        EXPERT INSIGHTS:
        {json.dumps(insights, indent=2)}
        
        Create a comprehensive action plan with specific recommendations organized by timeframe and priority:
        
        1. IMMEDIATE ACTIONS (next 30 days) - Critical items that need attention now
        2. SHORT-TERM GOALS (3-6 months) - Important improvements to work towards
        3. LONG-TERM STRATEGY (1+ years) - Strategic financial planning
        4. EMERGENCY IMPROVEMENTS (if needed) - Critical actions for financial stability
        5. OPTIMIZATION OPPORTUNITIES - Ways to improve an already good situation
        
        For each recommendation, provide:
        - Specific, actionable step
        - Expected impact/benefit
        - Priority level (critical/high/medium/low)
        - Estimated effort required (low/medium/high)
        
        Guidelines:
        - Make recommendations specific to their actual financial data
        - Consider their current financial health level
        - Prioritize based on biggest impact and urgency
        - Include specific dollar amounts or percentages where relevant
        - Focus on achievable, realistic steps
        
        Respond in this exact JSON format:
        {{
            "immediate_actions": [
                {{
                    "action": "specific action to take",
                    "impact": "expected result/benefit",
                    "priority": "critical|high|medium|low",
                    "effort": "low|medium|high"
                }}
            ],
            "short_term_goals": [
                {{
                    "action": "goal to work towards",
                    "impact": "expected benefit",
                    "priority": "high|medium|low",
                    "effort": "low|medium|high"
                }}
            ],
            "long_term_strategy": [
                {{
                    "action": "strategic initiative",
                    "impact": "long-term benefit",
                    "priority": "high|medium|low"
                }}
            ],
            "emergency_improvements": [
                {{
                    "action": "critical action for stability",
                    "impact": "immediate benefit for financial health"
                }}
            ],
            "optimization_opportunities": [
                {{
                    "action": "optimization opportunity",
                    "impact": "potential benefit",
                    "effort": "low|medium|high"
                }}
            ]
        }}
        """
    
    def _validate_insights(self, insights: Dict) -> Dict:
        """Validate and clean LLM insights"""
        required_keys = ['overall_assessment', 'strengths', 'weaknesses', 'risk_factors', 'behavioral_insights', 'trend_analysis', 'key_metrics_interpretation']
        
        validated = {}
        for key in required_keys:
            if key in insights:
                if key in ['overall_assessment', 'trend_analysis', 'key_metrics_interpretation']:
                    validated[key] = str(insights[key])[:500]  # Limit length
                else:
                    validated[key] = insights[key][:5] if isinstance(insights[key], list) else []
            else:
                validated[key] = "" if key in ['overall_assessment', 'trend_analysis', 'key_metrics_interpretation'] else []
        
        return validated
    
    def _validate_recommendations(self, recommendations: Dict) -> Dict:
        """Validate and clean LLM recommendations"""
        required_keys = ['immediate_actions', 'short_term_goals', 'long_term_strategy', 'emergency_improvements', 'optimization_opportunities']
        
        validated = {}
        for key in required_keys:
            if key in recommendations and isinstance(recommendations[key], list):
                validated[key] = recommendations[key][:5]  # Limit to 5 items per category
            else:
                validated[key] = []
        
        return validated
    
    def _default_insights(self) -> Dict:
        """Default insights when LLM analysis fails"""
        return {
            "overall_assessment": "Basic financial analysis completed. Enable LLM for detailed insights.",
            "strengths": [],
            "weaknesses": [],
            "risk_factors": [],
            "behavioral_insights": [],
            "trend_analysis": "Basic trend analysis available with LLM enhancement.",
            "key_metrics_interpretation": "Standard financial health metrics calculated."
        }
    
    def _default_recommendations(self) -> Dict:
        """Default recommendations when LLM analysis fails"""
        return {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_strategy": [],
            "emergency_improvements": [],
            "optimization_opportunities": []
        }


class HealthTrendAnalyzer:
    """Analyze financial health trends over time"""
    
    def __init__(self):
        self.llm = llm_manager.get_client()
    
    def analyze_trends(self, transactions: List[EnhancedTransaction], window_months: int = 3) -> Dict:
        """Analyze financial health trends over specified time window"""
        if len(transactions) < 10:
            return {'status': 'insufficient_data'}
        
        # Group transactions by time periods
        time_periods = self._group_transactions_by_period(transactions, window_months)
        
        # Calculate metrics for each period
        period_metrics = []
        for period, period_transactions in time_periods.items():
            if len(period_transactions) > 0:
                metrics = HealthCalculatorEngine.calculate_basic_metrics(period_transactions)
                period_metrics.append({
                    'period': period,
                    'metrics': metrics.to_dict(),
                    'transaction_count': len(period_transactions)
                })
        
        if len(period_metrics) < 2:
            return {'status': 'insufficient_periods'}
        
        # Analyze trends
        trend_analysis = self._calculate_trend_metrics(period_metrics)
        
        # Get LLM insights on trends if available
        if self.llm:
            trend_insights = self._generate_trend_insights(period_metrics, trend_analysis)
            trend_analysis['llm_insights'] = trend_insights
        
        return {
            'status': 'success',
            'period_metrics': period_metrics,
            'trend_analysis': trend_analysis,
            'analysis_window_months': window_months
        }
    
    def _group_transactions_by_period(self, transactions: List[EnhancedTransaction], window_months: int) -> Dict:
        """Group transactions into time periods"""
        periods = {}
        
        for tx in transactions:
            # Create period key based on year and month groups
            date = datetime.fromisoformat(tx.date)
            period_key = f"{date.year}-{(date.month - 1) // window_months + 1:02d}"
            
            if period_key not in periods:
                periods[period_key] = []
            periods[period_key].append(tx)
        
        return periods
    
    def _calculate_trend_metrics(self, period_metrics: List[Dict]) -> Dict:
        """Calculate trend metrics across time periods"""
        # Extract key metrics across periods
        overall_scores = [p['metrics']['overall_score'] for p in period_metrics]
        savings_scores = [p['metrics']['savings_ratio_score'] for p in period_metrics]
        cashflow_scores = [p['metrics']['cashflow_score'] for p in period_metrics]
        stability_scores = [p['metrics']['spending_stability_score'] for p in period_metrics]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            return (values[-1] - values[0]) / len(values)
        
        return {
            'overall_trend': calculate_trend(overall_scores),
            'savings_trend': calculate_trend(savings_scores),
            'cashflow_trend': calculate_trend(cashflow_scores),
            'stability_trend': calculate_trend(stability_scores),
            'trajectory': self._determine_trajectory(overall_scores),
            'volatility': np.std(overall_scores) if len(overall_scores) > 1 else 0,
            'best_period': max(period_metrics, key=lambda x: x['metrics']['overall_score'])['period'],
            'worst_period': min(period_metrics, key=lambda x: x['metrics']['overall_score'])['period']
        }
    
    def _determine_trajectory(self, scores: List[float]) -> str:
        """Determine overall financial trajectory"""
        if len(scores) < 2:
            return 'stable'
        
        trend = (scores[-1] - scores[0]) / len(scores)
        
        if trend > 2:
            return 'strongly_improving'
        elif trend > 0.5:
            return 'improving'
        elif trend > -0.5:
            return 'stable'
        elif trend > -2:
            return 'declining'
        else:
            return 'strongly_declining'
    
    def _generate_trend_insights(self, period_metrics: List[Dict], trend_analysis: Dict) -> Dict:
        """Generate LLM insights on financial trends"""
        try:
            prompt = f"""
            Analyze these financial health trends and provide insights:
            
            PERIOD METRICS:
            {json.dumps(period_metrics, indent=2)}
            
            TREND ANALYSIS:
            {json.dumps(trend_analysis, indent=2)}
            
            Provide insights on:
            1. Overall trajectory assessment
            2. Key trend drivers (what's causing changes)
            3. Concerning patterns or warning signs
            4. Positive developments to reinforce
            5. Recommendations based on trends
            
            Respond in JSON format:
            {{
                "trajectory_assessment": "assessment of overall direction",
                "trend_drivers": ["driver1", "driver2", "driver3"],
                "warning_signs": ["warning1", "warning2"],
                "positive_developments": ["positive1", "positive2"],
                "trend_recommendations": ["recommendation1", "recommendation2"]
            }}
            """
            
            messages = [
                SystemMessage(content="You are a financial analyst expert in trend analysis. Provide actionable insights based on financial health trends over time."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return json.loads(response.content)
            
        except Exception:
            return {
                "trajectory_assessment": "Unable to generate LLM trend insights",
                "trend_drivers": [],
                "warning_signs": [],
                "positive_developments": [],
                "trend_recommendations": []
            }