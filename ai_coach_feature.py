import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import re
from dateutil import parser
import csv
import io
import sqlite3
from pathlib import Path
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Enhanced logging with rotation
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Setup enhanced logging with rotation"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        "empowerfin_guardian_v2.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(message)s")
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class FinancialHealthMetrics:
    """Financial health scoring components with validation"""
    cashflow_score: float = 0.0
    savings_ratio_score: float = 0.0
    debt_to_income_score: float = 0.0
    spending_stability_score: float = 0.0
    emergency_fund_score: float = 0.0
    overall_score: float = 0.0
    risk_level: str = "Unknown"
    trend: str = "Stable"
    improvement_suggestions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate metric values to ensure they're between 0 and 100"""
        score_fields = [
            'cashflow_score', 'savings_ratio_score', 'debt_to_income_score',
            'spending_stability_score', 'emergency_fund_score', 'overall_score'
        ]
        
        for field_name in score_fields:
            value = getattr(self, field_name)
            if isinstance(value, (int, float)):
                # Ensure values are between 0 and 100
                validated_value = max(0.0, min(100.0, float(value)))
                setattr(self, field_name, validated_value)

@dataclass
class ConversationContext:
    """Enhanced context for conversational AI"""
    user_emotion: str = "neutral"
    conversation_history: List[Dict] = field(default_factory=list)
    current_focus: str = "general"
    user_preferences: Dict = field(default_factory=dict)
    financial_stress_level: float = 0.0
    session_id: str = ""
    interaction_count: int = 0
    topics_discussed: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize session_id if not provided"""
        if not self.session_id:
            self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

class EnhancedAgentState(Dict):
    """Enhanced state with better error handling and validation"""
    def __init__(self):
        super().__init__()
        try:
            self.update({
                'user_consent': False,
                'bank_statement': {},
                'transactions': [],
                'financial_goal': "",
                'health_metrics': FinancialHealthMetrics(),
                'conversation_context': ConversationContext(),
                'ai_insights': [],
                'recommendations': [],
                'alerts': [],
                'logs': [],
                'uploaded_file': None,
                'chat_history': [],
                'current_query': "",
                'personalized_response': "",
                'processing_status': "idle",
                'error_messages': [],
                'data_quality_score': 0.0,
                'last_updated': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error initializing EnhancedAgentState: {e}")
            # Fallback to basic initialization
            super().__init__()
            self.update({
                'user_consent': False,
                'bank_statement': {},
                'transactions': [],
                'financial_goal': "",
                'health_metrics': {'overall_score': 0.0, 'risk_level': 'Unknown'},
                'conversation_context': {'user_emotion': 'neutral'},
                'ai_insights': [],
                'recommendations': [],
                'logs': [],
                'processing_status': "idle",
                'error_messages': [f"Initialization warning: {str(e)}"]
            })
    
    def add_log(self, message: str, level: str = "info"):
        """Add log with timestamp and level"""
        if "logs" not in self:
            self["logs"] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        self["logs"].append(log_entry)
        
        # Log to file as well
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

class SafeLLMClient:
    """Thread-safe LLM client with retry logic and fallbacks"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client with error handling"""
        try:
            if not GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found, using fallback responses")
                self._client = None
                return
            
            self._client = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0.7,
                max_tokens=1000,
                timeout=self.timeout
            )
            logger.info("LLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self._client = None
    
    async def generate_response(self, messages: List, fallback_response: str = None) -> str:
        """Generate response with retry logic and fallbacks"""
        if not self._client:
            return fallback_response or "I'm currently experiencing technical difficulties. Please try again later."
        
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.wait_for(
                    self._client.ainvoke(messages),
                    timeout=self.timeout
                )
                return response.content.strip()
            
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return fallback_response or "I'm taking longer than usual to respond. Please try rephrasing your question."
            
            except Exception as e:
                logger.error(f"LLM error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return fallback_response or "I encountered an error processing your request. Please try again."
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return fallback_response or "I'm unable to respond right now. Please try again later."

# Initialize safe LLM client
llm_client = SafeLLMClient()

class FinancialHealthCalculator:
    """Enhanced financial health scoring with better validation"""
    
    @staticmethod
    def detect_financial_stress(text: str, health_score: float) -> float:
        """Detect financial stress level (0-1)"""
        try:
            if not text or not isinstance(text, str):
                return 0.0
            
            stress_keywords = [
                "worried", "anxious", "struggling", "difficult", "can't afford",
                "broke", "debt", "overwhelmed", "scared", "panic", "crisis",
                "emergency", "urgent", "desperate", "help", "trouble"
            ]
            
            stress_level = 0.0
            text_lower = text.lower()
            
            for keyword in stress_keywords:
                if keyword in text_lower:
                    stress_level += 0.15  # Each keyword adds stress
            
            # Factor in health score
            if health_score < 40:
                stress_level += 0.3
            elif health_score < 60:
                stress_level += 0.1
            
            return min(1.0, stress_level)
            
        except Exception as e:
            logger.error(f"Error detecting financial stress: {e}")
            return 0.0
    
    @staticmethod
    def validate_transactions(transactions: List[Dict]) -> List[Dict]:
        """Validate and clean transaction data"""
        valid_transactions = []
        
        for tx in transactions:
            try:
                # Validate required fields
                if not all(key in tx for key in ['date', 'description', 'amount']):
                    continue
                
                # Validate and convert amount
                amount = float(tx['amount'])
                if np.isnan(amount) or np.isinf(amount):
                    continue
                
                # Validate date
                try:
                    pd.to_datetime(tx['date'])
                except:
                    continue
                
                # Clean description
                description = str(tx['description']).strip()
                if not description:
                    continue
                
                valid_tx = {
                    'date': tx['date'],
                    'description': description,
                    'amount': amount,
                    'category': tx.get('category', 'other'),
                    'spending_type': tx.get('spending_type', 'regular_spending')
                }
                
                valid_transactions.append(valid_tx)
                
            except Exception as e:
                logger.warning(f"Skipping invalid transaction: {e}")
                continue
        
        return valid_transactions
    
    @staticmethod
    def calculate_cashflow_score(transactions: List[Dict]) -> float:
        """Calculate cashflow stability score with better error handling"""
        try:
            if not transactions:
                return 0.0
            
            # Group by month
            monthly_data = {}
            for tx in transactions:
                try:
                    month_key = pd.to_datetime(tx['date']).strftime('%Y-%m')
                    if month_key not in monthly_data:
                        monthly_data[month_key] = 0
                    monthly_data[month_key] += tx['amount']
                except:
                    continue
            
            if len(monthly_data) < 2:
                return 50.0  # Neutral score for insufficient data
            
            monthly_balances = list(monthly_data.values())
            mean_balance = np.mean(monthly_balances)
            
            if abs(mean_balance) < 0.01:  # Avoid division by zero
                return 50.0
            
            volatility = np.std(monthly_balances) / abs(mean_balance)
            stability_score = max(0, min(100, 100 - (volatility * 50)))
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Error calculating cashflow score: {e}")
            return 0.0
    
    @staticmethod
    def calculate_savings_ratio_score(transactions: List[Dict]) -> float:
        """Calculate savings ratio with improved logic"""
        try:
            if not transactions:
                return 0.0
            
            total_income = sum(tx['amount'] for tx in transactions if tx['amount'] > 0)
            total_expenses = sum(abs(tx['amount']) for tx in transactions if tx['amount'] < 0)
            
            if total_income <= 0:
                return 0.0
            
            savings_rate = (total_income - total_expenses) / total_income
            
            # More nuanced scoring
            if savings_rate >= 0.20:  # 20%+ savings rate
                return 100.0
            elif savings_rate >= 0.10:  # 10-20% savings rate
                return 60 + (savings_rate - 0.10) * 400  # Scale from 60-100
            elif savings_rate >= 0.05:  # 5-10% savings rate
                return 30 + (savings_rate - 0.05) * 600  # Scale from 30-60
            elif savings_rate > 0:  # 0-5% savings rate
                return savings_rate * 600  # Scale from 0-30
            else:  # Negative savings (spending more than earning)
                return max(0, 30 + savings_rate * 100)  # Penalize overspending
                
        except Exception as e:
            logger.error(f"Error calculating savings ratio: {e}")
            return 0.0
    
    @staticmethod
    def calculate_spending_categories_score(transactions: List[Dict]) -> Dict[str, float]:
        """Calculate scores for different spending categories"""
        try:
            category_totals = {}
            total_expenses = 0
            
            for tx in transactions:
                if tx['amount'] < 0:  # Only expenses
                    category = tx.get('category', 'other')
                    amount = abs(tx['amount'])
                    category_totals[category] = category_totals.get(category, 0) + amount
                    total_expenses += amount
            
            if total_expenses == 0:
                return {}
            
            # Calculate percentage and score each category
            category_scores = {}
            recommended_percentages = {
                'fixed_expenses': 0.50,  # Housing, utilities, insurance
                'food_dining': 0.15,     # Food and dining
                'transportation': 0.15,   # Transportation
                'shopping': 0.10,        # Discretionary shopping
                'entertainment': 0.05,   # Entertainment
                'other': 0.05           # Other expenses
            }
            
            for category, amount in category_totals.items():
                percentage = amount / total_expenses
                recommended = recommended_percentages.get(category, 0.05)
                
                # Score based on how close to recommended percentage
                if percentage <= recommended:
                    score = 100  # Good if at or below recommended
                else:
                    # Penalize overspending in category
                    overspend_ratio = percentage / recommended
                    score = max(0, 100 - (overspend_ratio - 1) * 50)
                
                category_scores[category] = {
                    'amount': amount,
                    'percentage': percentage * 100,
                    'recommended': recommended * 100,
                    'score': score
                }
            
            return category_scores
            
        except Exception as e:
            logger.error(f"Error calculating category scores: {e}")
            return {}

class EnhancedBankStatementConverter:
    """Improved bank statement converter with better format detection"""
    
    @staticmethod
    def detect_date_format(date_series: pd.Series) -> str:
        """Detect date format with multiple attempts"""
        sample_dates = date_series.dropna().astype(str).head(10)
        
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%d.%m.%Y', '%m.%d.%Y', '%Y.%m.%d',
            '%d %b %Y', '%b %d, %Y', '%Y-%m-%d %H:%M:%S'
        ]
        
        for fmt in date_formats:
            success_count = 0
            for date_str in sample_dates:
                try:
                    datetime.strptime(date_str, fmt)
                    success_count += 1
                except:
                    continue
            
            if success_count >= len(sample_dates) * 0.8:  # 80% success rate
                return fmt
        
        return 'auto'  # Fallback to pandas auto-detection
    
    @staticmethod
    def clean_amount_column(amount_series: pd.Series) -> pd.Series:
        """Clean and convert amount column with better handling"""
        cleaned = amount_series.copy()
        
        # Convert to string first
        cleaned = cleaned.astype(str)
        
        # Remove common currency symbols and whitespace
        cleaned = cleaned.str.replace(r'[$£€¥₹,\s]', '', regex=True)
        
        # Handle parentheses for negative numbers
        negative_mask = cleaned.str.contains(r'\(.*\)', regex=True, na=False)
        cleaned = cleaned.str.replace(r'[()]', '', regex=True)
        
        # Handle CR/DR indicators
        cr_mask = cleaned.str.contains(r'\bCR\b', regex=True, na=False)
        dr_mask = cleaned.str.contains(r'\bDR\b', regex=True, na=False)
        cleaned = cleaned.str.replace(r'\b(CR|DR)\b', '', regex=True)
        
        # Convert to numeric
        cleaned = pd.to_numeric(cleaned, errors='coerce')
        
        # Apply negative signs
        cleaned.loc[negative_mask] = -abs(cleaned.loc[negative_mask])
        cleaned.loc[dr_mask] = -abs(cleaned.loc[dr_mask])
        # CR entries remain positive
        
        return cleaned
    
    @staticmethod
    def validate_and_convert_file(uploaded_file) -> pd.DataFrame:
        """Enhanced file validation and conversion"""
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    
                    # Handle different file types
                    if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                        df = pd.read_excel(uploaded_file)
                    else:
                        # Try different separators for CSV
                        for sep in [',', ';', '\t', '|']:
                            try:
                                uploaded_file.seek(0)
                                df = pd.read_csv(uploaded_file, encoding=encoding, sep=sep)
                                if len(df.columns) > 1:  # Valid if more than 1 column
                                    break
                            except:
                                continue
                    
                    if df is not None and len(df.columns) > 1:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to read with encoding {encoding}: {e}")
                    continue
            
            if df is None or len(df.columns) <= 1:
                raise ValueError("Could not read file with any supported format or encoding")
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Auto-detect columns
            date_col = None
            desc_col = None
            amount_col = None
            
            # Find date column
            for col in df.columns:
                if any(keyword in col for keyword in ['date', 'time', 'posted', 'transaction_date']):
                    try:
                        # Test if column contains dates
                        pd.to_datetime(df[col].dropna().head(5))
                        date_col = col
                        break
                    except:
                        continue
            
            # Find description column
            for col in df.columns:
                if any(keyword in col for keyword in ['description', 'transaction', 'details', 'narration', 'memo', 'particulars']):
                    desc_col = col
                    break
            
            # Find amount column - look for numeric columns
            for col in df.columns:
                if col != date_col and col != desc_col:
                    try:
                        # Try to convert to numeric
                        test_series = EnhancedBankStatementConverter.clean_amount_column(df[col])
                        if not test_series.isna().all():
                            amount_col = col
                            break
                    except:
                        continue
            
            # Validate we found all required columns
            missing = []
            if not date_col:
                missing.append('date')
            if not desc_col:
                missing.append('description')
            if not amount_col:
                missing.append('amount')
            
            if missing:
                raise ValueError(f"Could not auto-detect columns: {', '.join(missing)}. "
                               f"Available columns: {', '.join(df.columns)}")
            
            # Create standardized DataFrame
            standard_df = pd.DataFrame()
            
            # Convert date
            try:
                date_format = EnhancedBankStatementConverter.detect_date_format(df[date_col])
                if date_format == 'auto':
                    standard_df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
                else:
                    standard_df['date'] = pd.to_datetime(df[date_col], format=date_format).dt.strftime('%Y-%m-%d')
            except Exception as e:
                raise ValueError(f"Could not convert date column '{date_col}': {e}")
            
            # Convert description
            standard_df['description'] = df[desc_col].astype(str).str.strip()
            
            # Convert amount
            try:
                standard_df['amount'] = EnhancedBankStatementConverter.clean_amount_column(df[amount_col])
            except Exception as e:
                raise ValueError(f"Could not convert amount column '{amount_col}': {e}")
            
            # Remove rows with missing critical data
            initial_rows = len(standard_df)
            standard_df = standard_df.dropna(subset=['date', 'amount'])
            standard_df = standard_df[standard_df['description'].str.len() > 0]
            
            if len(standard_df) == 0:
                raise ValueError("No valid transactions found after cleaning")
            
            final_rows = len(standard_df)
            if final_rows < initial_rows * 0.5:  # Lost more than 50% of data
                logger.warning(f"Data quality concern: {initial_rows - final_rows} rows removed during cleaning")
            
            logger.info(f"Successfully converted file: {final_rows} valid transactions")
            return standard_df
            
        except Exception as e:
            logger.error(f"File conversion failed: {e}")
            raise

class ConversationalAI:
    """Enhanced conversational AI with better context management"""
    
    @staticmethod
    def detect_emotion(text: str) -> str:
        """Enhanced emotion detection"""
        try:
            if not text or not isinstance(text, str):
                return "neutral"
            
            blob = TextBlob(text.lower())
            polarity = blob.sentiment.polarity
            
            # Check for specific emotional keywords
            stress_words = ['worried', 'anxious', 'stressed', 'scared', 'panic', 'overwhelmed']
            happy_words = ['great', 'excellent', 'happy', 'excited', 'wonderful', 'amazing']
            sad_words = ['sad', 'depressed', 'hopeless', 'terrible', 'awful', 'worst']
            
            text_lower = text.lower()
            
            if any(word in text_lower for word in stress_words):
                return "stressed"
            elif any(word in text_lower for word in happy_words):
                return "happy"
            elif any(word in text_lower for word in sad_words):
                return "sad"
            elif polarity > 0.3:
                return "positive"
            elif polarity < -0.3:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return "neutral"
    
    @staticmethod
    def extract_financial_topics(text: str) -> List[str]:
        """Extract financial topics from user query"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'budgeting': ['budget', 'budgeting', 'spending plan', 'allocate'],
            'saving': ['save', 'savings', 'emergency fund', 'nest egg'],
            'debt': ['debt', 'loan', 'credit', 'payment', 'owe'],
            'investment': ['invest', 'investment', 'portfolio', 'stocks', 'bonds'],
            'spending': ['spend', 'spending', 'expenses', 'cost'],
            'income': ['income', 'salary', 'earn', 'revenue'],
            'retirement': ['retirement', 'pension', '401k', 'ira'],
            'insurance': ['insurance', 'coverage', 'premium']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    @staticmethod
    async def generate_empathetic_response(query: str, context: ConversationContext, 
                                         health_metrics: FinancialHealthMetrics) -> str:
        """Generate contextually aware response with fallbacks"""
        
        if not query:
            return "I'm here to help with your financial questions. What would you like to discuss?"
        
        # Extract topics for better context
        topics = ConversationalAI.extract_financial_topics(query)
        context.topics_discussed.extend(topics)
        
        # Create contextual system prompt
        system_prompt = f"""
        You are EmpowerFin Guardian, a compassionate AI financial coach.
        
        User Context:
        - Emotion: {context.user_emotion}
        - Financial Health Score: {health_metrics.overall_score:.1f}/100
        - Risk Level: {health_metrics.risk_level}
        - Topics of Interest: {', '.join(topics)}
        - Interaction Count: {context.interaction_count}
        
        Guidelines:
        1. Be empathetic and supportive, especially if the user seems stressed
        2. Provide specific, actionable advice
        3. Keep responses under 200 words
        4. Focus on positive solutions and encouragement
        5. Reference their financial health score when relevant
        
        Adapt your tone based on their emotional state:
        - Stressed/Negative: Be extra supportive and reassuring
        - Positive/Happy: Be encouraging and build momentum
        - Neutral: Be informative but warm
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {query}")
        ]
        
        # Generate response with fallback
        fallback_responses = {
            'budgeting': "Creating a budget is a great first step! Try the 50/30/20 rule: 50% for needs, 30% for wants, and 20% for savings and debt payment.",
            'saving': "Building savings takes time, but every small amount counts. Start with a goal of saving just $25 per week - that's over $1,300 per year!",
            'debt': "Dealing with debt can feel overwhelming, but you have options. Consider the debt snowball method: pay minimum on all debts, then put extra money toward the smallest debt first.",
            'spending': "Understanding your spending patterns is key to financial health. Look for areas where you can cut back without impacting your quality of life significantly.",
            'general': "I'm here to help you build a stronger financial future. What specific area of your finances would you like to focus on today?"
        }
        
        primary_topic = topics[0] if topics else 'general'
        fallback = fallback_responses.get(primary_topic, fallback_responses['general'])
        
        try:
            response = await llm_client.generate_response(messages, fallback)
            return response
        except Exception as e:
            logger.error(f"Error generating empathetic response: {e}")
            return fallback

# Enhanced Agents

async def enhanced_data_processor(state: EnhancedAgentState) -> EnhancedAgentState:
    """Enhanced data processing with comprehensive validation"""
    try:
        state["processing_status"] = "processing"
        
        # Safe logging function
        def safe_log(message: str, level: str = "info"):
            try:
                if hasattr(state, 'add_log'):
                    state.add_log(message, level)
                else:
                    if "logs" not in state:
                        state["logs"] = []
                    state["logs"].append(f"{datetime.now().isoformat()} - {level.upper()}: {message}")
                    if level == "error":
                        logger.error(message)
                    else:
                        logger.info(message)
            except Exception as e:
                logger.error(f"Logging failed: {e}, Original message: {message}")
        
        safe_log("Starting enhanced data processing")
        
        if not state.get("user_consent", False):
            state["user_consent"] = True
        
        if "uploaded_file" not in state or state["uploaded_file"] is None:
            raise ValueError("No bank statement uploaded")
        
        uploaded_file = state["uploaded_file"]
        uploaded_file.seek(0)
        
        # Use enhanced converter
        df = EnhancedBankStatementConverter.validate_and_convert_file(uploaded_file)
        
        # Convert to transactions with enhanced categorization
        transactions = df.to_dict(orient="records")
        
        # Enhanced categorization logic
        for tx in transactions:
            desc = tx["description"].lower()
            amount = tx["amount"]
            
            # Smart categorization with better patterns
            if any(word in desc for word in ["grocery", "supermarket", "food", "restaurant", "cafe", "dining", "meal"]):
                tx["category"] = "food_dining"
            elif any(word in desc for word in ["rent", "mortgage", "insurance", "utility", "phone", "internet", "electric", "gas bill"]):
                tx["category"] = "fixed_expenses"
            elif any(word in desc for word in ["fuel", "gas station", "transport", "uber", "taxi", "bus", "train", "parking"]):
                tx["category"] = "transportation"
            elif any(word in desc for word in ["amazon", "shopping", "retail", "store", "mall", "clothes", "electronics"]):
                tx["category"] = "shopping"
            elif any(word in desc for word in ["salary", "wage", "income", "bonus", "payroll", "deposit"]):
                tx["category"] = "income"
            elif any(word in desc for word in ["transfer", "savings", "investment", "401k", "ira"]):
                tx["category"] = "savings_investment"
            elif any(word in desc for word in ["entertainment", "movie", "game", "subscription", "netflix", "spotify"]):
                tx["category"] = "entertainment"
            elif any(word in desc for word in ["medical", "doctor", "pharmacy", "hospital", "health"]):
                tx["category"] = "healthcare"
            else:
                tx["category"] = "other"
            
            # Enhanced spending type analysis
            if amount < 0:  # Expense
                if abs(amount) > 500:
                    tx["spending_type"] = "major_expense"
                elif any(word in desc for word in ["friday", "saturday", "sunday"]):
                    tx["spending_type"] = "weekend_spending"
                elif any(word in desc for word in ["impulse", "sale", "clearance"]):
                    tx["spending_type"] = "impulse_spending"
                else:
                    tx["spending_type"] = "regular_spending"
        
        # Validate transactions
        valid_transactions = FinancialHealthCalculator.validate_transactions(transactions)
        
        if not valid_transactions:
            raise ValueError("No valid transactions found after processing")
        
        state["transactions"] = valid_transactions
        state["bank_statement"]["balance"] = sum(tx["amount"] for tx in valid_transactions)
        
        # Calculate enhanced financial health metrics
        calculator = FinancialHealthCalculator()
        health_metrics = FinancialHealthMetrics()
        
        health_metrics.cashflow_score = calculator.calculate_cashflow_score(valid_transactions)
        health_metrics.savings_ratio_score = calculator.calculate_savings_ratio_score(valid_transactions)
        health_metrics.spending_stability_score = calculator.calculate_spending_stability_score(valid_transactions)
        
        # Enhanced emergency fund calculation
        monthly_expenses = sum(abs(tx["amount"]) for tx in valid_transactions if tx["amount"] < 0)
        num_months = len(set(tx["date"][:7] for tx in valid_transactions))
        avg_monthly_expenses = monthly_expenses / max(1, num_months)
        
        if avg_monthly_expenses > 0:
            months_covered = state["bank_statement"]["balance"] / avg_monthly_expenses
            health_metrics.emergency_fund_score = min(100, (months_covered / 6) * 100)  # 6 months = 100%
        else:
            health_metrics.emergency_fund_score = 100  # No expenses, perfect score
        
        # Calculate overall score with validation
        health_metrics = calculator.calculate_overall_health_score(health_metrics)
        
        # Generate improvement suggestions
        suggestions = []
        if health_metrics.savings_ratio_score < 60:
            suggestions.append("Try to increase your savings rate to at least 10% of income")
        if health_metrics.cashflow_score < 60:
            suggestions.append("Work on stabilizing your monthly cash flow")
        if health_metrics.emergency_fund_score < 60:
            suggestions.append("Build an emergency fund covering 3-6 months of expenses")
        if health_metrics.spending_stability_score < 60:
            suggestions.append("Create a budget to make your spending more predictable")
        
        health_metrics.improvement_suggestions = suggestions
        state["health_metrics"] = health_metrics
        
        # Calculate data quality score
        data_quality = min(100, (len(valid_transactions) / len(transactions)) * 100) if transactions else 0
        state["data_quality_score"] = data_quality
        
        state["processing_status"] = "completed"
        safe_log(f"Processing complete. Health Score: {health_metrics.overall_score:.1f}/100, Data Quality: {data_quality:.1f}%")
        
    except Exception as e:
        state["processing_status"] = "error"
        error_msg = f"Processing error: {str(e)}"
        
        # Safe error logging
        try:
            if hasattr(state, 'add_log'):
                state.add_log(error_msg, "error")
            else:
                if "logs" not in state:
                    state["logs"] = []
                state["logs"].append(f"{datetime.now().isoformat()} - ERROR: {error_msg}")
                logger.error(error_msg)
        except:
            logger.error(error_msg)
        
        if "error_messages" not in state:
            state["error_messages"] = []
        state["error_messages"].append(str(e))
        raise
    
    return state

async def conversational_coach_agent(state: EnhancedAgentState) -> EnhancedAgentState:
    """Enhanced conversational AI coach with better context management"""
    def safe_log(message: str, level: str = "info"):
        try:
            if "logs" not in state:
                state["logs"] = []
            state["logs"].append(f"{datetime.now().isoformat()} - {level.upper()}: {message}")
            if level == "error":
                logger.error(message)
            else:
                logger.info(message)
        except Exception as e:
            logger.error(f"Logging failed: {e}, Original message: {message}")
    
    try:
        query = state.get("current_query", "")
        context = state.get("conversation_context", ConversationContext())
        health_metrics = state.get("health_metrics", FinancialHealthMetrics())
        
        if query:
            # Update interaction count
            context.interaction_count += 1
            
            # Detect emotion and stress level
            context.user_emotion = ConversationalAI.detect_emotion(query)
            context.financial_stress_level = min(1.0, max(0.0, 
                ConversationalAI.detect_financial_stress(query, health_metrics.overall_score)))
            
            # Generate empathetic response
            response = await ConversationalAI.generate_empathetic_response(query, context, health_metrics)
            
            # Update conversation history with size limit
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_query": query,
                "ai_response": response,
                "emotion": context.user_emotion,
                "stress_level": context.financial_stress_level,
                "topics": ConversationalAI.extract_financial_topics(query)
            }
            
            context.conversation_history.append(conversation_entry)
            
            # Keep only last 10 conversations to manage memory
            if len(context.conversation_history) > 10:
                context.conversation_history = context.conversation_history[-10:]
            
            state["personalized_response"] = response
            state["conversation_context"] = context
            
        safe_log("Conversational coach agent completed")
        
    except Exception as e:
        safe_log(f"Conversational coach error: {e}", "error")
        state["personalized_response"] = "I'm here to help! Please let me know what financial questions you have."
    
    return state

async def intelligent_insights_agent(state: EnhancedAgentState) -> EnhancedAgentState:
    """Generate comprehensive intelligent insights"""
    def safe_log(message: str, level: str = "info"):
        try:
            if "logs" not in state:
                state["logs"] = []
            state["logs"].append(f"{datetime.now().isoformat()} - {level.upper()}: {message}")
            if level == "error":
                logger.error(message)
            else:
                logger.info(message)
        except Exception as e:
            logger.error(f"Logging failed: {e}, Original message: {message}")
    
    try:
        transactions = state.get("transactions", [])
        health_metrics = state.get("health_metrics", FinancialHealthMetrics())
        
        insights = []
        
        if not transactions:
            state["ai_insights"] = insights
            return state
        
        # Advanced spending analysis
        expenses = [tx for tx in transactions if tx["amount"] < 0]
        income_txs = [tx for tx in transactions if tx["amount"] > 0]
        
        total_income = sum(tx["amount"] for tx in income_txs)
        total_expenses = sum(abs(tx["amount"]) for tx in expenses)
        
        # 1. Income vs Expenses Analysis
        if total_income > 0:
            expense_ratio = total_expenses / total_income
            if expense_ratio > 0.9:
                insights.append({
                    "type": "warning",
                    "category": "spending",
                    "title": "High Expense Ratio",
                    "message": f"You're spending {expense_ratio*100:.1f}% of your income. Try to keep it below 80% for better financial health.",
                    "actionable": True,
                    "priority": "high",
                    "action_items": ["Create a detailed budget", "Identify non-essential expenses to cut"]
                })
        
        # 2. Category-wise spending insights
        category_totals = {}
        for tx in expenses:
            category = tx.get("category", "other")
            category_totals[category] = category_totals.get(category, 0) + abs(tx["amount"])
        
        if category_totals:
            largest_category = max(category_totals.items(), key=lambda x: x[1])
            if largest_category[1] > total_expenses * 0.4:  # More than 40% in one category
                insights.append({
                    "type": "insight",
                    "category": "spending_pattern",
                    "title": "Concentrated Spending",
                    "message": f"You spend {largest_category[1]/total_expenses*100:.1f}% on {largest_category[0].replace('_', ' ')}. Consider if this aligns with your priorities.",
                    "actionable": True,
                    "priority": "medium",
                    "action_items": [f"Review {largest_category[0].replace('_', ' ')} expenses", "Consider rebalancing your spending"]
                })
        
        # 3. Temporal spending patterns
        weekend_expenses = [tx for tx in expenses if tx.get("spending_type") == "weekend_spending"]
        if weekend_expenses:
            weekend_total = sum(abs(tx["amount"]) for tx in weekend_expenses)
            weekend_ratio = weekend_total / total_expenses
            
            if weekend_ratio > 0.25:  # More than 25% weekend spending
                insights.append({
                    "type": "behavioral",
                    "category": "spending_pattern",
                    "title": "Weekend Spending Pattern",
                    "message": f"You spend {weekend_ratio*100:.1f}% of your money on weekends. This might indicate emotional or social spending.",
                    "actionable": True,
                    "priority": "medium",
                    "action_items": ["Set a weekend spending limit", "Plan affordable weekend activities"]
                })
        
        # 4. Emergency fund analysis
        if isinstance(health_metrics, dict):
            emergency_score = health_metrics.get("emergency_fund_score", 0)
        else:
            emergency_score = getattr(health_metrics, "emergency_fund_score", 0)
            
        if emergency_score < 50:
            months_covered = (state.get("bank_statement", {}).get("balance", 0) / 
                            (total_expenses / max(1, len(set(tx["date"][:7] for tx in transactions)))))
            
            insights.append({
                "type": "goal",
                "category": "emergency_fund",
                "title": "Emergency Fund Goal",
                "message": f"Your emergency fund covers {months_covered:.1f} months of expenses. Aim for 3-6 months for financial security.",
                "actionable": True,
                "priority": "high",
                "action_items": [
                    "Set up automatic transfers to savings",
                    f"Save monthly to reach 3-month emergency fund goal"
                ]
            })
        
        # 5. Positive reinforcement
        if isinstance(health_metrics, dict):
            overall_score = health_metrics.get("overall_score", 0)
        else:
            overall_score = getattr(health_metrics, "overall_score", 0)
            
        if overall_score > 70:
            insights.append({
                "type": "positive",
                "category": "achievement",
                "title": "Strong Financial Health",
                "message": f"Excellent! Your financial health score of {overall_score:.1f} is above average. You're on the right track!",
                "actionable": False,
                "priority": "low",
                "action_items": ["Keep up the good habits", "Consider advanced financial planning"]
            })
        
        # 6. Savings opportunity analysis
        discretionary_categories = ["shopping", "entertainment", "other"]
        discretionary_spending = sum(category_totals.get(cat, 0) for cat in discretionary_categories)
        
        if discretionary_spending > total_expenses * 0.2:  # More than 20% discretionary
            potential_savings = discretionary_spending * 0.3  # Could save 30% of discretionary
            insights.append({
                "type": "opportunity",
                "category": "savings",
                "title": "Savings Opportunity",
                "message": f"You could potentially save ${potential_savings:.0f} by reducing discretionary spending by 30%.",
                "actionable": True,
                "priority": "medium",
                "action_items": [
                    "Review entertainment and shopping expenses",
                    "Set monthly limits for discretionary categories"
                ]
            })
        
        # 7. Income diversification
        if len(income_txs) <= 1:
            insights.append({
                "type": "risk",
                "category": "income",
                "title": "Single Income Source",
                "message": "Having only one income source increases financial risk. Consider diversifying your income streams.",
                "actionable": True,
                "priority": "medium",
                "action_items": ["Explore side hustles", "Develop passive income sources", "Build emergency fund"]
            })
        
        state["ai_insights"] = insights
        safe_log(f"Generated {len(insights)} intelligent insights")
        
    except Exception as e:
        safe_log(f"Insights agent error: {e}", "error")
        state["ai_insights"] = []
    
    return state

async def recommendations_agent(state: EnhancedAgentState) -> EnhancedAgentState:
    """Generate personalized financial recommendations"""
    def safe_log(message: str, level: str = "info"):
        try:
            if "logs" not in state:
                state["logs"] = []
            state["logs"].append(f"{datetime.now().isoformat()} - {level.upper()}: {message}")
            if level == "error":
                logger.error(message)
            else:
                logger.info(message)
        except Exception as e:
            logger.error(f"Logging failed: {e}, Original message: {message}")
    
    try:
        health_metrics = state.get("health_metrics", FinancialHealthMetrics())
        recommendations = []
        
        # Generate basic recommendations based on health score
        if isinstance(health_metrics, dict):
            overall_score = health_metrics.get("overall_score", 0)
        else:
            overall_score = health_metrics.overall_score
        
        if overall_score < 40:
            recommendations.append({
                "priority": "critical",
                "title": "Financial Health Critical",
                "description": "Your financial health needs immediate attention.",
                "actions": [
                    "Stop all non-essential spending immediately",
                    "Create a survival budget covering only necessities",
                    "Consider speaking with a financial counselor"
                ],
                "timeline": "This week"
            })
        
        state["recommendations"] = recommendations
        safe_log(f"Generated {len(recommendations)} personalized recommendations")
        
    except Exception as e:
        safe_log(f"Recommendations agent error: {e}", "error")
        state["recommendations"] = []
    
    return state to reach 3-month goal in a year"
                ]
            })
        
        # 5. Positive reinforcement
        if health_metrics.overall_score > 70:
            insights.append({
                "type": "positive",
                "category": "achievement",
                "title": "Strong Financial Health",
                "message": f"Excellent! Your financial health score of {health_metrics.overall_score:.1f} is above average. You're on the right track!",
                "actionable": False,
                "priority": "low",
                "action_items": ["Keep up the good habits", "Consider advanced financial planning"]
            })
        
        # 6. Savings opportunity analysis
        discretionary_categories = ["shopping", "entertainment", "other"]
        discretionary_spending = sum(category_totals.get(cat, 0) for cat in discretionary_categories)
        
        if discretionary_spending > total_expenses * 0.2:  # More than 20% discretionary
            potential_savings = discretionary_spending * 0.3  # Could save 30% of discretionary
            insights.append({
                "type": "opportunity",
                "category": "savings",
                "title": "Savings Opportunity",
                "message": f"You could potentially save ${potential_savings:.0f} by reducing discretionary spending by 30%.",
                "actionable": True,
                "priority": "medium",
                "action_items": [
                    "Review entertainment and shopping expenses",
                    "Set monthly limits for discretionary categories"
                ]
            })
        
        # 7. Income diversification
        if len(income_txs) == 1:
            insights.append({
                "type": "risk",
                "category": "income",
                "title": "Single Income Source",
                "message": "Having only one income source increases financial risk. Consider diversifying your income streams.",
                "actionable": True,
                "priority": "medium",
                "action_items": ["Explore side hustles", "Develop passive income sources", "Build emergency fund"]
            })
        
        state["ai_insights"] = insights
        state.add_log(f"Generated {len(insights)} intelligent insights")
        
    except Exception as e:
        state.add_log(f"Insights agent error: {e}", "error")
        state["ai_insights"] = []
    
    return state

async def recommendations_agent(state: EnhancedAgentState) -> EnhancedAgentState:
    """Generate personalized financial recommendations"""
    try:
        health_metrics = state.get("health_metrics", FinancialHealthMetrics())
        transactions = state.get("transactions", [])
        insights = state.get("ai_insights", [])
        
        recommendations = []
        
        # Priority-based recommendations
        if health_metrics.overall_score < 40:
            recommendations.append({
                "priority": "critical",
                "title": "Financial Health Critical",
                "description": "Your financial health needs immediate attention.",
                "actions": [
                    "Stop all non-essential spending immediately",
                    "Create a survival budget covering only necessities",
                    "Consider speaking with a financial counselor",
                    "Look for additional income sources"
                ],
                "timeline": "This week"
            })
        
        elif health_metrics.savings_ratio_score < 30:
            recommendations.append({
                "priority": "high",
                "title": "Improve Savings Rate",
                "description": "Building savings should be your top priority.",
                "actions": [
                    "Automate savings of at least 10% of income",
                    "Use the 50/30/20 budgeting rule",
                    "Cut unnecessary subscriptions and expenses",
                    "Consider a high-yield savings account"
                ],
                "timeline": "Next 30 days"
            })
        
        elif health_metrics.emergency_fund_score < 50:
            recommendations.append({
                "priority": "high",
                "title": "Build Emergency Fund",
                "description": "An emergency fund provides financial security.",
                "actions": [
                    "Set a goal of 3-6 months of expenses",
                    "Open a separate savings account for emergencies",
                    "Automate weekly transfers to emergency fund",
                    "Use windfalls (tax refunds, bonuses) for emergency fund"
                ],
                "timeline": "Next 6 months"
            })
        
        elif health_metrics.spending_stability_score < 60:
            recommendations.append({
                "priority": "medium",
                "title": "Stabilize Spending Patterns",
                "description": "Consistent spending helps with financial planning.",
                "actions": [
                    "Create and stick to a monthly budget",
                    "Use spending tracking apps",
                    "Set up automatic bill payments",
                    "Review and categorize expenses weekly"
                ],
                "timeline": "Next 60 days"
            })
        
        else:
            # Advanced recommendations for healthy finances
            recommendations.append({
                "priority": "optimization",
                "title": "Optimize Financial Strategy",
                "description": "Your finances are healthy. Focus on optimization and growth.",
                "actions": [
                    "Increase investment contributions",
                    "Consider tax-advantaged accounts (401k, IRA)",
                    "Review and optimize insurance coverage",
                    "Plan for long-term financial goals"
                ],
                "timeline": "Next 3 months"
            })
        
        # Category-specific recommendations
        expenses = [tx for tx in transactions if tx["amount"] < 0]
        if expenses:
            category_totals = {}
            total_expenses = sum(abs(tx["amount"]) for tx in expenses)
            
            for tx in expenses:
                category = tx.get("category", "other")
                category_totals[category] = category_totals.get(category, 0) + abs(tx["amount"])
            
            # Food & Dining recommendations
            food_spending = category_totals.get("food_dining", 0)
            if food_spending > total_expenses * 0.20:  # More than 20%
                recommendations.append({
                    "priority": "medium",
                    "title": "Optimize Food Spending",
                    "description": f"You spend {food_spending/total_expenses*100:.1f}% on food and dining.",
                    "actions": [
                        "Plan meals and create shopping lists",
                        "Cook more meals at home",
                        "Limit dining out to special occasions",
                        "Buy generic brands and use coupons"
                    ],
                    "timeline": "Start this week"
                })
        
        state["recommendations"] = recommendations
        state.add_log(f"Generated {len(recommendations)} personalized recommendations")
        
    except Exception as e:
        state.add_log(f"Recommendations agent error: {e}", "error")
        state["recommendations"] = []
    
    return state

# Enhanced UI Components

def render_financial_health_dashboard(state: EnhancedAgentState):
    """Render comprehensive financial health dashboard"""
    health_metrics = state.get("health_metrics", FinancialHealthMetrics())
    
    st.markdown("## 📊 Financial Health Dashboard")
    
    # Handle both dict and object types for health_metrics
    def get_metric_value(metrics, attr, default=0):
        if isinstance(metrics, dict):
            return metrics.get(attr, default)
        else:
            return getattr(metrics, attr, default)
    
    overall_score = get_metric_value(health_metrics, "overall_score", 0)
    risk_level = get_metric_value(health_metrics, "risk_level", "Unknown")
    trend = get_metric_value(health_metrics, "trend", "Stable")
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Health Score",
            f"{overall_score:.1f}/100",
            delta=f"{trend}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Risk Level",
            risk_level,
            delta=None
        )
    
    with col3:
        balance = state.get("bank_statement", {}).get("balance", 0)
        st.metric(
            "Current Balance",
            f"${balance:,.2f}",
            delta=None
        )
    
    with col4:
        data_quality = state.get("data_quality_score", 0)
        st.metric(
            "Data Quality",
            f"{data_quality:.1f}%",
            delta=None
        )
    
    # Detailed metrics breakdown
    if overall_score > 0:
        st.markdown("### Health Score Breakdown")
        
        cashflow_score = get_metric_value(health_metrics, "cashflow_score", 0)
        savings_score = get_metric_value(health_metrics, "savings_ratio_score", 0)
        stability_score = get_metric_value(health_metrics, "spending_stability_score", 0)
        emergency_score = get_metric_value(health_metrics, "emergency_fund_score", 0)
        
        metrics_data = {
            'Metric': ['Cashflow Stability', 'Savings Rate', 'Spending Stability', 'Emergency Fund'],
            'Score': [cashflow_score, savings_score, stability_score, emergency_score],
            'Color': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=metrics_data['Metric'],
                y=metrics_data['Score'],
                marker_color=metrics_data['Color'],
                text=[f"{score:.1f}" for score in metrics_data['Score']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Financial Health Component Scores",
            yaxis_title="Score (0-100)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Improvement suggestions
        improvement_suggestions = get_metric_value(health_metrics, "improvement_suggestions", [])
        if improvement_suggestions:
            st.markdown("### 💡 Improvement Suggestions")
            for suggestion in improvement_suggestions:
                st.info(f"• {suggestion}")

def render_spending_analysis(state: EnhancedAgentState):
    """Render detailed spending analysis"""
    transactions = state.get("transactions", [])
    
    if not transactions:
        return
    
    st.markdown("## 💰 Spending Analysis")
    
    expenses = [tx for tx in transactions if tx["amount"] < 0]
    income_txs = [tx for tx in transactions if tx["amount"] > 0]
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        if expenses:
            category_totals = {}
            for tx in expenses:
                category = tx.get("category", "other")
                category_totals[category] = category_totals.get(category, 0) + abs(tx["amount"])
            
            # Create pie chart
            fig = px.pie(
                values=list(category_totals.values()),
                names=[cat.replace('_', ' ').title() for cat in category_totals.keys()],
                title="Spending by Category"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if expenses and income_txs:
            # Monthly trend
            monthly_data = {}
            for tx in transactions:
                month_key = pd.to_datetime(tx['date']).strftime('%Y-%m')
                if month_key not in monthly_data:
                    monthly_data[month_key] = {'income': 0, 'expenses': 0}
                
                if tx['amount'] > 0:
                    monthly_data[month_key]['income'] += tx['amount']
                else:
                    monthly_data[month_key]['expenses'] += abs(tx['amount'])
            
            months = sorted(monthly_data.keys())
            income_trend = [monthly_data[month]['income'] for month in months]
            expense_trend = [monthly_data[month]['expenses'] for month in months]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=income_trend, name='Income', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=months, y=expense_trend, name='Expenses', line=dict(color='red')))
            
            fig.update_layout(
                title="Monthly Income vs Expenses Trend",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_conversational_interface(state: EnhancedAgentState):
    """Render enhanced conversational interface"""
    st.markdown("## 💬 AI Financial Coach")
    
    conversation_context = state.get("conversation_context", {})
    
    # Handle both dict and object types
    if isinstance(conversation_context, dict):
        conversation_history = conversation_context.get("conversation_history", [])
        user_emotion = conversation_context.get("user_emotion", "neutral")
    else:
        conversation_history = getattr(conversation_context, "conversation_history", [])
        user_emotion = getattr(conversation_context, "user_emotion", "neutral")
    
    # Display conversation history
    if conversation_history:
        st.markdown("### Recent Conversations")
        
        # Show last 3 conversations in an expandable format
        for i, chat in enumerate(conversation_history[-3:]):
            with st.expander(f"Conversation {len(conversation_history) - 2 + i}", expanded=(i == 2)):
                st.markdown(f"**You:** {chat.get('user_query', 'N/A')}")
                st.markdown(f"**AI Coach:** {chat.get('ai_response', 'N/A')}")
                
                # Show emotion and topics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Emotion: {chat.get('emotion', 'neutral')}")
                with col2:
                    st.caption(f"Stress Level: {chat.get('stress_level', 0):.1f}/1.0")
                with col3:
                    topics = chat.get('topics', [])
                    if topics:
                        st.caption(f"Topics: {', '.join(topics)}")
    
    # Current response
    current_response = state.get("personalized_response", "")
    if current_response:
        st.markdown("### AI Coach Response")
        st.success(current_response)
        
        # Show emotion context
        if user_emotion != "neutral":
            st.caption(f"Response tailored for {user_emotion} mood")

def render_insights_and_recommendations(state: EnhancedAgentState):
    """Render insights and recommendations"""
    insights = state.get("ai_insights", [])
    recommendations = state.get("recommendations", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🧠 AI Insights")
        if insights:
            for insight in insights:
                if insight["priority"] == "high":
                    st.error(f"**{insight['title']}**\n\n{insight['message']}")
                elif insight["priority"] == "medium":
                    st.warning(f"**{insight['title']}**\n\n{insight['message']}")
                else:
                    st.success(f"**{insight['title']}**\n\n{insight['message']}")
                
                if insight.get("actionable") and "action_items" in insight:
                    with st.expander("Action Items"):
                        for action in insight["action_items"]:
                            st.markdown(f"• {action}")
        else:
            st.info("Process your bank statement to see AI insights!")
    
    with col2:
        st.markdown("## 📋 Recommendations")
        if recommendations:
            for rec in recommendations:
                priority_color = {
                    "critical": "🔴",
                    "high": "🟠", 
                    "medium": "🟡",
                    "optimization": "🟢"
                }
                
                st.markdown(f"### {priority_color.get(rec['priority'], '🔵')} {rec['title']}")
                st.markdown(rec['description'])
                
                with st.expander("Action Plan"):
                    for action in rec['actions']:
                        st.markdown(f"• {action}")
                    st.caption(f"Timeline: {rec['timeline']}")
        else:
            st.info("Complete the analysis to see personalized recommendations!")

def render_enhanced_dashboard(state: EnhancedAgentState):
    """Render the complete enhanced dashboard"""
    
    st.markdown("# 🤖 EmpowerFin Guardian 2.0")
    st.markdown("*Your Advanced AI Financial Coach*")
    
    # Show processing status
    status = state.get("processing_status", "idle")
    if status == "processing":
        st.info("🔄 Processing your financial data...")
    elif status == "error":
        st.error("❌ Error processing data. Please check your file and try again.")
        if state.get("error_messages"):
            for error in state["error_messages"]:
                st.error(f"Error: {error}")
    elif status == "completed":
        st.success("✅ Analysis complete!")
    
    # Render dashboard components
    if state.get("transactions"):
        render_financial_health_dashboard(state)
        render_spending_analysis(state)
        render_conversational_interface(state)
        render_insights_and_recommendations(state)
        
        # Show logs in expander
        logs = state.get("logs", [])
        if logs:
            with st.expander("📋 Processing Logs"):
                for log in logs[-10:]:  # Show last 10 logs
                    if isinstance(log, dict):
                        st.caption(f"{log['timestamp']} - {log['level'].upper()}: {log['message']}")
                    else:
                        st.caption(str(log))

# Enhanced Workflow
def build_enhanced_workflow():
    """Build the enhanced conversational workflow"""
    workflow = StateGraph(EnhancedAgentState)
    
    workflow.add_node("process_data", enhanced_data_processor)
    workflow.add_node("conversational_coach", conversational_coach_agent)
    workflow.add_node("intelligent_insights", intelligent_insights_agent)
    workflow.add_node("recommendations", recommendations_agent)
    
    workflow.set_entry_point("process_data")
    workflow.add_edge("process_data", "conversational_coach")
    workflow.add_edge("conversational_coach", "intelligent_insights")
    workflow.add_edge("intelligent_insights", "recommendations")
    workflow.add_edge("recommendations", END)
    
    return workflow.compile()

# Enhanced Streamlit App
def main():
    st.set_page_config(
        page_title="EmpowerFin Guardian 2.0",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state with error handling
    try:
        if 'app_state' not in st.session_state:
            st.session_state.app_state = EnhancedAgentState()
    except Exception as e:
        st.error(f"Error initializing application: {e}")
        logger.error(f"Failed to initialize session state: {e}")
        return
    
    # Initialize uploaded file data if not exists
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🚀 Setup")
        
        # File upload section
        st.markdown("### 📄 Upload Bank Statement")
        st.markdown("""
        **Supported formats:**
        - CSV files (any delimiter)
        - Excel files (.xlsx, .xls)
        
        **Auto-detection includes:**
        - Transaction dates (various formats)
        - Descriptions/details
        - Amounts (with/without currency symbols)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your bank statement", 
            type=["csv", "xlsx", "xls"],
            help="Upload your bank statement in CSV or Excel format"
        )
        
        # File preview
        if uploaded_file is not None:
            try:
                with st.spinner("Validating file..."):
                    df = EnhancedBankStatementConverter.validate_and_convert_file(uploaded_file)
                    st.success("✅ File validated successfully!")
                    
                    with st.expander("📊 Data Preview"):
                        st.dataframe(df.head())
                        st.caption(f"Found {len(df)} transactions")
                
                # Store in session state
                temp_csv = io.StringIO()
                df.to_csv(temp_csv, index=False)
                temp_csv.seek(0)
                st.session_state.uploaded_file_data = temp_csv
                
            except Exception as e:
                st.error(f"❌ File validation failed: {str(e)}")
                st.session_state.uploaded_file_data = None
        
        # Financial goal input
        st.markdown("### 🎯 Financial Goal")
        financial_goal = st.text_area(
            "What's your main financial goal?",
            value="Build emergency fund and improve savings rate",
            help="This helps personalize your recommendations"
        )
        
        # Chat input
        st.markdown("### 💬 Ask Your AI Coach")
        user_query = st.text_area(
            "What would you like to discuss?",
            placeholder="e.g., I'm worried about my spending habits. How can I save more money?",
            help="Ask any financial question for personalized advice"
        )
        
        # Process button
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.app_state = EnhancedAgentState()
            st.session_state.uploaded_file_data = None
            st.rerun()
    
    # Main content area
    if process_button:
        if not hasattr(st.session_state, 'uploaded_file_data') or st.session_state.uploaded_file_data is None:
            st.error("""
            ❌ **Please upload a bank statement first**
            
            The system supports various file formats and will automatically:
            - Detect column types (dates, amounts, descriptions)
            - Convert different date formats
            - Handle currency symbols and formatting
            - Validate data quality
            """)
            return
        
        with st.spinner("🔄 Analyzing your financial data..."):
            try:
                # Update state with inputs
                st.session_state.app_state.update({
                    'financial_goal': financial_goal,
                    'uploaded_file': st.session_state.uploaded_file_data,
                    'current_query': user_query,
                    'processing_status': 'starting'
                })
                
                # Build and run workflow
                app = build_enhanced_workflow()
                
                # Try async execution
                try:
                    result = asyncio.run(app.ainvoke(st.session_state.app_state))
                except Exception as workflow_error:
                    logger.error(f"Async workflow failed: {workflow_error}")
                    # Fallback to synchronous processing
                    st.warning("Using simplified processing mode...")
                    
                    # Simple fallback processing
                    temp_state = st.session_state.app_state.copy()
                    temp_state = asyncio.run(enhanced_data_processor(temp_state))
                    temp_state = asyncio.run(conversational_coach_agent(temp_state))
                    temp_state = asyncio.run(intelligent_insights_agent(temp_state))
                    temp_state = asyncio.run(recommendations_agent(temp_state))
                    result = temp_state
                
                # Update session state
                st.session_state.app_state.update(result)
                st.rerun()
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                st.error(f"❌ Analysis failed: {str(e)}")
                
                # Show detailed error for debugging
                with st.expander("Error Details"):
                    st.text(f"Error type: {type(e).__name__}")
                    st.text(f"Error message: {str(e)}")
                    
                    # Show some logs if available
                    logs = st.session_state.app_state.get("logs", [])
                    if logs:
                        st.text("Recent logs:")
                        for log in logs[-5:]:
                            st.text(f"  {log}")
                
                st.session_state.app_state["processing_status"] = "error"
                if "error_messages" not in st.session_state.app_state:
                    st.session_state.app_state["error_messages"] = []
                st.session_state.app_state["error_messages"].append(str(e))
    
    # Render dashboard
    render_enhanced_dashboard(st.session_state.app_state)
    
    # Welcome screen for new users
    if not st.session_state.app_state.get("transactions"):
        st.markdown("""
        ## Welcome to EmpowerFin Guardian 2.0 🚀
        
        ### Your Advanced AI Financial Coach
        
        **What makes this special:**
        
        #### 🎯 **Intelligent Analysis**
        - **Real-time Financial Health Scoring** - Comprehensive analysis across 4 key metrics
        - **Smart Transaction Categorization** - Automatically categorizes your spending
        - **Predictive Insights** - Identifies patterns and potential issues
        
        #### 🤖 **Conversational AI Coach**
        - **Emotion-Aware Responses** - Detects your mood and adapts advice accordingly
        - **Personalized Guidance** - Tailored recommendations based on your financial situation
        - **Stress-Level Detection** - Provides extra support when you're financially stressed
        
        #### 📊 **Advanced Features**
        - **Multi-Format Support** - Works with any CSV or Excel bank statement
        - **Data Quality Scoring** - Ensures reliable analysis
        - **Behavioral Insights** - Understand your spending psychology
        - **Actionable Recommendations** - Clear next steps for improvement
        
        ### 🚀 Getting Started:
        
        1. **Upload Your Bank Statement** → Supports various formats with auto-detection
        2. **Set Your Financial Goal** → Helps personalize recommendations  
        3. **Ask Questions** → Chat naturally about your financial concerns
        4. **Get Insights** → Receive AI-powered analysis and advice
        
        ---
        
        ### 🆕 New Features Added:
        
        #### 🔧 **Technical Improvements**
        - Enhanced error handling and validation
        - Better file format detection and conversion
        - Improved data quality scoring
        - Thread-safe LLM client with retry logic
        - Rotating log files for better debugging
        
        #### 🧠 **AI Enhancements**
        - Advanced emotion detection and stress analysis
        - Multi-topic conversation tracking
        - Behavioral spending pattern analysis
        - Priority-based recommendation system
        - Category-specific financial advice
        
        #### 📈 **Analytics Upgrades**
        - Monthly income vs expense trends
        - Weekend and emotional spending detection
        - Emergency fund coverage calculations
        - Income diversification analysis
        - Savings opportunity identification
        
        #### 🎨 **UI/UX Improvements**
        - Interactive dashboard with drill-down capabilities
        - Expandable conversation history
        - Priority-coded insights and recommendations
        - Processing status indicators
        - Comprehensive error messaging
        
        **Ready to take control of your financial future?** Upload your bank statement to begin! 💪
        """)

# Suggested Additional Features for Future Enhancement

def suggest_new_features():
    """
    NEW FEATURE SUGGESTIONS FOR EMPOWERFIN GUARDIAN 3.0:
    
    1. **Goal Tracking & Progress Monitoring**
       - Set specific financial goals (e.g., save $10k in 12 months)
       - Track progress with visual indicators
       - Milestone celebrations and adjustments
       - Goal-based spending recommendations
    
    2. **Predictive Analytics & Forecasting**
       - Income and expense forecasting based on historical data
       - Cash flow predictions for next 3-6 months
       - "What-if" scenario analysis (salary increase, new expense, etc.)
       - Risk assessment for financial decisions
    
    3. **Advanced Budgeting Tools**
       - Interactive budget creation and management
       - Envelope budgeting system
       - Bill reminder and payment tracking
       - Budget vs actual spending comparisons
    
    4. **Investment Analysis Integration**
       - Portfolio analysis and asset allocation advice
       - Risk tolerance assessment
       - Investment recommendation engine
       - Retirement planning calculator
    
    5. **Bill & Subscription Management**
       - Automatic subscription detection
       - Unused subscription alerts
       - Bill optimization recommendations
       - Contract renewal reminders
    
    6. **Social & Comparative Features**
       - Anonymous peer comparisons by demographics
       - Financial health rankings
       - Community challenges and tips
       - Expert webinars and educational content
    
    7. **Mobile Integration & Notifications**
       - Mobile app companion
       - Real-time spending alerts
       - Weekly/monthly financial summaries
       - Achievement notifications
    
    8. **Advanced Security & Privacy**
       - End-to-end encryption for financial data
       - Biometric authentication
       - Privacy-preserving analytics
       - Data retention controls
    
    9. **Integration Ecosystem**
       - Direct bank API connections (with user consent)
       - Credit card integration
       - Investment account linking
       - Tax software integration
    
    10. **AI-Powered Financial Assistant**
        - Voice-enabled queries and responses
        - Natural language financial planning
        - Proactive financial advice
        - Learning from user feedback and preferences
    
    11. **Debt Management & Optimization**
        - Debt consolidation analysis
        - Payoff strategy optimization (avalanche vs snowball)
        - Credit score improvement suggestions
        - Refinancing opportunity alerts
    
    12. **Tax Optimization Features**
        - Tax-efficient spending recommendations
        - Deduction tracking and optimization
        - Tax bracket analysis
        - HSA/401k contribution optimization
    
    13. **Emergency & Crisis Support**
        - Financial crisis detection
        - Emergency resource recommendations
        - Hardship planning tools
        - Recovery roadmap creation
    
    14. **Family & Joint Financial Management**
        - Multi-user account management
        - Family budget coordination
        - Child financial education tools
        - Couple financial goal alignment
    
    15. **Advanced Reporting & Export**
        - Customizable financial reports
        - PDF export capabilities
        - Integration with accounting software
        - Financial advisor sharing tools
    """
    pass

if __name__ == "__main__":
    main()