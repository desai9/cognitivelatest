# ==============================================================================
# simple_launcher.py - Simplified EmpowerFin Guardian (Single File)
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import time
import re

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class Transaction:
    """Transaction data model"""
    date: str
    description: str
    amount: float
    category: str = "other"
    spending_type: str = "regular_spending"

@dataclass
class FinancialHealthMetrics:
    """Financial health scoring components"""
    cashflow_score: float = 0.0
    savings_ratio_score: float = 0.0
    spending_stability_score: float = 0.0
    emergency_fund_score: float = 0.0
    overall_score: float = 0.0
    risk_level: str = "Unknown"
    trend: str = "Stable"
    improvement_suggestions: List[str] = field(default_factory=list)

# ==============================================================================
# FILE PROCESSOR
# ==============================================================================

class SimpleFileProcessor:
    """Simplified file processing"""
    
    @staticmethod
    def process_file(uploaded_file):
        """Process uploaded bank statement file"""
        try:
            # Read file
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                # Try different separators
                for sep in [',', ';', '\t', '|']:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=sep)
                        if len(df.columns) > 1:
                            break
                    except:
                        continue
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Auto-detect columns
            date_col = SimpleFileProcessor._find_column(df, ['date', 'time', 'posted'])
            desc_col = SimpleFileProcessor._find_column(df, ['description', 'transaction', 'details', 'memo'])
            amount_col = SimpleFileProcessor._find_column(df, ['amount', 'value', 'debit', 'credit'])
            
            if not all([date_col, desc_col, amount_col]):
                st.error(f"Could not detect required columns. Available: {list(df.columns)}")
                return None
            
            # Create standard format
            result_df = pd.DataFrame()
            result_df['date'] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            result_df['description'] = df[desc_col].astype(str).str.strip()
            
            # Clean amount column
            amount_series = df[amount_col].astype(str)
            amount_series = amount_series.str.replace(r'[$£€,\s]', '', regex=True)
            
            # Handle negative numbers in parentheses
            negative_mask = amount_series.str.contains(r'\(.*\)', na=False)
            amount_series = amount_series.str.replace(r'[()]', '', regex=True)
            
            result_df['amount'] = pd.to_numeric(amount_series, errors='coerce')
            result_df.loc[negative_mask, 'amount'] = -abs(result_df.loc[negative_mask, 'amount'])
            
            # Remove invalid rows
            result_df = result_df.dropna()
            
            return result_df
            
        except Exception as e:
            st.error(f"File processing error: {e}")
            return None
    
    @staticmethod
    def _find_column(df, keywords):
        """Find column matching keywords"""
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None

# ==============================================================================
# TRANSACTION CATEGORIZER
# ==============================================================================

class SimpleTransactionCategorizer:
    """Simple transaction categorization"""
    
    def __init__(self):
        self.categories = {
            'food_dining': ['grocery', 'supermarket', 'food', 'restaurant', 'cafe', 'mcdonalds', 'starbucks'],
            'transportation': ['gas', 'fuel', 'uber', 'taxi', 'bus', 'metro', 'parking'],
            'fixed_expenses': ['rent', 'mortgage', 'insurance', 'utility', 'phone', 'internet', 'electric'],
            'shopping': ['amazon', 'walmart', 'target', 'shopping', 'store', 'mall'],
            'income': ['salary', 'wage', 'payroll', 'deposit', 'income', 'pay'],
            'entertainment': ['netflix', 'spotify', 'movie', 'game', 'entertainment'],
            'healthcare': ['pharmacy', 'doctor', 'medical', 'hospital', 'cvs', 'walgreens']
        }
    
    def categorize_transaction(self, description: str) -> str:
        """Categorize a single transaction"""
        desc_lower = description.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category
        
        return 'other'

# ==============================================================================
# HEALTH CALCULATOR
# ==============================================================================

class SimpleHealthCalculator:
    """Simplified health calculation"""
    
    @staticmethod
    def calculate_health(transactions: List[Transaction]) -> FinancialHealthMetrics:
        """Calculate financial health metrics"""
        metrics = FinancialHealthMetrics()
        
        if not transactions:
            return metrics
        
        # Basic calculations
        total_income = sum(tx.amount for tx in transactions if tx.amount > 0)
        total_expenses = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
        current_balance = sum(tx.amount for tx in transactions)
        
        # Savings ratio score
        if total_income > 0:
            savings_rate = (total_income - total_expenses) / total_income
            metrics.savings_ratio_score = min(100, max(0, savings_rate * 500))
        
        # Emergency fund score (assuming monthly expenses)
        num_months = len(set(tx.date[:7] for tx in transactions))
        avg_monthly_expenses = total_expenses / max(1, num_months)
        
        if avg_monthly_expenses > 0:
            months_covered = current_balance / avg_monthly_expenses
            metrics.emergency_fund_score = min(100, (months_covered / 6) * 100)
        
        # Spending stability (simplified)
        expenses = [abs(tx.amount) for tx in transactions if tx.amount < 0]
        if len(expenses) > 1:
            cv = np.std(expenses) / (np.mean(expenses) + 1)
            metrics.spending_stability_score = max(0, min(100, 100 - (cv * 50)))
        else:
            metrics.spending_stability_score = 50
        
        # Cashflow score (simplified)
        if len(transactions) > 5:
            recent_balance = sum(tx.amount for tx in transactions[-5:])
            metrics.cashflow_score = min(100, max(0, 50 + (recent_balance / 100)))
        else:
            metrics.cashflow_score = 50
        
        # Overall score
        metrics.overall_score = (
            metrics.cashflow_score * 0.25 +
            metrics.savings_ratio_score * 0.30 +
            metrics.spending_stability_score * 0.25 +
            metrics.emergency_fund_score * 0.20
        )
        
        # Risk level
        if metrics.overall_score >= 80:
            metrics.risk_level = "Low Risk"
        elif metrics.overall_score >= 60:
            metrics.risk_level = "Medium Risk"
        elif metrics.overall_score >= 40:
            metrics.risk_level = "High Risk"
        else:
            metrics.risk_level = "Critical Risk"
        
        # Improvement suggestions
        suggestions = []
        if metrics.savings_ratio_score < 50:
            suggestions.append("Try to increase your savings rate to at least 10% of income")
        if metrics.emergency_fund_score < 50:
            suggestions.append("Build an emergency fund covering 3-6 months of expenses")
        if metrics.spending_stability_score < 50:
            suggestions.append("Create a budget to make spending more predictable")
        
        metrics.improvement_suggestions = suggestions
        
        return metrics

# ==============================================================================
# CONVERSATIONAL AI
# ==============================================================================

class SimpleConversationalAI:
    """Simplified conversational AI with enhanced responses"""
    
    def __init__(self):
        self.responses = {
            'budgeting': {
                'stressed': "I understand budgeting can feel overwhelming. Let's start simple with the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Begin by tracking just one week of expenses to see where your money goes.",
                'default': "Great question about budgeting! The 50/30/20 rule is a solid foundation: 50% needs, 30% wants, 20% savings. I recommend starting with a simple expense tracker app or even a notebook to see your spending patterns."
            },
            'saving': {
                'stressed': "Building savings can feel impossible when money is tight, but even $5 per week adds up to $260 per year! Start micro-small: save loose change or round up purchases. Once you see it working, gradually increase.",
                'default': "Building savings is crucial for financial security. Start with the 'pay yourself first' approach: automatically save even 1% of your income, then increase by 1% every few months until you reach 10-20%."
            },
            'debt': {
                'stressed': "Dealing with debt is stressful, but you have options. Try the debt snowball method: pay minimums on all debts, then put every extra dollar toward the smallest debt first. The psychological wins will keep you motivated!",
                'default': "For debt management, you have two main strategies: 1) Debt Avalanche (pay highest interest first - saves more money), or 2) Debt Snowball (pay smallest balance first - builds momentum). Choose based on your personality."
            },
            'spending': {
                'stressed': "When spending feels out of control, try the 24-hour rule: wait a day before any non-essential purchase over $50. You'll be surprised how often you change your mind. Also, remove stored payment info from shopping apps.",
                'default': "To reduce spending, try these tactics: 1) The 24-hour rule for purchases over $50, 2) Unsubscribe from retailer emails, 3) Use cash for discretionary spending, 4) Find free alternatives for entertainment."
            },
            'income': {
                'default': "To increase income, consider: 1) Asking for a raise (research market rates first), 2) Side hustles that match your skills, 3) Freelancing in your spare time, 4) Selling unused items, 5) Passive income streams like cashback apps."
            },
            'investment': {
                'stressed': "Don't worry about investing until you have an emergency fund and stable income. Focus on the basics first: budget, save, pay off high-interest debt. Investing comes after you've built a solid foundation.",
                'default': "For beginner investing: 1) Max out any employer 401k match (free money!), 2) Open a Roth IRA, 3) Start with low-cost index funds, 4) Invest consistently, not all at once, 5) Don't try to time the market."
            },
            'emergency': {
                'default': "Emergency funds are crucial! Aim for 3-6 months of expenses. Start small: even $500 can cover many emergencies. Keep it in a separate high-yield savings account so you're not tempted to spend it."
            },
            'health_score': {
                'default': "To improve your financial health: 1) Increase your savings rate, 2) Build an emergency fund, 3) Make spending more consistent, 4) Improve cash flow stability. Focus on one area at a time for best results."
            },
            'general': {
                'stressed': "I understand financial stress can be overwhelming. Let's start with one small step: track your spending for just one week. Knowledge is power, and you can't improve what you don't measure. You've got this! 💪",
                'default': "I'm here to help with your financial journey! Whether it's budgeting, saving, debt management, or improving your financial health score, we can work together to build a stronger financial future. What's your biggest concern right now?"
            }
        }
        
        # Enhanced topic keywords
        self.topic_keywords = {
            'budgeting': ['budget', 'budgeting', 'plan', 'planning', 'allocate', 'allocation'],
            'saving': ['save', 'saving', 'savings', 'emergency fund', 'nest egg'],
            'debt': ['debt', 'loan', 'credit', 'owe', 'payment', 'payoff'],
            'spending': ['spend', 'spending', 'expenses', 'cost', 'reduce', 'cut'],
            'income': ['income', 'salary', 'earn', 'money', 'raise', 'side hustle'],
            'investment': ['invest', 'investing', 'stocks', 'portfolio', '401k', 'ira'],
            'emergency': ['emergency', 'emergency fund', 'rainy day'],
            'health_score': ['health score', 'financial health', 'improve score', 'score']
        }
        
        # Stress indicators
        self.stress_keywords = [
            'worried', 'anxious', 'stressed', 'scared', 'overwhelmed', 'struggling',
            'desperate', 'help', 'trouble', 'confused', 'lost', 'panic', 'crisis'
        ]
    
    def detect_emotion(self, text: str) -> str:
        """Enhanced emotion detection"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.stress_keywords):
            return 'stressed'
        
        # Check for positive emotions
        positive_words = ['good', 'great', 'excellent', 'happy', 'excited', 'ready']
        if any(word in text_lower for word in positive_words):
            return 'positive'
        
        return 'default'
    
    def detect_topics(self, text: str) -> List[str]:
        """Enhanced topic detection"""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def generate_response(self, query: str, health_score: float = 50) -> Dict[str, Any]:
        """Generate enhanced response to user query"""
        emotion = self.detect_emotion(query)
        topics = self.detect_topics(query)
        primary_topic = topics[0]
        
        # Get response template
        topic_responses = self.responses.get(primary_topic, self.responses['general'])
        response = topic_responses.get(emotion, topic_responses.get('default', 
                                     topic_responses[list(topic_responses.keys())[0]]))
        
        # Add context based on health score
        if health_score < 30:
            response += "\n\n🚨 Your financial health score suggests immediate attention is needed. Focus on the basics: stop non-essential spending, create a simple budget, and start building even a small emergency fund."
        elif health_score < 50:
            response += f"\n\n📊 With your health score of {health_score:.1f}, you're making progress but there's room for improvement. Small, consistent changes will make a big difference!"
        elif health_score < 70:
            response += f"\n\n✅ Your health score of {health_score:.1f} shows you're on the right track! Now it's time to optimize and fine-tune your financial strategy."
        elif health_score >= 70:
            response += f"\n\n🌟 Excellent! Your health score of {health_score:.1f} shows strong financial health. Consider advanced strategies like investment optimization or tax planning."
        
        # Add motivational closing for stressed users
        if emotion == 'stressed':
            encouragements = [
                "Remember: small steps lead to big changes! 🌱",
                "You're not alone in this journey! 💪",
                "Every financial expert started where you are now! 🚀",
                "Progress, not perfection, is the goal! ⭐"
            ]
            import random
            response += f"\n\n{random.choice(encouragements)}"
        
        return {
            'response': response,
            'emotion': emotion,
            'topics': topics,
            'timestamp': datetime.now().isoformat()
        }

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class EmpowerFinApp:
    """Main application class"""
    
    def __init__(self):
        self.file_processor = SimpleFileProcessor()
        self.categorizer = SimpleTransactionCategorizer()
        self.health_calculator = SimpleHealthCalculator()
        self.conversational_ai = SimpleConversationalAI()
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="EmpowerFin Guardian 2.0",
            page_icon="🤖",
            layout="wide"
        )
        
        # Header
        st.markdown("# 🤖 EmpowerFin Guardian 2.0")
        st.markdown("*Your AI Financial Intelligence Platform*")
        
        # Initialize session state
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.transactions:
            self._render_dashboard()
        else:
            self._render_welcome()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## 📁 Upload Bank Statement")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            help="Upload your bank statement in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            if st.sidebar.button("🔍 Process File", type="primary"):
                self._process_file(uploaded_file)
        
        st.sidebar.markdown("## 💬 Chat with AI")
        
        user_query = st.sidebar.text_area(
            "Ask a financial question",
            placeholder="e.g., How can I save more money?",
            height=100
        )
        
        if st.sidebar.button("💬 Ask AI") and user_query:
            self._process_chat(user_query)
        
        # Show system status
        st.sidebar.markdown("## 📊 System Status")
        st.sidebar.success("✅ All Services Online")
        
        if st.session_state.transactions:
            st.sidebar.metric("Transactions", len(st.session_state.transactions))
        
        if st.session_state.health_metrics:
            score = st.session_state.health_metrics.overall_score
            st.sidebar.metric("Health Score", f"{score:.1f}/100")
    
    def _process_file(self, uploaded_file):
        """Process uploaded file"""
        with st.spinner("🤖 Processing your bank statement..."):
            try:
                # Process file
                df = self.file_processor.process_file(uploaded_file)
                
                if df is not None:
                    # Convert to transactions
                    transactions = []
                    for _, row in df.iterrows():
                        transaction = Transaction(
                            date=row['date'],
                            description=row['description'],
                            amount=float(row['amount'])
                        )
                        # Categorize
                        transaction.category = self.categorizer.categorize_transaction(transaction.description)
                        transactions.append(transaction)
                    
                    # Calculate health metrics
                    health_metrics = self.health_calculator.calculate_health(transactions)
                    
                    # Store in session state
                    st.session_state.transactions = transactions
                    st.session_state.health_metrics = health_metrics
                    
                    st.success(f"✅ Processed {len(transactions)} transactions successfully!")
                    time.sleep(1)
                    st.rerun()
                
            except Exception as e:
                st.error(f"Processing failed: {e}")
    
    def _process_chat(self, query):
        """Process chat query"""
        try:
            health_score = 50
            if st.session_state.health_metrics:
                health_score = st.session_state.health_metrics.overall_score
            
            # Generate response
            chat_result = self.conversational_ai.generate_response(query, health_score)
            
            # Store conversation
            conversation_entry = {
                'user_query': query,
                'ai_response': chat_result['response'],
                'emotion': chat_result['emotion'],
                'topics': chat_result['topics'],
                'timestamp': chat_result['timestamp']
            }
            
            st.session_state.conversation_history.append(conversation_entry)
            
            st.success("✅ Response generated!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Chat processing failed: {e}")
    
    def _render_welcome(self):
        """Render welcome screen"""
        st.markdown("""
        ## Welcome to EmpowerFin Guardian 2.0! 🚀
        
        ### 🤖 Your AI Financial Intelligence Platform
        
        **Features:**
        - 📊 **Smart Analysis**: Upload any bank statement format
        - 🧠 **AI Health Scoring**: Comprehensive financial health assessment
        - 💬 **Conversational AI**: Natural language financial advice
        - 📈 **Visual Insights**: Interactive charts and analytics
        
        ### 🚀 Getting Started:
        1. **Upload** your bank statement using the sidebar
        2. **Review** your financial health dashboard
        3. **Chat** with the AI for personalized advice
        4. **Track** your progress over time
        
        ---
        
        *Ready to take control of your finances? Upload your bank statement to begin!* 💪
        """)
    
    def _render_dashboard(self):
        """Render main dashboard"""
        # Health metrics overview
        self._render_health_overview()
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Analysis", "💬 AI Chat", "📋 Transactions"])
        
        with tab1:
            self._render_analysis()
        
        with tab2:
            self._render_chat()
        
        with tab3:
            self._render_transactions()
    
    def _render_health_overview(self):
        """Render health metrics overview"""
        if not st.session_state.health_metrics:
            return
        
        metrics = st.session_state.health_metrics
        
        st.markdown("## 📊 Financial Health Dashboard")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Health", f"{metrics.overall_score:.1f}/100")
        
        with col2:
            st.metric("Risk Level", metrics.risk_level)
        
        with col3:
            balance = sum(tx.amount for tx in st.session_state.transactions)
            st.metric("Current Balance", f"${balance:,.2f}")
        
        with col4:
            st.metric("Transactions", len(st.session_state.transactions))
        
        # Health breakdown chart
        if metrics.overall_score > 0:
            component_scores = {
                'Cashflow': metrics.cashflow_score,
                'Savings Rate': metrics.savings_ratio_score,
                'Stability': metrics.spending_stability_score,
                'Emergency Fund': metrics.emergency_fund_score
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(component_scores.keys()),
                    y=list(component_scores.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )
            ])
            
            fig.update_layout(
                title="Health Score Breakdown",
                yaxis_title="Score (0-100)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Improvement suggestions
        if metrics.improvement_suggestions:
            st.markdown("### 💡 Improvement Suggestions")
            for suggestion in metrics.improvement_suggestions:
                st.info(f"• {suggestion}")
    
    def _render_analysis(self):
        """Render financial analysis"""
        transactions = st.session_state.transactions
        
        # Category breakdown
        expenses = [tx for tx in transactions if tx.amount < 0]
        
        if expenses:
            category_totals = {}
            for tx in expenses:
                category = tx.category.replace('_', ' ').title()
                amount = abs(tx.amount)
                category_totals[category] = category_totals.get(category, 0) + amount
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(category_totals.values()),
                    names=list(category_totals.keys()),
                    title="Spending by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly trend
                monthly_data = {}
                for tx in transactions:
                    month = tx.date[:7]  # YYYY-MM
                    if month not in monthly_data:
                        monthly_data[month] = {'income': 0, 'expenses': 0}
                    
                    if tx.amount > 0:
                        monthly_data[month]['income'] += tx.amount
                    else:
                        monthly_data[month]['expenses'] += abs(tx.amount)
                
                if len(monthly_data) > 1:
                    months = sorted(monthly_data.keys())
                    income_trend = [monthly_data[m]['income'] for m in months]
                    expense_trend = [monthly_data[m]['expenses'] for m in months]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=months, y=income_trend, name='Income', line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=months, y=expense_trend, name='Expenses', line=dict(color='red')))
                    fig.update_layout(title="Monthly Income vs Expenses")
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_chat(self):
        """Render chat interface"""
        st.markdown("## 💬 Chat with Your AI Financial Coach")
        
        # Display conversation history
        for chat in st.session_state.conversation_history[-5:]:  # Show last 5
            st.markdown(f"**You:** {chat['user_query']}")
            st.markdown(f"**AI:** {chat['ai_response']}")
            st.caption(f"Emotion: {chat['emotion']} | Topics: {', '.join(chat['topics'])}")
            st.divider()
        
        if not st.session_state.conversation_history:
            st.info("Start a conversation by asking a question in the sidebar!")
    
    def _render_transactions(self):
        """Render transaction table"""
        if not st.session_state.transactions:
            return
        
        st.markdown("## 📋 Transaction History")
        
        # Convert to DataFrame for display
        data = []
        for tx in st.session_state.transactions:
            data.append({
                'Date': tx.date,
                'Description': tx.description,
                'Amount': f"${tx.amount:,.2f}",
                'Category': tx.category.replace('_', ' ').title()
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

# ==============================================================================
# RUN APPLICATION
# ==============================================================================

def main():
    """Main function to run the application"""
    app = EmpowerFinApp()
    app.run()

if __name__ == "__main__":
    main()