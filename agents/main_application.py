# ==============================================================================
# main_application.py - Main Application Launcher
# ==============================================================================

import asyncio
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from pathlib import Path
import logging
import io

# Add all modules to path
sys.path.append(str(Path(__file__).parent))

from shared_models import (
    AgentMessage, SharedState, FinancialHealthMetrics, Transaction, UserProfile
)
from agent_orchestrator import OrchestratorService
from data_processing_agent import DataProcessingService
from health_calculation_agent import HealthCalculationService
from conversational_ai_agent import ConversationalAIService

# ==============================================================================
# STREAMLIT UI MANAGER
# ==============================================================================

class StreamlitUIManager:
    """Manages the Streamlit user interface for the multi-agent system"""
    
    def __init__(self):
        self.services = {}
        self.orchestrator = None
        self.is_initialized = False
    
    async def initialize_services(self):
        """Initialize all services"""
        if self.is_initialized:
            return
        
        try:
            # Initialize services
            self.services = {
                'orchestrator': OrchestratorService(),
                'data_processor': DataProcessingService(),
                'health_calculator': HealthCalculationService(),
                'conversational_ai': ConversationalAIService()
            }
            
            # Start orchestrator
            await self.services['orchestrator'].start()
            self.orchestrator = self.services['orchestrator'].orchestrator
            
            # Register agents with orchestrator
            await self._register_agents()
            
            self.is_initialized = True
            st.session_state.services_initialized = True
            
        except Exception as e:
            st.error(f"Failed to initialize services: {e}")
            logging.error(f"Service initialization error: {e}")
    
    async def _register_agents(self):
        """Register all agents with the orchestrator"""
        try:
            # Register data processing agent
            await self.orchestrator.register_agent(
                "DataProcessor", 
                ["process_file", "validate_data", "categorize_transactions"]
            )
            
            # Register health calculation agent
            await self.orchestrator.register_agent(
                "HealthCalculator",
                ["calculate_health", "update_metrics", "get_metrics"]
            )
            
            # Register conversational AI agent
            await self.orchestrator.register_agent(
                "ConversationalAI",
                ["chat", "analyze_emotion", "generate_advice"]
            )
            
            logging.info("All agents registered successfully")
            
        except Exception as e:
            logging.error(f"Agent registration failed: {e}")
    
    def render_main_dashboard(self):
        """Render the main application dashboard"""
        st.set_page_config(
            page_title="EmpowerFin Guardian 2.0 - Multi-Agent",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.markdown("# 🤖 EmpowerFin Guardian 2.0")
        st.markdown("*Multi-Agent Financial Intelligence Platform*")
        
        # Initialize services
        if 'services_initialized' not in st.session_state:
            st.session_state.services_initialized = False
        
        if not st.session_state.services_initialized:
            with st.spinner("🚀 Initializing AI Agent Services..."):
                try:
                    asyncio.run(self.initialize_services())
                except Exception as e:
                    st.error(f"Failed to initialize: {e}")
                    return
        
        # Service status indicator
        self._render_service_status()
        
        # Main interface
        self._render_sidebar_controls()
        self._render_main_content()
    
    def _render_service_status(self):
        """Render service status indicators"""
        if self.is_initialized and self.orchestrator:
            try:
                # Get system status
                status = asyncio.run(self.orchestrator.get_system_status())
                
                st.sidebar.markdown("## 🔧 System Status")
                
                # Orchestrator status
                orchestrator_status = status.get('orchestrator', {})
                if orchestrator_status.get('running', False):
                    st.sidebar.success("✅ Orchestrator: Online")
                else:
                    st.sidebar.error("❌ Orchestrator: Offline")
                
                # Agent status
                agents = status.get('agents', {})
                st.sidebar.markdown("### 🤖 Agents")
                
                for agent_name, agent_info in agents.items():
                    health = agent_info.get('health_status', 'unknown')
                    if health == 'healthy':
                        st.sidebar.success(f"✅ {agent_name}")
                    else:
                        st.sidebar.warning(f"⚠️ {agent_name}: {health}")
                
                # System stats
                stats = orchestrator_status.get('stats', {})
                st.sidebar.markdown("### 📊 Statistics")
                st.sidebar.metric("Messages Processed", stats.get('messages_processed', 0))
                st.sidebar.metric("Agents Registered", stats.get('agents_registered', 0))
                
            except Exception as e:
                st.sidebar.error(f"Status check failed: {e}")
        else:
            st.sidebar.warning("⚠️ Services not initialized")
    
    def _render_sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## 📁 File Upload")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Bank Statement",
            type=["csv", "xlsx", "xls"],
            help="Upload your bank statement in CSV or Excel format"
        )
        
        st.sidebar.markdown("## 🎯 Financial Goal")
        financial_goal = st.sidebar.text_area(
            "What's your financial goal?",
            value="Build emergency fund and improve savings rate",
            height=100
        )
        
        st.sidebar.markdown("## 💬 Chat with AI")
        user_query = st.sidebar.text_area(
            "Ask your AI financial team",
            placeholder="e.g., How can I reduce my spending?",
            height=100
        )
        
        # Action buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        with col2:
            chat_button = st.button("💬 Chat", use_container_width=True)
        
        # Process actions
        if analyze_button and uploaded_file and self.is_initialized:
            self._process_file_upload(uploaded_file, financial_goal)
        
        if chat_button and user_query and self.is_initialized:
            self._process_chat_query(user_query)
        
        # Store values in session state
        st.session_state.uploaded_file = uploaded_file
        st.session_state.financial_goal = financial_goal
        st.session_state.user_query = user_query
    
    def _process_file_upload(self, uploaded_file, financial_goal):
        """Process file upload through the agent system"""
        try:
            with st.spinner("🤖 AI Agents are processing your file..."):
                # Create user profile
                user_profile = {
                    "financial_goal": financial_goal,
                    "session_id": st.session_state.get("session_id", "web_session")
                }
                
                # Process through orchestrator
                success = asyncio.run(
                    self.orchestrator.process_file_upload(uploaded_file, user_profile)
                )
                
                if success:
                    st.success("✅ File processed successfully!")
                    # Wait a moment for processing to complete
                    import time
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ File processing failed")
                
        except Exception as e:
            st.error(f"Processing error: {e}")
            logging.error(f"File upload processing error: {e}")
    
    def _process_chat_query(self, query):
        """Process chat query through the agent system"""
        try:
            with st.spinner("🤖 AI is thinking..."):
                # Get current context
                context = self._get_current_context()
                
                # Process through orchestrator
                success = asyncio.run(
                    self.orchestrator.process_chat_query(query, context)
                )
                
                if success:
                    st.success("✅ Response generated!")
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Chat processing failed")
                
        except Exception as e:
            st.error(f"Chat error: {e}")
            logging.error(f"Chat processing error: {e}")
    
    def _get_current_context(self) -> Dict:
        """Get current context for chat processing"""
        try:
            if self.orchestrator:
                shared_state = self.orchestrator.shared_state
                return asyncio.run(shared_state.get_all())
            return {}
        except Exception as e:
            logging.error(f"Context retrieval error: {e}")
            return {}
    
    def _render_main_content(self):
        """Render main content area"""
        if not self.is_initialized:
            self._render_welcome_screen()
            return
        
        # Get current state
        try:
            current_state = self._get_current_context()
            processing_status = current_state.get("processing_status", "idle")
            
            if processing_status == "idle":
                self._render_welcome_screen()
            elif processing_status in ["data_processed", "health_calculated", "completed"]:
                self._render_analysis_results(current_state)
            else:
                st.info(f"🔄 Processing status: {processing_status}")
                
        except Exception as e:
            st.error(f"Error rendering content: {e}")
    
    def _render_welcome_screen(self):
        """Render welcome screen"""
        st.markdown("""
        ## Welcome to EmpowerFin Guardian 2.0! 🚀
        
        ### 🤖 **Your Multi-Agent Financial Intelligence Team**
        
        #### **🏗️ Distributed Agent Architecture**
        - **Agent Orchestrator**: Coordinates all agent communications
        - **Data Processing Agent**: Validates and categorizes transactions  
        - **Health Calculation Agent**: Computes comprehensive financial scores
        - **Conversational AI Agent**: Provides personalized financial advice
        
        #### **✨ Advanced Features**
        - **Real-time Communication**: Agents work together seamlessly
        - **Fault Tolerance**: System continues if individual agents fail
        - **Scalable Design**: Easy to add new specialized agents
        - **Load Balancing**: Automatically distributes work across agents
        
        #### **🎯 Intelligence Capabilities**
        - **Smart File Processing**: Auto-detects any bank statement format
        - **Behavioral Analysis**: Understands your spending psychology  
        - **Emotion-Aware AI**: Adapts responses to your emotional state
        - **Contextual Advice**: Remembers your conversation history
        
        ### 🚀 **Getting Started:**
        1. **Upload** your bank statement (any format supported)
        2. **Set** your financial goal in the sidebar
        3. **Chat** with your AI team about your finances
        4. **Get** personalized insights and recommendations
        
        ---
        
        *Powered by advanced multi-agent AI architecture for maximum reliability and intelligence* 🧠
        """)
    
    def _render_analysis_results(self, state: Dict):
        """Render analysis results"""
        # Financial Health Dashboard
        self._render_health_dashboard(state)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "💬 AI Chat", "🧠 Insights", "📋 Recommendations"])
        
        with tab1:
            self._render_financial_analysis(state)
        
        with tab2:
            self._render_chat_interface(state)
        
        with tab3:
            self._render_insights(state)
        
        with tab4:
            self._render_recommendations(state)
    
    def _render_health_dashboard(self, state: Dict):
        """Render financial health dashboard"""
        st.markdown("## 📊 Financial Health Dashboard")
        
        health_metrics = state.get("health_metrics", {})
        
        if not health_metrics:
            st.info("Upload your bank statement to see your financial health score")
            return
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_score = health_metrics.get('overall_score', 0)
            st.metric(
                "Overall Health",
                f"{overall_score:.1f}/100",
                delta=health_metrics.get('trend', 'Stable')
            )
        
        with col2:
            st.metric(
                "Risk Level",
                health_metrics.get('risk_level', 'Unknown')
            )
        
        with col3:
            transactions = state.get("transactions", [])
            balance = sum(tx.get('amount', 0) for tx in transactions)
            st.metric("Balance", f"${balance:,.2f}")
        
        with col4:
            agent_count = len(self.orchestrator.agent_registry.get_all_agents()) if self.orchestrator else 0
            st.metric("AI Agents", f"{agent_count}/4", delta="All systems operational")
        
        # Health breakdown chart
        if overall_score > 0:
            self._render_health_breakdown_chart(health_metrics)
    
    def _render_health_breakdown_chart(self, health_metrics: Dict):
        """Render health score breakdown chart"""
        component_scores = {
            'Cashflow': health_metrics.get('cashflow_score', 0),
            'Savings Rate': health_metrics.get('savings_ratio_score', 0),
            'Stability': health_metrics.get('spending_stability_score', 0),
            'Emergency Fund': health_metrics.get('emergency_fund_score', 0)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(component_scores.keys()),
                y=list(component_scores.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                text=[f"{score:.1f}" for score in component_scores.values()],
                textposition='auto'
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
    
    def _render_financial_analysis(self, state: Dict):
        """Render detailed financial analysis"""
        transactions = state.get("transactions", [])
        
        if not transactions:
            st.info("No transaction data available")
            return
        
        # Spending analysis
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_category_breakdown(transactions)
        
        with col2:
            self._render_spending_trends(transactions)
        
        # Transaction details
        st.markdown("### 📋 Recent Transactions")
        df = pd.DataFrame(transactions)
        if not df.empty:
            # Format for display
            df['amount'] = df['amount'].apply(lambda x: f"${x:,.2f}")
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Show last 20 transactions
            st.dataframe(
                df[['date', 'description', 'amount', 'category']].tail(20),
                use_container_width=True
            )
    
    def _render_category_breakdown(self, transactions: List[Dict]):
        """Render spending category breakdown"""
        expenses = [tx for tx in transactions if tx.get('amount', 0) < 0]
        
        if not expenses:
            st.info("No expense data available")
            return
        
        # Calculate category totals
        category_totals = {}
        for tx in expenses:
            category = tx.get('category', 'other').replace('_', ' ').title()
            amount = abs(tx.get('amount', 0))
            category_totals[category] = category_totals.get(category, 0) + amount
        
        # Create pie chart
        fig = px.pie(
            values=list(category_totals.values()),
            names=list(category_totals.keys()),
            title="Spending by Category"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_spending_trends(self, transactions: List[Dict]):
        """Render spending trends over time"""
        if not transactions:
            return
        
        # Group by month
        monthly_data = {}
        for tx in transactions:
            try:
                date = pd.to_datetime(tx['date'])
                month_key = date.strftime('%Y-%m')
                
                if month_key not in monthly_data:
                    monthly_data[month_key] = {'income': 0, 'expenses': 0}
                
                amount = tx.get('amount', 0)
                if amount > 0:
                    monthly_data[month_key]['income'] += amount
                else:
                    monthly_data[month_key]['expenses'] += abs(amount)
            except:
                continue
        
        if not monthly_data:
            return
        
        # Create trend chart
        months = sorted(monthly_data.keys())
        income_trend = [monthly_data[month]['income'] for month in months]
        expense_trend = [monthly_data[month]['expenses'] for month in months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=income_trend, name='Income', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=months, y=expense_trend, name='Expenses', line=dict(color='red')))
        
        fig.update_layout(
            title="Monthly Income vs Expenses",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_chat_interface(self, state: Dict):
        """Render chat interface"""
        st.markdown("## 💬 Chat with Your AI Financial Team")
        
        # Display conversation history
        conversation_history = state.get("conversation_history", [])
        
        if conversation_history:
            st.markdown("### Recent Conversations")
            
            for i, chat in enumerate(conversation_history[-5:]):  # Show last 5
                with st.container():
                    st.markdown(f"**You:** {chat.get('user_query', '')}")
                    st.markdown(f"**AI Team:** {chat.get('ai_response', '')}")
                    
                    # Show context
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"Emotion: {chat.get('emotion', 'neutral')}")
                    with col2:
                        stress = chat.get('stress_level', 0)
                        st.caption(f"Stress: {stress:.1f}/1.0")
                    with col3:
                        topics = chat.get('topics', [])
                        st.caption(f"Topics: {', '.join(topics)}")
                    
                    st.divider()
        
        # Current response
        current_response = state.get("personalized_response", "")
        if current_response:
            st.success(f"**Latest Response:** {current_response}")
    
    def _render_insights(self, state: Dict):
        """Render AI insights"""
        st.markdown("## 🧠 AI-Generated Insights")
        
        insights = state.get("insights", [])
        
        if insights:
            for insight in insights:
                insight_type = insight.get("type", "info")
                priority = insight.get("priority", "low")
                title = insight.get("title", "Insight")
                message = insight.get("message", "")
                
                # Display based on priority
                if priority == "high":
                    st.error(f"**{title}**\n\n{message}")
                elif priority == "medium":
                    st.warning(f"**{title}**\n\n{message}")
                else:
                    st.success(f"**{title}**\n\n{message}")
                
                # Show action items if available
                action_items = insight.get("action_items", [])
                if action_items:
                    with st.expander("💡 Action Items"):
                        for item in action_items:
                            st.write(f"• {item}")
        else:
            st.info("Upload your bank statement to see AI-generated insights!")
    
    def _render_recommendations(self, state: Dict):
        """Render recommendations"""
        st.markdown("## 📋 Personalized Recommendations")
        
        recommendations = state.get("recommendations", [])
        
        if recommendations:
            for rec in recommendations:
                priority = rec.get("priority", "low")
                title = rec.get("title", "Recommendation")
                description = rec.get("description", "")
                actions = rec.get("actions", [])
                timeline = rec.get("timeline", "")
                
                # Priority icons
                priority_icons = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }
                
                icon = priority_icons.get(priority, "🔵")
                
                with st.expander(f"{icon} {title}"):
                    st.write(description)
                    
                    if actions:
                        st.markdown("**Action Steps:**")
                        for action in actions:
                            st.write(f"• {action}")
                    
                    if timeline:
                        st.caption(f"⏱️ Timeline: {timeline}")
        else:
            st.info("Complete the analysis to see personalized recommendations!")

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    """Main application entry point"""
    try:
        # Initialize UI manager
        ui_manager = StreamlitUIManager()
        
        # Render main dashboard
        ui_manager.render_main_dashboard()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Main application error: {e}")
        
        # Show error details in expander
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())

# ==============================================================================
# CLI MODE
# ==============================================================================

async def run_cli_mode():
    """Run in CLI mode for testing"""
    print("EmpowerFin Guardian 2.0 - Multi-Agent System")
    print("=" * 50)
    
    # Initialize services
    ui_manager = StreamlitUIManager()
    await ui_manager.initialize_services()
    
    if ui_manager.is_initialized:
        print("✅ All services initialized successfully")
        
        # Show system status
        status = await ui_manager.orchestrator.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Orchestrator: {'Running' if status['orchestrator']['running'] else 'Stopped'}")
        print(f"  Agents: {len(status['agents'])}")
        print(f"  Messages Processed: {status['orchestrator']['stats']['messages_processed']}")
        
        print("\nPress Ctrl+C to stop...")
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            
        # Stop services
        for service in ui_manager.services.values():
            if hasattr(service, 'stop'):
                await service.stop()
    else:
        print("❌ Failed to initialize services")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run in CLI mode
        asyncio.run(run_cli_mode())
    else:
        # Run Streamlit app
        main()