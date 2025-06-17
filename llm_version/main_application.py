# ==============================================================================
# main_application.py - Main Streamlit Application Orchestrator
# ==============================================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime
from typing import Dict, List, Any

# Import all agents
from config_manager import llm_manager, app_config, session_manager
from data_models import EnhancedTransaction, FinancialHealthMetrics, ConversationEntry
from file_processor_agent import FileProcessorAgent, BatchFileProcessor
from categorization_agent import TransactionCategorizationAgent, CategorizationTrainer
from health_analyzer_agent import LLMHealthAnalyzer, HealthTrendAnalyzer
from conversational_agent import ConversationalAgent, QuickResponseGenerator, ConversationAnalytics


class EmpowerFinApplication:
    """Main application orchestrator"""
    
    def __init__(self):
        self.file_processor = FileProcessorAgent()
        self.batch_processor = BatchFileProcessor()
        self.categorizer = TransactionCategorizationAgent()
        self.health_analyzer = LLMHealthAnalyzer()
        self.trend_analyzer = HealthTrendAnalyzer()
        self.conversational_agent = ConversationalAgent()
        self.quick_responder = QuickResponseGenerator()
        self.conversation_analytics = ConversationAnalytics()
        self.categorization_trainer = CategorizationTrainer()
    
    def run(self):
        """Main application entry point"""
        self._setup_page_config()
        self._render_header()
        self._initialize_session()
        
        # Main application flow
        if st.session_state.transactions:
            self._render_main_dashboard()
        else:
            self._render_welcome_screen()
    
    def _setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title=f"{app_config.app_name} - v{app_config.version}",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _render_header(self):
        """Render application header"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"# 🤖 {app_config.app_name}")
            st.markdown(f"*{app_config.description}*")
        
        with col2:
            self._render_llm_status()
        
        with col3:
            if st.button("🔧 Settings"):
                self._show_settings_modal()
    
    def _render_llm_status(self):
        """Show LLM status indicator"""
        if llm_manager.is_available():
            st.success(f"🧠 LLM: {llm_manager.get_model_info()}")
        else:
            st.warning("⚠️ LLM: Not Available")
            if st.button("Setup LLM"):
                self._show_llm_setup()
    
    def _show_llm_setup(self):
        """Show LLM configuration setup"""
        with st.expander("🔧 LLM Configuration", expanded=True):
            st.markdown("""
            **Set your API keys as environment variables:**
            
            ```bash
            # For Groq (Recommended - Fast & Free Tier)
            export GROQ_API_KEY="your_groq_api_key"
            
            # For OpenAI  
            export OPENAI_API_KEY="your_openai_api_key"
            
            # For Anthropic Claude
            export ANTHROPIC_API_KEY="your_anthropic_api_key"
            ```
            
            **Current Status:**
            """)
            
            for provider, config in llm_manager.available_models.items():
                import os
                api_key = os.getenv(config['api_key_env'])
                status = "✅ Configured" if api_key else "❌ Not configured"
                st.write(f"- **{provider.title()}**: {status}")
            
            st.info("💡 **Tip**: Groq offers free API access with fast inference. Get your key at: https://console.groq.com/")
    
    def _show_settings_modal(self):
        """Show application settings"""
        with st.sidebar:
            st.markdown("## ⚙️ Application Settings")
            
            # LLM Settings
            st.markdown("### 🧠 LLM Configuration")
            if st.button("Test LLM Connection"):
                self._test_llm_connection()
            
            # Data Management
            st.markdown("### 📊 Data Management")
            if st.button("Clear All Data"):
                if st.checkbox("I understand this will delete all data"):
                    session_manager.clear_session()
                    st.success("All data cleared!")
                    st.rerun()
            
            # Export Options
            st.markdown("### 📁 Export Options")
            if st.session_state.transactions:
                if st.button("Export Transactions"):
                    self._export_transactions()
                
                if st.session_state.conversation_history:
                    if st.button("Export Conversations"):
                        self._export_conversations()
    
    def _test_llm_connection(self):
        """Test LLM connection"""
        if llm_manager.is_available():
            try:
                with st.spinner("Testing LLM connection..."):
                    from langchain.schema import SystemMessage, HumanMessage
                    messages = [
                        SystemMessage(content="You are a helpful assistant."),
                        HumanMessage(content="Respond with 'LLM connection successful!' if you receive this message.")
                    ]
                    response = llm_manager.get_client().invoke(messages)
                    st.success(f"✅ LLM Test Successful: {response.content}")
            except Exception as e:
                st.error(f"❌ LLM Test Failed: {e}")
        else:
            st.warning("⚠️ No LLM configured for testing")
    
    def _initialize_session(self):
        """Initialize session state"""
        session_manager.initialize_session_state()
    
    def _render_welcome_screen(self):
        """Render welcome screen for new users"""
        st.markdown("""
        ## Welcome to EmpowerFin Guardian 2.0! 🚀
        
        ### 🧠 **AI-Powered Financial Intelligence**
        
        Transform your financial management with cutting-edge LLM technology:
        
        #### **✨ Smart Features**
        - **🎯 Intelligent File Processing**: Automatically detects any bank statement format
        - **🏷️ AI Categorization**: Advanced transaction classification with confidence scoring
        - **🔍 Deep Health Analysis**: Comprehensive financial insights beyond basic metrics
        - **💬 Expert Conversations**: Context-aware financial coaching with memory
        - **📊 Personalized Recommendations**: Tailored action plans for your situation
        
        #### **🤖 AI Agents Working for You**
        - **File Processor**: Intelligently parses any bank statement
        - **Categorization Engine**: Smart transaction classification
        - **Health Analyzer**: Deep financial wellness insights
        - **Conversation Coach**: Expert financial advisor with personality
        
        ### 🚀 **Get Started:**
        """)
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📁 Upload Your Bank Statement")
            uploaded_files = st.file_uploader(
                "Choose files (CSV, Excel, or TXT)",
                type=["csv", "xlsx", "xls", "txt"],
                accept_multiple_files=True,
                help="Upload one or more bank statements - AI will automatically detect the format"
            )
            
            if uploaded_files:
                if len(uploaded_files) == 1:
                    if st.button("🧠 Process with AI", type="primary"):
                        self._process_single_file(uploaded_files[0])
                else:
                    if st.button("🧠 Process All Files with AI", type="primary"):
                        self._process_multiple_files(uploaded_files)
        
        with col2:
            st.markdown("#### 🎯 Quick Start Options")
            
            if st.button("📊 View Demo Data"):
                self._load_demo_data()
            
            if st.button("🧪 Test LLM Features"):
                self._show_llm_demo()
        
        # Sidebar with file processing status
        self._render_sidebar()
    
    def _render_main_dashboard(self):
        """Render main dashboard with all features"""
        # Sidebar
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Financial Health", 
            "🧠 AI Analysis", 
            "💬 AI Advisor", 
            "📈 Trends", 
            "🔧 Management"
        ])
        
        with tab1:
            self._render_health_dashboard()
        
        with tab2:
            self._render_ai_analysis()
        
        with tab3:
            self._render_ai_conversation()
        
        with tab4:
            self._render_trend_analysis()
        
        with tab5:
            self._render_data_management()
    
    def _render_sidebar(self):
        """Render comprehensive sidebar"""
        st.sidebar.markdown("## 📁 File Upload")
        
        # Single file upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Bank Statement",
            type=["csv", "xlsx", "xls", "txt"],
            help="AI will automatically detect the format"
        )
        
        if uploaded_file:
            if st.sidebar.button("🧠 Process with AI", type="primary"):
                self._process_single_file(uploaded_file)
        
        # Multiple file upload
        with st.sidebar.expander("📚 Batch Upload"):
            uploaded_files = st.file_uploader(
                "Upload Multiple Files",
                type=["csv", "xlsx", "xls", "txt"],
                accept_multiple_files=True,
                key="batch_upload"
            )
            
            if uploaded_files and len(uploaded_files) > 1:
                if st.button("🔄 Process All Files"):
                    self._process_multiple_files(uploaded_files)
        
        # AI Conversation Quick Start
        st.sidebar.markdown("## 💬 AI Financial Advisor")
        
        # Quick conversation starters
        if st.sidebar.button("🎯 Get Personalized Advice"):
            self._start_quick_conversation("personalized_advice")
        
        if st.sidebar.button("🔍 Analyze My Spending"):
            self._start_quick_conversation("spending_analysis")
        
        if st.sidebar.button("📈 Future Planning"):
            self._start_quick_conversation("future_planning")
        
        # Status indicators
        self._render_sidebar_status()
    
    def _render_sidebar_status(self):
        """Render status indicators in sidebar"""
        st.sidebar.markdown("## 📊 Status")
        
        # LLM Status
        if llm_manager.is_available():
            st.sidebar.success(f"🧠 AI: {llm_manager.get_model_info()}")
        else:
            st.sidebar.warning("⚠️ AI: Basic Mode")
        
        # Data Status
        if st.session_state.transactions:
            st.sidebar.metric("Transactions", len(st.session_state.transactions))
            
            # Analysis quality
            analysis_quality = "Basic"
            if st.session_state.health_metrics:
                analysis_quality = st.session_state.health_metrics.get('analysis_quality', 'basic').title()
            st.sidebar.metric("Analysis Quality", analysis_quality)
            
            # Health score
            if st.session_state.health_metrics:
                score = st.session_state.health_metrics['overall_score']
                st.sidebar.metric("Health Score", f"{score:.1f}/100")
        
        # Conversation Status
        if st.session_state.conversation_history:
            st.sidebar.metric("Conversations", len(st.session_state.conversation_history))
    
    def _process_single_file(self, uploaded_file):
        """Process a single uploaded file"""
        with st.spinner("🧠 AI is analyzing your bank statement..."):
            try:
                # Process file
                df, analysis_result = self.file_processor.process_uploaded_file(uploaded_file)
                
                if analysis_result.success and df is not None:
                    # Convert to transactions
                    transactions = self.file_processor.convert_to_enhanced_transactions(df)
                    
                    # Categorize transactions
                    categorized_transactions, cat_analysis = self.categorizer.categorize_transactions(transactions)
                    
                    # Analyze financial health
                    health_metrics = self.health_analyzer.analyze_financial_health(categorized_transactions)
                    
                    # Store in session
                    st.session_state.transactions = [tx.to_dict() for tx in categorized_transactions]
                    st.session_state.health_metrics = health_metrics.to_dict()
                    
                    # Show success message
                    st.success(f"✅ Successfully processed {len(categorized_transactions)} transactions!")
                    
                    # Show processing details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📁 File: {uploaded_file.name}")
                    with col2:
                        st.info(f"🧠 Confidence: {analysis_result.confidence.title()}")
                    with col3:
                        st.info(f"🎯 AI Quality: {health_metrics.analysis_quality.title()}")
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {analysis_result.error_message}")
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
    
    def _process_multiple_files(self, uploaded_files):
        """Process multiple uploaded files"""
        with st.spinner(f"🧠 AI is processing {len(uploaded_files)} files..."):
            try:
                results = self.batch_processor.process_multiple_files(uploaded_files)
                
                if results['successful']:
                    # Combine all transactions
                    all_transactions = results['combined_transactions']
                    
                    # Categorize all transactions
                    categorized_transactions, cat_analysis = self.categorizer.categorize_transactions(all_transactions)
                    
                    # Analyze financial health
                    health_metrics = self.health_analyzer.analyze_financial_health(categorized_transactions)
                    
                    # Store in session
                    st.session_state.transactions = [tx.to_dict() for tx in categorized_transactions]
                    st.session_state.health_metrics = health_metrics.to_dict()
                    
                    # Show results
                    st.success(f"✅ Successfully processed {len(results['successful'])} files with {len(categorized_transactions)} total transactions!")
                    
                    # Show processing summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", len(results['successful']))
                    with col2:
                        st.metric("Files Failed", len(results['failed']))
                    with col3:
                        st.metric("Total Transactions", len(categorized_transactions))
                    
                    if results['failed']:
                        with st.expander("⚠️ Failed Files"):
                            for failed in results['failed']:
                                st.error(f"❌ {failed['filename']}: {failed['error']}")
                    
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ No files were successfully processed")
                    
            except Exception as e:
                st.error(f"❌ Batch processing failed: {e}")
    
    def _start_quick_conversation(self, conversation_type):
        """Start a quick conversation based on type"""
        quick_queries = {
            'personalized_advice': "Based on my financial data, what are your top 3 personalized recommendations for improving my financial health?",
            'spending_analysis': "Analyze my spending patterns in detail and identify any concerning trends or opportunities for optimization.",
            'future_planning': "Based on my current financial health, what should my financial priorities be for the next 6 months?"
        }
        
        query = quick_queries.get(conversation_type, "How can you help me improve my financial situation?")
        self._process_conversation(query)
    
    def _process_conversation(self, query):
        """Process a conversation query"""
        try:
            with st.spinner("🧠 AI Advisor is thinking..."):
                context = {
                    'health_metrics': st.session_state.health_metrics,
                    'transactions': st.session_state.transactions
                }
                
                conversation_entry = self.conversational_agent.generate_response(query, context)
                
                st.success("✅ AI response generated!")
                time.sleep(0.5)
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Conversation failed: {e}")
    
    def _render_health_dashboard(self):
        """Render comprehensive health dashboard"""
        if not st.session_state.health_metrics:
            st.info("Upload your bank statement to see financial health analysis!")
            return
        
        metrics = st.session_state.health_metrics
        
        # Main health overview
        st.markdown("## 📊 Financial Health Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score_color = "🟢" if metrics['overall_score'] >= 70 else "🟡" if metrics['overall_score'] >= 40 else "🔴"
            st.metric(
                f"{score_color} Overall Health",
                f"{metrics['overall_score']:.1f}/100",
                delta=f"Risk: {metrics['risk_level']}"
            )
        
        with col2:
            st.metric("Analysis Quality", metrics.get('analysis_quality', 'basic').title())
        
        with col3:
            balance = sum(tx['amount'] for tx in st.session_state.transactions)
            st.metric("Current Balance", f"${balance:,.2f}")
        
        with col4:
            ai_status = "🧠 AI Enhanced" if llm_manager.is_available() else "📊 Basic Mode"
            st.metric("AI Status", ai_status)
        
        # Component scores chart
        self._render_health_components_chart(metrics)
        
        # Quick insights
        self._render_quick_health_insights(metrics)
    
    def _render_health_components_chart(self, metrics):
        """Render health components chart"""
        component_scores = {
            'Cashflow': metrics['cashflow_score'],
            'Savings Rate': metrics['savings_ratio_score'],
            'Stability': metrics['spending_stability_score'],
            'Emergency Fund': metrics['emergency_fund_score']
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
        
        analysis_quality = metrics.get('analysis_quality', 'basic')
        title_suffix = " (AI Enhanced)" if analysis_quality == 'llm_enhanced' else " (Basic Analysis)"
        
        fig.update_layout(
            title=f"Financial Health Component Scores{title_suffix}",
            yaxis_title="Score (0-100)",
            yaxis=dict(range=[0, 100]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_quick_health_insights(self, metrics):
        """Render quick health insights"""
        col1, col2 = st.columns(2)
        
        with col1:
            if 'llm_insights' in metrics and metrics['llm_insights'].get('strengths'):
                st.success("**💪 Key Strengths:**")
                for strength in metrics['llm_insights']['strengths'][:3]:
                    st.write(f"• {strength}")
        
        with col2:
            if 'llm_insights' in metrics and metrics['llm_insights'].get('weaknesses'):
                st.warning("**⚠️ Areas for Improvement:**")
                for weakness in metrics['llm_insights']['weaknesses'][:3]:
                    st.write(f"• {weakness}")
        
        # Overall assessment
        if 'llm_insights' in metrics and metrics['llm_insights'].get('overall_assessment'):
            st.info(f"**🎯 AI Assessment:** {metrics['llm_insights']['overall_assessment']}")
    
    def _render_ai_analysis(self):
        """Render AI analysis tab"""
        st.markdown("## 🧠 AI-Powered Analysis")
        
        if not st.session_state.transactions:
            st.info("Upload your bank statement to see AI analysis!")
            return
        
        # Analysis quality indicator
        if llm_manager.is_available():
            st.success("🧠 **Enhanced AI Analysis Active**")
        else:
            st.warning("⚠️ **Basic Analysis Mode** - Enable LLM for advanced insights")
        
        # Transaction categorization analysis
        self._render_categorization_analysis()
        
        # Spending pattern analysis
        self._render_spending_analysis()
        
        # AI insights and recommendations
        if st.session_state.health_metrics.get('llm_insights'):
            self._render_ai_insights()
        
        if st.session_state.health_metrics.get('llm_recommendations'):
            self._render_ai_recommendations()
    
    def _render_categorization_analysis(self):
        """Render categorization analysis"""
        st.markdown("### 🏷️ Smart Transaction Categorization")
        
        transactions = st.session_state.transactions
        df = pd.DataFrame(transactions)
        
        if not df.empty:
            # Category breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                # Category distribution
                category_counts = df['category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Transaction Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                if 'confidence' in df.columns:
                    avg_confidence = df['confidence'].mean()
                    st.metric("Average AI Confidence", f"{avg_confidence:.1%}")
                    
                    # Confidence histogram
                    fig = px.histogram(
                        df, 
                        x='confidence', 
                        title="AI Confidence Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Show some categorization examples
        if 'llm_reasoning' in df.columns:
            st.markdown("### 🧠 AI Reasoning Examples")
            
            examples = df[df['llm_reasoning'].notna()].head(5)
            for _, tx in examples.iterrows():
                with st.expander(f"Transaction: {tx['description'][:50]}..."):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Description:** {tx['description']}")
                        st.write(f"**AI Reasoning:** {tx['llm_reasoning']}")
                    with col2:
                        st.write(f"**Category:** {tx['category']}")
                        st.write(f"**Confidence:** {tx.get('confidence', 0):.1%}")
    
    def _render_spending_analysis(self):
        """Render spending pattern analysis"""
        st.markdown("### 💰 Spending Pattern Analysis")
        
        transactions = st.session_state.transactions
        df = pd.DataFrame(transactions)
        
        # Spending by category
        expense_df = df[df['amount'] < 0].copy()
        if not expense_df.empty:
            expense_df['amount_abs'] = expense_df['amount'].abs()
            
            category_spending = expense_df.groupby('category')['amount_abs'].sum().sort_values(ascending=False)
            
            # Top spending categories chart
            fig = go.Figure(data=[
                go.Bar(
                    y=category_spending.index[:10],
                    x=category_spending.values[:10],
                    orientation='h',
                    marker_color='lightcoral'
                )
            ])
            fig.update_layout(
                title="Top 10 Spending Categories",
                xaxis_title="Amount ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly spending trend
            df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_spending = expense_df.groupby('month')['amount_abs'].sum()
            
            if len(monthly_spending) > 1:
                fig = px.line(
                    x=monthly_spending.index.astype(str),
                    y=monthly_spending.values,
                    title="Monthly Spending Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_ai_insights(self):
        """Render AI insights section"""
        st.markdown("### 🧠 AI Financial Insights")
        
        insights = st.session_state.health_metrics['llm_insights']
        
        # Overall assessment
        if insights.get('overall_assessment'):
            st.markdown("#### 🎯 Overall Assessment")
            st.write(insights['overall_assessment'])
        
        # Detailed insights in columns
        col1, col2 = st.columns(2)
        
        with col1:
            if insights.get('strengths'):
                st.markdown("#### 💪 Financial Strengths")
                for i, strength in enumerate(insights['strengths'], 1):
                    st.success(f"{i}. {strength}")
            
            if insights.get('behavioral_insights'):
                st.markdown("#### 🧠 Behavioral Insights")
                for i, insight in enumerate(insights['behavioral_insights'], 1):
                    st.info(f"{i}. {insight}")
        
        with col2:
            if insights.get('weaknesses'):
                st.markdown("#### 🔧 Areas for Improvement")
                for i, weakness in enumerate(insights['weaknesses'], 1):
                    st.warning(f"{i}. {weakness}")
            
            if insights.get('risk_factors'):
                st.markdown("#### ⚠️ Risk Factors")
                for i, risk in enumerate(insights['risk_factors'], 1):
                    st.error(f"{i}. {risk}")
        
        # Trend analysis
        if insights.get('trend_analysis'):
            st.markdown("#### 📈 Trend Analysis")
            st.write(insights['trend_analysis'])
    
    def _render_ai_recommendations(self):
        """Render AI recommendations section"""
        st.markdown("### 📋 AI Recommendations")
        
        recommendations = st.session_state.health_metrics['llm_recommendations']
        
        # Immediate actions
        if recommendations.get('immediate_actions'):
            st.markdown("#### 🚨 Immediate Actions (Next 30 Days)")
            for i, action in enumerate(recommendations['immediate_actions'], 1):
                priority_colors = {
                    'critical': 'error',
                    'high': 'warning',
                    'medium': 'info',
                    'low': 'success'
                }
                
                priority = action.get('priority', 'medium')
                color = priority_colors.get(priority, 'info')
                
                with getattr(st, color)(f"Action {i}"):
                    st.write(f"**Do:** {action.get('action', 'No action specified')}")
                    st.write(f"**Impact:** {action.get('impact', 'Not specified')}")
                    if action.get('effort'):
                        st.write(f"**Effort:** {action.get('effort', 'Not specified')}")
        
        # Short-term goals
        if recommendations.get('short_term_goals'):
            st.markdown("#### 🎯 Short-term Goals (3-6 Months)")
            for i, goal in enumerate(recommendations['short_term_goals'], 1):
                with st.expander(f"Goal {i}: {goal.get('action', 'Goal')[:50]}..."):
                    st.write(f"**Goal:** {goal.get('action', 'No goal specified')}")
                    st.write(f"**Impact:** {goal.get('impact', 'Not specified')}")
                    st.write(f"**Priority:** {goal.get('priority', 'Not specified')}")
        
        # Optimization opportunities
        if recommendations.get('optimization_opportunities'):
            st.markdown("#### ⚡ Optimization Opportunities")
            for i, opportunity in enumerate(recommendations['optimization_opportunities'], 1):
                with st.success(f"Opportunity {i}"):
                    st.write(f"**Optimize:** {opportunity.get('action', 'No action specified')}")
                    st.write(f"**Benefit:** {opportunity.get('impact', 'Not specified')}")
    
    def _render_ai_conversation(self):
        """Render AI conversation interface"""
        st.markdown("## 💬 AI Financial Advisor")
        
        if not llm_manager.is_available():
            st.warning("⚠️ **LLM not available** - Enable AI for intelligent conversations")
            return
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### Recent Conversations")
            
            for i, chat in enumerate(reversed(st.session_state.conversation_history[-5:])):
                with st.container():
                    # User message
                    st.markdown(f"**🧑 You:** {chat['user_query']}")
                    
                    # AI response
                    st.markdown(f"**🤖 AI Advisor:** {chat['ai_response']}")
                    
                    # Metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        emotion_emoji = {"stressed": "😰", "confident": "😊", "confused": "🤔", "excited": "🎉"}.get(chat.get('emotion', 'neutral'), "😐")
                        st.caption(f"{emotion_emoji} Emotion: {chat.get('emotion', 'neutral')}")
                    with col2:
                        st.caption(f"🎯 Intent: {chat.get('intent', 'unknown')}")
                    with col3:
                        st.caption(f"🏷️ Topics: {', '.join(chat.get('topics', []))}")
                    with col4:
                        confidence_emoji = {"high": "🔥", "medium": "👍", "low": "🤷"}.get(chat.get('confidence', 'medium'), "👍")
                        st.caption(f"{confidence_emoji} Confidence: {chat.get('confidence', 'medium')}")
                    
                    st.divider()
        
        # Chat input
        st.markdown("### Ask Your AI Advisor")
        
        with st.form(key="ai_chat_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_query = st.text_area(
                    "Type your financial question:",
                    placeholder="e.g., Based on my spending patterns, what are my biggest financial vulnerabilities?",
                    height=100
                )
            
            with col2:
                st.markdown("**Quick Questions:**")
                if st.form_submit_button("🔍 Deep Analysis"):
                    user_query = "Perform a comprehensive analysis of my financial data and provide specific recommendations."
                
                if st.form_submit_button("📈 Future Planning"):
                    user_query = "Based on my current situation, create a detailed financial plan for the next 12 months."
                
                if st.form_submit_button("⚠️ Risk Assessment"):
                    user_query = "Analyze my financial risks and vulnerabilities. How should I prepare for potential challenges?"
            
            submitted = st.form_submit_button("🧠 Ask AI Advisor", type="primary")
            
            if submitted and user_query.strip():
                self._process_conversation(user_query.strip())
    
    def _render_trend_analysis(self):
        """Render trend analysis tab"""
        st.markdown("## 📈 Financial Trends Analysis")
        
        if not st.session_state.transactions:
            st.info("Upload your bank statement to see trend analysis!")
            return
        
        # Convert transactions for analysis
        transactions = [EnhancedTransaction.from_dict(tx) for tx in st.session_state.transactions]
        
        # Perform trend analysis
        trend_results = self.trend_analyzer.analyze_trends(transactions)
        
        if trend_results.get('status') == 'success':
            self._render_trend_results(trend_results)
        else:
            st.warning(f"Trend analysis unavailable: {trend_results.get('status', 'unknown error')}")
    
    def _render_trend_results(self, trend_results):
        """Render trend analysis results"""
        trend_analysis = trend_results['trend_analysis']
        period_metrics = trend_results['period_metrics']
        
        # Trend overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trajectory = trend_analysis['trajectory']
            trajectory_emoji = {
                'strongly_improving': '📈🔥',
                'improving': '📈',
                'stable': '➡️',
                'declining': '📉',
                'strongly_declining': '📉💥'
            }.get(trajectory, '➡️')
            st.metric("Trajectory", f"{trajectory_emoji} {trajectory.replace('_', ' ').title()}")
        
        with col2:
            st.metric("Overall Trend", f"{trend_analysis['overall_trend']:+.1f}")
        
        with col3:
            st.metric("Best Period", trend_analysis['best_period'])
        
        with col4:
            st.metric("Volatility", f"{trend_analysis['volatility']:.1f}")
        
        # Trend charts
        if len(period_metrics) > 1:
            # Health score over time
            periods = [p['period'] for p in period_metrics]
            scores = [p['metrics']['overall_score'] for p in period_metrics]
            
            fig = px.line(
                x=periods,
                y=scores,
                title="Financial Health Score Trend",
                markers=True
            )
            fig.update_layout(yaxis_title="Health Score", yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
            
            # Component trends
            components = ['cashflow_score', 'savings_ratio_score', 'spending_stability_score', 'emergency_fund_score']
            component_names = ['Cashflow', 'Savings Rate', 'Stability', 'Emergency Fund']
            
            fig = go.Figure()
            for component, name in zip(components, component_names):
                values = [p['metrics'][component] for p in period_metrics]
                fig.add_trace(go.Scatter(x=periods, y=values, mode='lines+markers', name=name))
            
            fig.update_layout(
                title="Component Score Trends",
                yaxis_title="Score",
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # AI trend insights
        if trend_results.get('trend_analysis', {}).get('llm_insights'):
            st.markdown("### 🧠 AI Trend Insights")
            insights = trend_results['trend_analysis']['llm_insights']
            
            if insights.get('trajectory_assessment'):
                st.info(f"**Trajectory Assessment:** {insights['trajectory_assessment']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if insights.get('positive_developments'):
                    st.success("**Positive Developments:**")
                    for dev in insights['positive_developments']:
                        st.write(f"• {dev}")
            
            with col2:
                if insights.get('warning_signs'):
                    st.warning("**Warning Signs:**")
                    for warning in insights['warning_signs']:
                        st.write(f"• {warning}")
    
    def _render_data_management(self):
        """Render data management tab"""
        st.markdown("## 🔧 Data Management")
        
        # Transaction management
        if st.session_state.transactions:
            st.markdown("### 📊 Transaction Data")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", len(st.session_state.transactions))
            with col2:
                dates = [tx['date'] for tx in st.session_state.transactions]
                st.metric("Date Range", f"{min(dates)} to {max(dates)}")
            with col3:
                if st.button("🗑️ Clear Transactions"):
                    st.session_state.transactions = []
                    st.session_state.health_metrics = None
                    st.rerun()
            
            # Transaction table
            with st.expander("📋 View Transaction Data"):
                df = pd.DataFrame(st.session_state.transactions)
                st.dataframe(df, use_container_width=True)
            
            # Export options
            st.markdown("### 📁 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export to CSV"):
                    self._export_transactions()
            
            with col2:
                if st.button("📊 Export Health Report"):
                    self._export_health_report()
        
        # Conversation management
        if st.session_state.conversation_history:
            st.markdown("### 💬 Conversation Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversations", len(st.session_state.conversation_history))
            with col2:
                if st.button("🗑️ Clear Conversations"):
                    st.session_state.conversation_history = []
                    st.rerun()
            
            # Conversation analytics
            analytics = self.conversation_analytics.analyze_user_engagement(
                [ConversationEntry.from_dict(conv) for conv in st.session_state.conversation_history]
            )
            
            if analytics:
                st.markdown("#### 📈 Conversation Analytics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Engagement Rate", f"{analytics.get('engagement_rate', 0):.1%}")
                with col2:
                    st.metric("Avg Query Length", f"{analytics.get('avg_query_length', 0):.0f} chars")
                with col3:
                    st.metric("Topic Diversity", analytics.get('topic_diversity', 0))
    
    def _export_transactions(self):
        """Export transactions to CSV"""
        if st.session_state.transactions:
            df = pd.DataFrame(st.session_state.transactions)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Transactions CSV",
                data=csv,
                file_name=f"empowerfin_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _export_conversations(self):
        """Export conversation history"""
        if st.session_state.conversation_history:
            df = pd.DataFrame(st.session_state.conversation_history)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="💬 Download Conversations CSV",
                data=csv,
                file_name=f"empowerfin_conversations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def _export_health_report(self):
        """Export comprehensive health report"""
        if st.session_state.health_metrics:
            import json
            
            report = {
                'export_date': datetime.now().isoformat(),
                'health_metrics': st.session_state.health_metrics,
                'transaction_summary': {
                    'total_transactions': len(st.session_state.transactions),
                    'date_range': f"{min(tx['date'] for tx in st.session_state.transactions)} to {max(tx['date'] for tx in st.session_state.transactions)}"
                }
            }
            
            json_str = json.dumps(report, indent=2)
            
            st.download_button(
                label="📊 Download Health Report JSON",
                data=json_str,
                file_name=f"empowerfin_health_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    def _load_demo_data(self):
        """Load demo data for testing"""
        # This would load sample transaction data
        st.info("Demo data loading would be implemented here")
    
    def _show_llm_demo(self):
        """Show LLM capabilities demo"""
        st.info("LLM feature demo would be implemented here")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main application entry point"""
    app = EmpowerFinApplication()
    app.run()


if __name__ == "__main__":
    main()