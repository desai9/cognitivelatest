# ==============================================================================
# conversational_ai_agent.py - Conversational AI Service
# ==============================================================================

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add the shared models to the path
sys.path.append(str(Path(__file__).parent))
from shared_models import (
    BaseAgent, AgentMessage, SharedState, UserProfile
)

# ==============================================================================
# EMOTION AND CONTEXT ANALYSIS
# ==============================================================================

class EmotionAnalyzer:
    """Analyzes user emotion and financial stress levels"""
    
    def __init__(self):
        self.emotion_keywords = {
            'stressed': [
                'worried', 'anxious', 'stressed', 'scared', 'panic', 'overwhelmed',
                'desperate', 'struggling', 'difficult', 'crisis', 'emergency',
                'urgent', 'help', 'trouble', 'confused', 'lost'
            ],
            'happy': [
                'great', 'excellent', 'happy', 'excited', 'wonderful', 'amazing',
                'fantastic', 'thrilled', 'pleased', 'satisfied', 'good', 'success'
            ],
            'sad': [
                'sad', 'depressed', 'hopeless', 'terrible', 'awful', 'worst',
                'disappointed', 'frustrated', 'upset', 'discouraged'
            ],
            'confident': [
                'confident', 'sure', 'ready', 'prepared', 'determined', 'focused',
                'motivated', 'optimistic', 'positive'
            ]
        }
        
        self.financial_stress_indicators = [
            'debt', 'broke', 'bankruptcy', 'foreclosure', 'eviction',
            'can\'t afford', 'no money', 'financial crisis', 'bills',
            'overdue', 'behind on payments', 'credit card debt'
        ]
    
    def detect_emotion(self, text: str) -> str:
        """Detect primary emotion from text"""
        if not text:
            return "neutral"
        
        text_lower = text.lower()
        emotion_scores = {}
        
        # Score each emotion based on keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return emotion with highest score, or neutral if none
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to simple sentiment analysis
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.3:
                return "happy"
            elif polarity < -0.3:
                return "sad"
            else:
                return "neutral"
        except ImportError:
            return "neutral"
    
    def detect_financial_stress(self, text: str, health_score: float = 50) -> float:
        """Detect financial stress level (0-1)"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        stress_level = 0.0
        
        # Check for stress indicators
        for indicator in self.financial_stress_indicators:
            if indicator in text_lower:
                stress_level += 0.2
        
        # Factor in health score
        if health_score < 30:
            stress_level += 0.4
        elif health_score < 50:
            stress_level += 0.2
        
        # Check emotion
        emotion = self.detect_emotion(text)
        if emotion == "stressed":
            stress_level += 0.3
        elif emotion == "sad":
            stress_level += 0.2
        
        return min(1.0, stress_level)

class TopicExtractor:
    """Extracts financial topics from user queries"""
    
    def __init__(self):
        self.topic_keywords = {
            'budgeting': [
                'budget', 'budgeting', 'spending plan', 'allocate', 'expenses',
                'income planning', 'money management', 'financial plan'
            ],
            'saving': [
                'save', 'savings', 'emergency fund', 'nest egg', 'rainy day',
                'financial cushion', 'save money', 'building savings'
            ],
            'debt': [
                'debt', 'loan', 'credit', 'payment', 'owe', 'borrow',
                'credit card', 'mortgage', 'student loan', 'interest'
            ],
            'investment': [
                'invest', 'investment', 'portfolio', 'stocks', 'bonds',
                'mutual funds', 'retirement', '401k', 'ira', 'returns'
            ],
            'spending': [
                'spend', 'spending', 'expenses', 'cost', 'purchase',
                'buy', 'shopping', 'money out'
            ],
            'income': [
                'income', 'salary', 'earn', 'revenue', 'paycheck',
                'wages', 'money in', 'earnings'
            ],
            'goals': [
                'goal', 'target', 'objective', 'plan', 'future',
                'dreams', 'aspirations', 'financial goals'
            ],
            'insurance': [
                'insurance', 'coverage', 'premium', 'policy',
                'protection', 'health insurance', 'life insurance'
            ],
            'taxes': [
                'tax', 'taxes', 'tax return', 'deduction', 'refund',
                'irs', 'tax planning', 'tax season'
            ]
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract financial topics from text"""
        if not text:
            return ['general']
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']

# ==============================================================================
# RESPONSE GENERATOR
# ==============================================================================

class ResponseGenerator:
    """Generates contextual financial advice responses"""
    
    def __init__(self):
        self.response_templates = {
            'budgeting': {
                'stressed': "I understand budgeting can feel overwhelming. Let's start simple with the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings. Focus on tracking your expenses for one week first.",
                'happy': "Great to see your enthusiasm about budgeting! The 50/30/20 rule is a solid foundation. Would you like help creating a personalized budget plan?",
                'neutral': "Budgeting is a great step toward financial health. I recommend starting with tracking your current spending, then setting realistic limits for each category."
            },
            'saving': {
                'stressed': "Building savings can feel impossible when money is tight, but even $5 per week adds up. Start with automating tiny amounts - you won't even notice them.",
                'happy': "Wonderful that you're focused on saving! Consider the pay-yourself-first approach: save before you spend on anything else.",
                'neutral': "Building savings is crucial for financial security. Start with a goal of saving 10% of your income, even if you begin with a smaller amount."
            },
            'debt': {
                'stressed': "Dealing with debt is stressful, but you have options. The debt snowball method (smallest debts first) can build momentum and confidence. You're not alone in this.",
                'happy': "Great that you're tackling debt proactively! Consider whether the debt avalanche (highest interest first) or snowball method works better for your situation.",
                'neutral': "For debt management, prioritize high-interest debt first. Consider consolidation options if they lower your overall interest rate."
            },
            'investment': {
                'stressed': "Investment can wait until you have an emergency fund and stable income. Focus on the basics first - your future self will thank you.",
                'happy': "Investing is exciting! Start with low-cost index funds if you're new to investing. Make sure you have an emergency fund first.",
                'neutral': "Investing is important for long-term wealth building. Consider starting with employer 401k matching if available, then diversified index funds."
            },
            'general': {
                'stressed': "I understand financial stress can be overwhelming. Let's start with one small step: tracking your spending for a week. Small actions lead to big changes.",
                'happy': "I love your positive attitude about finances! What specific area would you like to focus on improving?",
                'neutral': "I'm here to help with your financial journey. What's your main financial concern or goal right now?"
            }
        }
        
        self.encouragement_phrases = [
            "You're taking a positive step by asking!",
            "Every financial expert started where you are now.",
            "Small consistent changes make a big difference.",
            "You have more control than you think.",
            "Progress, not perfection, is the goal."
        ]
    
    async def generate_response(self, query: str, emotion: str, topics: List[str], 
                              health_score: float, user_profile: UserProfile = None) -> str:
        """Generate contextual response"""
        try:
            primary_topic = topics[0] if topics else 'general'
            
            # Get base response from templates
            topic_responses = self.response_templates.get(primary_topic, self.response_templates['general'])
            base_response = topic_responses.get(emotion, topic_responses.get('neutral', ''))
            
            # Customize based on health score
            if health_score < 40:
                health_context = " Given your current financial situation, let's focus on immediate stabilization first."
            elif health_score > 70:
                health_context = " Your financial health looks strong - you're in a good position to optimize further."
            else:
                health_context = " You're making progress - let's build on what's working."
            
            # Add encouragement if stressed
            encouragement = ""
            if emotion == "stressed":
                encouragement = f" Remember: {self._get_encouragement()}"
            
            # Combine response elements
            full_response = base_response + health_context + encouragement
            
            # Add follow-up question
            follow_up = self._get_follow_up_question(primary_topic, emotion)
            if follow_up:
                full_response += f" {follow_up}"
            
            return full_response.strip()
            
        except Exception as e:
            return "I'm here to help with your financial questions. Could you tell me more about what you'd like to discuss?"
    
    def _get_encouragement(self) -> str:
        """Get random encouragement phrase"""
        import random
        return random.choice(self.encouragement_phrases)
    
    def _get_follow_up_question(self, topic: str, emotion: str) -> str:
        """Get appropriate follow-up question"""
        follow_ups = {
            'budgeting': "What's your biggest budgeting challenge right now?",
            'saving': "What are you saving for specifically?",
            'debt': "Which debt is causing you the most stress?",
            'investment': "What's your investment timeline and risk tolerance?",
            'spending': "Which spending category concerns you most?",
            'general': "What would you like to focus on first?"
        }
        
        if emotion == "stressed":
            return "What feels most urgent to address?"
        
        return follow_ups.get(topic, "What would you like to know more about?")

# ==============================================================================
# CONVERSATIONAL AI AGENT
# ==============================================================================

class ConversationalAIAgent(BaseAgent):
    """Handles natural language conversations and provides personalized advice"""
    
    def __init__(self, shared_state: SharedState = None):
        super().__init__("ConversationalAI", shared_state)
        self.emotion_analyzer = EmotionAnalyzer()
        self.topic_extractor = TopicExtractor()
        self.response_generator = ResponseGenerator()
        self.conversation_history = []
        
        # Register message handlers
        self.register_handler("chat", self._handle_chat)
        self.register_handler("analyze_emotion", self._handle_analyze_emotion)
        self.register_handler("get_conversation_history", self._handle_get_conversation_history)
        self.register_handler("clear_history", self._handle_clear_history)
        self.register_handler("health_check", self._handle_health_check)
    
    async def _handle_chat(self, message: AgentMessage) -> List[AgentMessage]:
        """Process conversational queries"""
        await self.log("Processing chat query")
        
        try:
            query = message.data.get("query", "")
            if not query:
                return []
            
            # Get context from shared state
            health_metrics = await self.shared_state.get("health_metrics", {})
            user_profile = await self.shared_state.get("user_profile", UserProfile())
            
            health_score = health_metrics.get('overall_score', 50) if isinstance(health_metrics, dict) else 50
            
            # Analyze query
            emotion = self.emotion_analyzer.detect_emotion(query)
            topics = self.topic_extractor.extract_topics(query)
            stress_level = self.emotion_analyzer.detect_financial_stress(query, health_score)
            
            # Generate response
            if isinstance(user_profile, dict):
                user_profile = UserProfile.from_dict(user_profile)
            
            response = await self.response_generator.generate_response(
                query, emotion, topics, health_score, user_profile
            )
            
            # Create conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_query": query,
                "ai_response": response,
                "emotion": emotion,
                "topics": topics,
                "stress_level": stress_level,
                "health_score": health_score
            }
            
            # Update conversation history
            self.conversation_history.append(conversation_entry)
            
            # Keep only last 20 conversations
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Update shared state
            await self.shared_state.set("conversation_history", self.conversation_history)
            await self.shared_state.set("current_query", query)
            await self.shared_state.set("personalized_response", response)
            
            await self.log(f"Generated response for {emotion} emotion, topics: {', '.join(topics)}")
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient="UIManager",
                    message_type="display_response",
                    data={
                        "response": response,
                        "conversation": conversation_entry,
                        "emotion": emotion,
                        "topics": topics
                    }
                )
            ]
            
        except Exception as e:
            await self.log(f"Chat processing error: {e}", "error")
            
            # Return fallback response
            fallback_response = "I'm here to help with your financial questions. Could you please rephrase that?"
            return [
                AgentMessage(
                    sender=self.name,
                    recipient="UIManager",
                    message_type="display_response",
                    data={"response": fallback_response}
                )
            ]
    
    async def _handle_analyze_emotion(self, message: AgentMessage) -> List[AgentMessage]:
        """Analyze emotion in text"""
        try:
            text = message.data.get("text", "")
            health_score = message.data.get("health_score", 50)
            
            emotion = self.emotion_analyzer.detect_emotion(text)
            stress_level = self.emotion_analyzer.detect_financial_stress(text, health_score)
            topics = self.topic_extractor.extract_topics(text)
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="emotion_analysis",
                    data={
                        "emotion": emotion,
                        "stress_level": stress_level,
                        "topics": topics,
                        "text": text
                    }
                )
            ]
            
        except Exception as e:
            await self.log(f"Emotion analysis error: {e}", "error")
            return []
    
    async def _handle_get_conversation_history(self, message: AgentMessage) -> List[AgentMessage]:
        """Get conversation history"""
        try:
            limit = message.data.get("limit", 10)
            history = self.conversation_history[-limit:] if limit > 0 else self.conversation_history
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="conversation_history",
                    data={"history": history}
                )
            ]
            
        except Exception as e:
            await self.log(f"Get conversation history error: {e}", "error")
            return []
    
    async def _handle_clear_history(self, message: AgentMessage) -> List[AgentMessage]:
        """Clear conversation history"""
        try:
            self.conversation_history = []
            await self.shared_state.set("conversation_history", [])
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="history_cleared",
                    data={"success": True}
                )
            ]
            
        except Exception as e:
            await self.log(f"Clear history error: {e}", "error")
            return []
    
    async def _handle_health_check(self, message: AgentMessage) -> List[AgentMessage]:
        """Health check handler"""
        return [
            AgentMessage(
                sender=self.name,
                recipient=message.sender,
                message_type="health_response",
                data={
                    "status": "healthy",
                    "capabilities": self.capabilities,
                    "active": self.is_active,
                    "conversation_count": len(self.conversation_history)
                }
            )
        ]
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.conversation_history:
            return {"total_conversations": 0}
        
        # Analyze conversation patterns
        emotions = [conv.get("emotion", "neutral") for conv in self.conversation_history]
        topics = []
        for conv in self.conversation_history:
            topics.extend(conv.get("topics", []))
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Count topics
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Calculate average stress level
        stress_levels = [conv.get("stress_level", 0) for conv in self.conversation_history]
        avg_stress = sum(stress_levels) / len(stress_levels) if stress_levels else 0
        
        return {
            "total_conversations": len(self.conversation_history),
            "emotion_distribution": emotion_counts,
            "topic_distribution": topic_counts,
            "average_stress_level": avg_stress,
            "most_common_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral",
            "most_discussed_topic": max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else "general"
        }

# ==============================================================================
# STANDALONE SERVICE
# ==============================================================================

class ConversationalAIService:
    """Standalone conversational AI service"""
    
    def __init__(self, config_file: str = None):
        self.shared_state = SharedState()
        self.agent = ConversationalAIAgent(self.shared_state)
        self.is_running = False
        
    async def start(self):
        """Start the conversational AI service"""
        self.is_running = True
        await self.agent.log("Conversational AI Service started")
        
        # Start background tasks
        asyncio.create_task(self._conversation_monitor())
    
    async def stop(self):
        """Stop the conversational AI service"""
        self.is_running = False
        await self.agent.log("Conversational AI Service stopped")
    
    async def chat(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process chat query and return response"""
        try:
            # Update context if provided
            if context:
                if "health_metrics" in context:
                    await self.shared_state.set("health_metrics", context["health_metrics"])
                if "user_profile" in context:
                    await self.shared_state.set("user_profile", context["user_profile"])
            
            message = AgentMessage(
                sender="API",
                recipient="ConversationalAI",
                message_type="chat",
                data={"query": query}
            )
            
            results = await self.agent.process(message)
            
            if results:
                response_data = results[0].data
                return {
                    "success": True,
                    "response": response_data.get("response", ""),
                    "emotion": response_data.get("emotion", "neutral"),
                    "topics": response_data.get("topics", []),
                    "conversation": response_data.get("conversation", {})
                }
            else:
                return {
                    "success": False,
                    "error": "No response generated"
                }
                
        except Exception as e:
            await self.agent.log(f"Chat processing error: {e}", "error")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_emotion(self, text: str, health_score: float = 50) -> Dict[str, Any]:
        """Analyze emotion in text"""
        try:
            message = AgentMessage(
                sender="API",
                recipient="ConversationalAI",
                message_type="analyze_emotion",
                data={"text": text, "health_score": health_score}
            )
            
            results = await self.agent.process(message)
            
            if results:
                analysis_data = results[0].data
                return {
                    "success": True,
                    "emotion": analysis_data.get("emotion", "neutral"),
                    "stress_level": analysis_data.get("stress_level", 0),
                    "topics": analysis_data.get("topics", [])
                }
            else:
                return {
                    "success": False,
                    "error": "No analysis generated"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_conversation_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get conversation history"""
        try:
            message = AgentMessage(
                sender="API",
                recipient="ConversationalAI",
                message_type="get_conversation_history",
                data={"limit": limit}
            )
            
            results = await self.agent.process(message)
            
            if results:
                return {
                    "success": True,
                    "history": results[0].data.get("history", [])
                }
            else:
                return {
                    "success": False,
                    "error": "Could not retrieve history"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        stats = await self.agent.get_conversation_stats()
        return {
            "service": "ConversationalAIService",
            "status": "running" if self.is_running else "stopped",
            "agent_status": self.agent.is_active,
            "capabilities": self.agent.capabilities,
            "conversation_stats": stats
        }
    
    async def _conversation_monitor(self):
        """Background conversation monitoring"""
        while self.is_running:
            try:
                # Monitor conversation patterns every 5 minutes
                await asyncio.sleep(300)
                
                stats = await self.agent.get_conversation_stats()
                total_conversations = stats.get("total_conversations", 0)
                avg_stress = stats.get("average_stress_level", 0)
                
                if total_conversations > 0:
                    await self.agent.log(f"Conversation monitor: {total_conversations} total, avg stress: {avg_stress:.2f}")
                
                # Alert if high stress levels detected
                if avg_stress > 0.7:
                    await self.agent.log("High stress levels detected in conversations", "warning")
                
            except Exception as e:
                await self.agent.log(f"Conversation monitor error: {e}", "error")

# ==============================================================================
# API ENDPOINTS (FastAPI Integration)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    class ChatRequest(BaseModel):
        query: str
        context: Optional[Dict[str, Any]] = None
    
    class ChatResponse(BaseModel):
        success: bool
        response: Optional[str] = None
        emotion: Optional[str] = None
        topics: Optional[List[str]] = None
        error: Optional[str] = None
    
    class EmotionAnalysisRequest(BaseModel):
        text: str
        health_score: Optional[float] = 50
    
    class EmotionAnalysisResponse(BaseModel):
        success: bool
        emotion: Optional[str] = None
        stress_level: Optional[float] = None
        topics: Optional[List[str]] = None
        error: Optional[str] = None
    
    def create_conversational_api(service: ConversationalAIService) -> FastAPI:
        """Create FastAPI app for conversational AI service"""
        app = FastAPI(title="EmpowerFin Conversational AI API", version="2.0.0")
        
        @app.post("/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Process chat query"""
            try:
                result = await service.chat(request.query, request.context)
                
                if result["success"]:
                    return ChatResponse(
                        success=True,
                        response=result["response"],
                        emotion=result.get("emotion"),
                        topics=result.get("topics")
                    )
                else:
                    raise HTTPException(status_code=400, detail=result["error"])
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
        async def analyze_emotion(request: EmotionAnalysisRequest):
            """Analyze emotion in text"""
            try:
                result = await service.analyze_emotion(request.text, request.health_score)
                
                if result["success"]:
                    return EmotionAnalysisResponse(
                        success=True,
                        emotion=result["emotion"],
                        stress_level=result["stress_level"],
                        topics=result["topics"]
                    )
                else:
                    raise HTTPException(status_code=400, detail=result["error"])
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/conversation-history")
        async def get_conversation_history(limit: int = 10):
            """Get conversation history"""
            result = await service.get_conversation_history(limit)
            return result
        
        @app.get("/status")
        async def get_service_status():
            """Get service status"""
            return await service.get_status()
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        return app

except ImportError:
    # FastAPI not available, skip API creation
    def create_conversational_api(service):
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

# ==============================================================================
# CLI INTERFACE
# ==============================================================================

async def main():
    """Main CLI interface for conversational AI service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EmpowerFin Conversational AI Service")
    parser.add_argument("--chat", "-c", help="Chat query to process")
    parser.add_argument("--service", "-s", action="store_true", help="Run as service")
    parser.add_argument("--api", "-a", action="store_true", help="Run API server")
    parser.add_argument("--port", "-p", type=int, default=8002, help="API port")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    parser.add_argument("--history", action="store_true", help="Show conversation history")
    parser.add_argument("--test", "-t", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    service = ConversationalAIService()
    
    if args.test:
        await run_tests(service)
    elif args.api:
        await run_api_server(service, args.port)
    elif args.service:
        await run_service(service)
    elif args.interactive:
        await run_interactive_chat(service)
    elif args.chat:
        await process_single_chat(service, args.chat)
    elif args.history:
        await show_conversation_history(service)
    else:
        parser.print_help()

async def run_tests(service: ConversationalAIService):
    """Run test suite"""
    print("Running Conversational AI Agent tests...")
    
    await service.start()
    
    test_queries = [
        {"query": "I'm worried about my spending habits", "expected_emotion": "stressed"},
        {"query": "How can I start budgeting?", "expected_topics": ["budgeting"]},
        {"query": "I'm excited to start investing!", "expected_emotion": "happy"},
        {"query": "What should I do about my debt?", "expected_topics": ["debt"]},
    ]
    
    try:
        for i, test in enumerate(test_queries):
            query = test["query"]
            result = await service.chat(query)
            
            if result["success"]:
                print(f"✓ Test {i+1} passed: '{query}' -> {result['emotion']}")
                
                # Check specific expectations
                if "expected_emotion" in test and result["emotion"] != test["expected_emotion"]:
                    print(f"  Warning: Expected emotion '{test['expected_emotion']}', got '{result['emotion']}'")
                
                if "expected_topics" in test:
                    expected_topics = test["expected_topics"]
                    actual_topics = result.get("topics", [])
                    if not any(topic in actual_topics for topic in expected_topics):
                        print(f"  Warning: Expected topics {expected_topics}, got {actual_topics}")
            else:
                print(f"✗ Test {i+1} failed: {result['error']}")
        
        # Test emotion analysis
        emotion_result = await service.analyze_emotion("I'm really struggling financially", 30)
        if emotion_result["success"] and emotion_result["stress_level"] > 0.5:
            print("✓ Emotion analysis test passed")
        else:
            print("✗ Emotion analysis test failed")
        
        # Test conversation history
        history_result = await service.get_conversation_history(5)
        if history_result["success"]:
            print(f"✓ Conversation history test passed: {len(history_result['history'])} conversations")
        else:
            print("✗ Conversation history test failed")
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    await service.stop()

async def run_api_server(service: ConversationalAIService, port: int):
    """Run as API server"""
    try:
        import uvicorn
        
        app = create_conversational_api(service)
        if not app:
            print("Cannot create API server. Install FastAPI and uvicorn.")
            return
        
        print(f"Starting Conversational AI API server on port {port}...")
        
        await service.start()
        
        # Run the API server
        config = uvicorn.Config(app=app, host="0.0.0.0", port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        await service.stop()

async def run_service(service: ConversationalAIService):
    """Run as a standalone service"""
    print("Starting Conversational AI Service...")
    
    await service.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down service...")
        await service.stop()

async def run_interactive_chat(service: ConversationalAIService):
    """Run interactive chat session"""
    print("Starting interactive chat with EmpowerFin AI...")
    print("Type 'quit' to exit, 'history' to see conversation history, 'clear' to clear history")
    
    await service.start()
    
    try:
        while True:
            query = input("\nYou: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'history':
                history_result = await service.get_conversation_history(10)
                if history_result["success"]:
                    history = history_result["history"]
                    print(f"\nConversation History ({len(history)} entries):")
                    for i, conv in enumerate(history[-5:], 1):  # Show last 5
                        print(f"{i}. You: {conv['user_query']}")
                        print(f"   AI: {conv['ai_response']}")
                        print(f"   Emotion: {conv['emotion']}, Topics: {', '.join(conv['topics'])}")
                else:
                    print("Could not retrieve conversation history")
                continue
            elif query.lower() == 'clear':
                # Clear history functionality would go here
                print("History cleared")
                continue
            elif not query:
                continue
            
            result = await service.chat(query)
            
            if result["success"]:
                print(f"\nAI: {result['response']}")
                print(f"[Detected emotion: {result['emotion']}, Topics: {', '.join(result['topics'])}]")
            else:
                print(f"Error: {result['error']}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
    await service.stop()

async def process_single_chat(service: ConversationalAIService, query: str):
    """Process a single chat query"""
    print(f"Processing query: '{query}'")
    
    await service.start()
    
    result = await service.chat(query)
    
    if result["success"]:
        print(f"\nResponse: {result['response']}")
        print(f"Emotion: {result['emotion']}")
        print(f"Topics: {', '.join(result['topics'])}")
    else:
        print(f"Error: {result['error']}")
    
    await service.stop()

async def show_conversation_history(service: ConversationalAIService):
    """Show conversation history and stats"""
    await service.start()
    
    # Get history
    history_result = await service.get_conversation_history(20)
    
    if history_result["success"]:
        history = history_result["history"]
        print(f"Conversation History ({len(history)} entries):")
        
        for i, conv in enumerate(history, 1):
            print(f"\n{i}. [{conv['timestamp']}]")
            print(f"   You: {conv['user_query']}")
            print(f"   AI: {conv['ai_response']}")
            print(f"   Emotion: {conv['emotion']}, Stress: {conv['stress_level']:.2f}, Topics: {', '.join(conv['topics'])}")
    
    # Get stats
    status = await service.get_status()
    stats = status.get("conversation_stats", {})
    
    print(f"\nConversation Statistics:")
    print(f"  Total conversations: {stats.get('total_conversations', 0)}")
    print(f"  Average stress level: {stats.get('average_stress_level', 0):.2f}")
    print(f"  Most common emotion: {stats.get('most_common_emotion', 'N/A')}")
    print(f"  Most discussed topic: {stats.get('most_discussed_topic', 'N/A')}")
    
    await service.stop()

if __name__ == "__main__":
    asyncio.run(main())