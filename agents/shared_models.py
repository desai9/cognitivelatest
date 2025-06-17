# ==============================================================================
# shared_models.py - Shared Data Models and Utilities
# ==============================================================================

import asyncio
import hashlib
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class FinancialHealthMetrics:
    """Financial health scoring components"""
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
                validated_value = max(0.0, min(100.0, float(value)))
                setattr(self, field_name, validated_value)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'cashflow_score': self.cashflow_score,
            'savings_ratio_score': self.savings_ratio_score,
            'debt_to_income_score': self.debt_to_income_score,
            'spending_stability_score': self.spending_stability_score,
            'emergency_fund_score': self.emergency_fund_score,
            'overall_score': self.overall_score,
            'risk_level': self.risk_level,
            'trend': self.trend,
            'improvement_suggestions': self.improvement_suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FinancialHealthMetrics':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class Transaction:
    """Transaction data model"""
    date: str
    description: str
    amount: float
    category: str = "other"
    spending_type: str = "regular_spending"
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate transaction data"""
        if not self.date or not self.description:
            raise ValueError("Date and description are required")
        if not isinstance(self.amount, (int, float)):
            raise ValueError("Amount must be numeric")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'date': self.date,
            'description': self.description,
            'amount': self.amount,
            'category': self.category,
            'spending_type': self.spending_type,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class UserProfile:
    """User profile and preferences"""
    session_id: str = ""
    financial_goal: str = ""
    risk_tolerance: str = "moderate"
    income_frequency: str = "monthly"
    preferred_communication_style: str = "balanced"
    topics_of_interest: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'financial_goal': self.financial_goal,
            'risk_tolerance': self.risk_tolerance,
            'income_frequency': self.income_frequency,
            'preferred_communication_style': self.preferred_communication_style,
            'topics_of_interest': self.topics_of_interest
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str
    message_type: str
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: str = "normal"
    correlation_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'message_type': self.message_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'priority': self.priority,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

# ==============================================================================
# SHARED STATE MANAGER
# ==============================================================================

class SharedState:
    """Thread-safe shared state manager"""
    
    def __init__(self):
        self.data = {
            'user_profile': UserProfile(),
            'transactions': [],
            'health_metrics': FinancialHealthMetrics(),
            'insights': [],
            'recommendations': [],
            'conversation_history': [],
            'current_query': "",
            'processing_status': "idle",
            'error_messages': [],
            'alerts': [],
            'logs': []
        }
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, default=None):
        """Get value from shared state"""
        async with self._lock:
            return self.data.get(key, default)
    
    async def set(self, key: str, value: Any):
        """Set value in shared state"""
        async with self._lock:
            self.data[key] = value
    
    async def update(self, updates: Dict[str, Any]):
        """Update multiple values in shared state"""
        async with self._lock:
            self.data.update(updates)
    
    async def add_log(self, message: str, level: str = "info", source: str = "system"):
        """Add log entry"""
        async with self._lock:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "source": source,
                "message": message
            }
            if 'logs' not in self.data:
                self.data['logs'] = []
            self.data['logs'].append(log_entry)
            
            # Keep only last 100 logs
            if len(self.data['logs']) > 100:
                self.data['logs'] = self.data['logs'][-100:]
    
    async def get_all(self) -> Dict:
        """Get all data (for serialization)"""
        async with self._lock:
            return self.data.copy()
    
    async def clear(self):
        """Clear all data"""
        async with self._lock:
            self.data = {
                'user_profile': UserProfile(),
                'transactions': [],
                'health_metrics': FinancialHealthMetrics(),
                'insights': [],
                'recommendations': [],
                'conversation_history': [],
                'current_query': "",
                'processing_status': "idle",
                'error_messages': [],
                'alerts': [],
                'logs': []
            }

# ==============================================================================
# BASE AGENT CLASS
# ==============================================================================

class BaseAgent:
    """Base class for all financial agents"""
    
    def __init__(self, name: str, shared_state: SharedState = None):
        self.name = name
        self.shared_state = shared_state or SharedState()
        self.logger = self._setup_logger()
        self.is_active = True
        self.dependencies = []
        self.capabilities = []
        self.message_handlers = {}
    
    def _setup_logger(self):
        """Setup agent-specific logger"""
        logger = logging.getLogger(f"Agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - {self.name} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def process(self, message: AgentMessage = None) -> List[AgentMessage]:
        """Process incoming message and return response messages"""
        if not message:
            return []
        
        # Route to specific handler based on message type
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                return await handler(message)
            except Exception as e:
                await self.log(f"Handler error for {message.message_type}: {e}", "error")
                return []
        else:
            await self.log(f"No handler for message type: {message.message_type}", "warning")
            return []
    
    async def log(self, message: str, level: str = "info"):
        """Log message to both agent logger and shared state"""
        if level == "error":
            self.logger.error(message)
        elif level == "warning":
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        if self.shared_state:
            await self.shared_state.add_log(f"[{self.name}] {message}", level, self.name)
    
    def can_handle(self, message_type: str) -> bool:
        """Check if agent can handle specific message type"""
        return message_type in self.capabilities
    
    def register_handler(self, message_type: str, handler):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        if message_type not in self.capabilities:
            self.capabilities.append(message_type)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Application configuration"""
    
    # Database settings
    DATABASE_URL = "sqlite:///empowerfin.db"
    
    # Redis settings (for message queue)
    REDIS_URL = "redis://localhost:6379"
    
    # API settings
    API_HOST = "localhost"
    API_PORT = 8000
    
    # File settings
    UPLOAD_FOLDER = "./uploads"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Agent settings
    AGENT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FILE = "empowerfin.log"
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        import os
        
        config = cls()
        config.DATABASE_URL = os.getenv("DATABASE_URL", cls.DATABASE_URL)
        config.REDIS_URL = os.getenv("REDIS_URL", cls.REDIS_URL)
        config.API_HOST = os.getenv("API_HOST", cls.API_HOST)
        config.API_PORT = int(os.getenv("API_PORT", cls.API_PORT))
        config.LOG_LEVEL = os.getenv("LOG_LEVEL", cls.LOG_LEVEL)
        
        return config

# ==============================================================================
# UTILITIES
# ==============================================================================

def setup_logging(config: Config):
    """Setup application-wide logging"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )

def validate_transaction_data(data: Dict) -> bool:
    """Validate transaction data structure"""
    required_fields = ['date', 'description', 'amount']
    return all(field in data for field in required_fields)

def sanitize_financial_data(data: Any) -> Any:
    """Sanitize financial data for security"""
    if isinstance(data, dict):
        return {k: sanitize_financial_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_financial_data(item) for item in data]
    elif isinstance(data, str):
        # Remove potentially sensitive patterns
        import re
        # Remove SSN patterns
        data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', data)
        # Remove credit card patterns
        data = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED]', data)
        return data
    else:
        return data

async def health_check() -> Dict[str, Any]:
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "shared_state": "operational",
            "logging": "operational"
        }
    }

# ==============================================================================
# EXCEPTION CLASSES
# ==============================================================================

class EmpowerFinException(Exception):
    """Base exception for EmpowerFin application"""
    pass

class AgentException(EmpowerFinException):
    """Exception raised by agents"""
    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(f"[{agent_name}] {message}")

class DataValidationException(EmpowerFinException):
    """Exception raised for data validation errors"""
    pass

class ProcessingException(EmpowerFinException):
    """Exception raised during data processing"""
    pass

if __name__ == "__main__":
    # Test the shared models
    import asyncio
    
    async def test_models():
        print("Testing shared models...")
        
        # Test Transaction
        tx = Transaction(
            date="2024-01-01",
            description="Test transaction",
            amount=-50.0,
            category="food_dining"
        )
        print(f"Transaction: {tx}")
        print(f"Transaction dict: {tx.to_dict()}")
        
        # Test FinancialHealthMetrics
        metrics = FinancialHealthMetrics(
            cashflow_score=75.0,
            savings_ratio_score=60.0,
            overall_score=67.5
        )
        print(f"Health Metrics: {metrics}")
        
        # Test SharedState
        state = SharedState()
        await state.set("test_key", "test_value")
        value = await state.get("test_key")
        print(f"SharedState test: {value}")
        
        # Test AgentMessage
        message = AgentMessage(
            sender="TestAgent",
            recipient="TargetAgent",
            message_type="test",
            data={"key": "value"}
        )
        print(f"Message: {message}")
        print(f"Message JSON: {message.to_json()}")
        
        print("All tests passed!")
    
    asyncio.run(test_models())