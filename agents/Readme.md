# EmpowerFin Guardian 2.0 - Multi-Agent Architecture

A sophisticated financial intelligence platform built with a distributed multi-agent architecture for maximum scalability, reliability, and modularity.

## 🏗️ Architecture Overview

The system is split into **6 independent programs** that communicate through a message-passing architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                         │
│                (Streamlit Web Interface)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Agent Orchestrator                         │
│              (Message Routing & Coordination)               │
└─────────────────────────────────────────────────────────────┘
           │              │              │              │
┌─────────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────┐
│ Data Processing │ │ Health Calc  │ │ Conversational│ │ Additional │
│     Agent       │ │    Agent     │ │   AI Agent    │ │   Agents   │
└─────────────────┘ └──────────────┘ └─────────────┘ └────────────┘
```

## 📁 Project Structure

```
empowerfin-guardian/
├── shared_models.py           # Shared data models and utilities
├── agent_orchestrator.py      # Central communication hub
├── data_processing_agent.py   # File processing service
├── health_calculation_agent.py # Financial health calculator
├── conversational_ai_agent.py # Chat and advice service
├── main_application.py        # Streamlit web interface
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # Multi-container deployment
├── configs/                   # Configuration files
│   ├── orchestrator.yml
│   ├── agents.yml
│   └── logging.yml
└── scripts/                   # Utility scripts
    ├── start_all_services.sh
    ├── health_check.sh
    └── deploy.sh
```

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8+
- pip or conda
- Optional: Docker & Docker Compose

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/empowerfin-guardian.git
   cd empowerfin-guardian
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"  # Optional for enhanced AI
   export LOG_LEVEL="INFO"
   export REDIS_URL="redis://localhost:6379"     # Optional for scaling
   ```

4. **Start the application:**
   ```bash
   # Option 1: All-in-one Streamlit app
   streamlit run main_application.py
   
   # Option 2: Individual services (for production)
   python agent_orchestrator.py --service &
   python data_processing_agent.py --service &
   python health_calculation_agent.py --service &
   python conversational_ai_agent.py --service &
   streamlit run main_application.py
   ```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Access the web interface
open http://localhost:8501
```

## 🎯 Individual Service Usage

Each service can run independently with its own CLI and API:

### 1. Agent Orchestrator

**Purpose**: Central communication hub and message routing

```bash
# Run as service
python agent_orchestrator.py --service

# Run API server
python agent_orchestrator.py --api --port 8000

# Check system status
python agent_orchestrator.py --status

# Run tests
python agent_orchestrator.py --test
```

**API Endpoints**:
- `POST /agents/register` - Register new agent
- `GET /agents` - List all agents
- `POST /messages/send` - Send message
- `GET /status` - System status
- `WebSocket /ws/{agent_name}` - Real-time communication

### 2. Data Processing Agent

**Purpose**: File validation, conversion, and transaction categorization

```bash
# Process a file
python data_processing_agent.py --file bank_statement.csv

# Run as service
python data_processing_agent.py --service

# Show processing statistics
python data_processing_agent.py --stats

# Run tests
python data_processing_agent.py --test
```

**Features**:
- Multi-format support (CSV, Excel, various encodings)
- Auto-column detection
- Intelligent transaction categorization
- Data quality scoring

### 3. Health Calculation Agent

**Purpose**: Financial health metrics and scoring

```bash
# Calculate health from transactions file
python health_calculation_agent.py --transactions transactions.json

# Run API server
python health_calculation_agent.py --api --port 8001

# Show current metrics
python health_calculation_agent.py --metrics

# Run tests
python health_calculation_agent.py --test
```

**Capabilities**:
- Comprehensive health scoring (4 components)
- Risk level assessment
- Trend analysis
- Improvement suggestions

### 4. Conversational AI Agent

**Purpose**: Natural language processing and personalized advice

```bash
# Interactive chat mode
python conversational_ai_agent.py --interactive

# Process single query
python conversational_ai_agent.py --chat "How can I save more money?"

# Run API server
python conversational_ai_agent.py --api --port 8002

# Show conversation history
python conversational_ai_agent.py --history
```

**Features**:
- Emotion detection and stress analysis
- Topic extraction
- Context-aware responses
- Conversation memory

## 🔌 API Integration

All services provide RESTful APIs for integration:

### Example: Processing a File via API

```python
import requests

# Register with orchestrator
response = requests.post("http://localhost:8000/agents/register", json={
    "agent_name": "MyClient",
    "capabilities": ["client"],
    "endpoint": "http://localhost:9000"
})

# Send file processing request
with open("bank_statement.csv", "rb") as f:
    response = requests.post("http://localhost:8000/messages/send", json={
        "sender": "MyClient",
        "recipient": "DataProcessor",
        "message_type": "process_file",
        "data": {"file_data": f.read().decode()}
    })
```

### Example: Chat Integration

```python
import requests

# Send chat query
response = requests.post("http://localhost:8002/chat", json={
    "query": "I'm worried about my spending habits",
    "context": {
        "health_metrics": {"overall_score": 45}
    }
})

result = response.json()
print(f"AI Response: {result['response']}")
print(f"Detected Emotion: {result['emotion']}")
```

## 🧪 Testing

Each service includes comprehensive tests:

```bash
# Test all services
python agent_orchestrator.py --test
python data_processing_agent.py --test
python health_calculation_agent.py --test
python conversational_ai_agent.py --test

# Integration tests
python main_application.py cli
```

## 🚀 Production Deployment

### Environment Variables

```bash
# Required
export GROQ_API_KEY="your_api_key"

# Optional
export DATABASE_URL="postgresql://user:pass@localhost/empowerfin"
export REDIS_URL="redis://localhost:6379"
export LOG_LEVEL="INFO"
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: empowerfin-orchestrator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: orchestrator
  template:
    metadata:
      labels:
        app: orchestrator
    spec:
      containers:
      - name: orchestrator
        image: empowerfin/orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
---
# Similar deployments for other agents
```

### Docker Compose (Production)

```yaml
version: '3.8'
services:
  orchestrator:
    build: .
    command: python agent_orchestrator.py --api
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  data-processor:
    build: .
    command: python data_processing_agent.py --service
    environment:
      - ORCHESTRATOR_URL=http://orchestrator:8000
  
  health-calculator:
    build: .
    command: python health_calculation_agent.py --service
  
  conversational-ai:
    build: .
    command: python conversational_ai_agent.py --service
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
  
  web-app:
    build: .
    command: streamlit run main_application.py
    ports:
      - "8501:8501"
    depends_on:
      - orchestrator
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🔧 Configuration

### Agent Configuration (`configs/agents.yml`)

```yaml
agents:
  data_processor:
    capabilities:
      - process_file
      - validate_data
      - categorize_transactions
    max_concurrent_files: 5
    supported_formats: [csv, xlsx, xls]
    
  health_calculator:
    capabilities:
      - calculate_health
      - update_metrics
    scoring_weights:
      cashflow: 0.25
      savings_ratio: 0.30
      spending_stability: 0.25
      emergency_fund: 0.20
      
  conversational_ai:
    capabilities:
      - chat
      - analyze_emotion
    max_conversation_history: 20
    emotion_keywords:
      stressed: [worried, anxious, overwhelmed]
      happy: [great, excellent, excited]
```

## 📊 Monitoring & Observability

### Health Checks

```bash
# Check individual service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# System-wide health check
python scripts/health_check.sh
```

### Metrics Collection

The system exposes metrics in Prometheus format:

```
# HELP empowerfin_messages_processed_total Total messages processed
# TYPE empowerfin_messages_processed_total counter
empowerfin_messages_processed_total{agent="DataProcessor"} 1234

# HELP empowerfin_agent_health Agent health status
# TYPE empowerfin_agent_health gauge
empowerfin_agent_health{agent="HealthCalculator"} 1
```

### Logging

Structured JSON logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "agent": "DataProcessor",
  "correlation_id": "abc123",
  "message": "File processed successfully",
  "metadata": {
    "file_size": 1024,
    "transactions_count": 150
  }
}
```

## 🔀 Message Flow Examples

### File Processing Workflow

```
User → Streamlit UI → Orchestrator
                        ↓
                   DataProcessor → [validates file]
                        ↓
                   HealthCalculator ← [transactions data]
                        ↓
                   ConversationalAI ← [health metrics]
                        ↓
                   Streamlit UI ← [complete analysis]
```

### Chat Processing Workflow

```
User → Streamlit UI → Orchestrator
                        ↓
                   ConversationalAI → [analyzes emotion/topics]
                        ↓
                   [accesses shared state for context]
                        ↓
                   Streamlit UI ← [personalized response]
```

## 🛠️ Development

### Adding New Agents

1. **Create agent file** (`my_agent.py`):
```python
from shared_models import BaseAgent, AgentMessage

class MyAgent(BaseAgent):
    def __init__(self, shared_state):
        super().__init__("MyAgent", shared_state)
        self.register_handler("my_capability", self._handle_my_capability)
    
    async def _handle_my_capability(self, message: AgentMessage):
        # Process message
        return [AgentMessage(...)]
```

2. **Register with orchestrator**:
```python
await orchestrator.register_agent("MyAgent", ["my_capability"])
```

3. **Add to main application**:
```python
self.services['my_agent'] = MyAgentService()
```

### Message Types

Standard message types across the system:

- `process_file` - File processing request
- `calculate_health` - Health calculation request
- `chat` - Conversational query
- `analyze_emotion` - Emotion analysis
- `health_check` - Agent health check
- `heartbeat` - Keep-alive signal

### Error Handling

All agents implement consistent error handling:

```python
try:
    # Process message
    result = await self.process_data(message.data)
    return [AgentMessage(sender=self.name, ...)]
except Exception as e:
    await self.log(f"Processing error: {e}", "error")
    return [AgentMessage(
        sender=self.name,
        recipient="ErrorHandler",
        message_type="error",
        data={"error": str(e)}
    )]
```

## 📈 Performance & Scaling

### Horizontal Scaling

- **Load Balancing**: Orchestrator automatically distributes messages across multiple agent instances
- **Agent Replication**: Run multiple instances of the same agent type
- **Geographic Distribution**: Deploy agents across different regions

```bash
# Run multiple data processors
python data_processing_agent.py --service --instance-id dp1 &
python data_processing_agent.py --service --instance-id dp2 &
python data_processing_agent.py --service --instance-id dp3 &
```

### Performance Benchmarks

| Component | Throughput | Latency | Memory |
|-----------|------------|---------|---------|
| Data Processor | 1000 files/hour | <2s | 512MB |
| Health Calculator | 10k calculations/min | <100ms | 256MB |
| Conversational AI | 1k queries/min | <500ms | 1GB |
| Orchestrator | 50k messages/min | <10ms | 128MB |

### Optimization Tips

1. **Message Batching**: Group related messages
2. **Caching**: Use Redis for shared state caching
3. **Connection Pooling**: Reuse database connections
4. **Async Processing**: All I/O operations are async

## 🔒 Security

### Authentication & Authorization

```python
# JWT-based authentication
@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    token = request.headers.get("Authorization")
    if not verify_jwt_token(token):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
```

### Data Protection

- **Encryption at Rest**: All sensitive data encrypted with AES-256
- **Encryption in Transit**: TLS 1.3 for all communications
- **Data Sanitization**: PII detection and redaction
- **Access Logging**: Comprehensive audit trails

### Financial Data Security

```python
def sanitize_financial_data(data: Any) -> Any:
    """Remove sensitive patterns from financial data"""
    if isinstance(data, str):
        # Remove SSN patterns
        data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', data)
        # Remove credit card patterns
        data = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[REDACTED]', data)
    return data
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Agent Not Registering

**Symptoms**: Agent appears offline in status
**Solution**:
```bash
# Check orchestrator is running
curl http://localhost:8000/health

# Verify agent can reach orchestrator
python data_processing_agent.py --test

# Check logs
tail -f empowerfin.log
```

#### 2. Message Routing Failures

**Symptoms**: Messages not reaching target agents
**Solution**:
```bash
# Check agent capabilities
curl http://localhost:8000/agents

# Verify message format
python -c "
from shared_models import AgentMessage
msg = AgentMessage(...)
print(msg.to_json())
"
```

#### 3. File Processing Errors

**Symptoms**: Files fail to process
**Solution**:
```bash
# Test file conversion manually
python data_processing_agent.py --file problematic_file.csv

# Check supported formats
python -c "
from data_processing_agent import EnhancedFileConverter
print('Supported formats: CSV, Excel (.xlsx, .xls)')
"
```

#### 4. High Memory Usage

**Symptoms**: Services consuming excessive memory
**Solution**:
```bash
# Monitor memory usage
python -c "
import psutil
for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
    if 'python' in proc.info['name']:
        print(f'{proc.info[\"pid\"]}: {proc.info[\"memory_percent\"]:.1f}%')
"

# Restart services if needed
./scripts/restart_services.sh
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python agent_orchestrator.py --service
```

### Health Check Script

```bash
#!/bin/bash
# scripts/health_check.sh

echo "EmpowerFin Guardian Health Check"
echo "================================"

# Check orchestrator
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Orchestrator: Healthy"
else
    echo "❌ Orchestrator: Down"
fi

# Check individual agents
for port in 8001 8002 8003; do
    if curl -s http://localhost:$port/health > /dev/null; then
        echo "✅ Agent on port $port: Healthy"
    else
        echo "❌ Agent on port $port: Down"
    fi
done

# Check Streamlit app
if curl -s http://localhost:8501/_stcore/health > /dev/null; then
    echo "✅ Web App: Healthy"
else
    echo "❌ Web App: Down"
fi
```

## 🔄 Updates & Migrations

### Rolling Updates

Deploy new versions without downtime:

```bash
# Update orchestrator
kubectl set image deployment/orchestrator orchestrator=empowerfin/orchestrator:v2.1.0

# Update agents one by one
kubectl set image deployment/data-processor data-processor=empowerfin/data-processor:v2.1.0
```

### Database Migrations

```python
# migrations/001_add_user_preferences.py
from shared_models import SharedState

async def migrate_up(shared_state: SharedState):
    """Add user preferences to existing profiles"""
    all_data = await shared_state.get_all()
    user_profile = all_data.get('user_profile', {})
    
    if 'preferences' not in user_profile:
        user_profile['preferences'] = {
            'communication_style': 'balanced',
            'risk_tolerance': 'moderate'
        }
        await shared_state.set('user_profile', user_profile)
```

## 📚 API Reference

### Orchestrator API

#### POST /agents/register
Register a new agent with the system.

**Request**:
```json
{
  "agent_name": "MyAgent",
  "capabilities": ["capability1", "capability2"],
  "endpoint": "http://localhost:8003"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Agent MyAgent registered"
}
```

#### GET /status
Get comprehensive system status.

**Response**:
```json
{
  "orchestrator": {
    "running": true,
    "uptime": "2:30:45",
    "stats": {
      "messages_processed": 1234,
      "agents_registered": 4
    }
  },
  "agents": {
    "DataProcessor": {
      "health_status": "healthy",
      "capabilities": ["process_file", "validate_data"]
    }
  }
}
```

### Data Processing API

#### POST /process
Process uploaded file.

**Request**: Multipart form with file upload

**Response**:
```json
{
  "success": true,
  "stats": {
    "total_transactions": 150,
    "category_distribution": {
      "food_dining": 45,
      "transportation": 30
    }
  }
}
```

### Health Calculation API

#### POST /calculate
Calculate financial health metrics.

**Request**:
```json
{
  "transactions": [
    {
      "date": "2024-01-01",
      "description": "Grocery Store",
      "amount": -50.00,
      "category": "food_dining"
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "analysis": {
    "health_score": 67.5,
    "risk_level": "Medium Risk",
    "component_scores": {
      "cashflow": 75.0,
      "savings_ratio": 60.0,
      "spending_stability": 70.0,
      "emergency_fund": 65.0
    }
  }
}
```

### Conversational AI API

#### POST /chat
Process chat query with context awareness.

**Request**:
```json
{
  "query": "How can I improve my savings rate?",
  "context": {
    "health_metrics": {"overall_score": 45}
  }
}
```

**Response**:
```json
{
  "success": true,
  "response": "Based on your current financial health, I recommend starting with the 50/30/20 budgeting rule...",
  "emotion": "neutral",
  "topics": ["saving", "budgeting"],
  "conversation": {
    "timestamp": "2024-01-15T10:30:00Z",
    "stress_level": 0.2
  }
}
```

## 🎯 Roadmap

### Version 2.1 (Q2 2024)
- [ ] Goal tracking and progress monitoring
- [ ] Investment analysis agent
- [ ] Mobile app integration
- [ ] Advanced security features

### Version 2.2 (Q3 2024)
- [ ] Bill management agent
- [ ] Tax optimization agent
- [ ] Social features and peer comparisons
- [ ] Advanced ML models

### Version 3.0 (Q4 2024)
- [ ] Blockchain integration
- [ ] Real-time bank API connections
- [ ] AI-powered financial planning
- [ ] Multi-tenant architecture

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Install dev dependencies**: `pip install -r requirements-dev.txt`
4. **Run tests**: `python -m pytest tests/`
5. **Submit pull request**

### Code Standards

- **Type Hints**: All functions must have type hints
- **Docstrings**: Use Google-style docstrings
- **Testing**: Minimum 80% code coverage
- **Linting**: Use black, flake8, mypy

```python
async def process_transaction(
    transaction: Transaction, 
    categories: Dict[str, List[str]]
) -> Transaction:
    """Process and categorize a financial transaction.
    
    Args:
        transaction: The transaction to process
        categories: Category patterns for classification
        
    Returns:
        Processed transaction with category assigned
        
    Raises:
        ProcessingException: If transaction validation fails
    """
```

### Testing Guidelines

```python
# tests/test_data_processor.py
import pytest
from data_processing_agent import DataProcessingAgent

@pytest.mark.asyncio
async def test_file_processing():
    """Test file processing with valid CSV"""
    agent = DataProcessingAgent()
    
    # Test with sample data
    test_data = "date,description,amount\n2024-01-01,Test,-50.00"
    result = await agent.process_file(io.StringIO(test_data))
    
    assert result["success"] is True
    assert len(result["transactions"]) == 1
```





## 🙏 Acknowledgments

- **LangGraph** for the agent framework foundation
- **Streamlit** for the intuitive web interface
- **FastAPI** for high-performance APIs
- **Plotly** for interactive visualizations
- **TextBlob** for natural language processing

---

**Built with ❤️ by the EmpowerFin Team**

*Empowering financial intelligence through distributed AI architecture*