# ==============================================================================
# requirements.txt - Python Dependencies
# ==============================================================================

# Core Dependencies
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0

# Data Processing
openpyxl>=3.1.0
xlrd>=2.0.1

# LLM Integration (Optional but recommended)
langchain>=0.1.0
langchain-groq>=0.1.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0

# Development Dependencies (Optional)
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0

# ==============================================================================
# setup.py - Package Setup Configuration
# ==============================================================================

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="empowerfin-guardian",
    version="2.0.0",
    author="EmpowerFin Team",
    author_email="team@empowerfin.ai",
    description="LLM-Enhanced Financial Intelligence Platform with Specialized AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/empowerfin/guardian-2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "llm": [
            "langchain-groq>=0.1.0",
            "langchain-openai>=0.1.0",
            "langchain-anthropic>=0.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "empowerfin=launch:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)

# ==============================================================================
# README.md - Project Documentation
# ==============================================================================

# EmpowerFin Guardian 2.0 🤖

**LLM-Enhanced Financial Intelligence Platform with Specialized AI Agents**

Transform your financial management with cutting-edge AI technology. EmpowerFin Guardian 2.0 features multiple specialized AI agents working together to provide comprehensive financial intelligence, analysis, and personalized recommendations.

## 🚀 Features

### 🧠 **AI-Powered Agents**
- **File Processor Agent**: Intelligently parses any bank statement format
- **Categorization Agent**: Smart transaction classification with confidence scoring
- **Health Analyzer Agent**: Comprehensive financial wellness insights
- **Conversational Agent**: Expert financial advisor with memory and context
- **Trend Analyzer Agent**: Historical pattern analysis and predictions

### ✨ **Smart Capabilities**
- **🎯 Intelligent File Processing**: Automatically detects columns in any format
- **🏷️ Advanced Categorization**: AI-powered transaction classification
- **🔍 Deep Health Analysis**: Beyond basic metrics with AI insights
- **💬 Expert Conversations**: Context-aware financial coaching
- **📊 Personalized Recommendations**: Tailored action plans
- **📈 Trend Analysis**: Historical patterns and future projections

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/empowerfin/guardian-2.0.git
cd guardian-2.0

# Install dependencies
pip install -r requirements.txt

# Run setup wizard
python launch.py --setup

# Launch application
python launch.py
```

### Advanced Installation
```bash
# Install with LLM support
pip install -e ".[llm]"

# Install development dependencies
pip install -e ".[dev]"
```

## 🧠 LLM Configuration

For AI features, configure API keys:

```bash
# Groq (Recommended - Fast & Free Tier)
export GROQ_API_KEY="your_groq_api_key"

# OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

**💡 Tip**: Groq offers free API access with fast inference. Get your key at: https://console.groq.com/

## 🚀 Usage

### Launch Full Application
```bash
python launch.py                    # Full application
python launch.py --demo             # With demo data
```

### Launch Individual Agents
```bash
python launch.py --agent file_processor     # File processing agent
python launch.py --agent categorizer        # Categorization agent
python launch.py --agent health_analyzer    # Health analysis agent
python launch.py --agent conversational     # Conversational agent
python launch.py --agent trend_analyzer     # Trend analysis agent
```

### Management Commands
```bash
python launch.py --agents          # List available agents
python launch.py --setup           # Run setup wizard
python launch.py --help            # Show help
```

## 🏗️ Architecture

### Modular Agent Design
```
EmpowerFin Guardian 2.0/
├── config_manager.py          # Core configuration & LLM management
├── data_models.py              # Data schemas and models
├── file_processor_agent.py     # Intelligent file processing
├── categorization_agent.py     # Transaction categorization
├── health_analyzer_agent.py    # Financial health analysis
├── conversational_agent.py     # AI financial advisor
├── main_application.py         # Streamlit orchestrator
└── launch.py                   # Application launcher
```

### Agent Responsibilities
- **Config Manager**: LLM setup, app configuration, session management
- **Data Models**: Type-safe data structures and schemas
- **File Processor**: Parse any bank statement format with AI
- **Categorizer**: Classify transactions with confidence scoring
- **Health Analyzer**: Calculate metrics and generate AI insights
- **Conversational**: Provide expert financial advice with context
- **Main App**: Orchestrate all agents in Streamlit interface

## 🤖 AI Agent Details

### File Processor Agent
- **LLM Column Detection**: Automatically identifies transaction columns
- **Format Flexibility**: Handles CSV, Excel, TXT files
- **Intelligent Parsing**: Adapts to different bank statement formats
- **Data Validation**: Ensures data quality and consistency

### Categorization Agent
- **Smart Classification**: AI-powered transaction categorization
- **Confidence Scoring**: Provides reliability metrics
- **Subcategory Detection**: Detailed spending breakdown
- **Behavioral Analysis**: Identifies spending patterns
- **Custom Training**: Learn from user corrections

### Health Analyzer Agent
- **Comprehensive Metrics**: 4-component health scoring
- **AI Insights**: Deep analysis of financial patterns
- **Risk Assessment**: Identify vulnerabilities and threats
- **Trend Analysis**: Historical pattern recognition
- **Personalized Recommendations**: Tailored improvement plans

### Conversational Agent
- **Context Awareness**: Remembers conversation history
- **Emotion Detection**: Adapts to user emotional state
- **Intent Recognition**: Understands user goals and needs
- **Expert Knowledge**: Provides certified financial planner advice
- **Follow-up Suggestions**: Proactive conversation guidance

## 📊 Sample Outputs

### Financial Health Score
```
Overall Health: 78.5/100 (Medium Risk)
├── Cashflow Score: 82.3/100
├── Savings Rate: 71.2/100
├── Stability: 79.8/100
└── Emergency Fund: 80.1/100
```

### AI Insights
```
🧠 AI Assessment: Your financial health shows strong fundamentals 
with good cash flow management and steady savings habits. 
However, spending volatility in discretionary categories 
suggests opportunities for budgeting optimization.

💪 Key Strengths:
• Consistent income and positive cash flow
• Above-average savings rate at 18%
• Strong emergency fund coverage

⚠️ Areas for Improvement:
• High variability in entertainment spending
• Limited investment diversification
• Opportunity to optimize subscription costs
```

### Smart Categorization
```
🏷️ Transaction: "AMAZON.COM*MARKETPLACE"
├── Category: shopping (90% confidence)
├── Subcategory: online_shopping
├── Behavior: impulse
└── AI Reasoning: "Marketplace purchase pattern suggests 
    discretionary shopping with impulse characteristics"
```

## 🔧 Development

### Project Structure
- **Modular Design**: Each agent is independent and testable
- **Type Safety**: Comprehensive data models with validation
- **LLM Integration**: Pluggable LLM providers (Groq, OpenAI, Anthropic)
- **Error Handling**: Graceful degradation when AI is unavailable
- **Session Management**: Stateful user experience

### Adding New Agents
1. Create agent module following existing patterns
2. Implement core interface methods
3. Add to agent launcher configuration
4. Update main application orchestrator
5. Add tests and documentation

### Testing Individual Agents
```bash
# Test file processor
python launch.py --agent file_processor

# Test categorization
python launch.py --agent categorizer

# Test health analysis
python launch.py --agent health_analyzer
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.empowerfin.ai](https://docs.empowerfin.ai)
- **Issues**: [GitHub Issues](https://github.com/empowerfin/guardian-2.0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/empowerfin/guardian-2.0/discussions)
- **Email**: support@empowerfin.ai

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [LangChain](https://langchain.com/) for LLM integration
- Uses [Plotly](https://plotly.com/) for interactive visualizations
- Supported by [Groq](https://groq.com/) for fast LLM inference

---

**Transform your financial intelligence with AI. Start your journey today!** 🚀

# ==============================================================================
# .gitignore - Git Ignore Configuration
# ==============================================================================

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Application Specific
data/
exports/
logs/
.streamlit/
*.log

# API Keys and Secrets
.env.local
.env.production
secrets.toml
config.local.py

# Temporary Files
temp/
tmp/
*.tmp
*.bak

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/

# ==============================================================================
# docker-compose.yml - Docker Deployment (Optional)
# ==============================================================================

version: '3.8'

services:
  empowerfin-guardian:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./exports:/app/exports
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3

# ==============================================================================
# Dockerfile - Docker Container (Optional)
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data exports logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Run application
CMD ["python", "launch.py"]