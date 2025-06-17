# ==============================================================================
# launch.py - EmpowerFin Guardian 2.0 Application Launcher
# ==============================================================================

"""
EmpowerFin Guardian 2.0 - LLM-Enhanced Financial Intelligence Platform

This launcher script provides a unified entry point for the modular financial
intelligence platform with multiple specialized AI agents.

Usage:
    python launch.py                    # Launch full application
    python launch.py --agent health     # Launch specific agent
    python launch.py --demo             # Launch with demo data
    python launch.py --setup            # Setup wizard
"""

import sys
import os
import argparse
import importlib
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


class AgentLauncher:
    """Launcher for individual AI agents"""
    
    def __init__(self):
        self.available_agents = {
            'file_processor': {
                'module': 'file_processor_agent',
                'class': 'FileProcessorAgent',
                'description': 'Intelligent file processing with LLM-enhanced column detection'
            },
            'categorizer': {
                'module': 'categorization_agent',
                'class': 'TransactionCategorizationAgent',
                'description': 'Smart transaction categorization with confidence scoring'
            },
            'health_analyzer': {
                'module': 'health_analyzer_agent',
                'class': 'LLMHealthAnalyzer',
                'description': 'Comprehensive financial health analysis with AI insights'
            },
            'conversational': {
                'module': 'conversational_agent',
                'class': 'ConversationalAgent',
                'description': 'Expert financial advisor with memory and context awareness'
            },
            'trend_analyzer': {
                'module': 'health_analyzer_agent',
                'class': 'HealthTrendAnalyzer',
                'description': 'Financial health trend analysis over time'
            }
        }
    
    def launch_agent(self, agent_name: str, **kwargs) -> Any:
        """Launch a specific agent"""
        if agent_name not in self.available_agents:
            raise ValueError(f"Unknown agent: {agent_name}. Available: {list(self.available_agents.keys())}")
        
        agent_config = self.available_agents[agent_name]
        
        # Import the module
        module = importlib.import_module(agent_config['module'])
        
        # Get the class
        agent_class = getattr(module, agent_config['class'])
        
        # Instantiate and return
        return agent_class(**kwargs)
    
    def list_agents(self) -> Dict[str, str]:
        """List all available agents"""
        return {name: config['description'] for name, config in self.available_agents.items()}


class SetupWizard:
    """Setup wizard for first-time configuration"""
    
    def __init__(self):
        self.config = {}
    
    def run_setup(self):
        """Run the interactive setup wizard"""
        print("🚀 Welcome to EmpowerFin Guardian 2.0 Setup!")
        print("=" * 50)
        
        # Check dependencies
        self._check_dependencies()
        
        # LLM configuration
        self._setup_llm()
        
        # Directory setup
        self._setup_directories()
        
        # Configuration summary
        self._show_summary()
        
        print("\n✅ Setup complete! You can now run: python launch.py")
    
    def _check_dependencies(self):
        """Check if required dependencies are installed"""
        print("\n📦 Checking dependencies...")
        
        required_packages = [
            'streamlit',
            'pandas',
            'plotly',
            'numpy'
        ]
        
        optional_packages = [
            'langchain-groq',
            'langchain-openai', 
            'langchain-anthropic'
        ]
        
        missing_required = []
        missing_optional = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"  ✅ {package}")
            except ImportError:
                missing_required.append(package)
                print(f"  ❌ {package}")
        
        for package in optional_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"  ✅ {package} (LLM support)")
            except ImportError:
                missing_optional.append(package)
                print(f"  ⚠️  {package} (optional for LLM)")
        
        if missing_required:
            print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
            print("Install with: pip install " + " ".join(missing_required))
            sys.exit(1)
        
        if missing_optional:
            print(f"\n⚠️  Optional LLM packages not installed: {', '.join(missing_optional)}")
            print("For AI features, install with: pip install " + " ".join(missing_optional))
    
    def _setup_llm(self):
        """Setup LLM configuration"""
        print("\n🧠 LLM Configuration Setup")
        print("Configure API keys for AI features (optional but recommended)")
        
        llm_providers = {
            'groq': {
                'name': 'Groq (Recommended - Fast & Free)',
                'env_var': 'GROQ_API_KEY',
                'url': 'https://console.groq.com/'
            },
            'openai': {
                'name': 'OpenAI',
                'env_var': 'OPENAI_API_KEY',
                'url': 'https://platform.openai.com/api-keys'
            },
            'anthropic': {
                'name': 'Anthropic Claude',
                'env_var': 'ANTHROPIC_API_KEY',
                'url': 'https://console.anthropic.com/'
            }
        }
        
        for provider, info in llm_providers.items():
            current_key = os.getenv(info['env_var'])
            status = "✅ Configured" if current_key else "❌ Not configured"
            
            print(f"\n{info['name']}: {status}")
            print(f"  Environment variable: {info['env_var']}")
            print(f"  Get API key: {info['url']}")
            
            if not current_key:
                setup = input(f"  Setup {provider}? (y/n): ").lower().strip()
                if setup == 'y':
                    print(f"  Add this to your environment:")
                    print(f"  export {info['env_var']}='your_api_key_here'")
    
    def _setup_directories(self):
        """Setup required directories"""
        print("\n📁 Setting up directories...")
        
        directories = [
            'data',
            'exports',
            'logs'
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                print(f"  ✅ Created {directory}/")
            else:
                print(f"  ✅ {directory}/ already exists")
    
    def _show_summary(self):
        """Show setup summary"""
        print("\n📋 Setup Summary")
        print("=" * 30)
        print("✅ Dependencies checked")
        print("✅ Directories created")
        print("🧠 LLM configuration shown")
        print("\n🚀 Ready to launch EmpowerFin Guardian 2.0!")


class ApplicationLauncher:
    """Main application launcher"""
    
    def __init__(self):
        self.agent_launcher = AgentLauncher()
        self.setup_wizard = SetupWizard()
    
    def launch_full_app(self, demo: bool = False):
        """Launch the full Streamlit application"""
        print("🚀 Launching EmpowerFin Guardian 2.0...")
        
        try:
            import streamlit.web.cli as stcli
            import sys
            
            # Prepare streamlit arguments
            sys.argv = [
                "streamlit",
                "run",
                "main_application.py",
                "--server.port=8501",
                "--server.address=localhost",
                "--theme.base=light"
            ]
            
            if demo:
                sys.argv.append("--server.runOnSave=true")
            
            # Launch streamlit
            stcli.main()
            
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to launch application: {e}")
            sys.exit(1)
    
    def launch_agent_demo(self, agent_name: str):
        """Launch a specific agent in demo mode"""
        print(f"🤖 Launching {agent_name} agent demo...")
        
        try:
            agent = self.agent_launcher.launch_agent(agent_name)
            print(f"✅ {agent_name} agent loaded successfully!")
            
            # Agent-specific demo logic would go here
            self._run_agent_demo(agent_name, agent)
            
        except Exception as e:
            print(f"❌ Failed to launch {agent_name} agent: {e}")
            sys.exit(1)
    
    def _run_agent_demo(self, agent_name: str, agent):
        """Run agent-specific demo"""
        print(f"\n🎯 Running {agent_name} demo...")
        
        if agent_name == 'file_processor':
            print("Demo: File processing capabilities")
            print("- Intelligent column detection")
            print("- Multiple format support")
            print("- LLM-enhanced parsing")
        
        elif agent_name == 'categorizer':
            print("Demo: Transaction categorization")
            print("- Smart category detection")
            print("- Confidence scoring")
            print("- LLM reasoning")
        
        elif agent_name == 'health_analyzer':
            print("Demo: Financial health analysis")
            print("- Comprehensive metrics")
            print("- LLM insights")
            print("- Risk assessment")
        
        elif agent_name == 'conversational':
            print("Demo: Conversational AI advisor")
            print("- Context-aware responses")
            print("- Emotional intelligence")
            print("- Financial expertise")
        
        elif agent_name == 'trend_analyzer':
            print("Demo: Trend analysis")
            print("- Historical patterns")
            print("- Trajectory prediction")
            print("- LLM insights")
        
        print(f"\n💡 {agent_name} agent is ready for integration!")
    
    def show_agent_list(self):
        """Show available agents"""
        print("🤖 Available AI Agents:")
        print("=" * 30)
        
        agents = self.agent_launcher.list_agents()
        for name, description in agents.items():
            print(f"  {name:<15} - {description}")
        
        print(f"\nUsage: python launch.py --agent <agent_name>")
    
    def show_help(self):
        """Show help information"""
        print("🚀 EmpowerFin Guardian 2.0 - LLM-Enhanced Financial Intelligence")
        print("=" * 65)
        print("\nUsage:")
        print("  python launch.py                    # Launch full application")
        print("  python launch.py --agent <name>     # Launch specific agent")
        print("  python launch.py --demo             # Launch with demo data")
        print("  python launch.py --setup            # Run setup wizard")
        print("  python launch.py --agents           # List available agents")
        print("  python launch.py --help             # Show this help")
        
        print("\nAI Agents:")
        agents = self.agent_launcher.list_agents()
        for name, description in agents.items():
            print(f"  {name:<15} - {description}")
        
        print("\nExamples:")
        print("  python launch.py --agent health_analyzer")
        print("  python launch.py --demo")
        print("  python launch.py --setup")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EmpowerFin Guardian 2.0 - LLM-Enhanced Financial Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--agent',
        type=str,
        help='Launch specific agent (file_processor, categorizer, health_analyzer, conversational, trend_analyzer)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Launch application with demo data'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup wizard for first-time configuration'
    )
    
    parser.add_argument(
        '--agents',
        action='store_true',
        help='List all available agents'
    )
    
    args = parser.parse_args()
    
    launcher = ApplicationLauncher()
    
    # Handle different launch modes
    if args.setup:
        launcher.setup_wizard.run_setup()
    
    elif args.agents:
        launcher.show_agent_list()
    
    elif args.agent:
        launcher.launch_agent_demo(args.agent)
    
    elif args.demo:
        launcher.launch_full_app(demo=True)
    
    elif len(sys.argv) == 1:
        # No arguments - launch full app
        launcher.launch_full_app()
    
    else:
        launcher.show_help()


if __name__ == "__main__":
    main()