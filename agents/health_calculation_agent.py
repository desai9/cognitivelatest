# ==============================================================================
# health_calculation_agent.py - Financial Health Calculator Service
# ==============================================================================

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add the shared models to the path
sys.path.append(str(Path(__file__).parent))
from shared_models import (
    BaseAgent, AgentMessage, Transaction, FinancialHealthMetrics, SharedState,
    ProcessingException
)

# ==============================================================================
# FINANCIAL HEALTH CALCULATOR
# ==============================================================================

class FinancialHealthCalculator:
    """Advanced financial health scoring system"""
    
    def __init__(self):
        self.weights = {
            'cashflow': 0.25,
            'savings_ratio': 0.30,
            'spending_stability': 0.25,
            'emergency_fund': 0.20
        }
        
        # Benchmark thresholds
        self.benchmarks = {
            'excellent_savings_rate': 0.20,  # 20%+
            'good_savings_rate': 0.10,       # 10-20%
            'emergency_fund_months': 6,       # 6 months ideal
            'high_volatility_threshold': 0.3  # 30% coefficient of variation
        }
    
    async def calculate_comprehensive_health(self, transactions: List[Dict]) -> FinancialHealthMetrics:
        """Calculate comprehensive financial health metrics"""
        try:
            # Convert dict transactions to Transaction objects
            transaction_objects = []
            for tx_data in transactions:
                if isinstance(tx_data, dict):
                    transaction_objects.append(Transaction.from_dict(tx_data))
                else:
                    transaction_objects.append(tx_data)
            
            if not transaction_objects:
                return FinancialHealthMetrics()
            
            metrics = FinancialHealthMetrics()
            
            # Calculate individual scores
            metrics.cashflow_score = await self._calculate_cashflow_score(transaction_objects)
            metrics.savings_ratio_score = await self._calculate_savings_ratio_score(transaction_objects)
            metrics.spending_stability_score = await self._calculate_spending_stability_score(transaction_objects)
            metrics.emergency_fund_score = await self._calculate_emergency_fund_score(transaction_objects)
            
            # Calculate overall score
            metrics.overall_score = (
                metrics.cashflow_score * self.weights['cashflow'] +
                metrics.savings_ratio_score * self.weights['savings_ratio'] +
                metrics.spending_stability_score * self.weights['spending_stability'] +
                metrics.emergency_fund_score * self.weights['emergency_fund']
            )
            
            # Determine risk level and trend
            metrics.risk_level = self._determine_risk_level(metrics.overall_score)
            metrics.trend = await self._calculate_trend(transaction_objects)
            
            # Generate improvement suggestions
            metrics.improvement_suggestions = await self._generate_improvement_suggestions(metrics, transaction_objects)
            
            return metrics
            
        except Exception as e:
            raise ProcessingException(f"Health calculation failed: {str(e)}")
    
    async def _calculate_cashflow_score(self, transactions: List[Transaction]) -> float:
        """Calculate cashflow stability score (0-100)"""
        try:
            if not transactions:
                return 0.0
            
            # Group transactions by month
            monthly_data = {}
            for tx in transactions:
                month_key = tx.date[:7]  # YYYY-MM format
                if month_key not in monthly_data:
                    monthly_data[month_key] = 0
                monthly_data[month_key] += tx.amount
            
            if len(monthly_data) < 2:
                return 50.0  # Neutral score for insufficient data
            
            monthly_balances = list(monthly_data.values())
            
            # Calculate trend and volatility
            mean_balance = np.mean(monthly_balances)
            
            if abs(mean_balance) < 0.01:  # Avoid division by zero
                return 50.0
            
            # Calculate coefficient of variation (volatility measure)
            cv = np.std(monthly_balances) / abs(mean_balance)
            
            # Score based on stability (lower volatility = higher score)
            if cv <= 0.1:  # Very stable
                stability_score = 100
            elif cv <= 0.2:  # Stable
                stability_score = 80
            elif cv <= 0.3:  # Moderately stable
                stability_score = 60
            elif cv <= 0.5:  # Somewhat unstable
                stability_score = 40
            else:  # Very unstable
                stability_score = 20
            
            # Adjust for positive vs negative trend
            if len(monthly_balances) >= 3:
                recent_trend = np.mean(monthly_balances[-3:]) - np.mean(monthly_balances[:-3])
                if recent_trend > 0:
                    stability_score = min(100, stability_score + 10)  # Bonus for positive trend
                elif recent_trend < -abs(mean_balance) * 0.2:  # Significant decline
                    stability_score = max(0, stability_score - 20)
            
            return float(stability_score)
            
        except Exception as e:
            print(f"Cashflow calculation error: {e}")
            return 0.0
    
    async def _calculate_savings_ratio_score(self, transactions: List[Transaction]) -> float:
        """Calculate savings ratio score (0-100)"""
        try:
            if not transactions:
                return 0.0
            
            total_income = sum(tx.amount for tx in transactions if tx.amount > 0)
            total_expenses = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
            
            if total_income <= 0:
                return 0.0
            
            savings_rate = (total_income - total_expenses) / total_income
            
            # Score based on savings rate benchmarks
            if savings_rate >= self.benchmarks['excellent_savings_rate']:  # 20%+
                return 100.0
            elif savings_rate >= self.benchmarks['good_savings_rate']:  # 10-20%
                # Linear interpolation between 60-100
                ratio = (savings_rate - self.benchmarks['good_savings_rate']) / (
                    self.benchmarks['excellent_savings_rate'] - self.benchmarks['good_savings_rate']
                )
                return 60 + (ratio * 40)
            elif savings_rate >= 0.05:  # 5-10%
                # Linear interpolation between 30-60
                ratio = (savings_rate - 0.05) / (self.benchmarks['good_savings_rate'] - 0.05)
                return 30 + (ratio * 30)
            elif savings_rate > 0:  # 0-5%
                # Linear interpolation between 0-30
                return (savings_rate / 0.05) * 30
            else:  # Negative savings (spending more than earning)
                # Penalty for overspending
                return max(0, 30 + (savings_rate * 100))
                
        except Exception as e:
            print(f"Savings ratio calculation error: {e}")
            return 0.0
    
    async def _calculate_spending_stability_score(self, transactions: List[Transaction]) -> float:
        """Calculate spending pattern stability (0-100)"""
        try:
            expenses = [abs(tx.amount) for tx in transactions if tx.amount < 0]
            
            if len(expenses) < 2:
                return 50.0  # Neutral score for insufficient data
            
            # Calculate coefficient of variation for expenses
            mean_expense = np.mean(expenses)
            std_expense = np.std(expenses)
            
            if mean_expense == 0:
                return 50.0
            
            cv = std_expense / mean_expense
            
            # Score based on spending consistency
            if cv <= 0.3:  # Very consistent
                return 100.0
            elif cv <= 0.5:  # Fairly consistent
                return 80.0
            elif cv <= 0.7:  # Moderately consistent
                return 60.0
            elif cv <= 1.0:  # Somewhat inconsistent
                return 40.0
            else:  # Very inconsistent
                return 20.0
                
        except Exception as e:
            print(f"Spending stability calculation error: {e}")
            return 0.0
    
    async def _calculate_emergency_fund_score(self, transactions: List[Transaction]) -> float:
        """Calculate emergency fund adequacy score (0-100)"""
        try:
            if not transactions:
                return 0.0
            
            # Calculate current balance
            current_balance = sum(tx.amount for tx in transactions)
            
            # Calculate average monthly expenses
            expenses = [abs(tx.amount) for tx in transactions if tx.amount < 0]
            total_expenses = sum(expenses)
            
            # Estimate number of months of data
            unique_months = len(set(tx.date[:7] for tx in transactions))
            avg_monthly_expenses = total_expenses / max(1, unique_months)
            
            if avg_monthly_expenses <= 0:
                return 100.0  # No expenses, perfect score
            
            # Calculate months of expenses covered
            months_covered = current_balance / avg_monthly_expenses
            
            # Score based on emergency fund benchmarks
            target_months = self.benchmarks['emergency_fund_months']  # 6 months
            
            if months_covered >= target_months:
                return 100.0
            elif months_covered >= target_months * 0.5:  # 3+ months
                # Linear interpolation between 70-100
                ratio = (months_covered - target_months * 0.5) / (target_months * 0.5)
                return 70 + (ratio * 30)
            elif months_covered >= 1:  # 1-3 months
                # Linear interpolation between 40-70
                ratio = (months_covered - 1) / (target_months * 0.5 - 1)
                return 40 + (ratio * 30)
            elif months_covered >= 0:  # 0-1 month
                # Linear interpolation between 0-40
                return months_covered * 40
            else:  # Negative balance
                return 0.0
                
        except Exception as e:
            print(f"Emergency fund calculation error: {e}")
            return 0.0
    
    def _determine_risk_level(self, overall_score: float) -> str:
        """Determine risk level based on overall score"""
        if overall_score >= 80:
            return "Low Risk"
        elif overall_score >= 60:
            return "Medium Risk"
        elif overall_score >= 40:
            return "High Risk"
        else:
            return "Critical Risk"
    
    async def _calculate_trend(self, transactions: List[Transaction]) -> str:
        """Calculate financial trend"""
        try:
            if len(transactions) < 6:  # Need at least 6 transactions
                return "Stable"
            
            # Group by month and calculate monthly balances
            monthly_data = {}
            for tx in transactions:
                month_key = tx.date[:7]
                if month_key not in monthly_data:
                    monthly_data[month_key] = 0
                monthly_data[month_key] += tx.amount
            
            if len(monthly_data) < 3:
                return "Stable"
            
            months = sorted(monthly_data.keys())
            balances = [monthly_data[month] for month in months]
            
            # Calculate trend using linear regression slope
            x = np.arange(len(balances))
            slope = np.polyfit(x, balances, 1)[0]
            
            # Determine trend based on slope
            avg_balance = np.mean(balances)
            relative_slope = slope / (abs(avg_balance) + 1)  # Avoid division by zero
            
            if relative_slope > 0.1:
                return "Improving"
            elif relative_slope < -0.1:
                return "Declining"
            else:
                return "Stable"
                
        except Exception as e:
            print(f"Trend calculation error: {e}")
            return "Stable"
    
    async def _generate_improvement_suggestions(self, metrics: FinancialHealthMetrics, 
                                              transactions: List[Transaction]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        try:
            # Savings rate suggestions
            if metrics.savings_ratio_score < 60:
                if metrics.savings_ratio_score < 30:
                    suggestions.append("Critical: Immediately reduce expenses and increase income. Aim for at least 5% savings rate.")
                else:
                    suggestions.append("Increase your savings rate to at least 10% of income for better financial stability.")
            
            # Emergency fund suggestions
            if metrics.emergency_fund_score < 60:
                current_balance = sum(tx.amount for tx in transactions)
                if current_balance < 0:
                    suggestions.append("Priority: Build positive account balance before focusing on emergency fund.")
                else:
                    suggestions.append("Build emergency fund to cover 3-6 months of expenses for financial security.")
            
            # Cashflow suggestions
            if metrics.cashflow_score < 60:
                suggestions.append("Stabilize monthly cash flow by creating a consistent budget and reducing variable expenses.")
            
            # Spending stability suggestions
            if metrics.spending_stability_score < 60:
                suggestions.append("Create a detailed budget to make spending more predictable and controlled.")
            
            # Overall health suggestions
            if metrics.overall_score < 40:
                suggestions.append("Consider consulting with a financial advisor for personalized guidance.")
            elif metrics.overall_score > 80:
                suggestions.append("Great job! Consider advanced strategies like investment optimization and tax planning.")
            
            return suggestions
            
        except Exception as e:
            print(f"Suggestion generation error: {e}")
            return ["Focus on building a budget and emergency fund as first steps."]

# ==============================================================================
# HEALTH CALCULATION AGENT
# ==============================================================================

class HealthCalculationAgent(BaseAgent):
    """Calculates financial health metrics and scores"""
    
    def __init__(self, shared_state: SharedState = None):
        super().__init__("HealthCalculator", shared_state)
        self.calculator = FinancialHealthCalculator()
        
        # Register message handlers
        self.register_handler("calculate_health", self._handle_calculate_health)
        self.register_handler("update_metrics", self._handle_update_metrics)
        self.register_handler("get_metrics", self._handle_get_metrics)
        self.register_handler("health_check", self._handle_health_check)
    
    async def _handle_calculate_health(self, message: AgentMessage) -> List[AgentMessage]:
        """Calculate financial health metrics"""
        await self.log("Calculating financial health metrics")
        
        try:
            transactions = message.data.get("transactions", [])
            
            if not transactions:
                await self.log("No transactions provided for health calculation", "warning")
                return []
            
            # Calculate comprehensive health metrics
            health_metrics = await self.calculator.calculate_comprehensive_health(transactions)
            
            # Update shared state
            await self.shared_state.set("health_metrics", health_metrics.to_dict())
            await self.shared_state.set("processing_status", "health_calculated")
            
            await self.log(f"Health score calculated: {health_metrics.overall_score:.1f}/100 ({health_metrics.risk_level})")
            
            # Notify recommendation engine
            return [
                AgentMessage(
                    sender=self.name,
                    recipient="RecommendationEngine",
                    message_type="generate_recommendations",
                    data={
                        "health_metrics": health_metrics.to_dict(),
                        "transactions": transactions
                    }
                )
            ]
            
        except Exception as e:
            await self.log(f"Health calculation error: {e}", "error")
            await self.shared_state.set("processing_status", "error")
            return []
    
    async def _handle_update_metrics(self, message: AgentMessage) -> List[AgentMessage]:
        """Update specific health metrics"""
        await self.log("Updating health metrics")
        
        try:
            current_metrics = await self.shared_state.get("health_metrics", {})
            updates = message.data.get("updates", {})
            
            # Merge updates
            current_metrics.update(updates)
            await self.shared_state.set("health_metrics", current_metrics)
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="metrics_updated",
                    data={"health_metrics": current_metrics}
                )
            ]
            
        except Exception as e:
            await self.log(f"Metrics update error: {e}", "error")
            return []
    
    async def _handle_get_metrics(self, message: AgentMessage) -> List[AgentMessage]:
        """Get current health metrics"""
        try:
            health_metrics = await self.shared_state.get("health_metrics", {})
            
            return [
                AgentMessage(
                    sender=self.name,
                    recipient=message.sender,
                    message_type="metrics_response",
                    data={"health_metrics": health_metrics}
                )
            ]
            
        except Exception as e:
            await self.log(f"Get metrics error: {e}", "error")
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
                    "calculator_ready": self.calculator is not None
                }
            )
        ]
    
    async def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed health analysis"""
        try:
            health_metrics = await self.shared_state.get("health_metrics", {})
            transactions = await self.shared_state.get("transactions", [])
            
            if not health_metrics or not transactions:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate additional insights
            total_income = sum(tx.get('amount', 0) for tx in transactions if tx.get('amount', 0) > 0)
            total_expenses = sum(abs(tx.get('amount', 0)) for tx in transactions if tx.get('amount', 0) < 0)
            current_balance = sum(tx.get('amount', 0) for tx in transactions)
            
            # Category breakdown
            category_breakdown = {}
            for tx in transactions:
                if tx.get('amount', 0) < 0:  # Expenses only
                    category = tx.get('category', 'other')
                    category_breakdown[category] = category_breakdown.get(category, 0) + abs(tx.get('amount', 0))
            
            return {
                "health_score": health_metrics.get('overall_score', 0),
                "risk_level": health_metrics.get('risk_level', 'Unknown'),
                "trend": health_metrics.get('trend', 'Stable'),
                "financial_summary": {
                    "total_income": total_income,
                    "total_expenses": total_expenses,
                    "current_balance": current_balance,
                    "savings_rate": (total_income - total_expenses) / total_income if total_income > 0 else 0
                },
                "category_breakdown": category_breakdown,
                "component_scores": {
                    "cashflow": health_metrics.get('cashflow_score', 0),
                    "savings_ratio": health_metrics.get('savings_ratio_score', 0),
                    "spending_stability": health_metrics.get('spending_stability_score', 0),
                    "emergency_fund": health_metrics.get('emergency_fund_score', 0)
                },
                "improvement_suggestions": health_metrics.get('improvement_suggestions', [])
            }
            
        except Exception as e:
            await self.log(f"Detailed analysis error: {e}", "error")
            return {"error": str(e)}

# ==============================================================================
# STANDALONE SERVICE
# ==============================================================================

class HealthCalculationService:
    """Standalone health calculation service"""
    
    def __init__(self, config_file: str = None):
        self.shared_state = SharedState()
        self.agent = HealthCalculationAgent(self.shared_state)
        self.is_running = False
        
    async def start(self):
        """Start the health calculation service"""
        self.is_running = True
        await self.agent.log("Health Calculation Service started")
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
    
    async def stop(self):
        """Stop the health calculation service"""
        self.is_running = False
        await self.agent.log("Health Calculation Service stopped")
    
    async def calculate_health(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Calculate health metrics for given transactions"""
        try:
            message = AgentMessage(
                sender="API",
                recipient="HealthCalculator",
                message_type="calculate_health",
                data={"transactions": transactions}
            )
            
            results = await self.agent.process(message)
            
            # Get detailed analysis
            analysis = await self.agent.get_detailed_analysis()
            
            return {
                "success": True,
                "analysis": analysis,
                "messages": [msg.to_dict() for msg in results]
            }
            
        except Exception as e:
            await self.agent.log(f"Health calculation error: {e}", "error")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current health metrics"""
        try:
            message = AgentMessage(
                sender="API",
                recipient="HealthCalculator",
                message_type="get_metrics",
                data={}
            )
            
            results = await self.agent.process(message)
            
            if results:
                return {
                    "success": True,
                    "metrics": results[0].data.get("health_metrics", {})
                }
            else:
                return {
                    "success": False,
                    "error": "No metrics available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service": "HealthCalculationService",
            "status": "running" if self.is_running else "stopped",
            "agent_status": self.agent.is_active,
            "capabilities": self.agent.capabilities,
            "calculator_status": "ready"
        }
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                # Check system health every 60 seconds
                await asyncio.sleep(60)
                
                # Log health status
                metrics = await self.shared_state.get("health_metrics", {})
                if metrics:
                    score = metrics.get('overall_score', 0)
                    await self.agent.log(f"Health monitor: Current score {score:.1f}/100")
                
            except Exception as e:
                await self.agent.log(f"Health monitor error: {e}", "error")

# ==============================================================================
# API ENDPOINTS (FastAPI Integration)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    class TransactionModel(BaseModel):
        date: str
        description: str
        amount: float
        category: Optional[str] = "other"
        spending_type: Optional[str] = "regular_spending"
    
    class HealthCalculationRequest(BaseModel):
        transactions: List[TransactionModel]
    
    class HealthCalculationResponse(BaseModel):
        success: bool
        analysis: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
    
    def create_health_api(service: HealthCalculationService) -> FastAPI:
        """Create FastAPI app for health calculation service"""
        app = FastAPI(title="EmpowerFin Health Calculation API", version="2.0.0")
        
        @app.post("/calculate", response_model=HealthCalculationResponse)
        async def calculate_health(request: HealthCalculationRequest):
            """Calculate financial health metrics"""
            try:
                transactions = [tx.dict() for tx in request.transactions]
                result = await service.calculate_health(transactions)
                
                if result["success"]:
                    return HealthCalculationResponse(
                        success=True,
                        analysis=result["analysis"]
                    )
                else:
                    raise HTTPException(status_code=400, detail=result["error"])
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/metrics")
        async def get_current_metrics():
            """Get current health metrics"""
            result = await service.get_current_metrics()
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
    def create_health_api(service):
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

# ==============================================================================
# CLI INTERFACE
# ==============================================================================

async def main():
    """Main CLI interface for health calculation service"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="EmpowerFin Health Calculation Service")
    parser.add_argument("--transactions", "-t", help="JSON file with transactions")
    parser.add_argument("--service", "-s", action="store_true", help="Run as service")
    parser.add_argument("--api", "-a", action="store_true", help="Run API server")
    parser.add_argument("--port", "-p", type=int, default=8001, help="API port")
    parser.add_argument("--metrics", "-m", action="store_true", help="Show current metrics")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    service = HealthCalculationService()
    
    if args.test:
        await run_tests(service)
    elif args.api:
        await run_api_server(service, args.port)
    elif args.service:
        await run_service(service)
    elif args.transactions:
        await calculate_from_file(service, args.transactions)
    elif args.metrics:
        await show_metrics(service)
    else:
        parser.print_help()

async def run_tests(service: HealthCalculationService):
    """Run test suite"""
    print("Running Health Calculation Agent tests...")
    
    await service.start()
    
    # Test data
    test_transactions = [
        {"date": "2024-01-01", "description": "Salary", "amount": 3000.0, "category": "income"},
        {"date": "2024-01-02", "description": "Rent", "amount": -1000.0, "category": "fixed_expenses"},
        {"date": "2024-01-03", "description": "Groceries", "amount": -200.0, "category": "food_dining"},
        {"date": "2024-01-04", "description": "Gas", "amount": -50.0, "category": "transportation"},
        {"date": "2024-02-01", "description": "Salary", "amount": 3000.0, "category": "income"},
        {"date": "2024-02-02", "description": "Rent", "amount": -1000.0, "category": "fixed_expenses"},
    ]
    
    try:
        # Test health calculation
        result = await service.calculate_health(test_transactions)
        
        if result["success"]:
            analysis = result["analysis"]
            print(f"✓ Health calculation test passed")
            print(f"  Overall Score: {analysis['health_score']:.1f}/100")
            print(f"  Risk Level: {analysis['risk_level']}")
            print(f"  Savings Rate: {analysis['financial_summary']['savings_rate']:.1%}")
        else:
            print(f"✗ Health calculation test failed: {result['error']}")
        
        # Test metrics retrieval
        metrics_result = await service.get_current_metrics()
        if metrics_result["success"]:
            print("✓ Metrics retrieval test passed")
        else:
            print(f"✗ Metrics retrieval test failed: {metrics_result['error']}")
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    await service.stop()

async def run_api_server(service: HealthCalculationService, port: int):
    """Run as API server"""
    try:
        import uvicorn
        
        app = create_health_api(service)
        if not app:
            print("Cannot create API server. Install FastAPI and uvicorn.")
            return
        
        print(f"Starting Health Calculation API server on port {port}...")
        
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

async def run_service(service: HealthCalculationService):
    """Run as a standalone service"""
    print("Starting Health Calculation Service...")
    
    await service.start()
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down service...")
        await service.stop()

async def calculate_from_file(service: HealthCalculationService, file_path: str):
    """Calculate health from transactions file"""
    print(f"Calculating health from file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            transactions = json.load(f)
        
        await service.start()
        
        result = await service.calculate_health(transactions)
        
        if result["success"]:
            analysis = result["analysis"]
            print("✓ Health calculation completed")
            print(f"Overall Score: {analysis['health_score']:.1f}/100")
            print(f"Risk Level: {analysis['risk_level']}")
            print(f"Trend: {analysis['trend']}")
            print("\nComponent Scores:")
            for component, score in analysis['component_scores'].items():
                print(f"  {component.replace('_', ' ').title()}: {score:.1f}/100")
            
            if analysis['improvement_suggestions']:
                print("\nImprovement Suggestions:")
                for suggestion in analysis['improvement_suggestions']:
                    print(f"  • {suggestion}")
        else:
            print(f"✗ Calculation failed: {result['error']}")
        
        await service.stop()
        
    except Exception as e:
        print(f"✗ Error: {e}")

async def show_metrics(service: HealthCalculationService):
    """Show current metrics"""
    await service.start()
    
    result = await service.get_current_metrics()
    
    if result["success"]:
        metrics = result["metrics"]
        print("Current Health Metrics:")
        print(f"  Overall Score: {metrics.get('overall_score', 0):.1f}/100")
        print(f"  Risk Level: {metrics.get('risk_level', 'Unknown')}")
        print(f"  Trend: {metrics.get('trend', 'Stable')}")
        
        suggestions = metrics.get('improvement_suggestions', [])
        if suggestions:
            print("\nImprovement Suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
    else:
        print(f"✗ Error retrieving metrics: {result['error']}")
    
    await service.stop()

if __name__ == "__main__":
    asyncio.run(main())