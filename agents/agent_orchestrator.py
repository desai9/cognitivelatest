# ==============================================================================
# agent_orchestrator.py - Agent Communication Hub
# ==============================================================================

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add the shared models to the path
sys.path.append(str(Path(__file__).parent))
from shared_models import (
    AgentMessage, SharedState, Config, setup_logging, health_check
)

# ==============================================================================
# MESSAGE QUEUE MANAGER
# ==============================================================================

class MessageQueueManager:
    """Manages message queues for agent communication"""
    
    def __init__(self, max_queue_size: int = 1000):
        self.queues = {}
        self.max_queue_size = max_queue_size
        self.message_history = []
        self.stats = {
            "messages_processed": 0,
            "messages_failed": 0,
            "queues_created": 0
        }
    
    async def create_queue(self, agent_name: str) -> asyncio.Queue:
        """Create a message queue for an agent"""
        if agent_name not in self.queues:
            self.queues[agent_name] = asyncio.Queue(maxsize=self.max_queue_size)
            self.stats["queues_created"] += 1
        return self.queues[agent_name]
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to recipient agent's queue"""
        try:
            recipient_queue = await self.create_queue(message.recipient)
            
            # Add to queue (will block if queue is full)
            await recipient_queue.put(message)
            
            # Track message
            self.message_history.append({
                "timestamp": datetime.now().isoformat(),
                "sender": message.sender,
                "recipient": message.recipient,
                "message_type": message.message_type,
                "correlation_id": message.correlation_id
            })
            
            # Keep only last 1000 messages in history
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]
            
            self.stats["messages_processed"] += 1
            return True
            
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def receive_message(self, agent_name: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive message from agent's queue"""
        try:
            queue = await self.create_queue(agent_name)
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logging.error(f"Failed to receive message for {agent_name}: {e}")
            return None
    
    async def broadcast_message(self, message: AgentMessage, exclude_sender: bool = True) -> int:
        """Broadcast message to all agents"""
        sent_count = 0
        
        for agent_name in self.queues.keys():
            if exclude_sender and agent_name == message.sender:
                continue
            
            # Create copy with new recipient
            broadcast_msg = AgentMessage(
                sender=message.sender,
                recipient=agent_name,
                message_type=message.message_type,
                data=message.data,
                priority=message.priority,
                correlation_id=message.correlation_id
            )
            
            if await self.send_message(broadcast_msg):
                sent_count += 1
        
        return sent_count
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all queues"""
        status = {}
        for agent_name, queue in self.queues.items():
            status[agent_name] = {
                "queue_size": queue.qsize(),
                "queue_full": queue.full(),
                "queue_empty": queue.empty()
            }
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message queue statistics"""
        return {
            "stats": self.stats,
            "active_queues": len(self.queues),
            "recent_messages": len(self.message_history),
            "queue_status": self.get_queue_status()
        }

# ==============================================================================
# AGENT REGISTRY
# ==============================================================================

class AgentRegistry:
    """Registry for managing agent connections and capabilities"""
    
    def __init__(self):
        self.agents = {}
        self.capabilities = {}
        self.health_status = {}
        self.last_heartbeat = {}
    
    async def register_agent(self, agent_name: str, capabilities: List[str], 
                           endpoint: str = None, process_id: str = None) -> bool:
        """Register an agent with its capabilities"""
        try:
            self.agents[agent_name] = {
                "name": agent_name,
                "capabilities": capabilities,
                "endpoint": endpoint,
                "process_id": process_id,
                "registered_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            # Update capability mappings
            for capability in capabilities:
                if capability not in self.capabilities:
                    self.capabilities[capability] = []
                self.capabilities[capability].append(agent_name)
            
            # Initialize health status
            self.health_status[agent_name] = "healthy"
            self.last_heartbeat[agent_name] = datetime.now()
            
            logging.info(f"Agent {agent_name} registered with capabilities: {capabilities}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to register agent {agent_name}: {e}")
            return False
    
    async def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_name in self.agents:
                # Remove from capability mappings
                agent_caps = self.agents[agent_name].get("capabilities", [])
                for capability in agent_caps:
                    if capability in self.capabilities:
                        self.capabilities[capability] = [
                            name for name in self.capabilities[capability] 
                            if name != agent_name
                        ]
                        # Remove capability if no agents support it
                        if not self.capabilities[capability]:
                            del self.capabilities[capability]
                
                # Remove from all registries
                del self.agents[agent_name]
                if agent_name in self.health_status:
                    del self.health_status[agent_name]
                if agent_name in self.last_heartbeat:
                    del self.last_heartbeat[agent_name]
                
                logging.info(f"Agent {agent_name} unregistered")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to unregister agent {agent_name}: {e}")
            return False
    
    def find_agents_for_capability(self, capability: str) -> List[str]:
        """Find agents that support a specific capability"""
        return self.capabilities.get(capability, [])
    
    def get_agent_info(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        if agent_name in self.agents:
            agent_info = self.agents[agent_name].copy()
            agent_info["health_status"] = self.health_status.get(agent_name, "unknown")
            agent_info["last_heartbeat"] = self.last_heartbeat.get(agent_name)
            return agent_info
        return None
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get information about all registered agents"""
        all_agents = {}
        for agent_name in self.agents:
            all_agents[agent_name] = self.get_agent_info(agent_name)
        return all_agents
    
    async def update_heartbeat(self, agent_name: str, health_status: str = "healthy"):
        """Update agent heartbeat and health status"""
        if agent_name in self.agents:
            self.last_heartbeat[agent_name] = datetime.now()
            self.health_status[agent_name] = health_status
    
    async def check_agent_health(self, timeout_minutes: int = 5) -> Dict[str, str]:
        """Check health of all agents based on heartbeat timeout"""
        current_time = datetime.now()
        health_report = {}
        
        for agent_name, last_heartbeat in self.last_heartbeat.items():
            time_diff = (current_time - last_heartbeat).total_seconds() / 60  # minutes
            
            if time_diff > timeout_minutes:
                health_report[agent_name] = "timeout"
                self.health_status[agent_name] = "timeout"
            else:
                health_report[agent_name] = self.health_status.get(agent_name, "healthy")
        
        return health_report

# ==============================================================================
# ROUTING ENGINE
# ==============================================================================

class RoutingEngine:
    """Routes messages between agents based on capabilities and load balancing"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry
class RoutingEngine:
    """Routes messages between agents based on capabilities and load balancing"""
    
    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry
        self.routing_stats = {}
        self.load_balancer = {}
    
    async def route_message(self, message: AgentMessage) -> List[str]:
        """Route message to appropriate agents"""
        recipients = []
        
        # If recipient is specified and exists, use it
        if message.recipient != "auto" and message.recipient in self.agent_registry.agents:
            recipients.append(message.recipient)
        else:
            # Find agents based on message type capability
            capable_agents = self.agent_registry.find_agents_for_capability(message.message_type)
            
            if capable_agents:
                # Load balance among capable agents
                recipients.append(self._select_agent_for_load_balancing(capable_agents, message.message_type))
            else:
                logging.warning(f"No agents found for capability: {message.message_type}")
        
        # Update routing stats
        for recipient in recipients:
            if recipient not in self.routing_stats:
                self.routing_stats[recipient] = 0
            self.routing_stats[recipient] += 1
        
        return recipients
    
    def _select_agent_for_load_balancing(self, agents: List[str], message_type: str) -> str:
        """Select agent using round-robin load balancing"""
        if not agents:
            return None
        
        if message_type not in self.load_balancer:
            self.load_balancer[message_type] = 0
        
        # Filter healthy agents
        healthy_agents = [
            agent for agent in agents 
            if self.agent_registry.health_status.get(agent) == "healthy"
        ]
        
        if not healthy_agents:
            healthy_agents = agents  # Fallback to all agents if none are healthy
        
        # Round-robin selection
        selected_index = self.load_balancer[message_type] % len(healthy_agents)
        selected_agent = healthy_agents[selected_index]
        
        self.load_balancer[message_type] += 1
        
        return selected_agent
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "routing_stats": self.routing_stats,
            "load_balancer_state": self.load_balancer,
            "total_routes": sum(self.routing_stats.values())
        }

# ==============================================================================
# AGENT ORCHESTRATOR
# ==============================================================================

class AgentOrchestrator:
    """Main orchestrator for agent communication and coordination"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.shared_state = SharedState()
        self.message_queue = MessageQueueManager()
        self.agent_registry = AgentRegistry()
        self.routing_engine = RoutingEngine(self.agent_registry)
        
        self.is_running = False
        self.worker_tasks = []
        self.stats = {
            "orchestrator_started": None,
            "messages_processed": 0,
            "agents_registered": 0,
            "errors_encountered": 0
        }
    
    async def start(self):
        """Start the orchestrator"""
        self.is_running = True
        self.stats["orchestrator_started"] = datetime.now().isoformat()
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._stats_reporter())
        ]
        
        logging.info("Agent Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logging.info("Agent Orchestrator stopped")
    
    async def register_agent(self, agent_name: str, capabilities: List[str], 
                           endpoint: str = None) -> bool:
        """Register an agent with the orchestrator"""
        success = await self.agent_registry.register_agent(agent_name, capabilities, endpoint)
        if success:
            self.stats["agents_registered"] += 1
            
            # Create message queue for the agent
            await self.message_queue.create_queue(agent_name)
            
            # Send welcome message
            welcome_message = AgentMessage(
                sender="Orchestrator",
                recipient=agent_name,
                message_type="agent_registered",
                data={
                    "welcome": True,
                    "orchestrator_time": datetime.now().isoformat(),
                    "agent_id": agent_name
                }
            )
            await self.send_message(welcome_message)
        
        return success
    
    async def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent"""
        return await self.agent_registry.unregister_agent(agent_name)
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the orchestrator"""
        try:
            # Route the message
            recipients = await self.routing_engine.route_message(message)
            
            if not recipients:
                logging.warning(f"No recipients found for message: {message.message_type}")
                return False
            
            # Send to all recipients
            success_count = 0
            for recipient in recipients:
                # Update recipient in message
                routed_message = AgentMessage(
                    sender=message.sender,
                    recipient=recipient,
                    message_type=message.message_type,
                    data=message.data,
                    priority=message.priority,
                    correlation_id=message.correlation_id
                )
                
                if await self.message_queue.send_message(routed_message):
                    success_count += 1
            
            self.stats["messages_processed"] += success_count
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            self.stats["errors_encountered"] += 1
            return False
    
    async def broadcast_message(self, message: AgentMessage) -> int:
        """Broadcast message to all agents"""
        return await self.message_queue.broadcast_message(message)
    
    async def receive_message(self, agent_name: str, timeout: float = 1.0) -> Optional[AgentMessage]:
        """Receive message for a specific agent"""
        return await self.message_queue.receive_message(agent_name, timeout)
    
    async def process_file_upload(self, file_data, user_profile: Dict = None):
        """Process file upload workflow"""
        try:
            # Update shared state with user profile
            if user_profile:
                await self.shared_state.set("user_profile", user_profile)
            
            # Send file to data processor
            message = AgentMessage(
                sender="Orchestrator",
                recipient="DataProcessor",
                message_type="process_file",
                data={"file_data": file_data},
                priority="high"
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logging.error(f"File upload processing failed: {e}")
            return False
    
    async def process_chat_query(self, query: str, context: Dict = None):
        """Process chat query workflow"""
        try:
            # Prepare context data
            chat_data = {"query": query}
            if context:
                chat_data["context"] = context
            
            # Send to conversational AI
            message = AgentMessage(
                sender="Orchestrator",
                recipient="ConversationalAI",
                message_type="chat",
                data=chat_data
            )
            
            return await self.send_message(message)
            
        except Exception as e:
            logging.error(f"Chat query processing failed: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestrator": {
                "running": self.is_running,
                "stats": self.stats,
                "uptime": self._calculate_uptime()
            },
            "agents": self.agent_registry.get_all_agents(),
            "message_queue": self.message_queue.get_stats(),
            "routing": self.routing_engine.get_routing_stats(),
            "shared_state": await self._get_shared_state_summary()
        }
    
    async def _message_processor(self):
        """Background message processing worker"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)  # Process messages every 100ms
                
                # Process any orchestrator-level messages
                orchestrator_message = await self.message_queue.receive_message("Orchestrator", timeout=0.1)
                
                if orchestrator_message:
                    await self._handle_orchestrator_message(orchestrator_message)
                
            except Exception as e:
                logging.error(f"Message processor error: {e}")
                self.stats["errors_encountered"] += 1
    
    async def _health_monitor(self):
        """Background health monitoring worker"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check health every 30 seconds
                
                # Check agent health
                health_report = await self.agent_registry.check_agent_health()
                
                # Log any unhealthy agents
                for agent_name, status in health_report.items():
                    if status != "healthy":
                        logging.warning(f"Agent {agent_name} health status: {status}")
                
                # Update shared state with health report
                await self.shared_state.set("agent_health", health_report)
                
            except Exception as e:
                logging.error(f"Health monitor error: {e}")
    
    async def _stats_reporter(self):
        """Background statistics reporting worker"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Report stats every 5 minutes
                
                status = await self.get_system_status()
                logging.info(f"System stats: {self.stats['messages_processed']} messages processed, "
                           f"{self.stats['agents_registered']} agents registered")
                
            except Exception as e:
                logging.error(f"Stats reporter error: {e}")
    
    async def _handle_orchestrator_message(self, message: AgentMessage):
        """Handle messages sent to the orchestrator"""
        try:
            if message.message_type == "heartbeat":
                agent_name = message.sender
                health_status = message.data.get("health_status", "healthy")
                await self.agent_registry.update_heartbeat(agent_name, health_status)
                
            elif message.message_type == "error_report":
                logging.error(f"Agent {message.sender} reported error: {message.data}")
                self.stats["errors_encountered"] += 1
                
            elif message.message_type == "capability_update":
                # Handle dynamic capability updates
                agent_name = message.sender
                new_capabilities = message.data.get("capabilities", [])
                # Update agent capabilities (implementation would go here)
                
        except Exception as e:
            logging.error(f"Error handling orchestrator message: {e}")
    
    def _calculate_uptime(self) -> str:
        """Calculate orchestrator uptime"""
        if self.stats["orchestrator_started"]:
            start_time = datetime.fromisoformat(self.stats["orchestrator_started"])
            uptime = datetime.now() - start_time
            return str(uptime)
        return "0:00:00"
    
    async def _get_shared_state_summary(self) -> Dict[str, Any]:
        """Get summary of shared state"""
        try:
            all_data = await self.shared_state.get_all()
            return {
                "transactions_count": len(all_data.get("transactions", [])),
                "processing_status": all_data.get("processing_status", "idle"),
                "health_metrics_available": bool(all_data.get("health_metrics")),
                "conversation_history_count": len(all_data.get("conversation_history", [])),
                "insights_count": len(all_data.get("insights", [])),
                "recommendations_count": len(all_data.get("recommendations", []))
            }
        except Exception as e:
            return {"error": str(e)}

# ==============================================================================
# STANDALONE SERVICE
# ==============================================================================

class OrchestratorService:
    """Standalone orchestrator service"""
    
    def __init__(self, config_file: str = None):
        self.config = Config.from_env() if not config_file else Config()
        setup_logging(self.config)
        self.orchestrator = AgentOrchestrator(self.config)
        
    async def start(self):
        """Start the orchestrator service"""
        await self.orchestrator.start()
        logging.info("Orchestrator Service started")
    
    async def stop(self):
        """Stop the orchestrator service"""
        await self.orchestrator.stop()
        logging.info("Orchestrator Service stopped")
    
    async def run_forever(self):
        """Run the orchestrator service indefinitely"""
        try:
            while self.orchestrator.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
        finally:
            await self.stop()

# ==============================================================================
# API ENDPOINTS (FastAPI Integration)
# ==============================================================================

try:
    from fastapi import FastAPI, HTTPException, WebSocket
    from pydantic import BaseModel
    from typing import List, Optional
    
    class AgentRegistrationRequest(BaseModel):
        agent_name: str
        capabilities: List[str]
        endpoint: Optional[str] = None
    
    class MessageRequest(BaseModel):
        sender: str
        recipient: str
        message_type: str
        data: Dict[str, Any]
        priority: Optional[str] = "normal"
    
    def create_orchestrator_api(service: OrchestratorService) -> FastAPI:
        """Create FastAPI app for orchestrator service"""
        app = FastAPI(title="EmpowerFin Agent Orchestrator API", version="2.0.0")
        
        @app.post("/agents/register")
        async def register_agent(request: AgentRegistrationRequest):
            """Register a new agent"""
            success = await service.orchestrator.register_agent(
                request.agent_name, request.capabilities, request.endpoint
            )
            if success:
                return {"success": True, "message": f"Agent {request.agent_name} registered"}
            else:
                raise HTTPException(status_code=400, detail="Failed to register agent")
        
        @app.delete("/agents/{agent_name}")
        async def unregister_agent(agent_name: str):
            """Unregister an agent"""
            success = await service.orchestrator.unregister_agent(agent_name)
            if success:
                return {"success": True, "message": f"Agent {agent_name} unregistered"}
            else:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        @app.post("/messages/send")
        async def send_message(request: MessageRequest):
            """Send a message through the orchestrator"""
            message = AgentMessage(
                sender=request.sender,
                recipient=request.recipient,
                message_type=request.message_type,
                data=request.data,
                priority=request.priority
            )
            
            success = await service.orchestrator.send_message(message)
            if success:
                return {"success": True, "message": "Message sent"}
            else:
                raise HTTPException(status_code=400, detail="Failed to send message")
        
        @app.get("/status")
        async def get_system_status():
            """Get comprehensive system status"""
            return await service.orchestrator.get_system_status()
        
        @app.get("/agents")
        async def get_all_agents():
            """Get all registered agents"""
            return service.orchestrator.agent_registry.get_all_agents()
        
        @app.get("/agents/{agent_name}")
        async def get_agent_info(agent_name: str):
            """Get information about a specific agent"""
            agent_info = service.orchestrator.agent_registry.get_agent_info(agent_name)
            if agent_info:
                return agent_info
            else:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        @app.websocket("/ws/{agent_name}")
        async def websocket_endpoint(websocket: WebSocket, agent_name: str):
            """WebSocket endpoint for real-time agent communication"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive messages from agent
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    message = AgentMessage.from_dict(message_data)
                    await service.orchestrator.send_message(message)
                    
                    # Send any pending messages to agent
                    pending_message = await service.orchestrator.receive_message(agent_name, timeout=0.1)
                    if pending_message:
                        await websocket.send_text(pending_message.to_json())
                        
            except Exception as e:
                logging.error(f"WebSocket error for {agent_name}: {e}")
            finally:
                await websocket.close()
        
        @app.get("/health")
        async def health_check_endpoint():
            """Health check endpoint"""
            return await health_check()
        
        return app

except ImportError:
    def create_orchestrator_api(service):
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return None

# ==============================================================================
# CLI INTERFACE
# ==============================================================================

async def main():
    """Main CLI interface for orchestrator service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EmpowerFin Agent Orchestrator")
    parser.add_argument("--service", "-s", action="store_true", help="Run as service")
    parser.add_argument("--api", "-a", action="store_true", help="Run API server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="API port")
    parser.add_argument("--config", "-c", help="Configuration file")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--test", "-t", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    service = OrchestratorService(args.config)
    
    if args.test:
        await run_tests(service)
    elif args.api:
        await run_api_server(service, args.port)
    elif args.service:
        await run_service(service)
    elif args.status:
        await show_status(service)
    else:
        parser.print_help()

async def run_tests(service: OrchestratorService):
    """Run test suite"""
    print("Running Agent Orchestrator tests...")
    
    await service.start()
    
    try:
        # Test agent registration
        success = await service.orchestrator.register_agent(
            "TestAgent", ["test_capability"], "http://localhost:8001"
        )
        print(f"✓ Agent registration test: {'passed' if success else 'failed'}")
        
        # Test message sending
        test_message = AgentMessage(
            sender="TestSender",
            recipient="TestAgent",
            message_type="test_message",
            data={"test": True}
        )
        
        success = await service.orchestrator.send_message(test_message)
        print(f"✓ Message sending test: {'passed' if success else 'failed'}")
        
        # Test system status
        status = await service.orchestrator.get_system_status()
        print(f"✓ System status test: {'passed' if status else 'failed'}")
        
        # Test agent unregistration
        success = await service.orchestrator.unregister_agent("TestAgent")
        print(f"✓ Agent unregistration test: {'passed' if success else 'failed'}")
        
        print("All tests completed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    await service.stop()

async def run_api_server(service: OrchestratorService, port: int):
    """Run as API server"""
    try:
        import uvicorn
        
        app = create_orchestrator_api(service)
        if not app:
            print("Cannot create API server. Install FastAPI and uvicorn.")
            return
        
        print(f"Starting Orchestrator API server on port {port}...")
        
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

async def run_service(service: OrchestratorService):
    """Run as a standalone service"""
    print("Starting Agent Orchestrator Service...")
    await service.start()
    await service.run_forever()

async def show_status(service: OrchestratorService):
    """Show system status"""
    await service.start()
    
    status = await service.orchestrator.get_system_status()
    
    print("Agent Orchestrator Status:")
    print(f"  Running: {status['orchestrator']['running']}")
    print(f"  Uptime: {status['orchestrator']['uptime']}")
    print(f"  Messages Processed: {status['orchestrator']['stats']['messages_processed']}")
    print(f"  Agents Registered: {status['orchestrator']['stats']['agents_registered']}")
    
    print(f"\nRegistered Agents ({len(status['agents'])}):")
    for agent_name, agent_info in status['agents'].items():
        print(f"  {agent_name}: {agent_info['health_status']} ({', '.join(agent_info['capabilities'])})")
    
    print(f"\nMessage Queue Status:")
    queue_status = status['message_queue']['queue_status']
    for agent_name, queue_info in queue_status.items():
        print(f"  {agent_name}: {queue_info['queue_size']} messages queued")
    
    await service.stop()

if __name__ == "__main__":
    asyncio.run(main())