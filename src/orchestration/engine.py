"""
Workflow Engine.

Central orchestration component that manages workflow execution,
agent coordination, and state management.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from src.core.base_agent import BaseAgent, AgentContext, AgentResult
from src.core.state import StateManager, WorkflowState, WorkflowStatus, StepState
from src.core.audit import AuditLogger, AuditEvent, AuditLevel, AuditCategory
from src.core.message import MessageBus


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class RetryPolicy(BaseModel):
    """Policy for retrying failed steps."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for given attempt number."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


class WorkflowStep(BaseModel):
    """Definition of a single workflow step."""
    id: str
    name: str
    description: str = ""
    
    # Agent configuration
    agent_type: str  # Class name of agent to use
    agent_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Input/output mapping
    input_mapping: Dict[str, str] = Field(default_factory=dict)
    # Maps step input names to workflow data paths
    # e.g., {"transcript": "shared.transcript_analysis"}
    
    output_key: str = ""  # Where to store output in shared state
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Step IDs
    
    # Execution control
    timeout_seconds: float = 300.0
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    continue_on_failure: bool = False
    
    # Conditions
    skip_condition: Optional[str] = None  # Python expression to evaluate
    
    class Config:
        arbitrary_types_allowed = True


class Workflow(BaseModel):
    """Definition of a complete workflow."""
    id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Steps
    steps: List[WorkflowStep] = Field(default_factory=list)
    
    # Input/output schema
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuration
    timeout_seconds: float = 3600.0  # Total workflow timeout
    max_parallel_steps: int = 5
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)


class WorkflowExecutionResult(BaseModel):
    """Result of a workflow execution."""
    execution_id: str
    workflow_id: str
    status: str
    
    # Results
    output: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Step results
    step_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Errors
    error: Optional[str] = None
    failed_step: Optional[str] = None
    
    # Audit
    audit_trail_id: Optional[str] = None


class WorkflowEngine:
    """
    Central workflow orchestration engine.
    
    Responsibilities:
    - Workflow execution management
    - Agent coordination
    - State management
    - Error handling and recovery
    - Audit trail integration
    """
    
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        message_bus: Optional[MessageBus] = None
    ):
        self.state_manager = state_manager or StateManager()
        self.audit_logger = audit_logger or AuditLogger()
        self.message_bus = message_bus or MessageBus()
        
        # Agent registry
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._agent_instances: Dict[str, BaseAgent] = {}
        
        # Running workflows
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_results: Dict[str, WorkflowExecutionResult] = {}
    
    def register_agent(self, agent_type: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class for use in workflows."""
        self._agents[agent_type] = agent_class
    
    def register_agents(self, agents: Dict[str, Type[BaseAgent]]) -> None:
        """Register multiple agent classes."""
        self._agents.update(agents)
    
    async def execute(
        self,
        workflow: Workflow,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow with the given input data.
        
        Args:
            workflow: The workflow definition to execute
            input_data: Input data for the workflow
            execution_id: Optional execution ID (generated if not provided)
            
        Returns:
            WorkflowExecutionResult with outcomes and audit trail
        """
        execution_id = execution_id or str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        
        # Log start
        await self.audit_logger.log(AuditEvent(
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW,
            event_type="workflow_started",
            workflow_id=workflow.id,
            execution_id=execution_id,
            description=f"Started workflow: {workflow.name}"
        ))
        
        # Initialize workflow state
        state = await self.state_manager.create_workflow(
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            input_data=input_data,
            metadata={"version": workflow.version, "tags": workflow.tags}
        )
        
        # Initialize steps
        for step in workflow.steps:
            state.steps[step.id] = StepState(
                step_id=step.id,
                step_name=step.name,
                step_number=workflow.steps.index(step),
                status="pending",
                dependencies=step.depends_on
            )
        state.total_steps = len(workflow.steps)
        
        # Execute workflow
        try:
            result = await self._execute_workflow(workflow, state, execution_id)
            
            # Log completion
            await self.audit_logger.log(AuditEvent(
                level=AuditLevel.INFO,
                category=AuditCategory.WORKFLOW,
                event_type="workflow_completed",
                workflow_id=workflow.id,
                execution_id=execution_id,
                description=f"Completed workflow: {workflow.name}",
                metadata={"status": result.status, "duration": result.duration_seconds}
            ))
            
            return result
            
        except Exception as e:
            # Log failure
            await self.audit_logger.log(AuditEvent(
                level=AuditLevel.ERROR,
                category=AuditCategory.WORKFLOW,
                event_type="workflow_failed",
                workflow_id=workflow.id,
                execution_id=execution_id,
                description=f"Workflow failed: {str(e)}",
                error_message=str(e)
            ))
            
            completed_at = datetime.now(timezone.utc)
            return WorkflowExecutionResult(
                execution_id=execution_id,
                workflow_id=workflow.id,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                error=str(e)
            )
    
    async def _execute_workflow(
        self,
        workflow: Workflow,
        state: WorkflowState,
        execution_id: str
    ) -> WorkflowExecutionResult:
        """Internal workflow execution logic."""
        started_at = datetime.now(timezone.utc)
        step_results = {}
        
        # Build dependency graph
        dependency_graph = {step.id: set(step.depends_on) for step in workflow.steps}
        completed_steps = set()
        failed_step = None
        
        # Execute steps
        while len(completed_steps) < len(workflow.steps):
            # Find ready steps (dependencies satisfied)
            ready_steps = [
                step for step in workflow.steps
                if step.id not in completed_steps
                and all(dep in completed_steps for dep in step.depends_on)
            ]
            
            if not ready_steps:
                # Check for deadlock
                remaining = [s.id for s in workflow.steps if s.id not in completed_steps]
                raise RuntimeError(f"Workflow deadlock - steps remaining but none ready: {remaining}")
            
            # Execute ready steps (potentially in parallel)
            parallel_limit = min(len(ready_steps), workflow.max_parallel_steps)
            batch = ready_steps[:parallel_limit]
            
            tasks = [
                self._execute_step(step, state, execution_id)
                for step in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(batch, results):
                if isinstance(result, Exception):
                    if not step.continue_on_failure:
                        failed_step = step.id
                        raise result
                    step_results[step.id] = {"status": "failed", "error": str(result)}
                else:
                    step_results[step.id] = result.model_dump() if hasattr(result, 'model_dump') else result
                
                completed_steps.add(step.id)
                state.current_step = len(completed_steps)
        
        completed_at = datetime.now(timezone.utc)
        
        # Gather output
        output = {}
        for step in workflow.steps:
            if step.output_key and step.id in step_results:
                result = step_results[step.id]
                if isinstance(result, dict) and result.get("success"):
                    output[step.output_key] = result.get("data", {})
        
        return WorkflowExecutionResult(
            execution_id=execution_id,
            workflow_id=workflow.id,
            status="completed" if not failed_step else "partial",
            output=output,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            step_results=step_results,
            failed_step=failed_step
        )
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        state: WorkflowState,
        execution_id: str
    ) -> AgentResult:
        """Execute a single workflow step."""
        # Update step status
        await self.state_manager.update_step(
            execution_id=state.execution_id,
            step_id=step.id,
            updates={"status": "running", "started_at": datetime.now(timezone.utc)}
        )
        
        # Check skip condition
        if step.skip_condition:
            try:
                should_skip = eval(step.skip_condition, {"state": state.shared_data})
                if should_skip:
                    await self.state_manager.update_step(
                        execution_id=state.execution_id,
                        step_id=step.id,
                        updates={"status": "skipped"}
                    )
                    return AgentResult(success=True, data={"skipped": True})
            except:
                pass  # If condition fails to evaluate, continue with step
        
        # Get or create agent instance
        agent = await self._get_agent(step.agent_type, step.agent_config)
        
        # Prepare input data
        input_data = self._resolve_input(step.input_mapping, state)
        
        # Calculate step number from the step's position in workflow
        step_number = hash(step.id) % 10000  # Convert step ID to numeric
        
        # Create execution context
        context = AgentContext(
            workflow_id=state.workflow_id,
            execution_id=execution_id,
            step_number=step_number,
            timeout_remaining=step.timeout_seconds,
            shared_state=state.shared_data,
            audit_trail=[]
        )
        
        # Execute with retry
        attempt = 0
        last_error = None
        
        while attempt < step.retry_policy.max_retries:
            attempt += 1
            
            try:
                # Execute agent
                result = await asyncio.wait_for(
                    agent.execute(input_data, context),
                    timeout=step.timeout_seconds
                )
                
                if result.success:
                    # Store output
                    if step.output_key and result.data:
                        await self.state_manager.set_shared_data(
                            execution_id=state.execution_id,
                            key=step.output_key,
                            value=result.data,
                            changed_by=agent.name
                        )
                    
                    # Update step status
                    await self.state_manager.update_step(
                        execution_id=state.execution_id,
                        step_id=step.id,
                        updates={
                            "status": "completed",
                            "completed_at": datetime.now(timezone.utc),
                            "output_data": result.data
                        }
                    )
                    
                    return result
                
                last_error = Exception(result.error or "Step failed without error message")
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Step timed out after {step.timeout_seconds}s")
            except Exception as e:
                last_error = e
            
            # Wait before retry
            if attempt < step.retry_policy.max_retries:
                delay = step.retry_policy.get_delay(attempt - 1)
                await asyncio.sleep(delay)
                
                await self.audit_logger.log(AuditEvent(
                    level=AuditLevel.WARNING,
                    category=AuditCategory.WORKFLOW,
                    event_type="step_retry",
                    workflow_id=state.workflow_id,
                    execution_id=execution_id,
                    description=f"Retrying step {step.name} (attempt {attempt + 1})",
                    metadata={"step_id": step.id, "error": str(last_error)}
                ))
        
        # All retries exhausted
        await self.state_manager.update_step(
            execution_id=state.execution_id,
            step_id=step.id,
            updates={
                "status": "failed",
                "error": str(last_error),
                "retries": attempt
            }
        )
        
        raise last_error or Exception("Step failed after all retries")
    
    async def _get_agent(
        self,
        agent_type: str,
        config: Dict[str, Any]
    ) -> BaseAgent:
        """Get or create an agent instance."""
        # Check for existing instance
        cache_key = f"{agent_type}_{hash(str(sorted(config.items())))}"
        
        if cache_key in self._agent_instances:
            return self._agent_instances[cache_key]
        
        # Create new instance
        if agent_type not in self._agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agents[agent_type]
        agent = agent_class(**config) if config else agent_class()
        
        await agent.initialize()
        self._agent_instances[cache_key] = agent
        
        return agent
    
    def _resolve_input(
        self,
        mapping: Dict[str, str],
        state: WorkflowState
    ) -> Any:
        """Resolve input mapping to actual data."""
        if not mapping:
            return state.input_data
        
        resolved = {}
        for input_name, path in mapping.items():
            parts = path.split(".")
            value = None
            
            if parts[0] == "input":
                value = state.input_data
                parts = parts[1:]
            elif parts[0] == "shared":
                value = state.shared_data
                parts = parts[1:]
            else:
                value = state.shared_data.get(parts[0])
                parts = parts[1:]
            
            # Navigate path
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            
            resolved[input_name] = value
        
        return resolved
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        if execution_id in self._active_executions:
            task = self._active_executions[execution_id]
            task.cancel()
            
            await self.audit_logger.log(AuditEvent(
                level=AuditLevel.WARNING,
                category=AuditCategory.WORKFLOW,
                event_type="workflow_cancelled",
                execution_id=execution_id,
                description="Workflow cancelled by request"
            ))
            
            return True
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow execution."""
        result = self._execution_results.get(execution_id)
        if result:
            return result.model_dump()
        return None
    
    async def get_audit_trail(self, execution_id: str) -> List[AuditEvent]:
        """Get complete audit trail for an execution."""
        return await self.audit_logger.get_workflow_trail(execution_id)
    
    async def cleanup(self) -> None:
        """Cleanup engine resources."""
        # Cancel active executions
        for task in self._active_executions.values():
            task.cancel()
        
        # Cleanup agent instances
        for agent in self._agent_instances.values():
            await agent.cleanup()
        
        self._agent_instances.clear()
        self._active_executions.clear()
