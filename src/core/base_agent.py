"""
Base Agent Class - Foundation for all specialized agents.

Provides core functionality including:
- Lifecycle management
- Error handling with recovery
- Audit trail integration
- Inter-agent communication
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field


class AgentStatus(Enum):
    """Current operational status of an agent."""
    IDLE = auto()
    PROCESSING = auto()
    WAITING = auto()
    ERROR = auto()
    RECOVERING = auto()
    TERMINATED = auto()


class AgentCapability(Enum):
    """Capabilities that agents can declare."""
    EXTRACTION = "extraction"
    DECISION_MAKING = "decision_making"
    ACTION_EXECUTION = "action_execution"
    VERIFICATION = "verification"
    MONITORING = "monitoring"
    ESCALATION = "escalation"


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_base: float = Field(default=1.0, description="Base delay for exponential backoff")
    timeout_seconds: float = Field(default=30.0, description="Operation timeout")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for decisions")
    enable_self_correction: bool = Field(default=True, description="Enable automatic error correction")


class AgentResult(BaseModel):
    """Result from an agent operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = 0.0


class AgentContext(BaseModel):
    """Execution context passed to agent operations."""
    workflow_id: str
    execution_id: str
    step_number: int = 0
    parent_agent: Optional[str] = None
    timeout_remaining: float = 30.0
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class RecoveryStrategy:
    """Defines how to recover from a specific error type."""
    error_type: type
    handler: Callable
    max_attempts: int = 3
    description: str = ""


T = TypeVar('T')


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all agents in the system.
    
    Provides:
    - Lifecycle management (initialize, process, cleanup)
    - Built-in error handling and recovery
    - Audit trail integration
    - Inter-agent communication support
    
    Type parameter T represents the expected input type for the agent.
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        config: Optional[AgentConfig] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.capabilities = capabilities
        self.config = config or AgentConfig()
        self.status = AgentStatus.IDLE
        self._recovery_strategies: List[RecoveryStrategy] = []
        self._audit_events: List[Dict[str, Any]] = []
        self._created_at = datetime.utcnow()
        self._last_activity = datetime.utcnow()
        self._error_count = 0
        self._success_count = 0
        
    @property
    def agent_type(self) -> str:
        """Return the type name of this agent."""
        return self.__class__.__name__
    
    @abstractmethod
    async def process(self, input_data: T, context: AgentContext) -> AgentResult:
        """
        Main processing method - must be implemented by subclasses.
        
        Args:
            input_data: The input to process
            context: Execution context with workflow information
            
        Returns:
            AgentResult with processing outcome
        """
        pass
    
    async def execute(self, input_data: T, context: AgentContext) -> AgentResult:
        """
        Execute the agent with full error handling and audit trail.
        
        This wraps the process() method with:
        - Automatic retries
        - Error recovery
        - Audit logging
        - Performance tracking
        """
        start_time = datetime.utcnow()
        self.status = AgentStatus.PROCESSING
        self._last_activity = start_time
        
        # Log execution start
        self._log_audit_event(
            event_type="execution_start",
            context=context,
            details={"input_type": type(input_data).__name__}
        )
        
        attempt = 0
        last_error: Optional[Exception] = None
        result: Optional[AgentResult] = None
        
        while attempt < self.config.max_retries:
            attempt += 1
            
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.process(input_data, context),
                    timeout=self.config.timeout_seconds
                )
                
                # Validate result confidence
                if result.success and result.confidence < self.config.confidence_threshold:
                    self._log_audit_event(
                        event_type="low_confidence",
                        context=context,
                        details={
                            "confidence": result.confidence,
                            "threshold": self.config.confidence_threshold
                        }
                    )
                    result.warnings.append(
                        f"Low confidence result ({result.confidence:.2f})"
                    )
                
                if result.success:
                    self._success_count += 1
                    break
                    
                # Result indicates failure - attempt recovery
                if self.config.enable_self_correction and attempt < self.config.max_retries:
                    self.status = AgentStatus.RECOVERING
                    corrected = await self._attempt_self_correction(
                        input_data, context, result
                    )
                    if corrected:
                        result = corrected
                        break
                        
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Operation timed out after {self.config.timeout_seconds}s")
                self._log_audit_event(
                    event_type="timeout",
                    context=context,
                    details={"attempt": attempt, "timeout": self.config.timeout_seconds}
                )
                
            except Exception as e:
                last_error = e
                self._error_count += 1
                self._log_audit_event(
                    event_type="error",
                    context=context,
                    details={"attempt": attempt, "error": str(e), "error_type": type(e).__name__}
                )
                
                # Attempt recovery strategy
                if self.config.enable_self_correction:
                    recovery_result = await self._attempt_recovery(e, input_data, context)
                    if recovery_result:
                        result = recovery_result
                        break
            
            # Apply exponential backoff before retry
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay_base * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Construct final result if no successful result
        if result is None:
            result = AgentResult(
                success=False,
                error=str(last_error) if last_error else "Unknown error",
                execution_time_ms=execution_time,
                metadata={"attempts": attempt}
            )
        else:
            result.execution_time_ms = execution_time
            result.metadata["attempts"] = attempt
        
        # Log completion
        self._log_audit_event(
            event_type="execution_complete",
            context=context,
            details={
                "success": result.success,
                "attempts": attempt,
                "execution_time_ms": execution_time,
                "confidence": result.confidence
            }
        )
        
        self.status = AgentStatus.IDLE if result.success else AgentStatus.ERROR
        return result
    
    async def _attempt_self_correction(
        self,
        input_data: T,
        context: AgentContext,
        failed_result: AgentResult
    ) -> Optional[AgentResult]:
        """
        Attempt to self-correct after a non-successful result.
        Override in subclasses for custom correction logic.
        """
        return None
    
    async def _attempt_recovery(
        self,
        error: Exception,
        input_data: T,
        context: AgentContext
    ) -> Optional[AgentResult]:
        """Attempt to recover from an exception using registered strategies."""
        for strategy in self._recovery_strategies:
            if isinstance(error, strategy.error_type):
                self._log_audit_event(
                    event_type="recovery_attempt",
                    context=context,
                    details={
                        "strategy": strategy.description,
                        "error_type": type(error).__name__
                    }
                )
                try:
                    return await strategy.handler(error, input_data, context)
                except Exception as recovery_error:
                    self._log_audit_event(
                        event_type="recovery_failed",
                        context=context,
                        details={
                            "strategy": strategy.description,
                            "error": str(recovery_error)
                        }
                    )
        return None
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy for handling specific error types."""
        self._recovery_strategies.append(strategy)
    
    def _log_audit_event(
        self,
        event_type: str,
        context: AgentContext,
        details: Dict[str, Any]
    ) -> None:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.id,
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "event_type": event_type,
            "workflow_id": context.workflow_id,
            "execution_id": context.execution_id,
            "step_number": context.step_number,
            "details": details
        }
        self._audit_events.append(event)
        context.audit_trail.append(event)
    
    def get_audit_events(self) -> List[Dict[str, Any]]:
        """Return all audit events for this agent."""
        return self._audit_events.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return agent statistics."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.name,
            "capabilities": [c.value for c in self.capabilities],
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "success_count": self._success_count,
            "error_count": self._error_count,
            "total_operations": self._success_count + self._error_count
        }
    
    async def initialize(self) -> None:
        """
        Initialize the agent. Override for custom initialization.
        Called before the agent starts processing.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Cleanup agent resources. Override for custom cleanup.
        Called when the agent is being terminated.
        """
        self.status = AgentStatus.TERMINATED


class CompositeAgent(BaseAgent[T]):
    """
    An agent that delegates to multiple sub-agents and aggregates results.
    Useful for complex operations requiring multiple specialized agents.
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        sub_agents: List[BaseAgent],
        config: Optional[AgentConfig] = None
    ):
        super().__init__(name, capabilities, config)
        self.sub_agents = sub_agents
    
    async def process(self, input_data: T, context: AgentContext) -> AgentResult:
        """Process input through all sub-agents and aggregate results."""
        results = []
        all_successful = True
        combined_data = {}
        all_warnings = []
        
        for i, agent in enumerate(self.sub_agents):
            sub_context = AgentContext(
                workflow_id=context.workflow_id,
                execution_id=context.execution_id,
                step_number=context.step_number + i,
                parent_agent=self.name,
                timeout_remaining=context.timeout_remaining,
                shared_state=context.shared_state,
                audit_trail=context.audit_trail
            )
            
            result = await agent.execute(input_data, sub_context)
            results.append(result)
            
            if not result.success:
                all_successful = False
            if result.data:
                combined_data[agent.name] = result.data
            all_warnings.extend(result.warnings)
        
        # Calculate aggregate confidence
        valid_confidences = [r.confidence for r in results if r.success]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        
        return AgentResult(
            success=all_successful,
            data=combined_data,
            confidence=avg_confidence,
            warnings=all_warnings,
            metadata={
                "sub_agent_count": len(self.sub_agents),
                "sub_results": [
                    {"agent": a.name, "success": r.success}
                    for a, r in zip(self.sub_agents, results)
                ]
            }
        )
