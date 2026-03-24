"""
MEMO — Meeting Extraction, Monitoring & Orchestration.

A multi-agent system that takes ownership of complex, multi-step 
enterprise processes with built-in failure detection, self-correction,
and complete auditability.
"""

__version__ = "1.0.0"
__author__ = "Enterprise AI Team"

from .core import (
    BaseAgent,
    AgentCapability,
    AgentStatus,
    Message,
    MessageType,
    StateManager,
    WorkflowState,
    AuditLogger,
    AuditEvent,
)

from .orchestration import (
    WorkflowEngine,
    Workflow,
    WorkflowStep,
    TaskScheduler,
    WorkflowMonitor,
)

from .recovery import (
    RecoveryManager,
    CircuitBreaker,
)

__all__ = [
    # Core
    "BaseAgent",
    "AgentCapability",
    "AgentStatus",
    "Message",
    "MessageType",
    "StateManager",
    "WorkflowState",
    "AuditLogger",
    "AuditEvent",
    # Orchestration
    "WorkflowEngine",
    "Workflow",
    "WorkflowStep",
    "TaskScheduler",
    "WorkflowMonitor",
    # Recovery
    "RecoveryManager",
    "CircuitBreaker",
]
