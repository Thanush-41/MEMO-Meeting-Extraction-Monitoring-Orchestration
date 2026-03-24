"""
Audit Trail System.

Provides comprehensive logging and tracking of all agent decisions
and actions for full auditability and compliance requirements.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict

from pydantic import BaseModel, Field


class AuditLevel(Enum):
    """Severity/importance level of audit events."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    DECISION = auto()  # Special level for decision points


class AuditCategory(Enum):
    """Categories of audit events."""
    WORKFLOW = "workflow"
    AGENT = "agent"
    DECISION = "decision"
    ACTION = "action"
    ERROR = "error"
    RECOVERY = "recovery"
    ESCALATION = "escalation"
    INTEGRATION = "integration"
    SECURITY = "security"


class AuditEvent(BaseModel):
    """
    A single audit event capturing a decision or action.
    
    Every event includes:
    - What happened (event_type, description)
    - Who did it (agent_id, agent_name)
    - Why (reasoning, confidence)
    - Context (workflow, execution, step)
    - Impact (affected_entities, reversible)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Event classification
    level: AuditLevel
    category: AuditCategory
    event_type: str
    
    # Who/what triggered the event
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_type: Optional[str] = None
    
    # Context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    step_number: Optional[int] = None
    correlation_id: Optional[str] = None
    
    # Event details
    description: str
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    
    # Decision-specific fields
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list)
    decision_factors: Dict[str, Any] = Field(default_factory=dict)
    
    # Impact
    affected_entities: List[str] = Field(default_factory=list)
    reversible: bool = True
    rollback_info: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AuditQuery(BaseModel):
    """Query parameters for searching audit events."""
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    agent_id: Optional[str] = None
    level: Optional[AuditLevel] = None
    category: Optional[AuditCategory] = None
    event_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    limit: int = 100
    offset: int = 0


class DecisionRecord(BaseModel):
    """
    Detailed record of a decision made by an agent.
    
    Captures the full context and reasoning for audit and review.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    agent_id: str
    agent_name: str
    workflow_id: str
    execution_id: str
    step_number: int
    
    # Decision details
    decision_type: str
    decision_description: str
    decision_outcome: str
    
    # Reasoning
    input_data_summary: Dict[str, Any]
    reasoning_steps: List[str]
    confidence_score: float
    
    # Alternatives
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: Dict[str, str] = Field(default_factory=dict)
    
    # Validation
    validated: bool = False
    validator: Optional[str] = None
    validation_notes: Optional[str] = None
    
    # Human review
    requires_review: bool = False
    reviewed: bool = False
    reviewer: Optional[str] = None
    review_outcome: Optional[str] = None
    review_timestamp: Optional[datetime] = None


class AuditLogger:
    """
    Central audit logging system.
    
    Features:
    - Multi-destination logging (memory, file, database)
    - Structured event storage
    - Query and search capabilities
    - Export functionality
    - Real-time event streaming
    """
    
    def __init__(
        self,
        persist_to_file: bool = False,
        file_path: Optional[Path] = None,
        max_memory_events: int = 100000
    ):
        self._events: List[AuditEvent] = []
        self._decisions: List[DecisionRecord] = []
        self._persist_to_file = persist_to_file
        self._file_path = file_path or Path("audit_log.jsonl")
        self._max_memory_events = max_memory_events
        self._subscribers: List[Callable[[AuditEvent], None]] = []
        self._lock = asyncio.Lock()
        self._indexes: Dict[str, Dict[str, List[int]]] = {
            "workflow_id": defaultdict(list),
            "execution_id": defaultdict(list),
            "agent_id": defaultdict(list),
            "event_type": defaultdict(list),
        }
    
    async def log(self, event: AuditEvent) -> str:
        """
        Log an audit event.
        
        Returns the event ID.
        """
        async with self._lock:
            event_idx = len(self._events)
            self._events.append(event)
            
            # Update indexes
            if event.workflow_id:
                self._indexes["workflow_id"][event.workflow_id].append(event_idx)
            if event.execution_id:
                self._indexes["execution_id"][event.execution_id].append(event_idx)
            if event.agent_id:
                self._indexes["agent_id"][event.agent_id].append(event_idx)
            self._indexes["event_type"][event.event_type].append(event_idx)
            
            # Persist if enabled
            if self._persist_to_file:
                await self._persist_event(event)
            
            # Trim if necessary
            if len(self._events) > self._max_memory_events:
                await self._trim_events()
        
        # Notify subscribers (outside lock)
        await self._notify_subscribers(event)
        
        return event.id
    
    async def log_decision(self, decision: DecisionRecord) -> str:
        """Log a detailed decision record."""
        async with self._lock:
            self._decisions.append(decision)
        
        # Also create an audit event for the decision
        event = AuditEvent(
            level=AuditLevel.DECISION,
            category=AuditCategory.DECISION,
            event_type="agent_decision",
            agent_id=decision.agent_id,
            agent_name=decision.agent_name,
            workflow_id=decision.workflow_id,
            execution_id=decision.execution_id,
            step_number=decision.step_number,
            description=decision.decision_description,
            reasoning="\n".join(decision.reasoning_steps),
            confidence=decision.confidence_score,
            alternatives_considered=decision.alternatives,
            metadata={"decision_record_id": decision.id}
        )
        await self.log(event)
        
        return decision.id
    
    def log_sync(
        self,
        level: AuditLevel,
        category: AuditCategory,
        event_type: str,
        description: str,
        **kwargs
    ) -> AuditEvent:
        """Synchronous logging helper for convenience."""
        event = AuditEvent(
            level=level,
            category=category,
            event_type=event_type,
            description=description,
            **kwargs
        )
        self._events.append(event)
        return event
    
    async def query(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events with filtering."""
        results = self._events.copy()
        
        # Apply filters
        if query.workflow_id:
            indexes = self._indexes["workflow_id"].get(query.workflow_id, [])
            results = [self._events[i] for i in indexes if i < len(self._events)]
        
        if query.execution_id:
            results = [e for e in results if e.execution_id == query.execution_id]
        
        if query.agent_id:
            results = [e for e in results if e.agent_id == query.agent_id]
        
        if query.level:
            target_level = query.level.value if isinstance(query.level, AuditLevel) else query.level
            results = [e for e in results if e.level == target_level]
        
        if query.category:
            target_cat = query.category.value if isinstance(query.category, AuditCategory) else query.category
            results = [e for e in results if e.category == target_cat]
        
        if query.event_type:
            results = [e for e in results if e.event_type == query.event_type]
        
        if query.start_time:
            results = [e for e in results if e.timestamp >= query.start_time]
        
        if query.end_time:
            results = [e for e in results if e.timestamp <= query.end_time]
        
        if query.tags:
            results = [e for e in results if any(t in e.tags for t in query.tags)]
        
        # Apply pagination
        return results[query.offset:query.offset + query.limit]
    
    async def get_workflow_trail(
        self,
        execution_id: str
    ) -> List[AuditEvent]:
        """Get complete audit trail for a workflow execution."""
        return await self.query(AuditQuery(execution_id=execution_id, limit=10000))
    
    async def get_agent_history(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Get audit history for a specific agent."""
        return await self.query(AuditQuery(agent_id=agent_id, limit=limit))
    
    async def get_decisions(
        self,
        execution_id: Optional[str] = None,
        requires_review: Optional[bool] = None
    ) -> List[DecisionRecord]:
        """Get decision records with optional filtering."""
        results = self._decisions.copy()
        
        if execution_id:
            results = [d for d in results if d.execution_id == execution_id]
        
        if requires_review is not None:
            results = [d for d in results if d.requires_review == requires_review]
        
        return results
    
    async def mark_decision_reviewed(
        self,
        decision_id: str,
        reviewer: str,
        outcome: str,
        notes: Optional[str] = None
    ) -> bool:
        """Mark a decision as reviewed by a human."""
        for decision in self._decisions:
            if decision.id == decision_id:
                decision.reviewed = True
                decision.reviewer = reviewer
                decision.review_outcome = outcome
                decision.review_timestamp = datetime.utcnow()
                decision.validation_notes = notes
                return True
        return False
    
    def subscribe(self, callback: Callable[[AuditEvent], None]) -> None:
        """Subscribe to real-time audit events."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from audit events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    async def _notify_subscribers(self, event: AuditEvent) -> None:
        """Notify all subscribers of a new event."""
        for callback in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception:
                pass  # Don't let subscriber errors affect logging
    
    async def _persist_event(self, event: AuditEvent) -> None:
        """Persist an event to file."""
        try:
            with open(self._file_path, "a") as f:
                f.write(event.model_dump_json() + "\n")
        except Exception:
            pass  # Silently fail file writes to not disrupt main operation
    
    async def _trim_events(self) -> None:
        """Trim in-memory events to stay within limit."""
        # Keep most recent events
        trim_count = len(self._events) - self._max_memory_events // 2
        self._events = self._events[trim_count:]
        
        # Rebuild indexes
        self._indexes = {
            "workflow_id": defaultdict(list),
            "execution_id": defaultdict(list),
            "agent_id": defaultdict(list),
            "event_type": defaultdict(list),
        }
        for i, event in enumerate(self._events):
            if event.workflow_id:
                self._indexes["workflow_id"][event.workflow_id].append(i)
            if event.execution_id:
                self._indexes["execution_id"][event.execution_id].append(i)
            if event.agent_id:
                self._indexes["agent_id"][event.agent_id].append(i)
            self._indexes["event_type"][event.event_type].append(i)
    
    async def export_trail(
        self,
        execution_id: str,
        format: str = "json"
    ) -> str:
        """Export audit trail in specified format."""
        events = await self.get_workflow_trail(execution_id)
        
        if format == "json":
            return json.dumps([e.model_dump() for e in events], indent=2, default=str)
        elif format == "markdown":
            return self._format_markdown(events)
        else:
            return json.dumps([e.model_dump() for e in events], default=str)
    
    def _format_markdown(self, events: List[AuditEvent]) -> str:
        """Format events as markdown report."""
        lines = ["# Audit Trail Report\n"]
        lines.append(f"Generated: {datetime.utcnow().isoformat()}\n")
        lines.append(f"Total Events: {len(events)}\n")
        lines.append("\n## Events\n")
        
        for event in events:
            lines.append(f"### {event.timestamp.isoformat()} - {event.event_type}")
            lines.append(f"- **Level**: {event.level}")
            lines.append(f"- **Category**: {event.category}")
            lines.append(f"- **Agent**: {event.agent_name or 'N/A'}")
            lines.append(f"- **Description**: {event.description}")
            if event.reasoning:
                lines.append(f"- **Reasoning**: {event.reasoning}")
            if event.confidence is not None:
                lines.append(f"- **Confidence**: {event.confidence:.2%}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logger statistics."""
        level_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for event in self._events:
            level_counts[event.level] += 1
            category_counts[event.category] += 1
        
        return {
            "total_events": len(self._events),
            "total_decisions": len(self._decisions),
            "events_by_level": dict(level_counts),
            "events_by_category": dict(category_counts),
            "subscribers": len(self._subscribers),
            "persisting": self._persist_to_file
        }


# Convenience functions for audit logging
def create_agent_event(
    agent_id: str,
    agent_name: str,
    event_type: str,
    description: str,
    workflow_id: str,
    execution_id: str,
    **kwargs
) -> AuditEvent:
    """Create an agent-related audit event."""
    return AuditEvent(
        level=AuditLevel.INFO,
        category=AuditCategory.AGENT,
        event_type=event_type,
        agent_id=agent_id,
        agent_name=agent_name,
        workflow_id=workflow_id,
        execution_id=execution_id,
        description=description,
        **kwargs
    )


def create_decision_event(
    agent_id: str,
    agent_name: str,
    decision_type: str,
    description: str,
    reasoning: str,
    confidence: float,
    workflow_id: str,
    execution_id: str,
    alternatives: Optional[List[Dict]] = None,
    **kwargs
) -> AuditEvent:
    """Create a decision audit event."""
    return AuditEvent(
        level=AuditLevel.DECISION,
        category=AuditCategory.DECISION,
        event_type=decision_type,
        agent_id=agent_id,
        agent_name=agent_name,
        workflow_id=workflow_id,
        execution_id=execution_id,
        description=description,
        reasoning=reasoning,
        confidence=confidence,
        alternatives_considered=alternatives or [],
        **kwargs
    )


def create_error_event(
    error: Exception,
    agent_id: Optional[str],
    agent_name: Optional[str],
    workflow_id: str,
    execution_id: str,
    recovery_action: Optional[str] = None,
    **kwargs
) -> AuditEvent:
    """Create an error audit event."""
    import traceback
    
    return AuditEvent(
        level=AuditLevel.ERROR,
        category=AuditCategory.ERROR,
        event_type="error_occurred",
        agent_id=agent_id,
        agent_name=agent_name,
        workflow_id=workflow_id,
        execution_id=execution_id,
        description=str(error),
        error_type=type(error).__name__,
        error_message=str(error),
        stack_trace=traceback.format_exc(),
        recovery_action=recovery_action,
        **kwargs
    )
