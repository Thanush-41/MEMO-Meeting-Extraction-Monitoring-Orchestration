"""
State Management System.

Provides centralized state management for workflows and agents,
with support for transactions, snapshots, and state recovery.
"""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

from pydantic import BaseModel, Field


class WorkflowStatus(Enum):
    """Status of a workflow execution."""
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RECOVERING = auto()


class AgentState(BaseModel):
    """State information for a single agent."""
    agent_id: str
    agent_name: str
    status: str
    last_input: Optional[Dict[str, Any]] = None
    last_output: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    processing_count: int = 0
    error_count: int = 0
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    custom_state: Dict[str, Any] = Field(default_factory=dict)


class StepState(BaseModel):
    """State of a workflow step."""
    step_id: str
    step_name: str
    step_number: int
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    dependencies: List[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    """
    Complete state of a workflow execution.
    
    Tracks:
    - Workflow status and progress
    - Individual step states
    - Shared data between steps
    - Error information
    - Timing information
    """
    workflow_id: str
    workflow_name: str
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: WorkflowStatus = WorkflowStatus.PENDING
    
    # Progress tracking
    current_step: int = 0
    total_steps: int = 0
    steps: Dict[str, StepState] = Field(default_factory=dict)
    
    # Data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    shared_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    error: Optional[str] = None
    error_step: Optional[str] = None
    recovery_attempts: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class StateSnapshot(BaseModel):
    """A point-in-time snapshot of state."""
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    workflow_id: str
    execution_id: str
    step_number: int
    state_data: Dict[str, Any]
    reason: str = ""


class StateChange(BaseModel):
    """Record of a state change."""
    change_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    workflow_id: str
    execution_id: str
    path: str  # JSON path to changed value
    old_value: Any
    new_value: Any
    changed_by: str  # Agent or system ID
    reason: str = ""


class StateManager:
    """
    Central state management for all workflows and agents.
    
    Provides:
    - CRUD operations for workflow state
    - Transaction support
    - Snapshot/rollback capability
    - State change tracking
    - Concurrent access handling
    """
    
    def __init__(self):
        self._workflows: Dict[str, WorkflowState] = {}
        self._agents: Dict[str, AgentState] = {}
        self._snapshots: Dict[str, List[StateSnapshot]] = defaultdict(list)
        self._changes: Dict[str, List[StateChange]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
    # Workflow State Operations
    
    async def create_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Create a new workflow state."""
        async with self._locks[workflow_id]:
            state = WorkflowState(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                input_data=input_data or {},
                metadata=metadata or {}
            )
            self._workflows[state.execution_id] = state
            await self._notify_subscribers(state.execution_id, "created", state)
            return state
    
    async def get_workflow(self, execution_id: str) -> Optional[WorkflowState]:
        """Get workflow state by execution ID."""
        return self._workflows.get(execution_id)
    
    async def update_workflow(
        self,
        execution_id: str,
        updates: Dict[str, Any],
        changed_by: str = "system"
    ) -> Optional[WorkflowState]:
        """Update workflow state with change tracking."""
        async with self._locks[execution_id]:
            state = self._workflows.get(execution_id)
            if state is None:
                return None
            
            # Track changes
            for path, new_value in updates.items():
                old_value = getattr(state, path, None)
                if old_value != new_value:
                    change = StateChange(
                        workflow_id=state.workflow_id,
                        execution_id=execution_id,
                        path=path,
                        old_value=old_value,
                        new_value=new_value,
                        changed_by=changed_by
                    )
                    self._changes[execution_id].append(change)
                    setattr(state, path, new_value)
            
            await self._notify_subscribers(execution_id, "updated", state)
            return state
    
    async def update_step(
        self,
        execution_id: str,
        step_id: str,
        updates: Dict[str, Any],
        changed_by: str = "system"
    ) -> Optional[StepState]:
        """Update a specific step within a workflow."""
        async with self._locks[execution_id]:
            state = self._workflows.get(execution_id)
            if state is None or step_id not in state.steps:
                return None
            
            step = state.steps[step_id]
            for key, value in updates.items():
                if hasattr(step, key):
                    setattr(step, key, value)
            
            await self._notify_subscribers(execution_id, "step_updated", step)
            return step
    
    async def set_shared_data(
        self,
        execution_id: str,
        key: str,
        value: Any,
        changed_by: str = "system"
    ) -> bool:
        """Set a value in shared workflow data."""
        async with self._locks[execution_id]:
            state = self._workflows.get(execution_id)
            if state is None:
                return False
            
            old_value = state.shared_data.get(key)
            state.shared_data[key] = value
            
            change = StateChange(
                workflow_id=state.workflow_id,
                execution_id=execution_id,
                path=f"shared_data.{key}",
                old_value=old_value,
                new_value=value,
                changed_by=changed_by
            )
            self._changes[execution_id].append(change)
            
            return True
    
    async def get_shared_data(
        self,
        execution_id: str,
        key: Optional[str] = None
    ) -> Any:
        """Get shared data from workflow."""
        state = self._workflows.get(execution_id)
        if state is None:
            return None
        if key is None:
            return state.shared_data.copy()
        return state.shared_data.get(key)
    
    # Agent State Operations
    
    async def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        initial_status: str = "idle"
    ) -> AgentState:
        """Register an agent with the state manager."""
        state = AgentState(
            agent_id=agent_id,
            agent_name=agent_name,
            status=initial_status
        )
        self._agents[agent_id] = state
        return state
    
    async def update_agent_state(
        self,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> Optional[AgentState]:
        """Update agent state."""
        state = self._agents.get(agent_id)
        if state is None:
            return None
        
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        state.last_activity = datetime.utcnow()
        return state
    
    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state."""
        return self._agents.get(agent_id)
    
    # Snapshot Operations
    
    async def create_snapshot(
        self,
        execution_id: str,
        reason: str = ""
    ) -> Optional[StateSnapshot]:
        """Create a point-in-time snapshot of workflow state."""
        state = self._workflows.get(execution_id)
        if state is None:
            return None
        
        snapshot = StateSnapshot(
            workflow_id=state.workflow_id,
            execution_id=execution_id,
            step_number=state.current_step,
            state_data=state.model_dump(),
            reason=reason
        )
        self._snapshots[execution_id].append(snapshot)
        return snapshot
    
    async def rollback_to_snapshot(
        self,
        execution_id: str,
        snapshot_id: str
    ) -> Optional[WorkflowState]:
        """Rollback workflow state to a snapshot."""
        snapshots = self._snapshots.get(execution_id, [])
        snapshot = next((s for s in snapshots if s.snapshot_id == snapshot_id), None)
        
        if snapshot is None:
            return None
        
        async with self._locks[execution_id]:
            # Create current state snapshot before rollback
            await self.create_snapshot(execution_id, f"Pre-rollback to {snapshot_id}")
            
            # Restore state
            restored = WorkflowState(**snapshot.state_data)
            restored.recovery_attempts += 1
            self._workflows[execution_id] = restored
            
            await self._notify_subscribers(execution_id, "rolled_back", restored)
            return restored
    
    async def get_snapshots(
        self,
        execution_id: str
    ) -> List[StateSnapshot]:
        """Get all snapshots for an execution."""
        return self._snapshots.get(execution_id, []).copy()
    
    # Change History
    
    async def get_change_history(
        self,
        execution_id: str,
        limit: int = 100
    ) -> List[StateChange]:
        """Get state change history for a workflow."""
        changes = self._changes.get(execution_id, [])
        return changes[-limit:]
    
    # Subscriptions
    
    def subscribe(
        self,
        execution_id: str,
        callback: Callable[[str, str, Any], None]
    ) -> None:
        """Subscribe to state changes for a workflow."""
        self._subscribers[execution_id].append(callback)
    
    def unsubscribe(
        self,
        execution_id: str,
        callback: Callable
    ) -> None:
        """Unsubscribe from state changes."""
        if callback in self._subscribers[execution_id]:
            self._subscribers[execution_id].remove(callback)
    
    async def _notify_subscribers(
        self,
        execution_id: str,
        event_type: str,
        data: Any
    ) -> None:
        """Notify subscribers of state changes."""
        for callback in self._subscribers.get(execution_id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(execution_id, event_type, data)
                else:
                    callback(execution_id, event_type, data)
            except Exception:
                pass  # Don't let subscriber errors affect state management
    
    # Utilities
    
    async def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        workflow_id: Optional[str] = None
    ) -> List[WorkflowState]:
        """List workflow states with optional filtering."""
        workflows = list(self._workflows.values())
        
        if status is not None:
            workflows = [w for w in workflows if w.status == status]
        if workflow_id is not None:
            workflows = [w for w in workflows if w.workflow_id == workflow_id]
        
        return workflows
    
    async def cleanup_completed(
        self,
        older_than_hours: int = 24
    ) -> int:
        """Remove completed workflow states older than specified hours."""
        cutoff = datetime.utcnow()
        count = 0
        
        to_remove = []
        for exec_id, state in self._workflows.items():
            if state.status in (WorkflowStatus.COMPLETED, WorkflowStatus.CANCELLED):
                if state.completed_at and (cutoff - state.completed_at).total_seconds() > older_than_hours * 3600:
                    to_remove.append(exec_id)
        
        for exec_id in to_remove:
            del self._workflows[exec_id]
            self._snapshots.pop(exec_id, None)
            self._changes.pop(exec_id, None)
            self._locks.pop(exec_id, None)
            count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        status_counts = defaultdict(int)
        for state in self._workflows.values():
            status_counts[state.status.name if isinstance(state.status, WorkflowStatus) else state.status] += 1
        
        return {
            "total_workflows": len(self._workflows),
            "total_agents": len(self._agents),
            "workflows_by_status": dict(status_counts),
            "total_snapshots": sum(len(s) for s in self._snapshots.values()),
            "total_changes": sum(len(c) for c in self._changes.values())
        }
