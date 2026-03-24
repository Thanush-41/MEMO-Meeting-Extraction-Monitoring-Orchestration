"""
Task Scheduler.

Provides scheduling capabilities for workflows and tasks,
including periodic execution, delayed starts, and dependencies.
"""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from pydantic import BaseModel, Field


class ScheduleType(Enum):
    """Types of schedules."""
    ONCE = auto()
    RECURRING = auto()
    CRON = auto()


class RecurrenceInterval(BaseModel):
    """Defines a recurrence pattern."""
    minutes: int = 0
    hours: int = 0
    days: int = 0
    weeks: int = 0
    
    def to_timedelta(self) -> timedelta:
        return timedelta(
            minutes=self.minutes,
            hours=self.hours,
            days=self.days,
            weeks=self.weeks
        )


class ScheduledTask(BaseModel):
    """A scheduled task or workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    
    # What to run
    workflow_id: Optional[str] = None
    callback_name: Optional[str] = None  # For custom callbacks
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    # Schedule
    schedule_type: ScheduleType = ScheduleType.ONCE
    scheduled_time: datetime
    recurrence: Optional[RecurrenceInterval] = None
    max_occurrences: Optional[int] = None  # For recurring tasks
    
    # Execution tracking
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    
    # Status
    enabled: bool = True
    
    # Dependencies (wait for these to complete first)
    wait_for: List[str] = Field(default_factory=list)  # Task IDs
    
    class Config:
        use_enum_values = True


@dataclass(order=True)
class ScheduledItem:
    """Item in the scheduler priority queue."""
    scheduled_time: datetime
    task_id: str = field(compare=False)
    task: ScheduledTask = field(compare=False)


class TaskScheduler:
    """
    Scheduler for workflows and tasks.
    
    Features:
    - One-time scheduling
    - Recurring schedules
    - Task dependencies
    - Priority queue execution
    """
    
    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._queue: List[ScheduledItem] = []  # Priority queue (min-heap)
        self._callbacks: Dict[str, Callable] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._completed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
    
    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function for scheduled tasks."""
        self._callbacks[name] = callback
    
    async def schedule(self, task: ScheduledTask) -> str:
        """Schedule a new task."""
        async with self._lock:
            task.next_execution = task.scheduled_time
            self._tasks[task.id] = task
            
            # Add to queue if enabled
            if task.enabled:
                heapq.heappush(
                    self._queue,
                    ScheduledItem(task.scheduled_time, task.id, task)
                )
            
            return task.id
    
    async def schedule_workflow(
        self,
        workflow_id: str,
        scheduled_time: datetime,
        payload: Dict[str, Any],
        name: Optional[str] = None,
        recurrence: Optional[RecurrenceInterval] = None
    ) -> str:
        """Convenience method to schedule a workflow."""
        task = ScheduledTask(
            name=name or f"Workflow {workflow_id}",
            workflow_id=workflow_id,
            schedule_type=ScheduleType.RECURRING if recurrence else ScheduleType.ONCE,
            scheduled_time=scheduled_time,
            recurrence=recurrence,
            payload=payload
        )
        return await self.schedule(task)
    
    async def schedule_callback(
        self,
        callback_name: str,
        scheduled_time: datetime,
        payload: Dict[str, Any],
        name: Optional[str] = None
    ) -> str:
        """Schedule a callback function."""
        if callback_name not in self._callbacks:
            raise ValueError(f"Unknown callback: {callback_name}")
        
        task = ScheduledTask(
            name=name or f"Callback {callback_name}",
            callback_name=callback_name,
            scheduled_time=scheduled_time,
            payload=payload
        )
        return await self.schedule(task)
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].enabled = False
                del self._tasks[task_id]
                return True
            return False
    
    async def pause(self, task_id: str) -> bool:
        """Pause a scheduled task."""
        if task_id in self._tasks:
            self._tasks[task_id].enabled = False
            return True
        return False
    
    async def resume(self, task_id: str) -> bool:
        """Resume a paused task."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.enabled = True
                
                # Re-schedule if next execution is in the past
                if task.next_execution and task.next_execution < datetime.utcnow():
                    task.next_execution = datetime.utcnow() + timedelta(seconds=1)
                
                heapq.heappush(
                    self._queue,
                    ScheduledItem(task.next_execution, task.id, task)
                )
                return True
            return False
    
    async def start(self, workflow_executor: Optional[Callable] = None) -> None:
        """Start the scheduler."""
        self._running = True
        self._workflow_executor = workflow_executor
        self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def _worker_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._process_due_tasks()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue running
                print(f"Scheduler error: {e}")
                await asyncio.sleep(5)
    
    async def _process_due_tasks(self) -> None:
        """Process tasks that are due for execution."""
        now = datetime.utcnow()
        
        while self._queue and self._queue[0].scheduled_time <= now:
            async with self._lock:
                if not self._queue:
                    break
                    
                item = heapq.heappop(self._queue)
                task = self._tasks.get(item.task_id)
                
                if not task or not task.enabled:
                    continue
                
                # Check dependencies
                if task.wait_for:
                    pending_deps = [
                        dep for dep in task.wait_for
                        if dep not in self._completed_tasks
                    ]
                    if pending_deps:
                        # Re-schedule with delay
                        new_time = now + timedelta(seconds=30)
                        heapq.heappush(
                            self._queue,
                            ScheduledItem(new_time, task.id, task)
                        )
                        continue
                
                # Execute task
                asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task."""
        try:
            task.execution_count += 1
            task.last_execution = datetime.utcnow()
            
            if task.workflow_id and self._workflow_executor:
                await self._workflow_executor(task.workflow_id, task.payload)
            elif task.callback_name and task.callback_name in self._callbacks:
                callback = self._callbacks[task.callback_name]
                if asyncio.iscoroutinefunction(callback):
                    await callback(task.payload)
                else:
                    callback(task.payload)
            
            # Mark as completed
            self._completed_tasks.add(task.id)
            
            # Handle recurrence
            if task.schedule_type == ScheduleType.RECURRING and task.recurrence:
                if task.max_occurrences and task.execution_count >= task.max_occurrences:
                    task.enabled = False
                else:
                    # Schedule next occurrence
                    task.next_execution = task.last_execution + task.recurrence.to_timedelta()
                    async with self._lock:
                        heapq.heappush(
                            self._queue,
                            ScheduledItem(task.next_execution, task.id, task)
                        )
            else:
                # One-time task completed
                task.enabled = False
                
        except Exception as e:
            print(f"Task execution error for {task.id}: {e}")
    
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all pending tasks."""
        return [
            task for task in self._tasks.values()
            if task.enabled
        ]
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a specific task."""
        return self._tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        enabled = [t for t in self._tasks.values() if t.enabled]
        recurring = [t for t in enabled if t.schedule_type == ScheduleType.RECURRING]
        
        return {
            "total_tasks": len(self._tasks),
            "enabled_tasks": len(enabled),
            "recurring_tasks": len(recurring),
            "queue_size": len(self._queue),
            "completed_tasks": len(self._completed_tasks),
            "running": self._running
        }
