"""
Recovery Strategies.

Implements various strategies for recovering from errors
in workflow execution, including retry, fallback, rollback,
and escalation.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class RecoveryAction(Enum):
    """Actions that can be taken for recovery."""
    RETRY = auto()
    FALLBACK = auto()
    SKIP = auto()
    ROLLBACK = auto()
    ESCALATE = auto()
    ABORT = auto()


class RecoveryResult(BaseModel):
    """Result of a recovery attempt."""
    action_taken: RecoveryAction
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    attempts: int = 1
    duration_ms: float = 0.0
    
    # For escalation
    escalated_to: Optional[str] = None
    
    # For rollback
    rolled_back_steps: List[str] = Field(default_factory=list)


class RecoveryContext(BaseModel):
    """Context for recovery decisions."""
    error: str
    error_type: str
    step_id: str
    step_name: str
    workflow_id: str
    execution_id: str
    attempt_number: int
    
    # Historical data
    previous_errors: List[str] = Field(default_factory=list)
    previous_recovery_actions: List[str] = Field(default_factory=list)
    
    # Current state
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    completed_steps: List[str] = Field(default_factory=list)


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""
    
    @property
    @abstractmethod
    def action(self) -> RecoveryAction:
        """The action this strategy performs."""
        pass
    
    @abstractmethod
    def can_handle(self, context: RecoveryContext) -> bool:
        """Check if this strategy can handle the given error context."""
        pass
    
    @abstractmethod
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute the recovery strategy."""
        pass


class RetryStrategy(RecoveryStrategy):
    """
    Strategy that retries the failed operation.
    
    Supports:
    - Configurable max retries
    - Exponential backoff
    - Error type filtering
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_errors: Optional[List[str]] = None
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_errors = retryable_errors or [
            "TimeoutError",
            "ConnectionError",
            "TemporaryError",
            "ServiceUnavailable"
        ]
    
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.RETRY
    
    def can_handle(self, context: RecoveryContext) -> bool:
        # Check if we haven't exceeded retries
        if context.attempt_number >= self.max_retries:
            return False
        
        # Check if error type is retryable
        if self.retryable_errors:
            return context.error_type in self.retryable_errors
        
        return True
    
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        # Calculate delay
        delay = min(
            self.initial_delay * (self.exponential_base ** (context.attempt_number - 1)),
            self.max_delay
        )
        
        await asyncio.sleep(delay)
        
        return RecoveryResult(
            action_taken=RecoveryAction.RETRY,
            success=True,  # Indicates retry should proceed
            message=f"Retrying after {delay:.1f}s delay (attempt {context.attempt_number + 1}/{self.max_retries})",
            attempts=context.attempt_number + 1
        )


class FallbackStrategy(RecoveryStrategy):
    """
    Strategy that falls back to an alternative action.
    
    Supports:
    - Alternative agent execution
    - Default value provision
    - Degraded mode operation
    """
    
    def __init__(
        self,
        fallback_handler: Optional[Callable] = None,
        default_value: Optional[Any] = None,
        applicable_errors: Optional[List[str]] = None
    ):
        self.fallback_handler = fallback_handler
        self.default_value = default_value
        self.applicable_errors = applicable_errors
    
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.FALLBACK
    
    def can_handle(self, context: RecoveryContext) -> bool:
        if self.applicable_errors:
            return context.error_type in self.applicable_errors
        return self.fallback_handler is not None or self.default_value is not None
    
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        start_time = datetime.utcnow()
        
        try:
            if self.fallback_handler:
                if asyncio.iscoroutinefunction(self.fallback_handler):
                    result_data = await self.fallback_handler(context)
                else:
                    result_data = self.fallback_handler(context)
            else:
                result_data = self.default_value
            
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return RecoveryResult(
                action_taken=RecoveryAction.FALLBACK,
                success=True,
                message="Fallback executed successfully",
                data={"fallback_result": result_data},
                duration_ms=duration
            )
        except Exception as e:
            return RecoveryResult(
                action_taken=RecoveryAction.FALLBACK,
                success=False,
                message=f"Fallback failed: {str(e)}"
            )


class SkipStrategy(RecoveryStrategy):
    """
    Strategy that skips the failed step and continues.
    
    Used for non-critical steps that shouldn't block workflow progress.
    """
    
    def __init__(
        self,
        skippable_steps: Optional[List[str]] = None,
        skippable_errors: Optional[List[str]] = None
    ):
        self.skippable_steps = skippable_steps or []
        self.skippable_errors = skippable_errors or []
    
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.SKIP
    
    def can_handle(self, context: RecoveryContext) -> bool:
        # Check if step is marked as skippable
        if self.skippable_steps and context.step_id in self.skippable_steps:
            return True
        
        # Check if error type allows skipping
        if self.skippable_errors and context.error_type in self.skippable_errors:
            return True
        
        return False
    
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        return RecoveryResult(
            action_taken=RecoveryAction.SKIP,
            success=True,
            message=f"Skipped step {context.step_name} due to error: {context.error_type}",
            data={"skipped_step": context.step_id}
        )


class RollbackStrategy(RecoveryStrategy):
    """
    Strategy that rolls back to a previous state.
    
    Supports:
    - Full rollback to start
    - Partial rollback to checkpoint
    - Compensating actions for completed steps
    """
    
    def __init__(
        self,
        compensating_actions: Optional[Dict[str, Callable]] = None,
        rollback_on_errors: Optional[List[str]] = None
    ):
        self.compensating_actions = compensating_actions or {}
        self.rollback_on_errors = rollback_on_errors or [
            "DataIntegrityError",
            "ConstraintViolation",
            "CriticalError"
        ]
    
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.ROLLBACK
    
    def can_handle(self, context: RecoveryContext) -> bool:
        if self.rollback_on_errors:
            return context.error_type in self.rollback_on_errors
        return False
    
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        rolled_back = []
        
        # Execute compensating actions in reverse order
        for step_id in reversed(context.completed_steps):
            if step_id in self.compensating_actions:
                try:
                    action = self.compensating_actions[step_id]
                    if asyncio.iscoroutinefunction(action):
                        await action(context.shared_state)
                    else:
                        action(context.shared_state)
                    rolled_back.append(step_id)
                except Exception as e:
                    return RecoveryResult(
                        action_taken=RecoveryAction.ROLLBACK,
                        success=False,
                        message=f"Rollback failed at step {step_id}: {str(e)}",
                        rolled_back_steps=rolled_back
                    )
        
        return RecoveryResult(
            action_taken=RecoveryAction.ROLLBACK,
            success=True,
            message=f"Successfully rolled back {len(rolled_back)} steps",
            rolled_back_steps=rolled_back
        )


class EscalateStrategy(RecoveryStrategy):
    """
    Strategy that escalates to human intervention.
    
    Used when automated recovery isn't possible and
    human decision-making is required.
    """
    
    def __init__(
        self,
        escalation_handler: Optional[Callable] = None,
        escalation_threshold: int = 3,  # Escalate after this many failures
        escalation_targets: Optional[List[str]] = None
    ):
        self.escalation_handler = escalation_handler
        self.escalation_threshold = escalation_threshold
        self.escalation_targets = escalation_targets or ["team_lead", "on_call"]
    
    @property
    def action(self) -> RecoveryAction:
        return RecoveryAction.ESCALATE
    
    def can_handle(self, context: RecoveryContext) -> bool:
        # Escalate if we've tried multiple times
        return context.attempt_number >= self.escalation_threshold
    
    async def execute(self, context: RecoveryContext) -> RecoveryResult:
        escalation_target = self.escalation_targets[0] if self.escalation_targets else "admin"
        
        if self.escalation_handler:
            try:
                if asyncio.iscoroutinefunction(self.escalation_handler):
                    await self.escalation_handler(context, escalation_target)
                else:
                    self.escalation_handler(context, escalation_target)
            except Exception as e:
                return RecoveryResult(
                    action_taken=RecoveryAction.ESCALATE,
                    success=False,
                    message=f"Escalation failed: {str(e)}"
                )
        
        return RecoveryResult(
            action_taken=RecoveryAction.ESCALATE,
            success=True,
            message=f"Escalated to {escalation_target}",
            escalated_to=escalation_target,
            data={
                "error": context.error,
                "step": context.step_name,
                "workflow": context.workflow_id
            }
        )


class RecoveryManager:
    """
    Manages error recovery across the system.
    
    Coordinates multiple recovery strategies and selects
    the appropriate one based on error context.
    """
    
    def __init__(self):
        self._strategies: List[RecoveryStrategy] = []
        self._error_history: Dict[str, List[RecoveryContext]] = {}
        self._recovery_stats: Dict[str, int] = {
            "total_recoveries": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0
        }
    
    def register_strategy(
        self,
        strategy: RecoveryStrategy,
        priority: int = 0
    ) -> None:
        """Register a recovery strategy with optional priority."""
        self._strategies.append((priority, strategy))
        self._strategies.sort(key=lambda x: x[0], reverse=True)
    
    def register_default_strategies(self) -> None:
        """Register standard recovery strategies."""
        self.register_strategy(RetryStrategy(), priority=100)
        self.register_strategy(FallbackStrategy(), priority=50)
        self.register_strategy(SkipStrategy(), priority=30)
        self.register_strategy(EscalateStrategy(), priority=10)
    
    async def recover(
        self,
        context: RecoveryContext
    ) -> RecoveryResult:
        """
        Attempt to recover from an error using registered strategies.
        
        Tries strategies in priority order until one succeeds.
        """
        self._recovery_stats["total_recoveries"] += 1
        
        # Track error for analysis
        key = f"{context.workflow_id}:{context.step_id}"
        if key not in self._error_history:
            self._error_history[key] = []
        self._error_history[key].append(context)
        
        # Try each strategy in order
        for priority, strategy in self._strategies:
            if strategy.can_handle(context):
                result = await strategy.execute(context)
                
                if result.success:
                    self._recovery_stats["successful_recoveries"] += 1
                    return result
                else:
                    # Strategy failed, try next
                    continue
        
        # No strategy succeeded
        self._recovery_stats["failed_recoveries"] += 1
        
        return RecoveryResult(
            action_taken=RecoveryAction.ABORT,
            success=False,
            message=f"All recovery strategies exhausted for error: {context.error_type}"
        )
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights."""
        patterns = {}
        
        for key, contexts in self._error_history.items():
            error_types = [c.error_type for c in contexts]
            patterns[key] = {
                "total_errors": len(contexts),
                "error_types": list(set(error_types)),
                "most_common_error": max(set(error_types), key=error_types.count) if error_types else None
            }
        
        return patterns
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        total = self._recovery_stats["total_recoveries"]
        successful = self._recovery_stats["successful_recoveries"]
        
        return {
            **self._recovery_stats,
            "success_rate": successful / total if total > 0 else 0.0,
            "strategies_registered": len(self._strategies),
            "unique_error_locations": len(self._error_history)
        }
