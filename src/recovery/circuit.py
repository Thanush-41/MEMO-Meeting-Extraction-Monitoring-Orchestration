"""
Circuit Breaker Implementation.

Provides protection against cascading failures by temporarily
stopping requests to failing services.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import functools


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = auto()      # Normal operation, requests allowed
    OPEN = auto()        # Failure threshold reached, requests blocked
    HALF_OPEN = auto()   # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes in half-open before closing
    timeout_seconds: float = 30.0       # Time to wait before half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    
    # Optional sliding window
    sliding_window_size: int = 10       # Number of calls to track
    sliding_window_failure_rate: float = 0.5  # Failure rate to trigger


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation.
    
    Protects against cascading failures by:
    1. Monitoring call success/failure rates
    2. Opening (blocking calls) when failure threshold reached
    3. Allowing periodic test calls (half-open state)
    4. Closing (resuming normal operation) when service recovers
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._opened_at: Optional[datetime] = None
        self._half_open_calls = 0
        self._call_history: List[bool] = []  # Recent call outcomes
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    async def call(
        self,
        func: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: The function to call
            fallback: Optional fallback function if circuit is open
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        async with self._lock:
            # Check if call is allowed
            allowed, reason = await self._check_call_allowed()
            
            if not allowed:
                self._stats.rejected_calls += 1
                
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is {self._state.name}: {reason}"
                )
        
        # Execute the call
        self._stats.total_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _check_call_allowed(self) -> tuple[bool, str]:
        """Check if a call is allowed based on circuit state."""
        now = datetime.now(timezone.utc)
        
        if self._state == CircuitState.CLOSED:
            return True, ""
        
        elif self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._opened_at:
                elapsed = (now - self._opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    # Transition to half-open
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True, ""
            
            return False, f"Circuit open, retry after {self.config.timeout_seconds}s"
        
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True, ""
            
            return False, "Half-open call limit reached"
        
        return False, "Unknown state"
    
    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.successful_calls += 1
            self._stats.last_success_time = datetime.now(timezone.utc)
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            
            # Update sliding window
            self._call_history.append(True)
            if len(self._call_history) > self.config.sliding_window_size:
                self._call_history.pop(0)
            
            # Check for state transitions
            if self._state == CircuitState.HALF_OPEN:
                if self._stats.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failed_calls += 1
            self._stats.last_failure_time = datetime.now(timezone.utc)
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            
            # Update sliding window
            self._call_history.append(False)
            if len(self._call_history) > self.config.sliding_window_size:
                self._call_history.pop(0)
            
            # Check for state transitions
            if self._state == CircuitState.CLOSED:
                # Check threshold
                if self._stats.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
                # Also check sliding window failure rate
                elif len(self._call_history) >= self.config.sliding_window_size:
                    failure_rate = 1 - (sum(self._call_history) / len(self._call_history))
                    if failure_rate >= self.config.sliding_window_failure_rate:
                        await self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                await self._transition_to(CircuitState.OPEN)
    
    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        
        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now(timezone.utc)
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._stats.consecutive_failures = 0
        
        # Notify state change
        if self.on_state_change and old_state != new_state:
            if asyncio.iscoroutinefunction(self.on_state_change):
                await self.on_state_change(self.name, old_state, new_state)
            else:
                self.on_state_change(self.name, old_state, new_state)
    
    async def force_open(self) -> None:
        """Manually force the circuit open."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
    
    async def force_close(self) -> None:
        """Manually force the circuit closed."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
    
    async def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._opened_at = None
            self._half_open_calls = 0
            self._call_history.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        total = self._stats.total_calls
        return {
            "name": self.name,
            "state": self._state.name,
            "total_calls": total,
            "successful_calls": self._stats.successful_calls,
            "failed_calls": self._stats.failed_calls,
            "rejected_calls": self._stats.rejected_calls,
            "success_rate": self._stats.successful_calls / total if total > 0 else 1.0,
            "failure_rate": self._stats.failed_calls / total if total > 0 else 0.0,
            "consecutive_failures": self._stats.consecutive_failures,
            "last_failure": self._stats.last_failure_time.isoformat() if self._stats.last_failure_time else None,
            "last_success": self._stats.last_success_time.isoformat() if self._stats.last_success_time else None,
        }


class CircuitOpenError(Exception):
    """Raised when a call is rejected due to open circuit."""
    pass


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    async def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)
    
    async def remove(self, name: str) -> bool:
        """Remove a circuit breaker."""
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }
    
    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuits."""
        return [
            name for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]


def circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Optional[Callable] = None
):
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Usage:
        breaker = CircuitBreaker("my_service")
        
        @circuit_breaker(breaker)
        async def call_service():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, fallback=fallback, **kwargs)
        return wrapper
    return decorator
