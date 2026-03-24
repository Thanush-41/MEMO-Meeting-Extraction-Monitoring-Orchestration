"""
Inter-Agent Messaging System.

Provides a robust message passing system for agent communication
with support for different message types, priorities, and routing.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

from pydantic import BaseModel, Field


class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    REQUEST = auto()      # Request for another agent to perform work
    RESPONSE = auto()     # Response to a request
    EVENT = auto()        # Notification of something that happened
    COMMAND = auto()      # Direct command to an agent
    QUERY = auto()        # Query for information
    BROADCAST = auto()    # Message to all agents
    ERROR = auto()        # Error notification
    HEARTBEAT = auto()    # Health check message


class MessagePriority(Enum):
    """Priority levels for message processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Message(BaseModel):
    """
    A message passed between agents.
    
    Messages are the primary means of inter-agent communication
    and are fully trackable for audit purposes.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Routing
    sender_id: str
    sender_name: str
    recipient_id: Optional[str] = None  # None for broadcasts
    recipient_name: Optional[str] = None
    
    # Content
    subject: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    # Context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    correlation_id: Optional[str] = None  # Links related messages
    reply_to: Optional[str] = None  # ID of message being replied to
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[float] = None  # Time to live
    requires_ack: bool = False
    
    # Tracking
    delivered: bool = False
    acknowledged: bool = False
    delivery_attempts: int = 0
    
    class Config:
        use_enum_values = True


class MessageHandler:
    """Handler for processing messages of specific types."""
    
    def __init__(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any],
        filter_func: Optional[Callable[[Message], bool]] = None
    ):
        self.message_type = message_type
        self.handler = handler
        self.filter_func = filter_func or (lambda m: True)


class MessageBus:
    """
    Central message bus for agent communication.
    
    Provides:
    - Publish/subscribe messaging
    - Direct agent-to-agent messaging
    - Priority-based message queuing
    - Message delivery tracking
    - Dead letter queue for failed deliveries
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self._queues: Dict[str, asyncio.PriorityQueue] = defaultdict(
            lambda: asyncio.PriorityQueue(maxsize=max_queue_size)
        )
        self._handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> agent_ids
        self._message_history: List[Message] = []
        self._dead_letter_queue: List[Message] = []
        self._running = False
        self._processors: Dict[str, asyncio.Task] = {}
        self._max_history = 10000
        
    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        
    async def stop(self) -> None:
        """Stop the message bus and all processors."""
        self._running = False
        for task in self._processors.values():
            task.cancel()
        self._processors.clear()
    
    def register_agent(self, agent_id: str) -> None:
        """Register an agent to receive messages."""
        # Initialize queue for agent
        _ = self._queues[agent_id]
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self._queues:
            del self._queues[agent_id]
        if agent_id in self._handlers:
            del self._handlers[agent_id]
        # Remove from all subscriptions
        for subscribers in self._subscriptions.values():
            subscribers.discard(agent_id)
    
    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic for broadcast messages."""
        self._subscriptions[topic].add(agent_id)
    
    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic."""
        self._subscriptions[topic].discard(agent_id)
    
    def add_handler(
        self,
        agent_id: str,
        handler: MessageHandler
    ) -> None:
        """Add a message handler for an agent."""
        self._handlers[agent_id].append(handler)
    
    async def send(self, message: Message) -> bool:
        """
        Send a message to a specific agent.
        
        Returns True if message was queued successfully.
        """
        if message.recipient_id is None:
            return False
            
        message.delivery_attempts += 1
        
        try:
            queue = self._queues[message.recipient_id]
            # Priority queue uses tuple (priority, timestamp, message)
            # Lower number = higher priority, so invert
            priority = 5 - message.priority.value if isinstance(message.priority, MessagePriority) else 5 - message.priority
            await queue.put((priority, message.timestamp.timestamp(), message))
            message.delivered = True
            self._record_message(message)
            return True
        except asyncio.QueueFull:
            self._dead_letter_queue.append(message)
            return False
    
    async def broadcast(self, message: Message, topic: str) -> int:
        """
        Broadcast a message to all subscribers of a topic.
        
        Returns the number of agents the message was sent to.
        """
        message.type = MessageType.BROADCAST
        subscribers = self._subscriptions.get(topic, set())
        sent_count = 0
        
        for agent_id in subscribers:
            # Create a copy for each subscriber
            msg_copy = message.model_copy()
            msg_copy.recipient_id = agent_id
            msg_copy.id = str(uuid.uuid4())  # Unique ID for each copy
            msg_copy.correlation_id = message.id  # Link to original
            
            if await self.send(msg_copy):
                sent_count += 1
        
        return sent_count
    
    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive the next message for an agent.
        
        Returns None if timeout expires or no message available.
        """
        queue = self._queues.get(agent_id)
        if queue is None:
            return None
            
        try:
            if timeout:
                _, _, message = await asyncio.wait_for(queue.get(), timeout)
            else:
                _, _, message = queue.get_nowait()
            return message
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None
    
    async def receive_all(self, agent_id: str) -> List[Message]:
        """Receive all pending messages for an agent."""
        messages = []
        while True:
            msg = await self.receive(agent_id)
            if msg is None:
                break
            messages.append(msg)
        return messages
    
    async def acknowledge(self, message_id: str) -> bool:
        """Acknowledge receipt of a message."""
        for msg in self._message_history:
            if msg.id == message_id:
                msg.acknowledged = True
                return True
        return False
    
    def get_message_history(
        self,
        workflow_id: Optional[str] = None,
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Message]:
        """Get message history with optional filters."""
        filtered = self._message_history
        
        if workflow_id:
            filtered = [m for m in filtered if m.workflow_id == workflow_id]
        if sender_id:
            filtered = [m for m in filtered if m.sender_id == sender_id]
        if recipient_id:
            filtered = [m for m in filtered if m.recipient_id == recipient_id]
            
        return filtered[-limit:]
    
    def get_dead_letters(self) -> List[Message]:
        """Get messages that failed to deliver."""
        return self._dead_letter_queue.copy()
    
    def clear_dead_letters(self) -> int:
        """Clear and return count of dead letters."""
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        return count
    
    def _record_message(self, message: Message) -> None:
        """Record a message in history."""
        self._message_history.append(message)
        # Trim history if too large
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics."""
        return {
            "registered_agents": len(self._queues),
            "total_topics": len(self._subscriptions),
            "messages_in_history": len(self._message_history),
            "dead_letters": len(self._dead_letter_queue),
            "queue_sizes": {
                agent_id: queue.qsize()
                for agent_id, queue in self._queues.items()
            }
        }


class RequestResponsePattern:
    """
    Helper for request-response messaging pattern.
    
    Simplifies making requests to agents and waiting for responses.
    """
    
    def __init__(self, message_bus: MessageBus, timeout: float = 30.0):
        self.message_bus = message_bus
        self.timeout = timeout
        self._pending_requests: Dict[str, asyncio.Future] = {}
    
    async def request(
        self,
        sender_id: str,
        sender_name: str,
        recipient_id: str,
        subject: str,
        payload: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Optional[Message]:
        """
        Send a request and wait for response.
        
        Returns the response message or None if timeout.
        """
        request_msg = Message(
            type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            subject=subject,
            payload=payload,
            workflow_id=workflow_id,
            requires_ack=True
        )
        
        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_msg.id] = future
        
        # Send request
        await self.message_bus.send(request_msg)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(future, self.timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_requests.pop(request_msg.id, None)
    
    def handle_response(self, response: Message) -> bool:
        """
        Handle an incoming response message.
        
        Returns True if the response matched a pending request.
        """
        if response.reply_to and response.reply_to in self._pending_requests:
            future = self._pending_requests[response.reply_to]
            if not future.done():
                future.set_result(response)
                return True
        return False
