"""
Test Suite for MEMO — Meeting Extraction, Monitoring & Orchestration.

Covers:
- Agent unit tests
- Workflow integration tests
- Recovery mechanism tests
- Audit trail tests
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult
)
from src.core.state import StateManager, WorkflowStatus
from src.core.audit import AuditLogger, AuditEvent, AuditLevel, AuditCategory
from src.core.message import MessageBus, Message, MessageType, MessagePriority

from src.agents.extraction.transcript_agent import TranscriptAnalyzer, TranscriptInput
from src.agents.extraction.decision_parser import DecisionExtractor, DecisionExtractionInput
from src.agents.extraction.action_item_agent import ActionItemExtractor, ActionItemExtractionInput
from src.agents.decision.task_prioritizer import TaskPrioritizer, PrioritizationInput
from src.agents.decision.owner_assigner import OwnerAssigner, OwnerAssignmentInput, TeamMember
from src.agents.decision.escalation_decider import EscalationDecider, EscalationInput

from src.recovery.strategies import (
    RecoveryManager,
    RecoveryContext,
    RetryStrategy,
    FallbackStrategy,
    RecoveryAction
)
from src.recovery.circuit import CircuitBreaker, CircuitState, CircuitBreakerConfig


# Test fixtures
@pytest.fixture
def sample_transcript():
    return """
    Alice: Let's discuss the project timeline. I think we should move the deadline to next month.
    
    Bob: I agree with Alice. We need more time for testing.
    
    Alice: Great, let's decide to extend the deadline to March 15th. Bob, can you update the schedule?
    
    Bob: Sure, I'll have it ready by Friday. I'm also blocked on the API integration.
    
    Alice: Charlie, please help Bob with the API integration by tomorrow.
    
    Charlie: I'll take a look at it this afternoon. We also need to review the security requirements.
    
    Alice: Good point. Let's add that to our action items. Everyone agrees we prioritize security.
    """


@pytest.fixture
def agent_context():
    return AgentContext(
        workflow_id="test-workflow",
        execution_id="test-exec-001",
        step_number=1,
        shared_state={}
    )


# Agent Tests
class TestTranscriptAnalyzer:
    """Tests for TranscriptAnalyzer agent."""
    
    @pytest.mark.asyncio
    async def test_basic_analysis(self, sample_transcript, agent_context):
        """Test basic transcript analysis."""
        agent = TranscriptAnalyzer()
        input_data = TranscriptInput(
            transcript_text=sample_transcript,
            meeting_title="Test Meeting",
            known_participants=["Alice", "Bob", "Charlie"]
        )
        
        result = await agent.execute(input_data, agent_context)
        
        assert result.success
        assert result.confidence > 0.5
        assert result.data is not None
        assert "speakers" in result.data
        assert len(result.data["speakers"]) >= 2
    
    @pytest.mark.asyncio
    async def test_empty_transcript(self, agent_context):
        """Test handling of empty transcript."""
        agent = TranscriptAnalyzer()
        input_data = TranscriptInput(
            transcript_text="",
            meeting_title="Empty Meeting"
        )
        
        result = await agent.execute(input_data, agent_context)
        
        # Should handle empty input gracefully
        assert result.data is not None


class TestDecisionExtractor:
    """Tests for DecisionExtractor agent."""
    
    @pytest.mark.asyncio
    async def test_decision_extraction(self, sample_transcript, agent_context):
        """Test decision extraction from transcript."""
        # First analyze transcript
        analyzer = TranscriptAnalyzer()
        transcript_input = TranscriptInput(
            transcript_text=sample_transcript,
            known_participants=["Alice", "Bob", "Charlie"]
        )
        analysis_result = await analyzer.execute(transcript_input, agent_context)
        
        # Then extract decisions
        agent = DecisionExtractor()
        input_data = DecisionExtractionInput(
            transcript_analysis=analysis_result.data,
            participant_roles={"Alice": "manager", "Bob": "developer", "Charlie": "developer"}
        )
        
        result = await agent.execute(input_data, agent_context)
        
        assert result.success
        assert result.data is not None
        decisions = result.data.get("decisions", [])
        assert len(decisions) > 0


class TestActionItemExtractor:
    """Tests for ActionItemExtractor agent."""
    
    @pytest.mark.asyncio
    async def test_action_extraction(self, sample_transcript, agent_context):
        """Test action item extraction."""
        analyzer = TranscriptAnalyzer()
        transcript_input = TranscriptInput(
            transcript_text=sample_transcript,
            known_participants=["Alice", "Bob", "Charlie"]
        )
        analysis_result = await analyzer.execute(transcript_input, agent_context)
        
        agent = ActionItemExtractor()
        input_data = ActionItemExtractionInput(
            transcript_analysis=analysis_result.data,
            current_date=datetime.now()
        )
        
        result = await agent.execute(input_data, agent_context)
        
        assert result.success
        assert result.data is not None
        items = result.data.get("action_items", [])
        # Should find at least the explicit assignments
        assert len(items) >= 1


class TestTaskPrioritizer:
    """Tests for TaskPrioritizer agent."""
    
    @pytest.mark.asyncio
    async def test_prioritization(self, agent_context):
        """Test task prioritization."""
        agent = TaskPrioritizer()
        
        action_items = [
            {
                "id": "AI-001",
                "title": "Security audit",
                "description": "Complete security audit before launch",
                "urgency": "critical",
                "deadline": (datetime.now() + timedelta(days=3)).isoformat()
            },
            {
                "id": "AI-002",
                "title": "Update documentation",
                "description": "Update user documentation",
                "urgency": "low"
            },
            {
                "id": "AI-003",
                "title": "API integration",
                "description": "Complete API integration",
                "urgency": "high",
                "deadline": (datetime.now() + timedelta(days=7)).isoformat()
            }
        ]
        
        input_data = PrioritizationInput(
            action_items=action_items,
            current_date=datetime.now(),
            business_priorities=["security", "launch"]
        )
        
        result = await agent.execute(input_data, agent_context)
        
        assert result.success
        assert result.data is not None
        scores = result.data.get("scores", [])
        assert len(scores) == 3
        
        # Critical security item should be in top half of priorities
        top_item = scores[0]
        assert top_item["priority_level"] in ["P0", "P1", "P2"]


class TestOwnerAssigner:
    """Tests for OwnerAssigner agent."""
    
    @pytest.mark.asyncio
    async def test_owner_assignment(self, agent_context):
        """Test owner assignment."""
        agent = OwnerAssigner()
        
        action_items = [
            {
                "id": "AI-001",
                "title": "API integration",
                "description": "Complete the API integration with external service",
                "action_type": "task"
            },
            {
                "id": "AI-002",
                "title": "Review marketing content",
                "description": "Review and approve marketing materials",
                "action_type": "review"
            }
        ]
        
        team_members = [
            TeamMember(
                name="Alice",
                role="manager",
                skills=["review", "planning"],
                current_workload=2
            ),
            TeamMember(
                name="Bob",
                role="developer",
                skills=["api", "integration", "backend"],
                current_workload=1
            )
        ]
        
        input_data = OwnerAssignmentInput(
            action_items=action_items,
            team_members=team_members
        )
        
        result = await agent.execute(input_data, agent_context)
        
        assert result.success
        assert result.data is not None
        assignments = result.data.get("assignments", [])
        assert len(assignments) >= 1


# Recovery Tests
class TestRecoveryStrategies:
    """Tests for recovery strategies."""
    
    @pytest.mark.asyncio
    async def test_retry_strategy(self):
        """Test retry strategy."""
        strategy = RetryStrategy(max_retries=3)
        
        context = RecoveryContext(
            error="Connection failed",
            error_type="ConnectionError",
            step_id="api_call",
            step_name="API Call",
            workflow_id="test",
            execution_id="test-001",
            attempt_number=1
        )
        
        assert strategy.can_handle(context)
        result = await strategy.execute(context)
        assert result.success
        assert result.action_taken == RecoveryAction.RETRY
    
    @pytest.mark.asyncio
    async def test_recovery_manager(self):
        """Test recovery manager."""
        manager = RecoveryManager()
        manager.register_default_strategies()
        
        context = RecoveryContext(
            error="Timeout",
            error_type="TimeoutError",
            step_id="slow_step",
            step_name="Slow Processing",
            workflow_id="test",
            execution_id="test-001",
            attempt_number=1
        )
        
        result = await manager.recover(context)
        assert result.success


class TestCircuitBreaker:
    """Tests for circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_circuit_closed_allows_calls(self):
        """Test that closed circuit allows calls."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test that circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)
        
        async def failing_func():
            raise Exception("Error")
        
        # Cause failures
        for _ in range(2):
            try:
                await breaker.call(failing_func)
            except:
                pass
        
        assert breaker.state == CircuitState.OPEN


# Audit Tests
class TestAuditLogger:
    """Tests for audit logging."""
    
    @pytest.mark.asyncio
    async def test_log_event(self):
        """Test logging an audit event."""
        logger = AuditLogger()
        
        event = AuditEvent(
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW,
            event_type="test_event",
            workflow_id="test-workflow",
            execution_id="test-exec",
            description="Test description"
        )
        
        event_id = await logger.log(event)
        assert event_id is not None
        
        # Query back
        from src.core.audit import AuditQuery
        results = await logger.query(AuditQuery(
            execution_id="test-exec"
        ))
        
        assert len(results) == 1
        assert results[0].event_type == "test_event"
    
    @pytest.mark.asyncio
    async def test_audit_trail_export(self):
        """Test exporting audit trail."""
        logger = AuditLogger()
        
        # Log some events
        for i in range(3):
            await logger.log(AuditEvent(
                level=AuditLevel.INFO,
                category=AuditCategory.AGENT,
                event_type=f"event_{i}",
                execution_id="export-test",
                description=f"Event {i}"
            ))
        
        # Export
        json_export = await logger.export_trail("export-test", format="json")
        assert "event_0" in json_export
        
        markdown_export = await logger.export_trail("export-test", format="markdown")
        assert "Audit Trail Report" in markdown_export


# State Management Tests
class TestStateManager:
    """Tests for state management."""
    
    @pytest.mark.asyncio
    async def test_workflow_state_lifecycle(self):
        """Test workflow state creation and updates."""
        manager = StateManager()
        
        # Create workflow
        state = await manager.create_workflow(
            workflow_id="test-workflow",
            workflow_name="Test Workflow",
            input_data={"test": "data"}
        )
        
        assert state.workflow_id == "test-workflow"
        assert state.status == WorkflowStatus.PENDING
        
        # Update workflow
        updated = await manager.update_workflow(
            execution_id=state.execution_id,
            updates={"status": WorkflowStatus.RUNNING}
        )
        
        assert updated.status == WorkflowStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_shared_data(self):
        """Test shared data operations."""
        manager = StateManager()
        
        state = await manager.create_workflow(
            workflow_id="test-workflow",
            workflow_name="Test"
        )
        
        # Set shared data
        await manager.set_shared_data(
            execution_id=state.execution_id,
            key="result",
            value={"items": [1, 2, 3]}
        )
        
        # Get shared data
        result = await manager.get_shared_data(state.execution_id, "result")
        assert result == {"items": [1, 2, 3]}
    
    @pytest.mark.asyncio
    async def test_snapshots(self):
        """Test state snapshots and rollback."""
        manager = StateManager()
        
        state = await manager.create_workflow(
            workflow_id="test-workflow",
            workflow_name="Test"
        )
        
        # Create snapshot
        snapshot = await manager.create_snapshot(
            execution_id=state.execution_id,
            reason="Before risky operation"
        )
        
        assert snapshot is not None
        
        # Modify state
        await manager.set_shared_data(
            state.execution_id,
            "modified",
            True
        )
        
        # Rollback
        restored = await manager.rollback_to_snapshot(
            state.execution_id,
            snapshot.snapshot_id
        )
        
        assert restored is not None


# Message Bus Tests
class TestMessageBus:
    """Tests for message bus."""
    
    @pytest.mark.asyncio
    async def test_send_receive(self):
        """Test sending and receiving messages."""
        bus = MessageBus()
        
        bus.register_agent("agent-1")
        bus.register_agent("agent-2")
        
        message = Message(
            type=MessageType.REQUEST,
            sender_id="agent-1",
            sender_name="Agent 1",
            recipient_id="agent-2",
            subject="Test Message",
            payload={"data": "test"}
        )
        
        sent = await bus.send(message)
        assert sent
        
        received = await bus.receive("agent-2")
        assert received is not None
        assert received.subject == "Test Message"
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcast messaging."""
        bus = MessageBus()
        
        bus.register_agent("agent-1")
        bus.register_agent("agent-2")
        bus.register_agent("agent-3")
        
        bus.subscribe("agent-2", "notifications")
        bus.subscribe("agent-3", "notifications")
        
        message = Message(
            type=MessageType.BROADCAST,
            sender_id="agent-1",
            sender_name="Agent 1",
            subject="Broadcast",
            payload={"announcement": "Hello all"}
        )
        
        count = await bus.broadcast(message, "notifications")
        assert count == 2


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
