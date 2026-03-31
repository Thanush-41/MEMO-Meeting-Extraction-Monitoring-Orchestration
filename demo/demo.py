"""
Demo Application for MEMO — Meeting Extraction, Monitoring & Orchestration.

MEMO is a multi-agent system that autonomously processes meeting transcripts,
extracts decisions, assigns tasks, and maintains a full auditable decision trail.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Simulated imports (in real usage, these would be actual imports)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.base_agent import AgentConfig, AgentContext, AgentResult
from src.core.state import StateManager, WorkflowStatus
from src.core.audit import AuditLogger, AuditQuery, AuditLevel
from src.core.message import MessageBus
from src.orchestration.engine import WorkflowEngine
from src.orchestration.monitor import WorkflowMonitor, SLADefinition, AlertSeverity
from src.recovery.strategies import RecoveryManager
from src.recovery.circuit import CircuitBreakerRegistry

from src.agents.extraction.transcript_agent import TranscriptAnalyzer, TranscriptInput
from src.agents.extraction.decision_parser import DecisionExtractor, DecisionExtractionInput
from src.agents.extraction.action_item_agent import ActionItemExtractor, ActionItemExtractionInput
from src.agents.decision.task_prioritizer import TaskPrioritizer, PrioritizationInput
from src.agents.decision.owner_assigner import OwnerAssigner, OwnerAssignmentInput, TeamMember
from src.agents.decision.escalation_decider import EscalationDecider, EscalationInput
from src.agents.ai.gemini_enrichment import GeminiEnrichmentAgent, GeminiEnrichmentInput

from src.workflows.meeting_intelligence import (
    create_meeting_intelligence_workflow,
    MeetingIntelligenceConfig,
    aggregate_workflow_results
)


# Sample meeting transcript for demo
SAMPLE_TRANSCRIPT = """
Sarah Chen: Good morning everyone. Let's start our Q4 planning meeting. We have a lot to cover today.

John Smith: Thanks Sarah. First, I'd like to discuss the new product launch timeline. Based on customer feedback, I think we should move the launch date from December 15th to January 10th.

Maria Garcia: I agree with John. The engineering team needs more time to implement the security features that customers have been requesting.

Sarah Chen: Alright, let's decided to delay the launch to January 10th. Marketing will need to update all the campaign materials. Maria, can you confirm the engineering timeline?

Maria Garcia: Yes, I'll have the updated schedule by Friday. We're blocked on the third-party API integration though - we need Bob to help with that.

Bob Wilson: I can take a look at the API integration tomorrow. It's been causing some issues. I'll need to coordinate with the external vendor.

Sarah Chen: Good. Let's make that a priority. Bob, please send me a status update by end of day Wednesday.

John Smith: For the marketing side, we need someone to review the new landing page content. It's critical that we get the messaging right.

Sarah Chen: I'll review the landing page. We should also schedule a follow-up meeting next week to check on progress.

Maria Garcia: Agreed. One more thing - we have a security audit coming up. This is urgent and needs to be completed before the launch.

Sarah Chen: That's important. Maria, please prepare the documentation for the audit. We'll need it by next Monday.

Bob Wilson: I have a concern about the current deployment pipeline. There's a risk of delays if we don't address the CI/CD issues soon.

Sarah Chen: Good point. Let's add that to our backlog. John, can you coordinate with DevOps to investigate?

John Smith: Sure, I'll reach out to the DevOps team today.

Sarah Chen: Great discussion everyone. To summarize: we've decided to push the launch to January 10th, Bob is handling the API integration, Maria is updating the engineering schedule and preparing audit docs, and John is coordinating the DevOps investigation. Let's reconvene next Tuesday.

All: Sounds good. Thanks everyone!
"""


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_json(data: Any, indent: int = 2):
    """Print data as formatted JSON."""
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    print(json.dumps(data, indent=indent, default=str))


async def demo_individual_agents():
    """Demonstrate individual agent capabilities."""
    print_section("DEMO 1: Individual Agent Processing")
    
    # Create context
    context = AgentContext(
        workflow_id="demo-workflow",
        execution_id="demo-001",
        step_number=1,
        shared_state={}
    )
    
    # 1. Transcript Analysis
    print(">>> Step 1: Transcript Analysis")
    print("-" * 40)
    
    transcript_agent = TranscriptAnalyzer()
    transcript_input = TranscriptInput(
        transcript_text=SAMPLE_TRANSCRIPT,
        meeting_title="Q4 Planning Meeting",
        meeting_date=datetime.now(),
        known_participants=["Sarah Chen", "John Smith", "Maria Garcia", "Bob Wilson"]
    )
    
    result = await transcript_agent.execute(transcript_input, context)
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Reasoning: {result.reasoning}")
    
    if result.data:
        print(f"\nSpeakers found: {len(result.data.get('speakers', []))}")
        print(f"Topics identified: {len(result.data.get('topics', []))}")
        print(f"Key points: {len(result.data.get('key_points', []))}")
        print(f"\nSummary: {result.data.get('summary', '')[:200]}...")
    
    # Store for next steps
    transcript_analysis = result.data
    
    # 2. Decision Extraction
    print("\n>>> Step 2: Decision Extraction")
    print("-" * 40)
    
    decision_agent = DecisionExtractor()
    decision_input = DecisionExtractionInput(
        transcript_analysis=transcript_analysis,
        participant_roles={
            "Sarah Chen": "manager",
            "John Smith": "marketing_lead",
            "Maria Garcia": "engineering_lead",
            "Bob Wilson": "developer"
        }
    )
    
    context.step_number = 2
    result = await decision_agent.execute(decision_input, context)
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.data:
        decisions = result.data.get("decisions", [])
        print(f"\nDecisions extracted: {len(decisions)}")
        for d in decisions[:3]:
            print(f"  - [{d.get('decision_type', 'unknown')}] {d.get('description', '')[:60]}...")
    
    decisions_data = result.data
    
    # 3. Action Item Extraction
    print("\n>>> Step 3: Action Item Extraction")
    print("-" * 40)
    
    action_agent = ActionItemExtractor()
    action_input = ActionItemExtractionInput(
        transcript_analysis=transcript_analysis,
        decisions=decisions_data.get("decisions", []),
        current_date=datetime.now()
    )
    
    context.step_number = 3
    result = await action_agent.execute(action_input, context)
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.data:
        items = result.data.get("action_items", [])
        print(f"\nAction items extracted: {len(items)}")
        print(f"Assigned: {result.data.get('assigned_count', 0)}")
        print(f"With deadlines: {result.data.get('with_deadlines', 0)}")
        
        for item in items[:4]:
            assignee = item.get('assignee', 'Unassigned')
            print(f"  - [{item.get('urgency', 'normal')}] {item.get('title', '')[:40]}... -> {assignee}")
    
    action_items_data = result.data
    
    # 3.5 AI Enrichment (Gemini)
    print("\n>>> Step 3.5: AI Enrichment (Gemini)")
    print("-" * 40)
    
    import os
    if os.environ.get("GEMINI_API_KEY"):
        ai_agent = GeminiEnrichmentAgent()
        ai_input = GeminiEnrichmentInput(
            transcript_text=SAMPLE_TRANSCRIPT,
            transcript_analysis=transcript_analysis,
            decisions=decisions_data.get("decisions", []),
            action_items=action_items_data.get("action_items", [])
        )
        
        context.step_number = 35
        result = await ai_agent.execute(ai_input, context)
        print(f"Success: {result.success}")
        print(f"Confidence: {result.confidence:.2%}")
        if result.error:
            print(f"Error: {result.error}")
        
        if result.data:
            print(f"\nAI Summary: {result.data.get('ai_summary', '')[:200]}...")
            missed_d = result.data.get("missed_decisions", [])
            missed_a = result.data.get("missed_action_items", [])
            risks = result.data.get("risks_identified", [])
            insights = result.data.get("key_insights", [])
            sentiment = result.data.get("sentiment_analysis", {})
            
            print(f"\nMissed decisions found by AI: {len(missed_d)}")
            for d in missed_d:
                print(f"  - {d.get('description', '')[:60]}")
            
            print(f"\nMissed action items found by AI: {len(missed_a)}")
            for a in missed_a:
                print(f"  - [{a.get('urgency', 'normal')}] {a.get('title', '')[:50]} -> {a.get('assignee', '?')}")
            
            print(f"\nRisks identified: {len(risks)}")
            for r in risks:
                print(f"  - [{r.get('severity', '?')}] {r.get('risk', '')[:60]}")
            
            print(f"\nSentiment: {sentiment.get('overall', 'N/A')}")
            
            print(f"\nKey Insights:")
            for insight in insights:
                print(f"  • {insight[:80]}")
            
            # Show enriched totals
            enriched_d = result.data.get("enriched_decisions", [])
            enriched_a = result.data.get("enriched_action_items", [])
            print(f"\nTotal decisions (rule + AI): {len(enriched_d)}")
            print(f"Total action items (rule + AI): {len(enriched_a)}")
    else:
        print("  GEMINI_API_KEY not set — skipping AI enrichment")
        print("  Set it with: $env:GEMINI_API_KEY = 'your-key'")
    
    # 4. Task Prioritization
    print("\n>>> Step 4: Task Prioritization")
    print("-" * 40)
    
    prioritizer = TaskPrioritizer()
    priority_input = PrioritizationInput(
        action_items=action_items_data.get("action_items", []),
        decisions=decisions_data.get("decisions", []),
        current_date=datetime.now(),
        business_priorities=["security", "launch", "customer"]
    )
    
    context.step_number = 4
    result = await prioritizer.execute(priority_input, context)
    print(f"Success: {result.success}")
    
    if result.data:
        scores = result.data.get("scores", [])
        print(f"\nPrioritization complete:")
        p0 = len([s for s in scores if s.get("priority_level") == "P0"])
        p1 = len([s for s in scores if s.get("priority_level") == "P1"])
        p2 = len([s for s in scores if s.get("priority_level") == "P2"])
        print(f"  P0 (Critical): {p0}")
        print(f"  P1 (High): {p1}")
        print(f"  P2 (Medium): {p2}")
        
        if scores:
            print(f"\nTop priority item: {scores[0].get('item_id')} - Score: {scores[0].get('overall_score', 0):.1f}")
    
    return {
        "transcript_analysis": transcript_analysis,
        "decisions": decisions_data,
        "action_items": action_items_data
    }


async def demo_workflow_engine():
    """Demonstrate the full workflow engine."""
    print_section("DEMO 2: Workflow Engine Execution")
    
    # Initialize components
    state_manager = StateManager()
    audit_logger = AuditLogger(persist_to_file=False)
    message_bus = MessageBus()
    
    engine = WorkflowEngine(
        state_manager=state_manager,
        audit_logger=audit_logger,
        message_bus=message_bus
    )
    
    # Register agents
    engine.register_agents({
        "TranscriptAnalyzer": TranscriptAnalyzer,
        "DecisionExtractor": DecisionExtractor,
        "ActionItemExtractor": ActionItemExtractor,
        "TaskPrioritizer": TaskPrioritizer,
        "OwnerAssigner": OwnerAssigner,
        "EscalationDecider": EscalationDecider,
        "GeminiEnrichmentAgent": GeminiEnrichmentAgent,
    })
    
    # Create workflow
    config = MeetingIntelligenceConfig(
        auto_assign_owners=True,
        auto_prioritize=True,
        check_escalations=True,
        enable_ai_enrichment=bool(os.environ.get("GEMINI_API_KEY"))
    )
    workflow = create_meeting_intelligence_workflow(config)
    
    print(f"Workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    for step in workflow.steps:
        deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
        print(f"  {step.id}: {step.name}{deps}")
    
    # Prepare input
    input_data = {
        "transcript": SAMPLE_TRANSCRIPT,
        "title": "Q4 Planning Meeting",
        "participants": ["Sarah Chen", "John Smith", "Maria Garcia", "Bob Wilson"],
        "participant_roles": {
            "Sarah Chen": "manager",
            "John Smith": "marketing_lead",
            "Maria Garcia": "engineering_lead",
            "Bob Wilson": "developer"
        },
        "participant_info": {
            "Sarah Chen": {"role": "manager", "areas": ["planning", "coordination"]},
            "John Smith": {"role": "marketing", "areas": ["marketing", "campaigns"]},
            "Maria Garcia": {"role": "engineering", "areas": ["development", "security"]},
            "Bob Wilson": {"role": "developer", "areas": ["api", "integration"]}
        },
        "team_members": [
            TeamMember(
                name="Sarah Chen",
                role="manager",
                skills=["planning", "coordination"],
                current_workload=3
            ).model_dump(),
            TeamMember(
                name="John Smith",
                role="marketing",
                skills=["marketing", "content"],
                current_workload=2
            ).model_dump(),
            TeamMember(
                name="Maria Garcia",
                role="engineering",
                skills=["development", "security"],
                current_workload=4
            ).model_dump(),
            TeamMember(
                name="Bob Wilson",
                role="developer",
                skills=["api", "integration", "devops"],
                current_workload=1
            ).model_dump()
        ],
        "business_priorities": ["security", "launch", "customer"]
    }
    
    print("\n>>> Executing workflow...")
    print("-" * 40)
    
    start_time = datetime.now()
    result = await engine.execute(workflow, input_data)
    end_time = datetime.now()
    
    print(f"\nExecution completed in {(end_time - start_time).total_seconds():.2f}s")
    print(f"Status: {result.status}")
    print(f"Execution ID: {result.execution_id}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    # Show step results summary
    print("\n>>> Step Results:")
    for step_id, step_result in result.step_results.items():
        status = "✓" if step_result.get("success") else "✗"
        conf = step_result.get("confidence", 0)
        print(f"  {status} {step_id}: confidence={conf:.2%}")
    
    # Aggregate results
    if result.output:
        print("\n>>> Aggregated Results:")
        aggregated = aggregate_workflow_results({
            **result.output,
            "execution_id": result.execution_id
        })
        print(f"  Decisions: {aggregated.decision_count}")
        print(f"  Action Items: {aggregated.action_item_count}")
        print(f"  Assigned: {aggregated.assigned_count}")
        print(f"  Escalations: {aggregated.escalation_count}")
        print(f"  Avg Confidence: {aggregated.avg_confidence:.2%}")
        
        if aggregated.ai_summary:
            print(f"\n  AI Summary: {aggregated.ai_summary[:200]}...")
        if aggregated.ai_missed_decisions:
            print(f"  AI found {aggregated.ai_missed_decisions} additional decisions")
        if aggregated.ai_missed_action_items:
            print(f"  AI found {aggregated.ai_missed_action_items} additional action items")
        if aggregated.risks_identified:
            print(f"  Risks: {len(aggregated.risks_identified)}")
        if aggregated.key_insights:
            print(f"  Key Insights:")
            for insight in aggregated.key_insights:
                print(f"    • {insight[:80]}")
        
        if aggregated.items_needing_review:
            print(f"  Items needing review: {len(aggregated.items_needing_review)}")
    
    # Show audit trail
    print("\n>>> Audit Trail (last 10 events):")
    trail = await engine.get_audit_trail(result.execution_id)
    for event in trail[-10:]:
        print(f"  [{event.timestamp.strftime('%H:%M:%S')}] {event.event_type}")
    
    return result


async def demo_monitoring():
    """Demonstrate the monitoring system."""
    print_section("DEMO 3: Health Monitoring & Alerting")
    
    # Create monitor
    alerts_received = []
    
    def alert_handler(alert):
        alerts_received.append(alert)
        print(f"  🔔 ALERT: [{alert.severity.name}] {alert.title}")
    
    monitor = WorkflowMonitor(
        check_interval_seconds=5.0,
        alert_callback=alert_handler
    )
    
    # Register SLA
    monitor.register_sla(SLADefinition(
        name="meeting_processing",
        target_completion_hours=1.0,
        warning_threshold_percent=50.0
    ))
    
    # Simulate some workflow activity
    print("Simulating workflow activity...")
    
    # Track execution start
    monitor.track_execution_start("meeting_intelligence", "exec-001")
    monitor.track_execution_start("meeting_intelligence", "exec-002")
    monitor.track_execution_start("meeting_intelligence", "exec-003")
    
    # Track resource utilization
    monitor.track_resource_utilization("agent_pool", 75.0, queue_size=5)
    monitor.track_resource_utilization("database", 45.0)
    
    # Complete some executions
    monitor.track_execution_complete("meeting_intelligence", "exec-001", success=True)
    monitor.track_execution_complete("meeting_intelligence", "exec-002", success=True)
    
    # Get health summary
    print("\n>>> Health Summary:")
    summary = monitor.get_health_summary()
    print(f"  Overall Status: {summary['overall_status']}")
    print(f"  Active Executions: {summary['active_executions']}")
    print(f"  Active Alerts: {summary['active_alerts']}")
    
    # Get metrics
    print("\n>>> Workflow Metrics:")
    metrics = monitor.get_workflow_metrics()
    for wf_id, m in metrics.items():
        if m:
            print(f"  {wf_id}:")
            print(f"    Total: {m.total_executions}, Active: {m.active_executions}")
            print(f"    Success Rate: {m.success_rate:.2%}")
    
    return {"summary": summary, "alerts": len(alerts_received)}


async def demo_error_recovery():
    """Demonstrate error recovery capabilities."""
    print_section("DEMO 4: Error Recovery & Self-Correction")
    
    from src.recovery.strategies import (
        RecoveryManager, 
        RecoveryContext,
        RetryStrategy,
        FallbackStrategy,
        EscalateStrategy
    )
    
    # Create recovery manager with strategies
    recovery_manager = RecoveryManager()
    recovery_manager.register_default_strategies()
    
    # Simulate various error scenarios
    scenarios = [
        {
            "name": "Temporary API Failure",
            "context": RecoveryContext(
                error="Connection refused",
                error_type="ConnectionError",
                step_id="api_call",
                step_name="External API Call",
                workflow_id="demo",
                execution_id="exec-001",
                attempt_number=1
            )
        },
        {
            "name": "Repeated Failures (Escalation)",
            "context": RecoveryContext(
                error="Service unavailable",
                error_type="ServiceError",
                step_id="critical_step",
                step_name="Critical Processing",
                workflow_id="demo",
                execution_id="exec-002",
                attempt_number=4,
                previous_errors=["Timeout", "Timeout", "ServiceError"]
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\n>>> Scenario: {scenario['name']}")
        print("-" * 40)
        
        result = await recovery_manager.recover(scenario['context'])
        
        print(f"  Action: {result.action_taken.name}")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        if result.escalated_to:
            print(f"  Escalated to: {result.escalated_to}")
    
    # Show recovery stats
    print("\n>>> Recovery Statistics:")
    stats = recovery_manager.get_stats()
    print(f"  Total Recoveries: {stats['total_recoveries']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    return stats


async def demo_audit_trail():
    """Demonstrate the audit trail system."""
    print_section("DEMO 5: Audit Trail & Decision Tracking")
    
    from src.core.audit import (
        AuditLogger, 
        AuditEvent, 
        AuditLevel, 
        AuditCategory,
        DecisionRecord
    )
    
    logger = AuditLogger()
    
    # Log various events
    events = [
        AuditEvent(
            level=AuditLevel.INFO,
            category=AuditCategory.WORKFLOW,
            event_type="workflow_started",
            workflow_id="meeting_intelligence",
            execution_id="audit-demo-001",
            description="Meeting intelligence workflow started"
        ),
        AuditEvent(
            level=AuditLevel.DECISION,
            category=AuditCategory.DECISION,
            event_type="priority_assigned",
            workflow_id="meeting_intelligence",
            execution_id="audit-demo-001",
            agent_name="TaskPrioritizer",
            description="Assigned P1 priority to security audit task",
            reasoning="High business impact due to security keyword; deadline within 7 days",
            confidence=0.85
        ),
        AuditEvent(
            level=AuditLevel.WARNING,
            category=AuditCategory.ESCALATION,
            event_type="escalation_triggered",
            workflow_id="meeting_intelligence",
            execution_id="audit-demo-001",
            description="Task escalated due to approaching deadline",
            affected_entities=["task-003"],
            metadata={"escalation_level": 1}
        )
    ]
    
    for event in events:
        await logger.log(event)
    
    # Log a detailed decision
    decision = DecisionRecord(
        agent_id="agent-001",
        agent_name="OwnerAssigner",
        workflow_id="meeting_intelligence",
        execution_id="audit-demo-001",
        step_number=5,
        decision_type="task_assignment",
        decision_description="Assigned 'API Integration' task to Bob Wilson",
        decision_outcome="Bob Wilson",
        input_data_summary={
            "task_id": "AI-003",
            "task_type": "technical",
            "required_skills": ["api", "integration"]
        },
        reasoning_steps=[
            "Task requires API integration skills",
            "Bob Wilson has matching skills: api, integration",
            "Bob Wilson has lowest current workload (1 task)",
            "Selected Bob Wilson with confidence 0.85"
        ],
        confidence_score=0.85,
        alternatives=[
            {"owner": "Maria Garcia", "score": 65},
            {"owner": "John Smith", "score": 40}
        ]
    )
    
    await logger.log_decision(decision)
    
    # Query and display
    print(">>> Audit Events Logged:")
    all_events = await logger.query(AuditQuery(
        execution_id="audit-demo-001",
        limit=100
    ))
    
    for event in all_events:
        level_icon = {
            AuditLevel.INFO: "ℹ️",
            AuditLevel.DECISION: "🎯",
            AuditLevel.WARNING: "⚠️",
            AuditLevel.ERROR: "❌"
        }.get(event.level, "•")
        
        # Handle category as string or enum
        category_name = event.category.name if hasattr(event.category, 'name') else str(event.category)
        print(f"  {level_icon} [{category_name}] {event.event_type}")
        print(f"     {event.description}")
        if event.reasoning:
            print(f"     Reasoning: {event.reasoning[:60]}...")
        if event.confidence:
            print(f"     Confidence: {event.confidence:.2%}")
    
    # Show decisions
    print("\n>>> Detailed Decisions:")
    decisions = await logger.get_decisions(execution_id="audit-demo-001")
    for d in decisions:
        print(f"  Decision: {d.decision_type}")
        print(f"  Outcome: {d.decision_outcome}")
        print(f"  Reasoning:")
        for step in d.reasoning_steps:
            print(f"    - {step}")
        print(f"  Confidence: {d.confidence_score:.2%}")
    
    # Export trail
    print("\n>>> Exporting audit trail...")
    markdown_report = await logger.export_trail("audit-demo-001", format="markdown")
    print(f"  Generated {len(markdown_report)} characters of documentation")
    
    return logger.get_stats()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  MEMO — Meeting Extraction, Monitoring & Orchestration")
    print("  Multi-Agent Meeting Intelligence Demo")
    print("=" * 60)
    
    print("\nThis demo showcases:")
    print("  1. Individual agent processing")
    print("  2. Orchestrated workflow execution")
    print("  3. Health monitoring & alerting")
    print("  4. Error recovery & self-correction")
    print("  5. Complete audit trail")
    
    # Run demos
    await demo_individual_agents()
    await demo_workflow_engine()
    await demo_monitoring()
    await demo_error_recovery()
    await demo_audit_trail()
    
    print_section("DEMO COMPLETE")
    print("The multi-agent system demonstrated:")
    print("  ✓ Autonomous processing of meeting transcripts")
    print("  ✓ Extraction of decisions and action items")
    print("  ✓ Intelligent task prioritization and assignment")
    print("  ✓ Error detection and self-correction")
    print("  ✓ Complete audit trail of all decisions")
    print("  ✓ Health monitoring and alerting")


if __name__ == "__main__":
    asyncio.run(main())
