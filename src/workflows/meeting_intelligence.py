"""
Meeting Intelligence Workflow.

A complete workflow that processes meeting transcripts to:
1. Extract key decisions
2. Identify action items
3. Assign owners
4. Prioritize tasks
5. Detect escalation needs
6. Generate comprehensive reports
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.orchestration.engine import Workflow, WorkflowStep, RetryPolicy


class MeetingIntelligenceConfig(BaseModel):
    """Configuration for meeting intelligence workflow."""
    auto_assign_owners: bool = True
    auto_prioritize: bool = True
    check_escalations: bool = True
    generate_summary: bool = True
    confidence_threshold: float = 0.6


def create_meeting_intelligence_workflow(
    config: Optional[MeetingIntelligenceConfig] = None
) -> Workflow:
    """
    Create the meeting intelligence workflow.
    
    Pipeline:
    1. TranscriptAnalyzer - Analyze transcript structure
    2. DecisionExtractor - Extract decisions
    3. ActionItemExtractor - Extract action items
    4. TaskPrioritizer - Prioritize items (optional)
    5. OwnerAssigner - Assign owners (optional)
    6. EscalationDecider - Check for escalations (optional)
    """
    config = config or MeetingIntelligenceConfig()
    
    steps = [
        # Step 1: Analyze transcript
        WorkflowStep(
            id="analyze_transcript",
            name="Analyze Transcript",
            description="Parse and analyze the meeting transcript",
            agent_type="TranscriptAnalyzer",
            input_mapping={
                "transcript_text": "input.transcript",
                "meeting_title": "input.title",
                "known_participants": "input.participants"
            },
            output_key="transcript_analysis",
            timeout_seconds=120.0,
            retry_policy=RetryPolicy(max_retries=2)
        ),
        
        # Step 2: Extract decisions
        WorkflowStep(
            id="extract_decisions",
            name="Extract Decisions",
            description="Identify decisions made during the meeting",
            agent_type="DecisionExtractor",
            depends_on=["analyze_transcript"],
            input_mapping={
                "transcript_analysis": "shared.transcript_analysis",
                "participant_roles": "input.participant_roles"
            },
            output_key="decisions",
            timeout_seconds=90.0,
            retry_policy=RetryPolicy(max_retries=2)
        ),
        
        # Step 3: Extract action items
        WorkflowStep(
            id="extract_actions",
            name="Extract Action Items",
            description="Identify action items and tasks",
            agent_type="ActionItemExtractor",
            depends_on=["analyze_transcript", "extract_decisions"],
            input_mapping={
                "transcript_analysis": "shared.transcript_analysis",
                "decisions": "shared.decisions.decisions",
                "participant_info": "input.participant_info"
            },
            output_key="action_items",
            timeout_seconds=90.0,
            retry_policy=RetryPolicy(max_retries=2)
        ),
    ]
    
    # Optional: Prioritize tasks
    if config.auto_prioritize:
        steps.append(WorkflowStep(
            id="prioritize_tasks",
            name="Prioritize Tasks",
            description="Assign priorities to action items",
            agent_type="TaskPrioritizer",
            depends_on=["extract_actions", "extract_decisions"],
            input_mapping={
                "action_items": "shared.action_items.action_items",
                "decisions": "shared.decisions.decisions",
                "business_priorities": "input.business_priorities"
            },
            output_key="prioritized_items",
            timeout_seconds=60.0,
            continue_on_failure=True
        ))
    
    # Optional: Assign owners
    if config.auto_assign_owners:
        steps.append(WorkflowStep(
            id="assign_owners",
            name="Assign Owners",
            description="Assign owners to unassigned tasks",
            agent_type="OwnerAssigner",
            depends_on=["extract_actions"] + (["prioritize_tasks"] if config.auto_prioritize else []),
            input_mapping={
                "action_items": "shared.action_items.action_items",
                "team_members": "input.team_members"
            },
            output_key="assignments",
            timeout_seconds=60.0,
            continue_on_failure=True
        ))
    
    # Optional: Check escalations
    if config.check_escalations:
        steps.append(WorkflowStep(
            id="check_escalations",
            name="Check Escalations",
            description="Identify items requiring escalation",
            agent_type="EscalationDecider",
            depends_on=["extract_actions"] + (["prioritize_tasks"] if config.auto_prioritize else []),
            input_mapping={
                "action_items": "shared.action_items.action_items"
            },
            output_key="escalations",
            timeout_seconds=60.0,
            continue_on_failure=True
        ))
    
    return Workflow(
        id="meeting_intelligence",
        name="Meeting Intelligence Processor",
        description="Processes meeting transcripts to extract decisions, action items, and manage follow-ups",
        version="1.0.0",
        steps=steps,
        input_schema={
            "type": "object",
            "required": ["transcript"],
            "properties": {
                "transcript": {"type": "string", "description": "Meeting transcript text"},
                "title": {"type": "string", "description": "Meeting title"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "participant_roles": {"type": "object"},
                "participant_info": {"type": "object"},
                "team_members": {"type": "array"},
                "business_priorities": {"type": "array", "items": {"type": "string"}}
            }
        },
        timeout_seconds=600.0,
        max_parallel_steps=2,
        tags=["meeting", "intelligence", "automation"]
    )


class MeetingIntelligenceResult(BaseModel):
    """Aggregated result from meeting intelligence workflow."""
    execution_id: str
    meeting_title: Optional[str] = None
    processing_time_seconds: float = 0.0
    
    # Extracted data
    summary: str = ""
    speakers: List[Dict[str, Any]] = Field(default_factory=list)
    topics: List[Dict[str, Any]] = Field(default_factory=list)
    
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    decision_count: int = 0
    
    action_items: List[Dict[str, Any]] = Field(default_factory=list)
    action_item_count: int = 0
    assigned_count: int = 0
    
    # Prioritization
    priority_breakdown: Dict[str, int] = Field(default_factory=dict)
    recommended_order: List[str] = Field(default_factory=list)
    
    # Escalations
    escalations: List[Dict[str, Any]] = Field(default_factory=list)
    escalation_count: int = 0
    
    # Quality metrics
    avg_confidence: float = 0.0
    items_needing_review: List[str] = Field(default_factory=list)


def aggregate_workflow_results(
    workflow_output: Dict[str, Any]
) -> MeetingIntelligenceResult:
    """
    Aggregate workflow step outputs into a unified result.
    """
    # Extract from workflow output
    transcript_analysis = workflow_output.get("transcript_analysis", {})
    decisions_data = workflow_output.get("decisions", {})
    action_items_data = workflow_output.get("action_items", {})
    prioritized_data = workflow_output.get("prioritized_items", {})
    assignments_data = workflow_output.get("assignments", {})
    escalations_data = workflow_output.get("escalations", {})
    
    # Build result
    result = MeetingIntelligenceResult(
        execution_id=workflow_output.get("execution_id", ""),
        meeting_title=transcript_analysis.get("meeting_title"),
        summary=transcript_analysis.get("summary", ""),
        speakers=transcript_analysis.get("speakers", []),
        topics=transcript_analysis.get("topics", []),
        decisions=decisions_data.get("decisions", []),
        decision_count=decisions_data.get("total_found", 0),
        action_items=action_items_data.get("action_items", []),
        action_item_count=action_items_data.get("total_found", 0),
        assigned_count=action_items_data.get("assigned_count", 0),
    )
    
    # Add prioritization data
    if prioritized_data:
        scores = prioritized_data.get("scores", [])
        priority_counts = {}
        for score in scores:
            level = score.get("priority_level", "P4")
            priority_counts[level] = priority_counts.get(level, 0) + 1
        result.priority_breakdown = priority_counts
        result.recommended_order = prioritized_data.get("recommended_order", [])
    
    # Add escalation data
    if escalations_data:
        result.escalations = escalations_data.get("escalations", [])
        result.escalation_count = len(result.escalations)
    
    # Calculate aggregate confidence
    confidences = []
    for decision in result.decisions:
        if "confidence" in decision:
            confidences.append(decision["confidence"])
    for item in result.action_items:
        if "confidence" in item:
            confidences.append(item["confidence"])
    
    if confidences:
        result.avg_confidence = sum(confidences) / len(confidences)
    
    # Items needing review
    review_items = []
    for decision in result.decisions:
        if decision.get("confidence", 1.0) < 0.6:
            review_items.append(decision.get("id", "unknown"))
    for item in result.action_items:
        if item.get("confidence", 1.0) < 0.6:
            review_items.append(item.get("id", "unknown"))
    result.items_needing_review = review_items
    
    return result
