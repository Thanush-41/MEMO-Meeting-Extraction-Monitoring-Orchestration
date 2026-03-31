"""
Task Prioritizer Agent.

Analyzes action items and assigns priority scores based on
various factors including urgency, dependencies, and business impact.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)
from src.core.audit import DecisionRecord


class PriorityScore(BaseModel):
    """Detailed priority scoring for an action item."""
    item_id: str
    overall_score: float = Field(ge=0, le=100)
    priority_level: str  # P0, P1, P2, P3, P4
    
    # Score components
    urgency_score: float = 0.0
    impact_score: float = 0.0
    dependency_score: float = 0.0
    effort_score: float = 0.0
    
    # Factors
    factors: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    
    # Recommendations
    suggested_order: int = 0
    can_be_parallelized: bool = False
    parallel_with: List[str] = Field(default_factory=list)


class PrioritizationInput(BaseModel):
    """Input for task prioritization."""
    action_items: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    current_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Context for prioritization
    team_capacity: Dict[str, float] = Field(default_factory=dict)  # person -> available hours
    business_priorities: List[str] = Field(default_factory=list)  # Keywords indicating priority
    blocked_items: List[str] = Field(default_factory=list)  # Item IDs that are blocked


class PrioritizationResult(BaseModel):
    """Result of task prioritization."""
    scores: List[PriorityScore] = Field(default_factory=list)
    recommended_order: List[str] = Field(default_factory=list)  # Item IDs in priority order
    p0_items: List[str] = Field(default_factory=list)  # Critical items
    parallel_groups: List[List[str]] = Field(default_factory=list)  # Groups that can run together


class TaskPrioritizer(BaseAgent[PrioritizationInput]):
    """
    Agent that prioritizes action items based on multiple factors.
    
    Scoring factors:
    - Urgency: Deadline proximity and explicit urgency markers
    - Impact: Business value and decision importance
    - Dependencies: Items that block others get higher priority
    - Effort: Quick wins may be prioritized for momentum
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="TaskPrioritizer",
            capabilities=[AgentCapability.DECISION_MAKING],
            config=config or AgentConfig(
                confidence_threshold=0.7,
                max_retries=2
            )
        )
        
        # Priority level thresholds
        self._priority_thresholds = {
            "P0": 90,  # Critical - must be done immediately
            "P1": 75,  # High - should be done this week
            "P2": 50,  # Medium - should be done this sprint
            "P3": 25,  # Low - nice to have
            "P4": 0,   # Backlog
        }
        
        # Urgency weights
        self._urgency_weights = {
            "critical": 40,
            "high": 30,
            "normal": 15,
            "low": 5
        }
        
        # Impact keywords
        self._high_impact_keywords = [
            "customer", "revenue", "security", "compliance", "deadline",
            "launch", "production", "critical", "blocker", "executive"
        ]
    
    async def process(
        self,
        input_data: PrioritizationInput,
        context: AgentContext
    ) -> AgentResult:
        """Prioritize action items."""
        if isinstance(input_data, dict):
            input_data = PrioritizationInput(**input_data)
        
        self._log_audit_event(
            event_type="prioritization_start",
            context=context,
            details={
                "item_count": len(input_data.action_items),
                "has_capacity_info": len(input_data.team_capacity) > 0
            }
        )
        
        try:
            scores = []
            decision_records = []
            
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(input_data.action_items)
            blockers = self._find_blockers(dependency_graph)
            
            for item in input_data.action_items:
                # Calculate component scores
                urgency = self._calculate_urgency_score(item, input_data.current_date)
                impact = self._calculate_impact_score(item, input_data)
                dependency = self._calculate_dependency_score(item, blockers)
                effort = self._calculate_effort_score(item)
                
                # Calculate overall score (weighted average)
                overall = (
                    urgency * 0.35 +
                    impact * 0.30 +
                    dependency * 0.25 +
                    effort * 0.10
                )
                
                # Determine priority level
                priority_level = self._score_to_priority(overall)
                
                # Build reasoning
                reasoning = self._build_reasoning(
                    item, urgency, impact, dependency, effort, overall
                )
                
                score = PriorityScore(
                    item_id=item["id"],
                    overall_score=overall,
                    priority_level=priority_level,
                    urgency_score=urgency,
                    impact_score=impact,
                    dependency_score=dependency,
                    effort_score=effort,
                    factors={
                        "has_deadline": item.get("deadline") is not None,
                        "is_blocker": item["id"] in blockers,
                        "has_assignee": item.get("assignee") is not None
                    },
                    reasoning=reasoning
                )
                scores.append(score)
                
                # Create decision record for audit
                decision_records.append(DecisionRecord(
                    agent_id=self.id,
                    agent_name=self.name,
                    workflow_id=context.workflow_id,
                    execution_id=context.execution_id,
                    step_number=context.step_number,
                    decision_type="priority_assignment",
                    decision_description=f"Assigned priority {priority_level} to '{item.get('title', item['id'])}'",
                    decision_outcome=f"Score: {overall:.1f}, Level: {priority_level}",
                    input_data_summary={
                        "item_id": item["id"],
                        "urgency": item.get("urgency", "normal"),
                        "has_deadline": item.get("deadline") is not None
                    },
                    reasoning_steps=[
                        f"Urgency score: {urgency:.1f}",
                        f"Impact score: {impact:.1f}",
                        f"Dependency score: {dependency:.1f}",
                        f"Effort score: {effort:.1f}",
                        f"Overall priority: {overall:.1f} ({priority_level})"
                    ],
                    confidence_score=0.8
                ))
            
            # Sort by score and assign order
            scores.sort(key=lambda s: s.overall_score, reverse=True)
            for i, score in enumerate(scores):
                score.suggested_order = i + 1
            
            # Find parallel groups
            parallel_groups = self._find_parallel_groups(scores, input_data.action_items)
            for group in parallel_groups:
                for score in scores:
                    if score.item_id in group and len(group) > 1:
                        score.can_be_parallelized = True
                        score.parallel_with = [id for id in group if id != score.item_id]
            
            result = PrioritizationResult(
                scores=scores,
                recommended_order=[s.item_id for s in scores],
                p0_items=[s.item_id for s in scores if s.priority_level == "P0"],
                parallel_groups=parallel_groups
            )
            
            self._log_audit_event(
                event_type="prioritization_complete",
                context=context,
                details={
                    "p0_count": len(result.p0_items),
                    "p1_count": len([s for s in scores if s.priority_level == "P1"]),
                    "parallel_opportunities": len(parallel_groups)
                }
            )
            
            # Store decision records in context
            context.shared_state["prioritization_decisions"] = [
                d.model_dump() for d in decision_records
            ]
            
            return AgentResult(
                success=True,
                data=result.model_dump(),
                confidence=0.85,
                reasoning=f"Prioritized {len(scores)} items: {len(result.p0_items)} P0, identified {len(parallel_groups)} parallel groups",
                metadata={"decision_count": len(decision_records)}
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="prioritization_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _calculate_urgency_score(
        self,
        item: Dict[str, Any],
        current_date: datetime
    ) -> float:
        """Calculate urgency score based on deadline and markers."""
        score = 0.0
        
        # Base score from urgency field
        urgency = item.get("urgency", "normal")
        score += self._urgency_weights.get(urgency, 15)
        
        # Deadline proximity bonus
        deadline = item.get("deadline")
        if deadline:
            if isinstance(deadline, str):
                try:
                    deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                except:
                    deadline = None
            
            if deadline:
                days_until = (deadline - current_date).days
                if days_until <= 0:
                    score += 50  # Overdue
                elif days_until <= 1:
                    score += 40
                elif days_until <= 3:
                    score += 30
                elif days_until <= 7:
                    score += 20
                elif days_until <= 14:
                    score += 10
        
        return min(score, 100)
    
    def _calculate_impact_score(
        self,
        item: Dict[str, Any],
        input_data: PrioritizationInput
    ) -> float:
        """Calculate impact score based on business value."""
        score = 30  # Base score
        
        # Check for high-impact keywords in title/description
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        
        keyword_matches = sum(1 for kw in self._high_impact_keywords if kw in text)
        score += keyword_matches * 10
        
        # Check against business priorities
        for priority in input_data.business_priorities:
            if priority.lower() in text:
                score += 20
        
        # Related decision bonus
        if item.get("related_decision_id"):
            score += 15
        
        return min(score, 100)
    
    def _calculate_dependency_score(
        self,
        item: Dict[str, Any],
        blockers: set
    ) -> float:
        """Calculate dependency score - blockers get higher priority."""
        score = 30  # Base score
        
        if item["id"] in blockers:
            score += 50  # This item blocks others
        
        if item.get("depends_on"):
            score -= 10  # This item is blocked
        
        return max(0, min(score, 100))
    
    def _calculate_effort_score(self, item: Dict[str, Any]) -> float:
        """Calculate effort score - quick wins get a bonus."""
        # Estimate effort from description length (simple heuristic)
        description = item.get("description", "")
        
        if len(description) < 50:
            return 80  # Quick task
        elif len(description) < 150:
            return 60  # Medium task
        else:
            return 40  # Larger task
    
    def _build_dependency_graph(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Build a graph of dependencies."""
        graph = {}
        for item in items:
            graph[item["id"]] = item.get("depends_on", [])
        return graph
    
    def _find_blockers(self, graph: Dict[str, List[str]]) -> set:
        """Find items that block other items."""
        blockers = set()
        for item_id, dependencies in graph.items():
            for dep in dependencies:
                blockers.add(dep)
        return blockers
    
    def _score_to_priority(self, score: float) -> str:
        """Convert score to priority level."""
        for level, threshold in self._priority_thresholds.items():
            if score >= threshold:
                return level
        return "P4"
    
    def _build_reasoning(
        self,
        item: Dict[str, Any],
        urgency: float,
        impact: float,
        dependency: float,
        effort: float,
        overall: float
    ) -> str:
        """Build human-readable reasoning for the priority."""
        reasons = []
        
        if urgency >= 70:
            reasons.append("high urgency due to deadline")
        elif urgency >= 40:
            reasons.append("moderate time pressure")
        
        if impact >= 70:
            reasons.append("high business impact")
        
        if dependency >= 70:
            reasons.append("blocks other tasks")
        
        if effort >= 70:
            reasons.append("quick win opportunity")
        
        if not reasons:
            reasons.append("standard priority task")
        
        return f"Priority assigned based on: {', '.join(reasons)}. Overall score: {overall:.1f}"
    
    def _find_parallel_groups(
        self,
        scores: List[PriorityScore],
        items: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Find groups of items that can be worked on in parallel."""
        groups = []
        item_map = {i["id"]: i for i in items}
        
        # Group by assignee (different people can work in parallel)
        assignee_groups = {}
        for item in items:
            assignee = item.get("assignee", "unassigned")
            if assignee not in assignee_groups:
                assignee_groups[assignee] = []
            assignee_groups[assignee].append(item["id"])
        
        # Items with no mutual dependencies within same priority can be parallel
        # Simplified: items with different assignees and similar priority
        for i, score1 in enumerate(scores):
            for score2 in scores[i+1:]:
                item1 = item_map.get(score1.item_id, {})
                item2 = item_map.get(score2.item_id, {})
                
                # Different assignees
                if item1.get("assignee") != item2.get("assignee"):
                    # No dependency conflicts
                    deps1 = set(item1.get("depends_on", []))
                    deps2 = set(item2.get("depends_on", []))
                    
                    if score1.item_id not in deps2 and score2.item_id not in deps1:
                        # Similar priority (within 20 points)
                        if abs(score1.overall_score - score2.overall_score) < 20:
                            # Add to or create group
                            found_group = False
                            for group in groups:
                                if score1.item_id in group or score2.item_id in group:
                                    if score1.item_id not in group:
                                        group.append(score1.item_id)
                                    if score2.item_id not in group:
                                        group.append(score2.item_id)
                                    found_group = True
                                    break
                            
                            if not found_group:
                                groups.append([score1.item_id, score2.item_id])
        
        return groups
