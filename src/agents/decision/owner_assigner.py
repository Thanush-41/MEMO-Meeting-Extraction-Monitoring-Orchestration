"""
Owner Assigner Agent.

Intelligently assigns owners to unassigned tasks based on
expertise, workload, and historical patterns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)
from src.core.audit import DecisionRecord


class TeamMember(BaseModel):
    """Information about a team member."""
    name: str
    email: Optional[str] = None
    role: str = ""
    skills: List[str] = Field(default_factory=list)
    areas_of_expertise: List[str] = Field(default_factory=list)
    current_workload: float = 0.0  # Hours or points
    max_capacity: float = 40.0
    availability: float = 1.0  # 0-1 multiplier
    
    # Historical data
    past_assignments: List[str] = Field(default_factory=list)  # Task types completed
    success_rate: float = 0.9
    avg_completion_time: float = 1.0  # Multiplier of estimated time


class AssignmentRecommendation(BaseModel):
    """Recommendation for task assignment."""
    item_id: str
    recommended_owner: str
    confidence: float
    
    # Scoring
    fit_score: float = 0.0  # How well skills match
    capacity_score: float = 0.0  # Available capacity
    history_score: float = 0.0  # Past success with similar tasks
    
    # Alternatives
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Reasoning
    reasoning: str = ""
    match_factors: List[str] = Field(default_factory=list)


class OwnerAssignmentInput(BaseModel):
    """Input for owner assignment."""
    action_items: List[Dict[str, Any]]  # Items needing owners
    team_members: List[TeamMember]
    existing_assignments: Dict[str, str] = Field(default_factory=dict)  # item_id -> owner
    
    # Constraints
    max_items_per_person: int = 10
    balance_workload: bool = True


class OwnerAssignmentResult(BaseModel):
    """Result of owner assignment."""
    assignments: List[AssignmentRecommendation] = Field(default_factory=list)
    assigned_count: int = 0
    unassignable_items: List[str] = Field(default_factory=list)
    workload_distribution: Dict[str, int] = Field(default_factory=dict)


class OwnerAssigner(BaseAgent[OwnerAssignmentInput]):
    """
    Agent that assigns owners to tasks based on skills and capacity.
    
    Considers:
    - Skill/expertise match
    - Current workload
    - Historical performance
    - Task distribution balance
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="OwnerAssigner",
            capabilities=[AgentCapability.DECISION_MAKING],
            config=config or AgentConfig(
                confidence_threshold=0.6,
                max_retries=2
            )
        )
        
        # Skill mappings for common task types
        self._task_skill_map = {
            "review": ["reviewer", "senior", "lead", "architect"],
            "approval": ["manager", "lead", "director"],
            "communication": ["coordinator", "pm", "manager"],
            "follow_up": ["coordinator", "pm"],
            "task": [],  # Generic, match by keywords
        }
    
    async def process(
        self,
        input_data: OwnerAssignmentInput,
        context: AgentContext
    ) -> AgentResult:
        """Assign owners to tasks."""
        
        self._log_audit_event(
            event_type="owner_assignment_start",
            context=context,
            details={
                "items_count": len(input_data.action_items),
                "team_size": len(input_data.team_members)
            }
        )
        
        try:
            if not input_data.team_members:
                return AgentResult(
                    success=False,
                    error="No team members available for assignment",
                    confidence=0.0
                )
            
            # Build skill index
            skill_index = self._build_skill_index(input_data.team_members)
            
            # Track assignments for load balancing
            workload_tracker = defaultdict(int)
            for item_id, owner in input_data.existing_assignments.items():
                workload_tracker[owner] += 1
            
            assignments = []
            unassignable = []
            decision_records = []
            
            # Sort items by priority (if available) to assign important items first
            items = sorted(
                input_data.action_items,
                key=lambda x: x.get("urgency", "normal") != "critical",
            )
            
            for item in items:
                # Skip if already assigned
                if item["id"] in input_data.existing_assignments:
                    continue
                
                # Find best owner
                recommendation = self._find_best_owner(
                    item=item,
                    team_members=input_data.team_members,
                    skill_index=skill_index,
                    workload_tracker=workload_tracker,
                    max_items=input_data.max_items_per_person,
                    balance_workload=input_data.balance_workload
                )
                
                if recommendation:
                    assignments.append(recommendation)
                    workload_tracker[recommendation.recommended_owner] += 1
                    
                    # Create decision record
                    decision_records.append(DecisionRecord(
                        agent_id=self.id,
                        agent_name=self.name,
                        workflow_id=context.workflow_id,
                        execution_id=context.execution_id,
                        step_number=context.step_number,
                        decision_type="owner_assignment",
                        decision_description=f"Assigned '{item.get('title', item['id'])}' to {recommendation.recommended_owner}",
                        decision_outcome=recommendation.recommended_owner,
                        input_data_summary={
                            "item_id": item["id"],
                            "action_type": item.get("action_type", "task")
                        },
                        reasoning_steps=recommendation.match_factors + [recommendation.reasoning],
                        confidence_score=recommendation.confidence,
                        alternatives=[
                            {"owner": alt["owner"], "score": alt["score"]}
                            for alt in recommendation.alternatives[:3]
                        ]
                    ))
                else:
                    unassignable.append(item["id"])
            
            result = OwnerAssignmentResult(
                assignments=assignments,
                assigned_count=len(assignments),
                unassignable_items=unassignable,
                workload_distribution=dict(workload_tracker)
            )
            
            avg_confidence = (
                sum(a.confidence for a in assignments) / len(assignments)
                if assignments else 0.0
            )
            
            self._log_audit_event(
                event_type="owner_assignment_complete",
                context=context,
                details={
                    "assigned": len(assignments),
                    "unassignable": len(unassignable),
                    "avg_confidence": avg_confidence
                }
            )
            
            context.shared_state["assignment_decisions"] = [
                d.model_dump() for d in decision_records
            ]
            
            return AgentResult(
                success=True,
                data=result.model_dump(),
                confidence=avg_confidence,
                reasoning=f"Assigned {len(assignments)} items across {len(set(a.recommended_owner for a in assignments))} team members",
                warnings=[f"{len(unassignable)} items could not be assigned"] if unassignable else []
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="owner_assignment_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _build_skill_index(
        self,
        team_members: List[TeamMember]
    ) -> Dict[str, List[str]]:
        """Build index of skills to team members."""
        index = defaultdict(list)
        
        for member in team_members:
            for skill in member.skills:
                index[skill.lower()].append(member.name)
            for area in member.areas_of_expertise:
                index[area.lower()].append(member.name)
            # Index role
            index[member.role.lower()].append(member.name)
        
        return dict(index)
    
    def _find_best_owner(
        self,
        item: Dict[str, Any],
        team_members: List[TeamMember],
        skill_index: Dict[str, List[str]],
        workload_tracker: Dict[str, int],
        max_items: int,
        balance_workload: bool
    ) -> Optional[AssignmentRecommendation]:
        """Find the best owner for a task."""
        candidates = []
        
        # Extract keywords from task
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        action_type = item.get("action_type", "task")
        
        for member in team_members:
            # Check capacity
            current_load = workload_tracker.get(member.name, 0)
            if current_load >= max_items:
                continue
            
            # Calculate scores
            fit_score = self._calculate_fit_score(member, text, action_type, skill_index)
            capacity_score = self._calculate_capacity_score(member, current_load, max_items)
            history_score = self._calculate_history_score(member, action_type)
            
            # Combined score
            overall = (
                fit_score * 0.45 +
                capacity_score * 0.35 +
                history_score * 0.20
            )
            
            # Adjust for workload balancing
            if balance_workload and current_load > 0:
                # Slightly penalize those with more tasks
                overall *= (1 - current_load * 0.05)
            
            candidates.append({
                "member": member,
                "owner": member.name,
                "score": overall,
                "fit_score": fit_score,
                "capacity_score": capacity_score,
                "history_score": history_score
            })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        best = candidates[0]
        
        # Build match factors
        match_factors = []
        if best["fit_score"] > 60:
            match_factors.append(f"Strong skill match (score: {best['fit_score']:.0f})")
        if best["capacity_score"] > 80:
            match_factors.append("Good capacity available")
        if best["history_score"] > 70:
            match_factors.append("Successful history with similar tasks")
        
        return AssignmentRecommendation(
            item_id=item["id"],
            recommended_owner=best["owner"],
            confidence=best["score"] / 100,
            fit_score=best["fit_score"],
            capacity_score=best["capacity_score"],
            history_score=best["history_score"],
            alternatives=[
                {"owner": c["owner"], "score": c["score"]}
                for c in candidates[1:4]
            ],
            reasoning=f"Selected {best['owner']} based on skill match and availability",
            match_factors=match_factors
        )
    
    def _calculate_fit_score(
        self,
        member: TeamMember,
        task_text: str,
        action_type: str,
        skill_index: Dict[str, List[str]]
    ) -> float:
        """Calculate how well a member fits the task."""
        score = 30  # Base score
        
        # Check role requirements
        required_roles = self._task_skill_map.get(action_type, [])
        if required_roles:
            if member.role.lower() in required_roles:
                score += 30
        
        # Check skill matches in task text
        member_keywords = (
            [s.lower() for s in member.skills] +
            [a.lower() for a in member.areas_of_expertise]
        )
        
        matches = sum(1 for kw in member_keywords if kw in task_text)
        score += min(matches * 15, 40)
        
        return min(score, 100)
    
    def _calculate_capacity_score(
        self,
        member: TeamMember,
        current_items: int,
        max_items: int
    ) -> float:
        """Calculate capacity score based on workload."""
        # Based on current vs max items
        utilization = current_items / max_items if max_items > 0 else 1.0
        base_score = (1 - utilization) * 80
        
        # Adjust by availability
        base_score *= member.availability
        
        # Bonus if under capacity
        if current_items < max_items * 0.5:
            base_score += 20
        
        return min(base_score, 100)
    
    def _calculate_history_score(
        self,
        member: TeamMember,
        action_type: str
    ) -> float:
        """Calculate score based on historical success."""
        score = 50  # Base
        
        # Check past assignments
        if action_type in member.past_assignments:
            score += 20
        
        # Success rate bonus
        if member.success_rate >= 0.9:
            score += 20
        elif member.success_rate >= 0.8:
            score += 10
        
        # Efficiency bonus
        if member.avg_completion_time <= 0.9:
            score += 10
        
        return min(score, 100)
