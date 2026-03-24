"""
Escalation Decider Agent.

Determines when tasks or workflows require escalation based on
SLA violations, stalls, risks, and other triggers.
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


class EscalationRule(BaseModel):
    """Rule defining when to escalate."""
    name: str
    description: str
    trigger_type: str  # sla_breach, stall, risk, manual
    threshold: float  # e.g., hours overdue, days stalled
    escalation_level: int  # 1, 2, 3 (higher = more severe)
    notify_roles: List[str] = Field(default_factory=list)


class EscalationTrigger(BaseModel):
    """A triggered escalation."""
    item_id: str
    item_title: str
    trigger_type: str
    trigger_reason: str
    escalation_level: int
    
    # Timing
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    response_required_by: Optional[datetime] = None
    
    # Recipients
    notify: List[str] = Field(default_factory=list)
    escalate_to: Optional[str] = None
    
    # Context
    current_owner: Optional[str] = None
    age_hours: float = 0.0
    deadline_status: str = "on_track"  # on_track, at_risk, overdue
    
    # Recommendations
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class EscalationInput(BaseModel):
    """Input for escalation decisions."""
    action_items: List[Dict[str, Any]]
    current_date: datetime = Field(default_factory=datetime.utcnow)
    
    # SLA configuration
    sla_hours: Dict[str, float] = Field(default_factory=lambda: {
        "critical": 4,
        "high": 24,
        "normal": 72,
        "low": 168
    })
    
    # Stall detection
    stall_threshold_hours: float = 48.0  # No update for this long = stalled
    
    # Custom rules
    custom_rules: List[EscalationRule] = Field(default_factory=list)
    
    # Status information
    item_statuses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # item_id -> {status, last_update, progress, blockers}


class EscalationResult(BaseModel):
    """Result of escalation analysis."""
    escalations: List[EscalationTrigger] = Field(default_factory=list)
    at_risk_items: List[str] = Field(default_factory=list)
    level_1_count: int = 0
    level_2_count: int = 0
    level_3_count: int = 0
    no_escalation_needed: List[str] = Field(default_factory=list)


class EscalationDecider(BaseAgent[EscalationInput]):
    """
    Agent that decides when escalation is needed.
    
    Monitors:
    - SLA compliance
    - Task stalls (no progress)
    - Approaching deadlines
    - Blocked items
    - Risk indicators
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="EscalationDecider",
            capabilities=[AgentCapability.DECISION_MAKING, AgentCapability.MONITORING],
            config=config or AgentConfig(
                confidence_threshold=0.7,
                max_retries=2
            )
        )
        
        # Default escalation roles by level
        self._escalation_roles = {
            1: ["team_lead", "owner"],
            2: ["manager", "team_lead"],
            3: ["director", "manager", "executive"]
        }
        
        # Risk indicators
        self._risk_keywords = [
            "blocked", "stuck", "waiting", "dependency", "risk",
            "delay", "issue", "problem", "unable", "cannot"
        ]
    
    async def process(
        self,
        input_data: EscalationInput,
        context: AgentContext
    ) -> AgentResult:
        """Analyze items and determine escalation needs."""
        
        self._log_audit_event(
            event_type="escalation_analysis_start",
            context=context,
            details={
                "items_count": len(input_data.action_items),
                "custom_rules": len(input_data.custom_rules)
            }
        )
        
        try:
            escalations = []
            at_risk = []
            no_escalation = []
            decision_records = []
            
            for item in input_data.action_items:
                item_status = input_data.item_statuses.get(item["id"], {})
                
                # Check various escalation triggers
                triggers = []
                
                # 1. SLA breach check
                sla_trigger = self._check_sla_breach(item, input_data)
                if sla_trigger:
                    triggers.append(sla_trigger)
                
                # 2. Stall check
                stall_trigger = self._check_stall(item, item_status, input_data)
                if stall_trigger:
                    triggers.append(stall_trigger)
                
                # 3. Deadline risk check
                deadline_trigger = self._check_deadline_risk(item, input_data)
                if deadline_trigger:
                    if deadline_trigger.escalation_level < 2:
                        at_risk.append(item["id"])
                    triggers.append(deadline_trigger)
                
                # 4. Blocker check
                blocker_trigger = self._check_blockers(item, item_status)
                if blocker_trigger:
                    triggers.append(blocker_trigger)
                
                # 5. Custom rules
                for rule in input_data.custom_rules:
                    custom_trigger = self._apply_custom_rule(item, item_status, rule, input_data)
                    if custom_trigger:
                        triggers.append(custom_trigger)
                
                if triggers:
                    # Use highest level trigger
                    triggers.sort(key=lambda t: t.escalation_level, reverse=True)
                    best_trigger = triggers[0]
                    
                    # Merge recommended actions from all triggers
                    all_actions = []
                    for t in triggers:
                        all_actions.extend(t.recommended_actions)
                    best_trigger.recommended_actions = list(set(all_actions))
                    
                    escalations.append(best_trigger)
                    
                    # Create decision record
                    decision_records.append(DecisionRecord(
                        agent_id=self.id,
                        agent_name=self.name,
                        workflow_id=context.workflow_id,
                        execution_id=context.execution_id,
                        step_number=context.step_number,
                        decision_type="escalation_triggered",
                        decision_description=f"Escalating '{item.get('title', item['id'])}' - {best_trigger.trigger_reason}",
                        decision_outcome=f"Level {best_trigger.escalation_level} escalation",
                        input_data_summary={
                            "item_id": item["id"],
                            "urgency": item.get("urgency", "normal"),
                            "trigger_count": len(triggers)
                        },
                        reasoning_steps=[
                            f"Trigger type: {best_trigger.trigger_type}",
                            f"Reason: {best_trigger.trigger_reason}",
                            f"Level: {best_trigger.escalation_level}",
                            f"Other triggers: {len(triggers) - 1}"
                        ],
                        confidence_score=best_trigger.confidence
                    ))
                else:
                    no_escalation.append(item["id"])
            
            # Count by level
            level_1 = len([e for e in escalations if e.escalation_level == 1])
            level_2 = len([e for e in escalations if e.escalation_level == 2])
            level_3 = len([e for e in escalations if e.escalation_level == 3])
            
            result = EscalationResult(
                escalations=escalations,
                at_risk_items=at_risk,
                level_1_count=level_1,
                level_2_count=level_2,
                level_3_count=level_3,
                no_escalation_needed=no_escalation
            )
            
            self._log_audit_event(
                event_type="escalation_analysis_complete",
                context=context,
                details={
                    "total_escalations": len(escalations),
                    "level_1": level_1,
                    "level_2": level_2,
                    "level_3": level_3,
                    "at_risk": len(at_risk)
                }
            )
            
            context.shared_state["escalation_decisions"] = [
                d.model_dump() for d in decision_records
            ]
            
            return AgentResult(
                success=True,
                data=result.model_dump(),
                confidence=0.85,
                reasoning=f"Analyzed {len(input_data.action_items)} items: {len(escalations)} escalations needed, {len(at_risk)} at risk",
                warnings=[f"{level_3} critical (Level 3) escalations"] if level_3 > 0 else []
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="escalation_analysis_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _check_sla_breach(
        self,
        item: Dict[str, Any],
        input_data: EscalationInput
    ) -> Optional[EscalationTrigger]:
        """Check if item has breached SLA."""
        urgency = item.get("urgency", "normal")
        sla_hours = input_data.sla_hours.get(urgency, 72)
        
        # Calculate age
        created_at = item.get("created_at")
        if created_at:
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    created_at = input_data.current_date - timedelta(hours=24)
            age_hours = (input_data.current_date - created_at).total_seconds() / 3600
        else:
            age_hours = 24  # Default assumption
        
        if item.get("status", "pending") != "completed" and age_hours > sla_hours:
            hours_over = age_hours - sla_hours
            
            # Determine level based on how much over
            if hours_over > sla_hours:  # 2x over
                level = 3
            elif hours_over > sla_hours * 0.5:  # 1.5x over
                level = 2
            else:
                level = 1
            
            return EscalationTrigger(
                item_id=item["id"],
                item_title=item.get("title", item["id"]),
                trigger_type="sla_breach",
                trigger_reason=f"SLA breached by {hours_over:.1f} hours",
                escalation_level=level,
                notify=self._escalation_roles.get(level, []),
                current_owner=item.get("assignee"),
                age_hours=age_hours,
                deadline_status="overdue",
                recommended_actions=[
                    "Investigate cause of delay",
                    "Reassign if owner is blocked",
                    "Update stakeholders"
                ],
                confidence=0.9
            )
        
        return None
    
    def _check_stall(
        self,
        item: Dict[str, Any],
        status: Dict[str, Any],
        input_data: EscalationInput
    ) -> Optional[EscalationTrigger]:
        """Check if item has stalled (no updates)."""
        if item.get("status") == "completed":
            return None
        
        last_update = status.get("last_update")
        if last_update:
            if isinstance(last_update, str):
                try:
                    last_update = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                except:
                    return None
            
            hours_since_update = (input_data.current_date - last_update).total_seconds() / 3600
            
            if hours_since_update > input_data.stall_threshold_hours:
                return EscalationTrigger(
                    item_id=item["id"],
                    item_title=item.get("title", item["id"]),
                    trigger_type="stall",
                    trigger_reason=f"No updates for {hours_since_update:.0f} hours",
                    escalation_level=1 if hours_since_update < input_data.stall_threshold_hours * 2 else 2,
                    notify=self._escalation_roles.get(1, []),
                    current_owner=item.get("assignee"),
                    recommended_actions=[
                        "Check with owner for status",
                        "Identify blockers",
                        "Consider reassignment"
                    ],
                    confidence=0.85
                )
        
        return None
    
    def _check_deadline_risk(
        self,
        item: Dict[str, Any],
        input_data: EscalationInput
    ) -> Optional[EscalationTrigger]:
        """Check if deadline is at risk."""
        deadline = item.get("deadline")
        if not deadline:
            return None
        
        if isinstance(deadline, str):
            try:
                deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            except:
                return None
        
        hours_until = (deadline - input_data.current_date).total_seconds() / 3600
        
        if hours_until < 0:
            return EscalationTrigger(
                item_id=item["id"],
                item_title=item.get("title", item["id"]),
                trigger_type="deadline_breached",
                trigger_reason=f"Deadline passed {abs(hours_until):.0f} hours ago",
                escalation_level=2,
                deadline_status="overdue",
                recommended_actions=[
                    "Communicate delay to stakeholders",
                    "Set new realistic deadline",
                    "Identify root cause"
                ],
                confidence=0.95
            )
        elif hours_until < 24:
            return EscalationTrigger(
                item_id=item["id"],
                item_title=item.get("title", item["id"]),
                trigger_type="deadline_risk",
                trigger_reason=f"Deadline in {hours_until:.0f} hours",
                escalation_level=1,
                deadline_status="at_risk",
                recommended_actions=[
                    "Verify completion plan",
                    "Clear blockers",
                    "Prepare contingency"
                ],
                confidence=0.8
            )
        
        return None
    
    def _check_blockers(
        self,
        item: Dict[str, Any],
        status: Dict[str, Any]
    ) -> Optional[EscalationTrigger]:
        """Check for blockers."""
        blockers = status.get("blockers", [])
        
        if blockers:
            return EscalationTrigger(
                item_id=item["id"],
                item_title=item.get("title", item["id"]),
                trigger_type="blocked",
                trigger_reason=f"Blocked by: {', '.join(blockers[:3])}",
                escalation_level=1,
                recommended_actions=[
                    "Resolve blocking issues",
                    "Reassign or rescope task",
                    "Escalate blocker resolution"
                ],
                confidence=0.85
            )
        
        # Check for risk keywords in description
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        risk_found = [kw for kw in self._risk_keywords if kw in text]
        
        if risk_found:
            return EscalationTrigger(
                item_id=item["id"],
                item_title=item.get("title", item["id"]),
                trigger_type="risk_identified",
                trigger_reason=f"Risk indicators: {', '.join(risk_found[:3])}",
                escalation_level=1,
                recommended_actions=[
                    "Review and assess risk",
                    "Create mitigation plan"
                ],
                confidence=0.6
            )
        
        return None
    
    def _apply_custom_rule(
        self,
        item: Dict[str, Any],
        status: Dict[str, Any],
        rule: EscalationRule,
        input_data: EscalationInput
    ) -> Optional[EscalationTrigger]:
        """Apply a custom escalation rule."""
        # Custom rules can be extended based on needs
        # This is a simplified implementation
        
        if rule.trigger_type == "manual":
            # Check if manual escalation flag is set
            if status.get("manual_escalation"):
                return EscalationTrigger(
                    item_id=item["id"],
                    item_title=item.get("title", item["id"]),
                    trigger_type="manual",
                    trigger_reason=f"Manual escalation: {rule.description}",
                    escalation_level=rule.escalation_level,
                    notify=rule.notify_roles,
                    recommended_actions=["Address per manual escalation request"],
                    confidence=1.0
                )
        
        return None
