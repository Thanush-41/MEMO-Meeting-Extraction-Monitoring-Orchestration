"""
Action Item Extractor Agent.

Extracts action items and tasks from meeting transcripts,
identifying what needs to be done, by whom, and deadlines.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import hashlib

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)


class ActionItem(BaseModel):
    """An action item extracted from the meeting."""
    id: str
    title: str
    description: str
    
    # Assignment
    assignee: Optional[str] = None
    assignee_email: Optional[str] = None
    assigned_by: Optional[str] = None
    
    # Timing
    deadline: Optional[datetime] = None
    deadline_text: Optional[str] = None  # Original text e.g., "by Friday"
    urgency: str = "normal"  # low, normal, high, critical
    
    # Classification
    action_type: str  # task, follow_up, review, approval, communication
    category: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)  # Other action item IDs
    blocks: List[str] = Field(default_factory=list)
    related_decision_id: Optional[str] = None
    
    # Status
    status: str = "pending"
    confidence: float = 0.0
    
    # Context
    context: str = ""
    source_segment_index: Optional[int] = None


class ActionItemExtractionInput(BaseModel):
    """Input for action item extraction."""
    transcript_analysis: Dict[str, Any]
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    participant_info: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_date: datetime = Field(default_factory=datetime.utcnow)


class ActionItemExtractionResult(BaseModel):
    """Result of action item extraction."""
    action_items: List[ActionItem] = Field(default_factory=list)
    total_found: int = 0
    assigned_count: int = 0
    unassigned_count: int = 0
    with_deadlines: int = 0
    high_urgency_count: int = 0


class ActionItemExtractor(BaseAgent[ActionItemExtractionInput]):
    """
    Agent that extracts action items from meeting transcripts.
    
    Identifies:
    - Explicit tasks ("John will do X by Friday")
    - Implicit tasks (from decisions and discussions)
    - Follow-up items
    - Deadlines and urgency
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="ActionItemExtractor",
            capabilities=[AgentCapability.EXTRACTION],
            config=config or AgentConfig(
                confidence_threshold=0.6,
                max_retries=2
            )
        )
        
        # Action patterns
        self._action_patterns = [
            # Direct assignments with person
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:will|should|needs?\s+to|has\s+to|must)\s+(.+?)(?:\.|$)", "task"),
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:is\s+)?(?:responsible\s+for|in\s+charge\s+of|owns?|handling)\s+(.+?)(?:\.|$)", "ownership"),
            (r"(?:assigned?\s+to|goes\s+to|belongs\s+to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)[:\s]+(.+?)(?:\.|$)", "assignment"),
            
            # Action without specific person
            (r"(?:we|someone)\s+(?:need|should|must|have\s+to)\s+(.+?)(?:\.|$)", "unassigned_task"),
            (r"(?:action\s+item|todo|task)[:\s]+(.+?)(?:\.|$)", "explicit_action"),
            (r"(?:follow\s+up|check\s+back|revisit)\s+(?:on|about|regarding)?\s*(.+?)(?:\.|$)", "follow_up"),
            
            # Communication tasks
            (r"(?:send|email|notify|inform|update|share\s+with)\s+(.+?)(?:\.|$)", "communication"),
            
            # Review/approval tasks
            (r"(?:review|approve|sign\s+off|validate)\s+(.+?)(?:\.|$)", "review"),
        ]
        
        # Deadline patterns
        self._deadline_patterns = [
            (r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "weekday"),
            (r"by\s+(end\s+of\s+(?:day|week|month|quarter))", "relative"),
            (r"by\s+(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", "date"),
            (r"(?:before|no\s+later\s+than)\s+(next\s+(?:week|month|meeting))", "relative"),
            (r"(?:in|within)\s+(\d+)\s*(day|week|hour|month)s?", "duration"),
            (r"(?:asap|immediately|urgent(?:ly)?)", "urgent"),
            (r"(?:today|tomorrow|tonight)", "immediate"),
        ]
        
        # Urgency indicators
        self._urgency_patterns = {
            "critical": ["critical", "emergency", "urgent", "asap", "immediately", "blocker"],
            "high": ["important", "priority", "soon", "quickly", "time-sensitive"],
            "low": ["when possible", "nice to have", "eventually", "if time permits"]
        }
    
    async def process(
        self,
        input_data: ActionItemExtractionInput,
        context: AgentContext
    ) -> AgentResult:
        """Extract action items from transcript analysis."""
        
        self._log_audit_event(
            event_type="action_extraction_start",
            context=context,
            details={
                "has_decisions": len(input_data.decisions) > 0,
                "has_participant_info": len(input_data.participant_info) > 0
            }
        )
        
        try:
            analysis = input_data.transcript_analysis
            segments = analysis.get("segments", [])
            speakers = {s.get("name", ""): s for s in analysis.get("speakers", [])}
            
            action_items = []
            
            # Extract from transcript segments
            for i, segment in enumerate(segments):
                text = segment.get("text", "")
                speaker = segment.get("speaker", "Unknown")
                
                items = self._extract_from_text(
                    text=text,
                    segment_index=i,
                    speaker=speaker,
                    input_data=input_data
                )
                action_items.extend(items)
            
            # Extract from decisions
            for decision in input_data.decisions:
                if decision.get("requires_follow_up"):
                    item = self._create_from_decision(decision, input_data)
                    if item:
                        action_items.append(item)
            
            # Assign IDs and deduplicate
            action_items = self._deduplicate_items(action_items)
            for i, item in enumerate(action_items):
                item.id = f"AI-{i+1:03d}"
            
            # Try to match unassigned items to participants
            action_items = self._infer_assignments(action_items, input_data)
            
            # Build result
            result = ActionItemExtractionResult(
                action_items=action_items,
                total_found=len(action_items),
                assigned_count=len([a for a in action_items if a.assignee]),
                unassigned_count=len([a for a in action_items if not a.assignee]),
                with_deadlines=len([a for a in action_items if a.deadline]),
                high_urgency_count=len([a for a in action_items if a.urgency in ["high", "critical"]])
            )
            
            avg_confidence = (
                sum(a.confidence for a in action_items) / len(action_items)
                if action_items else 0.0
            )
            
            self._log_audit_event(
                event_type="action_extraction_complete",
                context=context,
                details={
                    "items_found": len(action_items),
                    "assigned": result.assigned_count,
                    "with_deadlines": result.with_deadlines,
                    "high_urgency": result.high_urgency_count
                }
            )
            
            return AgentResult(
                success=True,
                data=result.model_dump(),
                confidence=avg_confidence,
                reasoning=f"Extracted {len(action_items)} action items ({result.assigned_count} assigned, {result.high_urgency_count} high priority)",
                warnings=[f"{result.unassigned_count} items need owner assignment"] if result.unassigned_count > 0 else []
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="action_extraction_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _extract_from_text(
        self,
        text: str,
        segment_index: int,
        speaker: str,
        input_data: ActionItemExtractionInput
    ) -> List[ActionItem]:
        """Extract action items from a text segment."""
        items = []
        
        for pattern, action_type in self._action_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                
                # Determine assignee and description based on pattern type
                if len(groups) == 2:
                    assignee_name, description = groups
                else:
                    assignee_name = None
                    description = groups[0]
                
                # Skip if description is too short
                if len(description.strip()) < 5:
                    continue
                
                # Extract deadline
                deadline, deadline_text = self._extract_deadline(text, input_data.current_date)
                
                # Determine urgency
                urgency = self._determine_urgency(text)
                
                # Create title from description
                title = self._create_title(description)
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    has_assignee=assignee_name is not None,
                    has_deadline=deadline is not None,
                    action_type=action_type,
                    text_length=len(text)
                )
                
                item = ActionItem(
                    id="",  # Will be assigned later
                    title=title,
                    description=description.strip(),
                    assignee=assignee_name,
                    assigned_by=speaker if assignee_name else None,
                    deadline=deadline,
                    deadline_text=deadline_text,
                    urgency=urgency,
                    action_type=action_type,
                    confidence=confidence,
                    context=text[:200],
                    source_segment_index=segment_index
                )
                items.append(item)
        
        return items
    
    def _extract_deadline(
        self,
        text: str,
        current_date: datetime
    ) -> tuple[Optional[datetime], Optional[str]]:
        """Extract deadline from text."""
        text_lower = text.lower()
        
        for pattern, deadline_type in self._deadline_patterns:
            match = re.search(pattern, text_lower)
            if match:
                deadline_text = match.group(0)
                
                if deadline_type == "urgent":
                    # ASAP - set to end of today
                    return current_date.replace(hour=23, minute=59), deadline_text
                
                elif deadline_type == "immediate":
                    if "tomorrow" in deadline_text:
                        return current_date + timedelta(days=1), deadline_text
                    else:
                        return current_date.replace(hour=23, minute=59), deadline_text
                
                elif deadline_type == "weekday":
                    weekday_name = match.group(1).lower()
                    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    target_weekday = weekdays.index(weekday_name)
                    current_weekday = current_date.weekday()
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0:
                        days_ahead += 7
                    return current_date + timedelta(days=days_ahead), deadline_text
                
                elif deadline_type == "duration":
                    amount = int(match.group(1))
                    unit = match.group(2).lower()
                    if unit.startswith("day"):
                        return current_date + timedelta(days=amount), deadline_text
                    elif unit.startswith("week"):
                        return current_date + timedelta(weeks=amount), deadline_text
                    elif unit.startswith("hour"):
                        return current_date + timedelta(hours=amount), deadline_text
                    elif unit.startswith("month"):
                        return current_date + timedelta(days=amount*30), deadline_text
                
                elif deadline_type == "relative":
                    captured = match.group(1).lower()
                    if "end of day" in captured:
                        return current_date.replace(hour=23, minute=59), deadline_text
                    elif "end of week" in captured:
                        days_to_friday = 4 - current_date.weekday()
                        if days_to_friday <= 0:
                            days_to_friday += 7
                        return current_date + timedelta(days=days_to_friday), deadline_text
                    elif "next week" in captured:
                        return current_date + timedelta(weeks=1), deadline_text
                    elif "next month" in captured:
                        return current_date + timedelta(days=30), deadline_text
        
        return None, None
    
    def _determine_urgency(self, text: str) -> str:
        """Determine urgency level from text."""
        text_lower = text.lower()
        
        for urgency, indicators in self._urgency_patterns.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return urgency
        
        return "normal"
    
    def _create_title(self, description: str) -> str:
        """Create a concise title from description."""
        # Take first significant part
        words = description.split()[:8]
        title = ' '.join(words)
        
        # Remove common prefixes
        prefixes = ["to ", "the ", "a "]
        for prefix in prefixes:
            if title.lower().startswith(prefix):
                title = title[len(prefix):]
        
        # Capitalize first letter
        title = title[0].upper() + title[1:] if title else description[:50]
        
        # Add ellipsis if truncated
        if len(description.split()) > 8:
            title += "..."
        
        return title
    
    def _calculate_confidence(
        self,
        has_assignee: bool,
        has_deadline: bool,
        action_type: str,
        text_length: int
    ) -> float:
        """Calculate confidence score for action item."""
        confidence = 0.5
        
        if has_assignee:
            confidence += 0.2
        
        if has_deadline:
            confidence += 0.1
        
        if action_type in ["task", "assignment", "explicit_action"]:
            confidence += 0.15
        
        if text_length > 50:
            confidence += 0.05
        
        return min(confidence, 0.95)
    
    def _create_from_decision(
        self,
        decision: Dict[str, Any],
        input_data: ActionItemExtractionInput
    ) -> Optional[ActionItem]:
        """Create action item from a decision requiring follow-up."""
        if not decision.get("requires_follow_up"):
            return None
        
        description = decision.get("description", "")
        impacted = decision.get("impacted_parties", [])
        
        assignee = impacted[0] if impacted else None
        
        return ActionItem(
            id="",
            title=f"Follow up on: {description[:50]}...",
            description=f"Follow up on decision: {description}",
            assignee=assignee,
            action_type="follow_up",
            related_decision_id=decision.get("id"),
            confidence=0.7,
            context=decision.get("context", "")
        )
    
    def _deduplicate_items(
        self,
        items: List[ActionItem]
    ) -> List[ActionItem]:
        """Remove duplicate action items."""
        seen_hashes = set()
        unique = []
        
        for item in items:
            # Create hash from key fields
            key_parts = [
                item.title.lower(),
                item.assignee.lower() if item.assignee else "",
                item.action_type
            ]
            item_hash = hashlib.md5("||".join(key_parts).encode()).hexdigest()
            
            if item_hash not in seen_hashes:
                seen_hashes.add(item_hash)
                unique.append(item)
        
        return unique
    
    def _infer_assignments(
        self,
        items: List[ActionItem],
        input_data: ActionItemExtractionInput
    ) -> List[ActionItem]:
        """Try to infer assignments for unassigned items."""
        participant_info = input_data.participant_info
        
        for item in items:
            if item.assignee:
                continue
            
            # Look for keywords matching participant roles/areas
            for name, info in participant_info.items():
                role = info.get("role", "").lower()
                areas = info.get("areas", [])
                
                desc_lower = item.description.lower()
                
                # Check if description matches role
                if role and role in desc_lower:
                    item.assignee = name
                    item.confidence = min(item.confidence + 0.1, 0.9)
                    break
                
                # Check if description matches areas
                for area in areas:
                    if area.lower() in desc_lower:
                        item.assignee = name
                        item.confidence = min(item.confidence + 0.1, 0.9)
                        break
        
        return items
