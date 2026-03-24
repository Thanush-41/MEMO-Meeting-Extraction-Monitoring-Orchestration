"""
Decision Extractor Agent.

Extracts decisions made during meetings from transcript analysis,
identifying what was decided, who made the decision, and impacted parties.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)


class Decision(BaseModel):
    """A decision extracted from the meeting."""
    id: str
    description: str
    decision_type: str  # approval, rejection, deferral, assignment, policy, technical
    made_by: List[str] = Field(default_factory=list)
    impacted_parties: List[str] = Field(default_factory=list)
    context: str = ""  # Surrounding discussion
    confidence: float = 0.0
    
    # Extracted details
    rationale: Optional[str] = None
    alternatives_discussed: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)  # "if", "unless", "provided that"
    
    # Metadata
    timestamp_in_meeting: Optional[str] = None
    requires_follow_up: bool = False
    follow_up_notes: Optional[str] = None


class DecisionExtractionInput(BaseModel):
    """Input for decision extraction."""
    transcript_analysis: Dict[str, Any]  # From TranscriptAnalyzer
    participant_roles: Dict[str, str] = Field(default_factory=dict)  # name -> role


class DecisionExtractionResult(BaseModel):
    """Result of decision extraction."""
    decisions: List[Decision] = Field(default_factory=list)
    total_found: int = 0
    high_confidence_count: int = 0
    requires_review: List[str] = Field(default_factory=list)  # Decision IDs needing review


class DecisionExtractor(BaseAgent[DecisionExtractionInput]):
    """
    Agent that extracts decisions from meeting transcript analysis.
    
    Identifies:
    - Explicit decisions ("we decided to...")
    - Implicit decisions (consensus reached)
    - Approvals and rejections
    - Assignments and delegations
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="DecisionExtractor",
            capabilities=[AgentCapability.EXTRACTION],
            config=config or AgentConfig(
                confidence_threshold=0.65,
                max_retries=2
            )
        )
        
        # Decision indicator patterns
        self._explicit_patterns = [
            (r"(?:we|let's|i)\s+(?:decided?|agree[d]?)\s+(?:to|that)\s+(.+?)(?:\.|$)", "explicit"),
            (r"(?:the\s+)?decision\s+(?:is|was)\s+(?:to|that)\s+(.+?)(?:\.|$)", "explicit"),
            (r"(?:we're|we\s+are)\s+going\s+(?:to|with)\s+(.+?)(?:\.|$)", "explicit"),
            (r"(?:approved?|approve[d]?)\s+(?:the|to)\s+(.+?)(?:\.|$)", "approval"),
            (r"(?:rejected?|reject[ed]?|declined?)\s+(?:the|to)\s+(.+?)(?:\.|$)", "rejection"),
            (r"(?:postponed?|defer(?:red)?|tabled?)\s+(?:the|to)\s+(.+?)(?:\.|$)", "deferral"),
        ]
        
        self._assignment_patterns = [
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+will\s+(?:be\s+)?(?:responsible\s+for|handle|take\s+care\s+of|own)\s+(.+?)(?:\.|$)", "assignment"),
            (r"assigned?\s+(?:to\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[:\-]?\s*(.+?)(?:\.|$)", "assignment"),
            (r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:to|should|needs\s+to)\s+(.+?)(?:\.|$)", "task"),
        ]
        
        self._consensus_indicators = [
            "everyone agrees", "all agreed", "consensus", "unanimous",
            "no objections", "sounds good to everyone", "all in favor"
        ]
        
        self._condition_patterns = [
            r"(?:if|unless|provided\s+that|as\s+long\s+as|assuming)\s+(.+?)(?:\.|,|$)",
            r"(?:subject\s+to|contingent\s+on|depending\s+on)\s+(.+?)(?:\.|,|$)",
        ]
    
    async def process(
        self,
        input_data: DecisionExtractionInput,
        context: AgentContext
    ) -> AgentResult:
        """Extract decisions from transcript analysis."""
        
        self._log_audit_event(
            event_type="decision_extraction_start",
            context=context,
            details={"has_roles": len(input_data.participant_roles) > 0}
        )
        
        try:
            analysis = input_data.transcript_analysis
            segments = analysis.get("segments", [])
            speakers = {s["name"] for s in analysis.get("speakers", [])}
            
            decisions = []
            decision_id = 0
            
            # Process each segment looking for decisions
            for i, segment in enumerate(segments):
                text = segment.get("text", "")
                speaker = segment.get("speaker", "Unknown")
                
                # Get context (surrounding segments)
                context_text = self._get_context(segments, i)
                
                # Extract explicit decisions
                for pattern, decision_type in self._explicit_patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        decision_id += 1
                        decision = self._create_decision(
                            decision_id=f"DEC-{decision_id:03d}",
                            description=match.group(1).strip(),
                            decision_type=decision_type,
                            speaker=speaker,
                            context=context_text,
                            full_text=text,
                            participant_roles=input_data.participant_roles
                        )
                        decisions.append(decision)
                
                # Extract assignments
                for pattern, decision_type in self._assignment_patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        decision_id += 1
                        assignee = match.group(1)
                        task = match.group(2)
                        decision = Decision(
                            id=f"DEC-{decision_id:03d}",
                            description=f"Assigned to {assignee}: {task}",
                            decision_type="assignment",
                            made_by=[speaker],
                            impacted_parties=[assignee],
                            context=context_text,
                            confidence=0.7,
                            requires_follow_up=True,
                            follow_up_notes=f"Track completion by {assignee}"
                        )
                        decisions.append(decision)
                
                # Check for consensus decisions
                text_lower = text.lower()
                for indicator in self._consensus_indicators:
                    if indicator in text_lower:
                        decision_id += 1
                        # Find what was agreed to
                        agreed_topic = self._extract_agreed_topic(segments, i)
                        if agreed_topic:
                            decision = Decision(
                                id=f"DEC-{decision_id:03d}",
                                description=agreed_topic,
                                decision_type="consensus",
                                made_by=list(speakers),
                                context=context_text,
                                confidence=0.65
                            )
                            decisions.append(decision)
                        break
            
            # Deduplicate similar decisions
            decisions = self._deduplicate_decisions(decisions)
            
            # Calculate overall confidence and identify items needing review
            high_confidence = [d for d in decisions if d.confidence >= 0.7]
            needs_review = [d.id for d in decisions if d.confidence < 0.6]
            
            result = DecisionExtractionResult(
                decisions=decisions,
                total_found=len(decisions),
                high_confidence_count=len(high_confidence),
                requires_review=needs_review
            )
            
            avg_confidence = (
                sum(d.confidence for d in decisions) / len(decisions)
                if decisions else 0.0
            )
            
            self._log_audit_event(
                event_type="decision_extraction_complete",
                context=context,
                details={
                    "decisions_found": len(decisions),
                    "high_confidence": len(high_confidence),
                    "needs_review": len(needs_review),
                    "avg_confidence": avg_confidence
                }
            )
            
            return AgentResult(
                success=True,
                data=result.model_dump(),
                confidence=avg_confidence,
                reasoning=f"Extracted {len(decisions)} decisions, {len(high_confidence)} with high confidence",
                warnings=[f"{len(needs_review)} decisions need human review"] if needs_review else []
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="decision_extraction_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _get_context(
        self,
        segments: List[Dict],
        current_index: int,
        window: int = 2
    ) -> str:
        """Get context text from surrounding segments."""
        start = max(0, current_index - window)
        end = min(len(segments), current_index + window + 1)
        
        context_parts = []
        for i in range(start, end):
            if i != current_index:
                context_parts.append(segments[i].get("text", "")[:100])
        
        return " ... ".join(context_parts)
    
    def _create_decision(
        self,
        decision_id: str,
        description: str,
        decision_type: str,
        speaker: str,
        context: str,
        full_text: str,
        participant_roles: Dict[str, str]
    ) -> Decision:
        """Create a decision object with analysis."""
        # Extract conditions
        conditions = []
        for pattern in self._condition_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                conditions.append(match.group(1).strip())
        
        # Determine confidence based on explicitness
        confidence = 0.8 if decision_type == "explicit" else 0.65
        
        # Check if decision maker has authority (if roles provided)
        role = participant_roles.get(speaker, "")
        if role.lower() in ["manager", "director", "lead", "owner", "executive"]:
            confidence += 0.1
        
        # Cap confidence
        confidence = min(confidence, 0.95)
        
        return Decision(
            id=decision_id,
            description=description,
            decision_type=decision_type,
            made_by=[speaker],
            context=context,
            conditions=conditions,
            confidence=confidence,
            requires_follow_up=decision_type in ["assignment", "task"]
        )
    
    def _extract_agreed_topic(
        self,
        segments: List[Dict],
        consensus_index: int
    ) -> Optional[str]:
        """Extract what was agreed to from context."""
        # Look at previous segments for topic
        for i in range(consensus_index - 1, max(-1, consensus_index - 4), -1):
            text = segments[i].get("text", "")
            # Look for proposal patterns
            proposal_patterns = [
                r"(?:propose|suggest|recommend)\s+(?:that\s+)?(?:we\s+)?(.+?)(?:\.|$)",
                r"(?:how\s+about|what\s+if)\s+(?:we\s+)?(.+?)(?:\?|$)",
                r"(?:should\s+we|let's)\s+(.+?)(?:\?|$)",
            ]
            for pattern in proposal_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return None
    
    def _deduplicate_decisions(
        self,
        decisions: List[Decision]
    ) -> List[Decision]:
        """Remove duplicate or very similar decisions."""
        if not decisions:
            return decisions
        
        unique = []
        seen_descriptions = set()
        
        for decision in decisions:
            # Normalize description
            normalized = decision.description.lower().strip()
            words = set(normalized.split())
            
            # Check similarity with seen descriptions
            is_duplicate = False
            for seen in seen_descriptions:
                seen_words = set(seen.split())
                overlap = len(words & seen_words) / max(len(words), 1)
                if overlap > 0.7:  # 70% word overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(decision)
                seen_descriptions.add(normalized)
        
        return unique
