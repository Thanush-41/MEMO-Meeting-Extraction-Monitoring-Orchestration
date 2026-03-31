"""
Gemini AI Enrichment Agent.

Uses Google Gemini to enhance rule-based extraction results by:
- Finding implicit decisions and action items the regex missed
- Generating better summaries
- Detecting sentiment nuance (sarcasm, hedging)
- Identifying risks and blockers
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)


class GeminiEnrichmentInput(BaseModel):
    """Input for AI enrichment — takes the raw transcript plus rule-based results."""
    transcript_text: str
    transcript_analysis: Dict[str, Any] = Field(default_factory=dict)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    action_items: List[Dict[str, Any]] = Field(default_factory=list)


class GeminiEnrichmentResult(BaseModel):
    """Result of AI enrichment pass."""
    ai_summary: str = ""
    missed_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    missed_action_items: List[Dict[str, Any]] = Field(default_factory=list)
    risks_identified: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_analysis: Dict[str, Any] = Field(default_factory=dict)
    key_insights: List[str] = Field(default_factory=list)
    enriched_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    enriched_action_items: List[Dict[str, Any]] = Field(default_factory=list)


_ENRICHMENT_PROMPT = """You are an expert meeting analyst. You are given a meeting transcript and the results of a rule-based extraction system. Your job is to find what the rules MISSED and enrich the existing results.

## MEETING TRANSCRIPT
{transcript}

## RULE-BASED RESULTS ALREADY FOUND

### Decisions Found ({decision_count}):
{decisions_text}

### Action Items Found ({action_count}):
{action_items_text}

## YOUR TASK

Analyze the transcript carefully and respond with a JSON object containing EXACTLY these keys:

1. "ai_summary": A concise 2-3 sentence meeting summary capturing the main outcome and next steps.

2. "missed_decisions": Array of decisions the rule-based system MISSED. Each object: {{"description": "...", "made_by": "...", "reasoning": "why this is a decision"}}. Return [] if none missed.

3. "missed_action_items": Array of action items the rules MISSED (implicit commitments, vague promises, etc). Each object: {{"title": "...", "assignee": "...", "deadline_text": "...", "urgency": "low|normal|high|critical"}}. Return [] if none missed.

4. "risks_identified": Array of risks or concerns raised in the meeting. Each object: {{"risk": "...", "raised_by": "...", "severity": "low|medium|high"}}. Return [] if none.

5. "sentiment_analysis": Object with speaker-level sentiment: {{"overall": "positive|neutral|negative|mixed", "speakers": {{"Name": {{"tone": "...", "concerns": ["..."]}}}}}}

6. "key_insights": Array of 2-4 high-level insights about the meeting dynamics, team alignment, or potential issues.

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""


class GeminiEnrichmentAgent(BaseAgent[GeminiEnrichmentInput]):
    """
    Agent that uses Google Gemini to enrich rule-based extraction results.

    Finds implicit decisions, missed action items, sentiment nuance,
    and risks that regex patterns cannot detect.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="GeminiEnrichmentAgent",
            capabilities=[
                AgentCapability.EXTRACTION,
                AgentCapability.VERIFICATION,
            ],
            config=config or AgentConfig(timeout_seconds=60.0),
        )
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY environment variable is not set. "
                    "Set it before using the AI enrichment agent."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_prompt(self, input_data: GeminiEnrichmentInput) -> str:
        """Build the prompt from input data."""
        decisions_text = "\n".join(
            f"  - {d.get('description', d.get('title', 'N/A'))}"
            for d in input_data.decisions
        ) or "  (none found)"

        action_items_text = "\n".join(
            f"  - [{a.get('urgency', 'normal')}] {a.get('title', 'N/A')} -> {a.get('assignee', 'Unassigned')}"
            for a in input_data.action_items
        ) or "  (none found)"

        return _ENRICHMENT_PROMPT.format(
            transcript=input_data.transcript_text,
            decision_count=len(input_data.decisions),
            decisions_text=decisions_text,
            action_count=len(input_data.action_items),
            action_items_text=action_items_text,
        )

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse the JSON response from Gemini, handling markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = cleaned.split("\n")
            lines = lines[1:]  # remove opening ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)

    async def process(
        self, input_data: GeminiEnrichmentInput, context: AgentContext
    ) -> AgentResult:
        """Call Gemini and merge AI findings with rule-based results."""
        start = datetime.now(timezone.utc)

        # Handle dict input from workflow engine
        if isinstance(input_data, dict):
            input_data = GeminiEnrichmentInput(**input_data)

        try:
            client = self._get_client()
            prompt = self._build_prompt(input_data)

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )

            ai_data = self._parse_response(response.text)

            # Merge enriched lists: original + AI-discovered items
            enriched_decisions = list(input_data.decisions) + [
                {**d, "source": "ai", "confidence": 0.75}
                for d in ai_data.get("missed_decisions", [])
            ]
            enriched_action_items = list(input_data.action_items) + [
                {**a, "source": "ai", "confidence": 0.70}
                for a in ai_data.get("missed_action_items", [])
            ]

            result_data = GeminiEnrichmentResult(
                ai_summary=ai_data.get("ai_summary", ""),
                missed_decisions=ai_data.get("missed_decisions", []),
                missed_action_items=ai_data.get("missed_action_items", []),
                risks_identified=ai_data.get("risks_identified", []),
                sentiment_analysis=ai_data.get("sentiment_analysis", {}),
                key_insights=ai_data.get("key_insights", []),
                enriched_decisions=enriched_decisions,
                enriched_action_items=enriched_action_items,
            )

            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            return AgentResult(
                success=True,
                data=result_data.model_dump(),
                confidence=0.85,
                reasoning=(
                    f"AI enrichment found {len(result_data.missed_decisions)} missed decisions, "
                    f"{len(result_data.missed_action_items)} missed action items, "
                    f"and {len(result_data.risks_identified)} risks"
                ),
                execution_time_ms=elapsed,
            )

        except json.JSONDecodeError as e:
            return AgentResult(
                success=False,
                error=f"Failed to parse Gemini response as JSON: {e}",
                confidence=0.0,
                execution_time_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Gemini API error: {e}",
                confidence=0.0,
                execution_time_ms=(datetime.now(timezone.utc) - start).total_seconds() * 1000,
            )
