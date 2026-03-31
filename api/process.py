"""
MEMO API — /api/process
Runs the full 6-agent pipeline on a meeting transcript and returns real results.
"""
import sys
import os
import asyncio
import json
from http.server import BaseHTTPRequestHandler
from datetime import datetime

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.base_agent import AgentContext
from src.agents.extraction.transcript_agent import TranscriptAnalyzer, TranscriptInput
from src.agents.extraction.decision_parser import DecisionExtractor, DecisionExtractionInput
from src.agents.extraction.action_item_agent import ActionItemExtractor, ActionItemExtractionInput
from src.agents.decision.task_prioritizer import TaskPrioritizer, PrioritizationInput
from src.agents.decision.owner_assigner import OwnerAssigner, OwnerAssignmentInput, TeamMember
from src.agents.decision.escalation_decider import EscalationDecider, EscalationInput
from src.agents.ai.gemini_enrichment import GeminiEnrichmentAgent, GeminiEnrichmentInput


async def run_full_pipeline(transcript_text: str, participants: list, participant_roles: dict) -> dict:
    exec_id = f"api-{int(datetime.now().timestamp())}"
    steps = []

    def ctx(n: int) -> AgentContext:
        return AgentContext(
            workflow_id="memo-api",
            execution_id=exec_id,
            step_number=n,
            shared_state={}
        )

    # ── Step 1: Transcript Analysis
    t0 = datetime.now()
    analyzer = TranscriptAnalyzer()
    t_result = await analyzer.execute(
        TranscriptInput(
            transcript_text=transcript_text,
            known_participants=participants,
            meeting_date=datetime.now()
        ),
        ctx(1)
    )
    steps.append({
        "agent": "TranscriptAnalyzer",
        "success": t_result.success,
        "confidence": round(t_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "speakers": len(t_result.data.get("speakers", [])) if t_result.data else 0,
            "topics": len(t_result.data.get("topics", [])) if t_result.data else 0,
            "key_points": len(t_result.data.get("key_points", [])) if t_result.data else 0,
            "summary": (t_result.data.get("summary", "") or "")[:200] if t_result.data else ""
        }
    })

    # ── Step 2: Decision Extraction
    t0 = datetime.now()
    d_agent = DecisionExtractor()
    d_result = await d_agent.execute(
        DecisionExtractionInput(
            transcript_analysis=t_result.data,
            participant_roles=participant_roles
        ),
        ctx(2)
    )
    decisions = d_result.data.get("decisions", []) if d_result.data else []
    steps.append({
        "agent": "DecisionExtractor",
        "success": d_result.success,
        "confidence": round(d_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "total": len(decisions),
            "explicit": len([d for d in decisions if d.get("decision_type") == "explicit"]),
            "assignments": len([d for d in decisions if d.get("decision_type") == "assignment"]),
            "items": [
                {"type": d.get("decision_type"), "description": (d.get("description") or "")[:120]}
                for d in decisions[:10]
            ]
        }
    })

    # ── Step 3: Action Item Extraction
    t0 = datetime.now()
    a_agent = ActionItemExtractor()
    a_result = await a_agent.execute(
        ActionItemExtractionInput(
            transcript_analysis=t_result.data,
            decisions=decisions,
            current_date=datetime.now()
        ),
        ctx(3)
    )
    action_items = a_result.data.get("action_items", []) if a_result.data else []
    steps.append({
        "agent": "ActionItemExtractor",
        "success": a_result.success,
        "confidence": round(a_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "total": len(action_items),
            "assigned": a_result.data.get("assigned_count", 0) if a_result.data else 0,
            "with_deadlines": a_result.data.get("with_deadlines", 0) if a_result.data else 0,
            "items": [
                {
                    "title": (i.get("title") or "")[:60],
                    "assignee": i.get("assignee"),
                    "urgency": i.get("urgency", "normal"),
                    "deadline": str(i.get("deadline", "")) if i.get("deadline") else None
                }
                for i in action_items[:8]
            ]
        }
    })

    # ── Step 3.5: AI Enrichment (Gemini)
    ai_step = None
    ai_data = {}
    try:
        t0 = datetime.now()
        ai_agent = GeminiEnrichmentAgent()
        ai_result = await ai_agent.execute(
            GeminiEnrichmentInput(
                transcript_text=transcript_text,
                transcript_analysis=t_result.data or {},
                decisions=decisions,
                action_items=action_items,
            ),
            ctx(7)
        )
        if ai_result.success and ai_result.data:
            ai_data = ai_result.data
            # Merge AI-discovered items into the pipeline
            missed_actions = ai_data.get("missed_action_items", [])
            missed_decisions = ai_data.get("missed_decisions", [])
            action_items = ai_data.get("enriched_action_items", action_items)
            decisions = ai_data.get("enriched_decisions", decisions)

        ai_step = {
            "agent": "GeminiEnrichmentAgent",
            "success": ai_result.success,
            "confidence": round(ai_result.confidence * 100, 1),
            "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
            "output": {
                "ai_summary": ai_data.get("ai_summary", ""),
                "missed_decisions": len(ai_data.get("missed_decisions", [])),
                "missed_action_items": len(ai_data.get("missed_action_items", [])),
                "risks": ai_data.get("risks_identified", []),
                "sentiment": ai_data.get("sentiment_analysis", {}),
                "key_insights": ai_data.get("key_insights", []),
            }
        }
    except Exception:
        ai_step = None

    if ai_step:
        steps.append(ai_step)

    # ── Step 4: Prioritization
    t0 = datetime.now()
    p_agent = TaskPrioritizer()
    p_result = await p_agent.execute(
        PrioritizationInput(
            action_items=action_items,
            decisions=decisions,
            current_date=datetime.now(),
            business_priorities=["security", "launch", "compliance", "customer"]
        ),
        ctx(4)
    )
    scores = p_result.data.get("scores", []) if p_result.data else []
    priority_counts = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "P4": 0}
    for s in scores:
        lvl = s.get("priority_level", "P4")
        priority_counts[lvl] = priority_counts.get(lvl, 0) + 1
    steps.append({
        "agent": "TaskPrioritizer",
        "success": p_result.success,
        "confidence": round(p_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "priority_counts": priority_counts,
            "top_items": [
                {"id": s.get("item_id"), "priority": s.get("priority_level"), "score": round(s.get("overall_score", 0), 1)}
                for s in scores[:5]
            ]
        }
    })

    # ── Step 5: Owner Assignment
    t0 = datetime.now()
    team_members = [
        TeamMember(name=p, role=participant_roles.get(p, "team_member"),
                   skills=[], current_workload=2)
        for p in participants
    ]
    oa_agent = OwnerAssigner()
    oa_result = await oa_agent.execute(
        OwnerAssignmentInput(action_items=action_items, team_members=team_members),
        ctx(5)
    )
    assignments = oa_result.data.get("assignments", []) if oa_result.data else []
    steps.append({
        "agent": "OwnerAssigner",
        "success": oa_result.success,
        "confidence": round(oa_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "assigned": len(assignments),
            "unassigned": oa_result.data.get("unassigned_count", 0) if oa_result.data else 0,
            "assignments": [
                {"task": (a.get("task_title") or "")[:50], "owner": a.get("assigned_to")}
                for a in assignments[:5]
            ]
        }
    })

    # ── Step 6: Escalation Check
    t0 = datetime.now()
    es_agent = EscalationDecider()
    es_result = await es_agent.execute(
        EscalationInput(
            action_items=action_items,
            priority_scores=scores,
            current_date=datetime.now()
        ),
        ctx(6)
    )
    escalations = es_result.data.get("escalations", []) if es_result.data else []
    steps.append({
        "agent": "EscalationDecider",
        "success": es_result.success,
        "confidence": round(es_result.confidence * 100, 1),
        "duration_ms": int((datetime.now() - t0).total_seconds() * 1000),
        "output": {
            "total_escalations": len(escalations),
            "escalations": [
                {"reason": e.get("reason", ""), "item": (e.get("item_title") or "")[:50]}
                for e in escalations[:3]
            ]
        }
    })

    return {
        "execution_id": exec_id,
        "steps": steps,
        "summary": {
            "speakers": steps[0]["output"]["speakers"],
            "topics": steps[0]["output"]["topics"],
            "decisions": steps[1]["output"]["total"],
            "action_items": steps[2]["output"]["total"],
            "assigned": steps[2]["output"]["assigned"],
            "with_deadlines": steps[2]["output"]["with_deadlines"],
            "priority_counts": steps[3 if not ai_step else 4]["output"]["priority_counts"],
            "escalations": len(escalations),
            "top_action_items": steps[2]["output"]["items"],
            "top_assignments": steps[4 if not ai_step else 5]["output"]["assignments"],
            "ai_enrichment": {
                "enabled": ai_step is not None and ai_step.get("success", False),
                "ai_summary": ai_data.get("ai_summary", ""),
                "missed_decisions": ai_data.get("missed_decisions", []),
                "missed_action_items": ai_data.get("missed_action_items", []),
                "risks": ai_data.get("risks_identified", []),
                "sentiment": ai_data.get("sentiment_analysis", {}),
                "key_insights": ai_data.get("key_insights", []),
                "total_decisions_with_ai": len(decisions),
                "total_actions_with_ai": len(action_items),
            }
        }
    }


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default request logging

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {"ok": True})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))

            transcript = body.get("transcript", "")
            participants = body.get("participants", [])
            participant_roles = body.get("participant_roles", {})

            if not transcript.strip():
                self._send_json(400, {"error": "transcript is required"})
                return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                run_full_pipeline(transcript, participants, participant_roles)
            )
            loop.close()

            self._send_json(200, result)

        except Exception as e:
            self._send_json(500, {"error": str(e), "type": type(e).__name__})
