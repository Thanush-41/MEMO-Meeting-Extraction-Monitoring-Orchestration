# MEMO — Meeting Extraction, Monitoring & Orchestration

## Overview

A multi-agent system that takes ownership of complex, multi-step enterprise processes with built-in failure detection, self-correction, and complete auditability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ Workflow Engine │  │ State Manager    │  │ SLA Monitor                 │ │
│  └─────────────────┘  └──────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐  ┌───────────────────────┐  ┌───────────────────────────┐
│ EXTRACTION AGENTS │  │ DECISION AGENTS       │  │ ACTION AGENTS             │
│ ─────────────────│  │ ─────────────────────  │  │ ─────────────────────     │
│ • TranscriptAgent │  │ • TaskPrioritizer     │  │ • TaskCreator             │
│ • DecisionParser  │  │ • OwnerAssigner       │  │ • NotificationAgent       │
│ • ActionItemAgent │  │ • EscalationDecider   │  │ • IntegrationAgent        │
└───────────────────┘  └───────────────────────┘  └───────────────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERIFICATION LAYER                                 │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ Completion Tracker  │  │ Quality Validator  │  │ Anomaly Detector    │   │
│  └─────────────────────┘  └────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AUDIT & RECOVERY                                   │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ Decision Logger     │  │ Error Recovery     │  │ Human Escalation    │   │
│  └─────────────────────┘  └────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Deep Autonomy
- Multi-step workflows complete without human intervention
- Agents collaborate to handle complex processes
- Intelligent task routing and assignment

### 2. Error Recovery
- Automatic retry with exponential backoff
- Alternative strategy selection on failure
- Graceful degradation when recovery fails
- Circuit breaker pattern for external services

### 3. Full Auditability
- Every decision logged with reasoning
- Complete execution trace
- Confidence scores for all outputs
- Reversible actions with rollback support

### 4. Workflow Health Monitoring
- Real-time SLA tracking
- Bottleneck prediction
- Automatic escalation
- Process drift detection

## Use Cases Implemented

### Meeting Intelligence System
Automatically processes meeting recordings to:
1. Extract key decisions made
2. Identify action items with deadlines
3. Assign owners based on context and availability
4. Create tasks in project management systems
5. Track completion and escalate stalls
6. Generate meeting summaries

## Project Structure

```
agent-et/
├── src/
│   ├── core/              # Core agent framework
│   │   ├── base_agent.py  # Base agent class
│   │   ├── message.py     # Inter-agent messaging
│   │   ├── state.py       # State management
│   │   └── audit.py       # Audit trail system
│   ├── agents/            # Specialized agents
│   │   ├── extraction/    # Data extraction agents
│   │   ├── decision/      # Decision-making agents
│   │   ├── action/        # Action execution agents
│   │   └── verification/  # Verification agents
│   ├── orchestration/     # Workflow orchestration
│   │   ├── engine.py      # Workflow engine
│   │   ├── scheduler.py   # Task scheduler
│   │   └── monitor.py     # Health monitor
│   ├── recovery/          # Error recovery
│   │   ├── strategies.py  # Recovery strategies
│   │   └── circuit.py     # Circuit breaker
│   └── integrations/      # External integrations
├── workflows/             # Workflow definitions
├── tests/                 # Test suite
└── demo/                  # Demo application
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.orchestration.engine import WorkflowEngine
from src.workflows.meeting_intelligence import MeetingIntelligenceWorkflow

# Initialize the workflow engine
engine = WorkflowEngine()

# Load and run a workflow
workflow = MeetingIntelligenceWorkflow()
result = await engine.execute(workflow, {
    "meeting_transcript": "path/to/transcript.txt",
    "participants": ["alice@company.com", "bob@company.com"]
})

# Access audit trail
audit_log = engine.get_audit_trail(result.execution_id)
```

## Configuration

See `config/` directory for configuration options including:
- Agent behavior parameters
- SLA thresholds
- Integration credentials
- Escalation rules

## License

MIT License
#   M E M O - M e e t i n g - E x t r a c t i o n - M o n i t o r i n g - O r c h e s t r a t i o n  
 