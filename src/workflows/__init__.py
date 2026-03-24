"""
Workflows Package.

Contains workflow definitions for various enterprise processes.
"""

from .meeting_intelligence import (
    create_meeting_intelligence_workflow,
    MeetingIntelligenceConfig,
    MeetingIntelligenceResult,
    aggregate_workflow_results,
)

__all__ = [
    "create_meeting_intelligence_workflow",
    "MeetingIntelligenceConfig",
    "MeetingIntelligenceResult",
    "aggregate_workflow_results",
]
