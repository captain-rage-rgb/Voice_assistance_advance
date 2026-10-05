"""
Agent package exports (Phases 1-10).
"""

from app.agent.action import Action, ActionStatus
from app.agent.action_queue import ActionQueue
from app.agent.cancellation import CancellationToken
from app.agent.context import ActionResult, ExecutionOutcome, Observation, TaskContext, TaskMilestone
from app.agent.events import Event, EventBus, EventHandler, EventType
from app.agent.executor import ActionExecutor, MockExecutor
from app.agent.health import HealthMonitor, SubsystemHealth, SubsystemStatus
from app.agent.orchestration import (
    ConfirmationManager,
    ConfirmationRequest,
    ConfirmationStatus,
    GoalOrchestrator,
    GoalStatus,
    SubGoal,
    TaskGoal,
)
from app.agent.overlay import FloatingOverlay, OverlayDisplayState
from app.agent.run_manager import RunLifecycleState, RunManager
from app.agent.runtime import AgentRuntime
from app.agent.validator import ActionValidator
from app.agent.verifier import Verifier
from app.agent.visual import (
    ComputerObservation,
    TargetingConfidence,
    TargetingResult,
    UIElement,
    UITargeter,
    VisualVerifier,
)

__all__ = [
    "Action",
    "ActionStatus",
    "ActionQueue",
    "ActionResult",
    "ExecutionOutcome",
    "Observation",
    "TaskContext",
    "TaskMilestone",
    "ActionValidator",
    "Verifier",
    "CancellationToken",
    "Event",
    "EventBus",
    "EventHandler",
    "EventType",
    "ActionExecutor",
    "MockExecutor",
    "FloatingOverlay",
    "OverlayDisplayState",
    "AgentRuntime",
    "RunManager",
    "RunLifecycleState",
    "HealthMonitor",
    "SubsystemHealth",
    "SubsystemStatus",
    # Phase 11 Visual
    "UIElement",
    "ComputerObservation",
    "UITargeter",
    "TargetingConfidence",
    "TargetingResult",
    "VisualVerifier",
    # Phase 12 Orchestration
    "GoalStatus",
    "SubGoal",
    "TaskGoal",
    "GoalOrchestrator",
    "ConfirmationRequest",
    "ConfirmationStatus",
    "ConfirmationManager",
]
