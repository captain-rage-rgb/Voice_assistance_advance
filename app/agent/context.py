"""
Lightweight Execution Context for active agent runs (Phases 7-10).
Maintains structured tracking of task objectives, completed actions, milestones,
observations, conversational references, and recovery state without unbounded memory growth.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from app.agent.action import Action


class ExecutionOutcome(str, Enum):
    """Structured outcomes for an executed action."""

    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    VERIFIED = "VERIFIED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"
    STALE = "STALE"


@dataclass
class Observation:
    """Represents a lightweight computer state observation between steps."""

    observation_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActionResult:
    """Structured outcome report for an executed computer action."""

    action_id: str
    action_type: str
    outcome: ExecutionOutcome
    action_arguments: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    observation_before: Optional[Observation] = None
    observation_after: Optional[Observation] = None
    error: Optional[str] = None
    recovery_attempts: int = 0
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_success(self) -> bool:
        """True if action successfully finished and passed verification."""
        return self.outcome in {ExecutionOutcome.COMPLETED, ExecutionOutcome.VERIFIED, ExecutionOutcome.RECOVERED}


@dataclass
class TaskMilestone:
    """Represents a distinct completed milestone within a long-running task."""

    name: str
    milestone_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    completed: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    """
    Lightweight, bounded task execution context for an active agent run.
    Ensures the agent tracks what has happened across multi-step commands
    without repeating completed steps or leaking state between runs.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_intent: str = ""
    current_objective: str = ""
    current_step_index: int = 0
    max_history_entries: int = 20

    # Structured history of completed steps and milestones
    completed_actions: List[ActionResult] = field(default_factory=list)
    milestones: List[TaskMilestone] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    recent_visual_observations: List[Any] = field(default_factory=list)
    pending_work: List[str] = field(default_factory=list)
    recent_failures: List[str] = field(default_factory=list)

    # Conversational continuity & reference resolution
    reference_context: Dict[str, Any] = field(default_factory=dict)

    # High-level goal orchestration (Phase 12)
    active_goal: Optional[Any] = None

    # State tracking
    is_paused: bool = False
    resumed_from_milestone: Optional[str] = None
    recovery_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def latest_visual_observation(self) -> Optional[Any]:
        """Return the most recent visual observation, if any."""
        return self.recent_visual_observations[-1] if self.recent_visual_observations else None

    def record_visual_observation(self, observation: Any, max_entries: int = 5) -> None:
        """
        Record a visual observation, strictly bounding memory and discarding obsolete ones.
        Screenshots are never persisted indefinitely.
        """
        self.recent_visual_observations.append(observation)
        if len(self.recent_visual_observations) > max_entries:
            self.recent_visual_observations.pop(0)

    def record_completed_action(self, result: ActionResult) -> None:
        """Record completed action, update reference context, and prune history."""
        self.completed_actions.append(result)
        if len(self.completed_actions) > self.max_history_entries:
            self.completed_actions.pop(0)
        self.current_step_index += 1

        # Automatically update conversational references
        if result.is_success:
            self.reference_context["last_action_type"] = result.action_type
            self.reference_context["last_action_arguments"] = dict(result.action_arguments)
            if "application" in result.action_arguments:
                self.reference_context["last_application"] = result.action_arguments["application"]
            if "query" in result.action_arguments:
                self.reference_context["last_query"] = result.action_arguments["query"]
            if "destination" in result.action_arguments or "url" in result.action_arguments:
                self.reference_context["last_destination"] = result.action_arguments.get("destination") or result.action_arguments.get("url")

    def record_milestone(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskMilestone:
        """Record a completed milestone, keeping milestone history bounded."""
        milestone = TaskMilestone(
            name=name,
            description=description,
            metadata=metadata or {},
        )
        self.milestones.append(milestone)
        if len(self.milestones) > self.max_history_entries:
            self.milestones.pop(0)
        return milestone

    def record_observation(self, observation: Observation) -> None:
        """Record an observation, keeping history bounded."""
        self.observations.append(observation)
        if len(self.observations) > self.max_history_entries:
            self.observations.pop(0)

    def record_failure(self, failure_reason: str) -> None:
        """Record a recent failure for diagnostic and recovery tracking."""
        self.recent_failures.append(failure_reason)
        if len(self.recent_failures) > self.max_history_entries:
            self.recent_failures.pop(0)

    def is_action_already_completed(self, action_type: str, arguments: Optional[Dict[str, Any]] = None) -> bool:
        """Check if an equivalent action was already successfully executed in this run."""
        args = arguments or {}
        for past in self.completed_actions:
            if past.is_success and past.action_type == action_type:
                # If exact arguments match
                if past.action_arguments == args:
                    return True
                # Or if key application/target arguments match
                key_fields = ("application", "destination", "url", "query", "text")
                matches = True
                found_key = False
                for k in key_fields:
                    if k in args or k in past.action_arguments:
                        found_key = True
                        if args.get(k) != past.action_arguments.get(k):
                            matches = False
                            break
                if found_key and matches:
                    return True
        return False

    def pause(self) -> None:
        """Pause task execution without terminating task context."""
        self.is_paused = True

    def resume(self) -> Optional[str]:
        """Resume task execution from the latest valid milestone or step."""
        self.is_paused = False
        if self.milestones:
            self.resumed_from_milestone = self.milestones[-1].name
        return self.resumed_from_milestone

    def resolve_reference(self, reference_text: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an ambiguous or anaphoric reference ('it', 'that', 'the first one',
        'the cheapest one', 'do that again') using bounded conversation context.
        """
        ref_lower = reference_text.lower().strip()

        # Repeat previous action: "do that again", "repeat", "again"
        if ref_lower in ("do that again", "repeat", "again", "repeat that"):
            if "last_action_type" in self.reference_context:
                return {
                    "action_type": self.reference_context["last_action_type"],
                    "arguments": dict(self.reference_context.get("last_action_arguments", {})),
                }

        # Targeting previous application: "it", "that", "the app", "the browser"
        if ref_lower in ("it", "that", "the browser", "the app", "the document"):
            if "last_application" in self.reference_context:
                return {"application": self.reference_context["last_application"]}

        # Specific item selectors: "the cheapest one", "the first one", "the first result"
        if any(w in ref_lower for w in ("cheapest", "first", "previous result", "top result")):
            return {
                "target_selector": ref_lower,
                "context_query": self.reference_context.get("last_query", ""),
                "parent_application": self.reference_context.get("last_application", ""),
            }

        return None
