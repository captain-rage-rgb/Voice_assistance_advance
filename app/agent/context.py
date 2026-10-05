"""
Lightweight Execution Context for active agent runs (Phase 7).
Maintains structured tracking of task objective, completed actions, observations,
and recovery state without unbounded memory growth.
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

    # Structured history of completed steps
    completed_actions: List[ActionResult] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    recovery_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def record_completed_action(self, result: ActionResult) -> None:
        """Record completed action, pruning older entries if history exceeds maximum."""
        self.completed_actions.append(result)
        if len(self.completed_actions) > self.max_history_entries:
            self.completed_actions.pop(0)
        self.current_step_index += 1

    def record_observation(self, observation: Observation) -> None:
        """Record an observation, keeping history bounded."""
        self.observations.append(observation)
        if len(self.observations) > self.max_history_entries:
            self.observations.pop(0)

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

