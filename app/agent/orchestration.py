"""
Safe High-Level Task Orchestration & Confirmation Gate (Phase 12).
Provides structured goal and sub-goal hierarchies, explicit state transitions,
dependency management, dynamic replanning, concise operational status reporting,
and strict run-isolated confirmation gates for sensitive actions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from app.agent.action import Action
from app.agent.context import ActionResult


class GoalStatus(str, Enum):
    """Explicit states for high-level tasks and sub-goals."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"


@dataclass
class SubGoal:
    """
    Represents a discrete, verifiable sub-goal within a high-level task.
    Requires explicit verification rather than passive execution.
    """

    description: str
    subgoal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.PENDING
    prerequisites: List[str] = field(default_factory=list)  # IDs of required sub-goals
    actions: List[Action] = field(default_factory=list)
    current_step: int = 0
    result: Optional[ActionResult] = None
    verification_state: Optional[str] = None
    is_consequential: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, result: Optional[ActionResult] = None, verification_state: str = "Verified") -> None:
        """Mark this sub-goal as completed based on verified outcome."""
        self.status = GoalStatus.COMPLETED
        self.result = result
        self.verification_state = verification_state

    def mark_failed(self, error: str) -> None:
        """Mark this sub-goal as failed."""
        self.status = GoalStatus.FAILED
        self.verification_state = f"Failed: {error}"


@dataclass
class TaskGoal:
    """
    Structured shallow goal hierarchy representing a high-level user objective.
    Maintains bounded sub-goals and dependency tracking.
    """

    description: str
    run_id: str
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.PENDING
    subgoals: List[SubGoal] = field(default_factory=list)
    max_subgoals: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def add_subgoal(
        self,
        description: str,
        prerequisites: Optional[List[str]] = None,
        actions: Optional[List[Action]] = None,
        is_consequential: bool = False,
        subgoal_id: Optional[str] = None,
    ) -> SubGoal:
        """Add a bounded sub-goal to the hierarchy."""
        if len(self.subgoals) >= self.max_subgoals:
            raise ValueError(f"Max sub-goals limit ({self.max_subgoals}) reached for goal {self.goal_id}")

        sg = SubGoal(
            description=description,
            subgoal_id=subgoal_id or str(uuid.uuid4()),
            prerequisites=prerequisites or [],
            actions=actions or [],
            is_consequential=is_consequential,
        )
        self.subgoals.append(sg)
        return sg

    def get_subgoal(self, subgoal_id: str) -> Optional[SubGoal]:
        """Find a sub-goal by its identifier."""
        for sg in self.subgoals:
            if sg.subgoal_id == subgoal_id:
                return sg
        return None

    def all_prerequisites_met(self, subgoal: SubGoal) -> bool:
        """Check if all prerequisite sub-goals have been successfully completed."""
        for prereq_id in subgoal.prerequisites:
            prereq = self.get_subgoal(prereq_id)
            if not prereq or prereq.status != GoalStatus.COMPLETED:
                return False
        return True

    def get_next_executable_subgoal(self) -> Optional[SubGoal]:
        """
        Return the next pending sub-goal whose prerequisites are satisfied.
        Prevents executing dependent sub-goals prematurely.
        """
        for sg in self.subgoals:
            if sg.status == GoalStatus.PENDING and self.all_prerequisites_met(sg):
                return sg
        return None

    def get_active_subgoal(self) -> Optional[SubGoal]:
        """Return the currently active sub-goal."""
        for sg in self.subgoals:
            if sg.status == GoalStatus.ACTIVE:
                return sg
        return None

    def is_completed(self) -> bool:
        """Check if all non-skipped sub-goals have completed successfully."""
        if not self.subgoals:
            return False
        return all(sg.status in (GoalStatus.COMPLETED, GoalStatus.SKIPPED) for sg in self.subgoals)

    def mark_completed(self) -> None:
        """Mark root task goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)

    def get_status_summary(self) -> str:
        """
        Generate a concise, user-facing operational summary.
        Strictly excludes internal reasoning or chain-of-thought.
        """
        active_sg = self.get_active_subgoal()
        completed_count = sum(1 for sg in self.subgoals if sg.status == GoalStatus.COMPLETED)
        total_count = len(self.subgoals)

        if self.status == GoalStatus.COMPLETED:
            return f"I've completed all {total_count} steps for your task."
        if self.status == GoalStatus.FAILED:
            return "I encountered an issue and couldn't complete the task."
        if active_sg:
            return f"I'm currently working on: {active_sg.description} ({completed_count}/{total_count} completed)."
        return f"I'm preparing to work on: {self.description}."


class ConfirmationStatus(str, Enum):
    """States for consequential action confirmations."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass
class ConfirmationRequest:
    """
    Represents an explicit confirmation gate for a sensitive action.
    Strictly isolated to a specific active run ID to prevent stale run reuse.
    """

    run_id: str
    action: Action
    prompt: str
    confirmation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subgoal_id: Optional[str] = None
    timeout_seconds: float = 30.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    resolved_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if the confirmation request exceeded its timeout."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds() > self.timeout_seconds

    def is_valid(self, active_run_id: Optional[str]) -> bool:
        """
        Validate whether this confirmation is still active and matches the current active run.
        Guarantees that an old confirmation from Run A cannot authorize anything in Run B.
        """
        if self.status != ConfirmationStatus.PENDING:
            return False
        if self.is_expired:
            self.status = ConfirmationStatus.EXPIRED
            return False
        if active_run_id is None or self.run_id != active_run_id:
            return False
        return True


class ConfirmationManager:
    """
    Manages safety confirmation gates for consequential actions.
    Ensures actions like deleting data, sending messages, or purchasing
    require explicit user authorization before execution.
    """

    CONSEQUENTIAL_ACTIONS: Set[str] = {
        "DELETE",
        "delete",
        "PURCHASE",
        "purchase",
        "SEND_MESSAGE",
        "send_message",
        "SYSTEM_SETTING",
        "system_setting",
        "FORMAT_DRIVE",
        "format_drive",
        "EMPTY_TRASH",
        "empty_trash",
    }

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.confirmation")
        self._pending_confirmations: Dict[str, ConfirmationRequest] = {}

    def is_consequential(self, action: Action) -> bool:
        """Check if an action is deemed sensitive or consequential."""
        if action.action_type in self.CONSEQUENTIAL_ACTIONS:
            return True
        if isinstance(action.arguments, dict):
            if action.arguments.get("requires_confirmation", False):
                return True
            if action.arguments.get("consequential", False):
                return True
            if action.arguments.get("is_sensitive", False):
                return True
        return False

    def request_confirmation(
        self,
        run_id: str,
        action: Action,
        prompt: Optional[str] = None,
        subgoal_id: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> ConfirmationRequest:
        """Create and register a pending confirmation gate."""
        default_prompt = f"Please confirm if you want me to execute '{action.action_type}'."
        req = ConfirmationRequest(
            run_id=run_id,
            action=action,
            prompt=prompt or default_prompt,
            subgoal_id=subgoal_id,
            timeout_seconds=timeout_seconds,
        )
        self._pending_confirmations[req.confirmation_id] = req
        self.logger.info(
            "CONFIRMATION REQUESTED: [ID: %s] Run: %s, Action: %s",
            req.confirmation_id,
            run_id,
            action.action_type,
        )
        return req

    def resolve_confirmation(
        self,
        confirmation_id: str,
        approved: bool,
        caller_run_id: Optional[str],
    ) -> Tuple[bool, str]:
        """
        Resolve a pending confirmation request.
        Validates caller run ID to reject stale authorizations.
        """
        req = self._pending_confirmations.get(confirmation_id)
        if not req:
            return False, "Confirmation request not found"

        if not req.is_valid(active_run_id=caller_run_id):
            if req.run_id != caller_run_id:
                reason = f"Confirmation belongs to run {req.run_id}, not current run {caller_run_id}"
                self.logger.warning("STALE CONFIRMATION REJECTED: %s", reason)
                req.status = ConfirmationStatus.CANCELLED
                return False, reason
            if req.is_expired:
                req.status = ConfirmationStatus.EXPIRED
                return False, "Confirmation request expired"
            return False, f"Confirmation invalid (status: {req.status})"

        req.resolved_at = datetime.now(timezone.utc)
        if approved:
            req.status = ConfirmationStatus.APPROVED
            self.logger.info("CONFIRMATION APPROVED: [ID: %s] for Run: %s", confirmation_id, caller_run_id)
            return True, "Confirmation approved"
        else:
            req.status = ConfirmationStatus.REJECTED
            self.logger.info("CONFIRMATION REJECTED: [ID: %s] for Run: %s", confirmation_id, caller_run_id)
            return True, "Confirmation rejected by user"

    def invalidate_run(self, run_id: str) -> None:
        """Cancel and purge all pending confirmations for a given run ID."""
        for cid, req in list(self._pending_confirmations.items()):
            if req.run_id == run_id and req.status == ConfirmationStatus.PENDING:
                req.status = ConfirmationStatus.CANCELLED
                req.resolved_at = datetime.now(timezone.utc)
                self.logger.info("CONFIRMATION CANCELLED: Stale confirmation %s invalidated for run %s", cid, run_id)

    def get_pending_request(self, confirmation_id: str) -> Optional[ConfirmationRequest]:
        """Retrieve a confirmation request by ID."""
        return self._pending_confirmations.get(confirmation_id)


class GoalOrchestrator:
    """
    Coordinates multi-step sub-goals, dependency resolution, bounded planning,
    and dynamic replanning.
    """

    def __init__(
        self,
        confirmation_manager: Optional[ConfirmationManager] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.confirmation_manager = confirmation_manager or ConfirmationManager()
        self.logger = logger or logging.getLogger("voice_agent.orchestrator")
        self._active_goals: Dict[str, TaskGoal] = {}

    def register_goal(self, goal: TaskGoal) -> None:
        """Register a high-level task goal."""
        self._active_goals[goal.run_id] = goal

    def get_goal(self, run_id: str) -> Optional[TaskGoal]:
        """Get active goal for a run ID."""
        return self._active_goals.get(run_id)

    def advance_subgoal_completion(
        self,
        goal: TaskGoal,
        subgoal: SubGoal,
        result: ActionResult,
    ) -> bool:
        """
        Mark sub-goal complete upon verified execution and advance goal.
        Returns True if root goal is now completely finished.
        """
        if result.is_success:
            subgoal.mark_completed(result, verification_state="Verified")
            self.logger.info(
                "SUBGOAL COMPLETED: '%s' (ID: %s) in Goal %s",
                subgoal.description,
                subgoal.subgoal_id,
                goal.goal_id,
            )
            if goal.is_completed():
                goal.mark_completed()
                self.logger.info("GOAL FULLY COMPLETED: '%s' (Goal ID: %s)", goal.description, goal.goal_id)
                return True
        else:
            subgoal.mark_failed(result.error or "Action failed")
            self.logger.warning(
                "SUBGOAL FAILED: '%s' (ID: %s): %s",
                subgoal.description,
                subgoal.subgoal_id,
                result.error,
            )
        return False

    def replan_subgoal(
        self,
        goal: TaskGoal,
        subgoal: SubGoal,
        new_actions: List[Action],
        reason: str = "Dynamic replan due to state change",
    ) -> None:
        """
        Dynamically update actions for a sub-goal when state changes unexpectedly.
        Never blindly replays the stale action sequence.
        """
        self.logger.info(
            "DYNAMIC REPLANNING: Updating sub-goal '%s' with %d new actions (Reason: %s)",
            subgoal.description,
            len(new_actions),
            reason,
        )
        subgoal.actions = new_actions
        subgoal.current_step = 0
        subgoal.status = GoalStatus.ACTIVE
        subgoal.metadata["replanned_reason"] = reason
