"""
Adaptive Recovery, Resilience & Conversational Continuity (Phase 14).
Classifies failures into normalized categories, manages a structured, bounded recovery hierarchy,
refreshes evidence before retrying, and prevents infinite loops or zombie executions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.agent.action import Action
from app.agent.context import ActionResult, TaskContext


class FailureCategory(str, Enum):
    """Normalized categories of failures across the agent execution lifecycle."""

    INVALID_INTENT = "INVALID_INTENT"
    AMBIGUOUS_INTENT = "AMBIGUOUS_INTENT"
    UNAVAILABLE_TARGET = "UNAVAILABLE_TARGET"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    ACTION_FAILURE = "ACTION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    UNEXPECTED_UI_CHANGE = "UNEXPECTED_UI_CHANGE"
    TIMEOUT = "TIMEOUT"
    PERMISSION_ISSUE = "PERMISSION_ISSUE"
    CANCELLATION = "CANCELLATION"
    SUBSYSTEM_FAILURE = "SUBSYSTEM_FAILURE"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"


class FinalRunState(str, Enum):
    """Explicit, verifiable final termination states for any agent run."""

    COMPLETED = "completed"
    CANCELLED = "canceled"
    FAILED = "failed"
    WAITING_FOR_USER = "waiting_for_user"
    TIMED_OUT = "timed_out"


class FailureClassifier:
    """Classifies runtime errors and execution failures into normalized categories."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.failure_classifier")

    def classify(self, error: Any, action: Optional[Action] = None) -> FailureCategory:
        """Categorize an error string or exception into a normalized FailureCategory."""
        msg = str(error).lower()

        if any(w in msg for w in ("cancel", "interrupted", "barge-in", "stop", "stopped")):
            return FailureCategory.CANCELLATION

        if any(w in msg for w in ("stale", "expired", "outdated observation")):
            return FailureCategory.STALE_OBSERVATION

        if any(w in msg for w in ("ambiguous", "multiple items", "multiple candidates")):
            return FailureCategory.AMBIGUOUS_INTENT

        if any(w in msg for w in ("not found", "no matching", "unavailable target", "target not resolved")):
            return FailureCategory.UNAVAILABLE_TARGET

        if any(w in msg for w in ("timed out", "timeout")):
            return FailureCategory.TIMEOUT

        if any(w in msg for w in ("visual mismatch", "ui did not transition", "unexpected state")):
            return FailureCategory.UNEXPECTED_UI_CHANGE

        if any(w in msg for w in ("verification failed", "verification")):
            return FailureCategory.VERIFICATION_FAILURE

        if any(w in msg for w in ("permission", "access denied", "forbidden", "unauthorized")):
            return FailureCategory.PERMISSION_ISSUE

        if any(w in msg for w in ("unsupported", "unknown action", "invalid action")):
            return FailureCategory.UNSUPPORTED_OPERATION

        if any(w in msg for w in ("subsystem", "audio error", "stt error", "driver error")):
            return FailureCategory.SUBSYSTEM_FAILURE

        return FailureCategory.ACTION_FAILURE


@dataclass
class RecoveryPlan:
    """Prescribes bounded next recovery step."""

    category: FailureCategory
    recommended_action: Optional[Action] = None
    requires_fresh_observation: bool = True
    should_replan: bool = False
    clarification_question: Optional[str] = None
    abort_task: bool = False
    reason: str = ""


class AdaptiveRecoveryManager:
    """
    Orchestrates bounded, hierarchical self-correction.
    Enforces the recovery hierarchy:
      1. Verify whether action actually succeeded
      2. Invalidate stale state & refresh observation
      3. Retry semantic target if still valid
      4. Alternative generic interaction method
      5. Replan current sub-goal
      6. Ask the user
      7. Stop safely
    """

    def __init__(
        self,
        max_action_retries: int = 2,
        max_observation_retries: int = 3,
        max_replanning_attempts: int = 3,
        max_total_recovery_duration: float = 15.0,
        classifier: Optional[FailureClassifier] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.max_action_retries = max_action_retries
        self.max_observation_retries = max_observation_retries
        self.max_replanning_attempts = max_replanning_attempts
        self.max_total_recovery_duration = max_total_recovery_duration
        self.classifier = classifier or FailureClassifier()
        self.logger = logger or logging.getLogger("voice_agent.recovery_manager")

    def plan_recovery(
        self,
        action: Action,
        error: str,
        attempt_num: int,
        context: TaskContext,
    ) -> RecoveryPlan:
        """
        Determine justified recovery step based on failure category and bounded limits.
        Never retries blindly without fresh evidence.
        """
        category = self.classifier.classify(error, action=action)

        # 1. Immediate termination conditions
        if category == FailureCategory.CANCELLATION:
            return RecoveryPlan(
                category=category,
                abort_task=True,
                reason="Task was cancelled by user or unified cancellation",
            )

        if category == FailureCategory.PERMISSION_ISSUE:
            return RecoveryPlan(
                category=category,
                abort_task=True,
                clarification_question="I don't have permission to perform this action. Should I stop?",
                reason="Permission denied",
            )

        if category == FailureCategory.AMBIGUOUS_INTENT:
            return RecoveryPlan(
                category=category,
                clarification_question="I found multiple possible targets. Could you specify which one to choose?",
                abort_task=False,
                reason="Ambiguous intent/target requires user disambiguation",
            )

        # Check bounded retry limits
        if attempt_num > self.max_action_retries:
            self.logger.warning("RECOVERY EXHAUSTED: Exceeded %d retries for %s", self.max_action_retries, action.action_type)
            return RecoveryPlan(
                category=category,
                abort_task=True,
                reason=f"Recovery exhausted after {attempt_num} attempts: {error}",
            )

        # 2. Stale observation or unexpected UI change -> Invalidate and re-observe
        if category in (FailureCategory.STALE_OBSERVATION, FailureCategory.UNEXPECTED_UI_CHANGE):
            return RecoveryPlan(
                category=category,
                recommended_action=Action("observe_state", {}, run_id=action.run_id),
                requires_fresh_observation=True,
                should_replan=True,
                reason="Observation was stale or UI changed; refreshing state before replanning",
            )

        # 3. Target unavailable -> Replan or alternate action
        if category == FailureCategory.UNAVAILABLE_TARGET:
            return RecoveryPlan(
                category=category,
                recommended_action=Action("observe_state", {}, run_id=action.run_id),
                requires_fresh_observation=True,
                should_replan=True,
                reason="Target unavailable on screen; observing before replanning",
            )

        # 4. Standard action or verification failure -> Attempt alternative or bounded retry
        retry_args = dict(action.arguments)
        retry_args["recovery_attempt"] = attempt_num
        if not retry_args.get("persist_verification_failure"):
            retry_args.pop("simulate_verification_failure", None)

        return RecoveryPlan(
            category=category,
            recommended_action=None,  # Delegate to Verifier.create_recovery_action
            requires_fresh_observation=True,
            reason=f"Retrying action with fresh evidence (attempt {attempt_num})",
        )
