"""
Multimodal State Fusion, Evidence Separation & Adaptive Interaction (Phase 13).
Maintains structured interaction state, separates evidence types,
enforces observation freshness, and resolves natural voice + visual spatial references.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from app.agent.action import Action
from app.agent.context import ActionResult
from app.agent.visual import ComputerObservation, TargetingConfidence, TargetingResult, UIElement


class EvidenceType(str, Enum):
    """Explicitly separated categories of evidence in the agent lifecycle."""

    USER_INPUT = "USER_INPUT"
    PLANNER_INTERPRETATION = "PLANNER_INTERPRETATION"
    ATTEMPTED_ACTION = "ATTEMPTED_ACTION"
    OBSERVED_STATE = "OBSERVED_STATE"
    VERIFIED_RESULT = "VERIFIED_RESULT"
    UNRESOLVED_UNCERTAINTY = "UNRESOLVED_UNCERTAINTY"


@dataclass
class EvidenceItem:
    """Discrete evidence unit with explicit provenance and confidence."""

    evidence_type: EvidenceType
    data: Dict[str, Any]
    confidence: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UnifiedInteractionState:
    """
    Unified multimodal interaction state tracking all relevant context for a run.
    Guarantees strict run isolation and prevents conflating inference with fact.
    """

    run_id: str
    current_goal: Optional[str] = None
    current_subgoal: Optional[str] = None
    latest_user_intent: Optional[str] = None
    execution_state: str = "IDLE"
    latest_valid_observation: Optional[ComputerObservation] = None
    observation_freshness: float = 0.0
    last_action: Optional[Action] = None
    action_result: Optional[ActionResult] = None
    verification_result: Optional[str] = None
    recovery_count: int = 0
    interruption_state: bool = False
    confirmation_state: Optional[str] = None
    confidence_metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_history: List[EvidenceItem] = field(default_factory=list)
    max_evidence_history: int = 20

    def add_evidence(
        self,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
        confidence: float = 1.0,
        source: str = "",
    ) -> EvidenceItem:
        """Record bounded evidence item."""
        item = EvidenceItem(
            evidence_type=evidence_type,
            data=data,
            confidence=confidence,
            source=source,
        )
        self.evidence_history.append(item)
        if len(self.evidence_history) > self.max_evidence_history:
            self.evidence_history.pop(0)
        return item

    def update_observation(self, obs: ComputerObservation) -> None:
        """Update active observation and refresh age."""
        if obs.run_id and obs.run_id != self.run_id:
            # Reject observation belonging to a different run
            return
        self.latest_valid_observation = obs
        self.observation_freshness = obs.age_seconds
        self.add_evidence(
            EvidenceType.OBSERVED_STATE,
            {"observation_id": obs.observation_id, "window": obs.window_title, "elements": len(obs.elements)},
            confidence=1.0,
            source="visual_sensor",
        )

    def invalidate_observation(self, reason: str = "State change or cancellation") -> None:
        """Invalidate the current visual observation."""
        self.latest_valid_observation = None
        self.observation_freshness = 999.0
        self.add_evidence(
            EvidenceType.UNRESOLVED_UNCERTAINTY,
            {"invalidated_reason": reason},
            confidence=0.0,
            source="invalidation",
        )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a privacy-safe operational summary of the interaction state."""
        return {
            "run_id": self.run_id,
            "goal": self.current_goal,
            "subgoal": self.current_subgoal,
            "execution_state": self.execution_state,
            "has_observation": self.latest_valid_observation is not None,
            "obs_age": round(self.observation_freshness, 2),
            "recovery_count": self.recovery_count,
            "interrupted": self.interruption_state,
            "confirmation": self.confirmation_state,
        }


class MultimodalReferenceResolver:
    """
    Resolves natural voice + visual spatial references against the latest computer observation.
    Examples:
      - 'click that' -> most salient or focused element
      - 'open the one on the right' -> element with maximum x-coordinate
      - 'the one on the left' -> element with minimum x-coordinate
      - 'select the second option' -> element at index 1 in top-down reading order
      - 'first option' -> element at index 0
      - 'the top one' -> minimum y-coordinate
      - 'the bottom one' -> maximum y-coordinate
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.multimodal_resolver")

    def resolve_reference(
        self,
        reference_phrase: str,
        observation: Optional[ComputerObservation],
        context_arguments: Optional[Dict[str, Any]] = None,
        max_age_seconds: float = 5.0,
    ) -> TargetingResult:
        """Resolve multimodal reference phrase to an actionable UIElement or clarify if ambiguous."""
        ref = reference_phrase.strip().lower()

        if not observation:
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason="No current visual observation available to resolve reference",
                clarification_question=f"I don't have a clear view of the screen. Could you describe what you'd like to click?",
            )

        if observation.is_stale(max_age_seconds=max_age_seconds):
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason=f"Observation is stale ({observation.age_seconds:.1f}s)",
                clarification_question="The screen may have changed. Should I refresh and check again?",
            )

        interactive_elements = [e for e in observation.elements if e.is_interactive]
        if not interactive_elements:
            interactive_elements = observation.elements

        if not interactive_elements:
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason="No elements found on screen",
                clarification_question="I couldn't identify any interactive elements on the screen. Could you clarify?",
            )

        # 1. Spatial: "on the right" / "right one"
        if "on the right" in ref or "right one" in ref:
            # Sort by center x descending
            sorted_x = sorted(interactive_elements, key=lambda e: e.center_coords[0], reverse=True)
            top_elem = sorted_x[0]
            # Check if there is a distinct candidate on the right
            if len(sorted_x) > 1 and (sorted_x[0].center_coords[0] - sorted_x[1].center_coords[0]) < 20:
                return TargetingResult(
                    confidence=TargetingConfidence.AMBIGUOUS,
                    score=0.7,
                    candidates=[(e, 0.7) for e in sorted_x[:2]],
                    reason="Multiple elements on the right with similar coordinates",
                    clarification_question=f"I see '{sorted_x[0].label}' and '{sorted_x[1].label}' on the right. Which one did you mean?",
                )
            return TargetingResult(
                confidence=TargetingConfidence.HIGH,
                score=0.9,
                target_element=top_elem,
                candidates=[(top_elem, 0.9)],
                reason=f"Resolved '{ref}' to rightmost element '{top_elem.label}'",
            )

        # 2. Spatial: "on the left" / "left one"
        if "on the left" in ref or "left one" in ref:
            sorted_x = sorted(interactive_elements, key=lambda e: e.center_coords[0])
            top_elem = sorted_x[0]
            if len(sorted_x) > 1 and (sorted_x[1].center_coords[0] - sorted_x[0].center_coords[0]) < 20:
                return TargetingResult(
                    confidence=TargetingConfidence.AMBIGUOUS,
                    score=0.7,
                    candidates=[(e, 0.7) for e in sorted_x[:2]],
                    reason="Multiple elements on the left with similar coordinates",
                    clarification_question=f"I see '{sorted_x[0].label}' and '{sorted_x[1].label}' on the left. Which one did you mean?",
                )
            return TargetingResult(
                confidence=TargetingConfidence.HIGH,
                score=0.9,
                target_element=top_elem,
                candidates=[(top_elem, 0.9)],
                reason=f"Resolved '{ref}' to leftmost element '{top_elem.label}'",
            )

        # 3. Positional order: "second option" / "second one"
        if "second option" in ref or "second one" in ref or "the second" in ref:
            # Sort in top-down reading order: y then x
            sorted_reading = sorted(interactive_elements, key=lambda e: (e.center_coords[1], e.center_coords[0]))
            if len(sorted_reading) >= 2:
                target = sorted_reading[1]
                return TargetingResult(
                    confidence=TargetingConfidence.HIGH,
                    score=0.9,
                    target_element=target,
                    candidates=[(target, 0.9)],
                    reason=f"Resolved '{ref}' to second element '{target.label}'",
                )
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason="Less than two elements available on screen",
                clarification_question="I only see one matching option on screen. Would you like me to select that?",
            )

        # 4. Positional order: "first option" / "first one"
        if "first option" in ref or "first one" in ref or "the first" in ref:
            sorted_reading = sorted(interactive_elements, key=lambda e: (e.center_coords[1], e.center_coords[0]))
            target = sorted_reading[0]
            return TargetingResult(
                confidence=TargetingConfidence.HIGH,
                score=0.9,
                target_element=target,
                candidates=[(target, 0.9)],
                reason=f"Resolved '{ref}' to first element '{target.label}'",
            )

        # 5. Generic deictic: "click that" / "select that" / "open this"
        if ref in ("click that", "select that", "that", "this", "open this", "close this"):
            # If target has a close button and ref says "close this"
            if "close" in ref:
                close_candidates = [e for e in interactive_elements if "close" in e.label.lower() or "cancel" in e.label.lower()]
                if close_candidates:
                    return TargetingResult(
                        confidence=TargetingConfidence.HIGH,
                        score=0.9,
                        target_element=close_candidates[0],
                        candidates=[(close_candidates[0], 0.9)],
                        reason=f"Resolved '{ref}' to close element '{close_candidates[0].label}'",
                    )

            # If exactly 1 interactive element exists, high confidence
            if len(interactive_elements) == 1:
                return TargetingResult(
                    confidence=TargetingConfidence.HIGH,
                    score=0.95,
                    target_element=interactive_elements[0],
                    candidates=[(interactive_elements[0], 0.95)],
                    reason=f"Single interactive element '{interactive_elements[0].label}' on screen",
                )

            # If multiple exist, asking clarification is mandatory to prevent accidental clicks
            return TargetingResult(
                confidence=TargetingConfidence.AMBIGUOUS,
                score=0.5,
                candidates=[(e, 0.5) for e in interactive_elements[:3]],
                reason=f"Multiple elements on screen for ambiguous reference '{ref}'",
                clarification_question="There are multiple items on screen. Which one would you like me to click?",
            )

        return TargetingResult(
            confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
            score=0.0,
            reason=f"Unrecognized reference phrase '{reference_phrase}'",
            clarification_question=f"Could you clarify what you mean by '{reference_phrase}'?",
        )
