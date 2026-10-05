"""
Unified Computer Observation Model & Visual UI Targeting (Phase 11).
Enables generic visual understanding, semantic/accessibility-preferred UI targeting,
freshness validation, ambiguity detection, and visual verification without hard-coded coordinates.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid


class TargetingConfidence(str, Enum):
    """Explicit confidence levels for visual UI targeting."""

    HIGH = "HIGH"
    SUFFICIENT = "SUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass
class UIElement:
    """
    Structured representation of an interactive or visible UI element.
    Derived from accessibility metadata, screen OCR, or visual bounding boxes.
    """

    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    element_type: str = "element"  # button, link, input, icon, menu_item, tab, etc.
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    center_coords: Tuple[int, int] = (0, 0)
    confidence: float = 1.0  # 0.0 to 1.0
    is_interactive: bool = True
    source: str = "accessibility"  # "accessibility", "ocr", "visual"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Automatically compute center coordinates from bounding box if not explicitly set
        if self.center_coords == (0, 0) and self.bounding_box != (0, 0, 0, 0):
            x, y, w, h = self.bounding_box
            self.center_coords = (x + w // 2, y + h // 2)


@dataclass
class ComputerObservation:
    """
    Unified computer observation model capturing current computer/screen state.
    Maintains bounded memory, privacy protections, and explicit freshness indicators.
    """

    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    window_title: Optional[str] = None
    active_application: Optional[str] = None
    elements: List[UIElement] = field(default_factory=list)
    recognized_text: List[Dict[str, Any]] = field(default_factory=list)  # [{"text": "...", "box": (x,y,w,h)}]
    screenshot_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Returns the age of this observation in seconds relative to current UTC time."""
        now = datetime.now(timezone.utc)
        return max(0.0, (now - self.timestamp).total_seconds())

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if observation exceeds the freshness threshold."""
        return self.age_seconds > max_age_seconds

    def summary(self) -> Dict[str, Any]:
        """
        Return a privacy-safe summary of the observation without exposing full screenshots
        or sensitive screen content.
        """
        return {
            "observation_id": self.observation_id,
            "run_id": self.run_id,
            "age_seconds": round(self.age_seconds, 2),
            "window_title": self.window_title,
            "active_application": self.active_application,
            "element_count": len(self.elements),
            "text_segment_count": len(self.recognized_text),
            "screenshot_hash": self.screenshot_hash,
        }


@dataclass
class TargetingResult:
    """Result of targeting resolution over an observation."""

    confidence: TargetingConfidence
    score: float
    target_element: Optional[UIElement] = None
    candidates: List[Tuple[UIElement, float]] = field(default_factory=list)
    reason: str = ""
    clarification_question: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        """True if confidence is high or sufficient and a valid target exists."""
        return (
            self.confidence in (TargetingConfidence.HIGH, TargetingConfidence.SUFFICIENT)
            and self.target_element is not None
        )


class UITargeter:
    """
    Generic UI target identification and candidate selection engine.
    Resolves targets using a hierarchical preference:
      1. Accessibility/semantic metadata (preferred)
      2. Visible screen text / OCR matches
      3. Visual appearance / coordinate fallback
    Enforces strict ambiguity detection and confidence thresholds.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.targeter")

    def resolve_target(
        self,
        target_query: str,
        observation: ComputerObservation,
        confidence_threshold: float = 0.7,
        ambiguity_threshold: float = 0.15,
        max_age_seconds: float = 5.0,
    ) -> TargetingResult:
        """
        Identify the target UI element matching the user's natural language query.
        Returns a TargetingResult with explicit confidence.
        """
        query_clean = target_query.strip().lower()
        if not query_clean:
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason="Target query is empty",
                clarification_question="What element would you like me to interact with?",
            )

        # Check observation freshness
        if observation.is_stale(max_age_seconds=max_age_seconds):
            self.logger.warning(
                "STALE OBSERVATION: Observation %s is %.2fs old (max allowed: %.1fs)",
                observation.observation_id,
                observation.age_seconds,
                max_age_seconds,
            )
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason=f"Observation is stale ({observation.age_seconds:.1f}s old)",
                clarification_question="The screen state has changed or is outdated. Should I refresh and check again?",
            )

        # Check multimodal spatial or relative references (Phase 13)
        spatial_triggers = (
            "on the right",
            "on the left",
            "second option",
            "first option",
            "click that",
            "right one",
            "left one",
            "close this",
            "the second",
            "the first",
        )
        if any(trigger in query_clean for trigger in spatial_triggers) or query_clean in ("that", "this"):
            from app.agent.multimodal import MultimodalReferenceResolver
            resolver = MultimodalReferenceResolver(logger=self.logger)
            return resolver.resolve_reference(
                reference_phrase=target_query,
                observation=observation,
                max_age_seconds=max_age_seconds,
            )

        scored_candidates: List[Tuple[UIElement, float]] = []

        # Helper to score an element
        for elem in observation.elements:
            score = self._compute_element_match(query_clean, elem)
            if score > 0.2:
                scored_candidates.append((elem, score))

        # Check OCR/recognized text segments if elements didn't yield matches
        if not scored_candidates and observation.recognized_text:
            for text_seg in observation.recognized_text:
                txt = text_seg.get("text", "").lower()
                box = text_seg.get("box", (0, 0, 0, 0))
                seg_conf = text_seg.get("confidence", 0.85)
                if query_clean in txt or any(w in txt for w in query_clean.split()):
                    synthetic_elem = UIElement(
                        label=text_seg.get("text", ""),
                        element_type="text",
                        bounding_box=box,
                        confidence=seg_conf,
                        source="ocr",
                    )
                    score = self._compute_element_match(query_clean, synthetic_elem)
                    if score > 0.2:
                        scored_candidates.append((synthetic_elem, score))

        # Sort candidates descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        if not scored_candidates:
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=0.0,
                reason=f"No UI elements found matching '{target_query}'",
                clarification_question=f"I couldn't find '{target_query}' on the screen. Could you clarify where it is?",
            )

        top_elem, top_score = scored_candidates[0]

        # Check for ambiguity among top candidates
        if len(scored_candidates) > 1:
            second_elem, second_score = scored_candidates[1]
            score_diff = top_score - second_score
            # If both candidates have meaningful scores and the difference is small
            if second_score >= 0.5 and score_diff < ambiguity_threshold:
                self.logger.info(
                    "TARGET AMBIGUITY DETECTED: '%s' vs '%s' (diff: %.3f < %.3f)",
                    top_elem.label,
                    second_elem.label,
                    score_diff,
                    ambiguity_threshold,
                )
                clarification = (
                    f"I found multiple items matching '{target_query}' "
                    f"('{top_elem.label}' and '{second_elem.label}'). Which one did you mean?"
                )
                return TargetingResult(
                    confidence=TargetingConfidence.AMBIGUOUS,
                    score=top_score,
                    target_element=None,
                    candidates=scored_candidates,
                    reason=f"Ambiguous match between '{top_elem.label}' and '{second_elem.label}'",
                    clarification_question=clarification,
                )

        if top_score < confidence_threshold:
            return TargetingResult(
                confidence=TargetingConfidence.INSUFFICIENT_INFORMATION,
                score=top_score,
                target_element=top_elem,
                candidates=scored_candidates,
                reason=f"Best match '{top_elem.label}' score {top_score:.2f} is below threshold {confidence_threshold:.2f}",
                clarification_question=f"Did you mean '{top_elem.label}' for '{target_query}'?",
            )

        conf_level = TargetingConfidence.HIGH if top_score >= 0.85 else TargetingConfidence.SUFFICIENT
        return TargetingResult(
            confidence=conf_level,
            score=top_score,
            target_element=top_elem,
            candidates=scored_candidates,
            reason=f"Matched '{top_elem.label}' via {top_elem.source} (score: {top_score:.2f})",
        )

    def _compute_element_match(self, query: str, elem: UIElement) -> float:
        """Compute matching score between query and element considering source preference."""
        label_clean = elem.label.strip().lower()
        if not label_clean:
            return 0.0

        # Base similarity score
        match_score = 0.0
        if query == label_clean:
            match_score = 1.0
        elif query in label_clean:
            # Fraction of length
            match_score = 0.85 + (len(query) / max(len(label_clean), 1)) * 0.1
        elif label_clean in query:
            match_score = 0.80
        else:
            # Word token overlap
            q_words = set(query.split())
            l_words = set(label_clean.split())
            common = q_words.intersection(l_words)
            if common:
                match_score = 0.5 + 0.3 * (len(common) / max(len(q_words), 1))

        if match_score == 0.0:
            return 0.0

        # Semantic/Accessibility source preference
        source_weight = 1.0
        if elem.source == "accessibility":
            source_weight = 1.0  # highest priority
        elif elem.source == "ocr":
            source_weight = 0.95
        elif elem.source == "visual":
            source_weight = 0.90

        # Element type alignment boost (e.g. user asks to "click" and element is interactive)
        type_boost = 1.05 if elem.is_interactive else 0.95

        final_score = min(1.0, match_score * source_weight * type_boost * elem.confidence)
        return final_score


class VisualVerifier:
    """
    Verifies meaningful UI state changes using computer observations.
    Detects whether an expected visual state change occurred or if a visual mismatch happened.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.visual_verifier")

    def verify_change(
        self,
        obs_before: Optional[ComputerObservation],
        obs_after: ComputerObservation,
        expected_change: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Verify that obs_after reflects the expected change relative to obs_before or expected specs.
        Returns:
            (is_verified, reason_message)
        """
        # 1. Expected element appeared
        if "expected_element_appeared" in expected_change:
            target_label = expected_change["expected_element_appeared"].lower()
            found = any(target_label in e.label.lower() for e in obs_after.elements)
            if found:
                return True, f"Expected element '{target_label}' appeared on screen"
            return False, f"Expected element '{target_label}' did not appear"

        # 2. Expected element disappeared
        if "expected_element_disappeared" in expected_change:
            target_label = expected_change["expected_element_disappeared"].lower()
            still_present = any(target_label in e.label.lower() for e in obs_after.elements)
            if not still_present:
                return True, f"Expected element '{target_label}' disappeared as expected"
            return False, f"Element '{target_label}' is still visible on screen"

        # 3. Expected window title
        if "expected_window_title" in expected_change:
            expected_title = expected_change["expected_window_title"].lower()
            actual_title = (obs_after.window_title or "").lower()
            if expected_title in actual_title:
                return True, f"Window title matches expected: '{obs_after.window_title}'"
            return False, f"Window title '{obs_after.window_title}' does not match expected '{expected_title}'"

        # 4. Expected text appeared
        if "expected_text" in expected_change:
            expected_txt = expected_change["expected_text"].lower()
            found_in_text = any(
                expected_txt in seg.get("text", "").lower() for seg in obs_after.recognized_text
            )
            found_in_elem = any(expected_txt in e.label.lower() for e in obs_after.elements)
            if found_in_text or found_in_elem:
                return True, f"Expected text '{expected_txt}' recognized on screen"
            return False, f"Expected text '{expected_txt}' not found on screen"

        # 5. Generic state change check (e.g. screenshot hash changed or active application changed)
        if obs_before is not None:
            if obs_before.screenshot_hash and obs_after.screenshot_hash:
                if obs_before.screenshot_hash != obs_after.screenshot_hash:
                    return True, "Visual state changed (screenshot hash difference)"
                return False, "Visual state unchanged (screenshot hash identical)"
            if obs_before.window_title != obs_after.window_title:
                return True, f"Window title changed from '{obs_before.window_title}' to '{obs_after.window_title}'"
            if len(obs_before.elements) != len(obs_after.elements):
                return True, "UI element layout changed"

        return True, "Visual verification passed"
