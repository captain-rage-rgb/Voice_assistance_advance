"""
Incremental Intent Detector analyzing streaming transcripts and extracting actionable intents.
Supports segmentation across natural boundaries ('and', 'then', 'next', 'after that'),
distinguishes complete vs incomplete intents, handles revisions, and suppresses duplicates.
"""

from dataclasses import dataclass
import logging
import re
from typing import List, Optional, Set
from app.intelligence.intent import Intent, IntentType


@dataclass
class _SegmentCandidate:
    """Internal candidate representation of a parsed segment."""

    intent_type: IntentType
    parameters: dict
    source_text: str
    confidence: float
    is_complete: bool


class IncrementalIntentDetector:
    """
    Analyzes partial and final streaming speech transcripts incrementally.
    Produces structured, actionable Intent objects as soon as each sub-intent is complete.
    """

    # Common command boundary connectors
    BOUNDARY_PATTERN = re.compile(
        r"\b(?:and then|after that|and|then|next)\b", re.IGNORECASE
    )

    # Incomplete trailing words indicating the user hasn't finished the clause
    INCOMPLETE_TRAILERS = {
        "and", "then", "next", "after", "that", "the", "a", "an", "to", "for", "in", "on", "of", "with"
    }

    # Known canonical application aliases
    KNOWN_APPS = {
        "chrome": "Chrome",
        "google chrome": "Chrome",
        "brave": "Brave",
        "brave browser": "Brave",
        "firefox": "Firefox",
        "mozilla firefox": "Firefox",
        "edge": "Edge",
        "microsoft edge": "Edge",
        "vs code": "VS Code",
        "vscode": "VS Code",
        "visual studio code": "VS Code",
        "powerpoint": "PowerPoint",
        "power point": "PowerPoint",
        "spotify": "Spotify",
        "file explorer": "File Explorer",
        "explorer": "File Explorer",
        "notepad": "Notepad",
        "calculator": "Calculator",
        "terminal": "Terminal",
    }

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("voice_agent.intelligence.detector")
        # Track signatures of intents already emitted during the current utterance/session
        self._emitted_signatures: Set[str] = set()
        self._last_processed_text: str = ""

    def reset_context(self) -> None:
        """Reset short-term utterance context and emitted signatures."""
        self._emitted_signatures.clear()
        self._last_processed_text = ""

    def process_transcript(self, text: str, is_final: bool = False) -> List[Intent]:
        """
        Analyze a transcript update and return any newly actionable Intent objects.

        Args:
            text: The normalized speech transcript text.
            is_final: True if this is a final transcript segment.

        Returns:
            List of newly confirmed, non-duplicate Intent objects.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        # Split transcript into sequential clauses along connectors
        clauses = self._split_clauses(clean_text)
        newly_emitted_intents: List[Intent] = []

        for idx, clause in enumerate(clauses):
            is_last_clause = (idx == len(clauses) - 1)
            # A clause must be evaluated as potentially incomplete only if it is the trailing clause
            # and the transcript is not yet final. Preceding clauses are considered closed by their connector.
            must_be_complete = not is_last_clause or is_final

            candidate = self._parse_clause(clause, allow_partial=not must_be_complete)
            if candidate and candidate.is_complete:
                sig = self._intent_signature(candidate.intent_type, candidate.parameters)
                if sig not in self._emitted_signatures:
                    intent = Intent(
                        intent_type=candidate.intent_type,
                        parameters=candidate.parameters,
                        source_text=candidate.source_text,
                        confidence=candidate.confidence,
                    )
                    self._emitted_signatures.add(sig)
                    newly_emitted_intents.append(intent)
                    self.logger.info(
                        "INTENT CONFIRMED: %s -> %s (confidence: %.2f)",
                        candidate.source_text,
                        candidate.parameters,
                        candidate.confidence,
                    )
                else:
                    self.logger.debug("Duplicate intent signature ignored: %s", sig)

        if is_final:
            self.reset_context()

        return newly_emitted_intents

    def _split_clauses(self, text: str) -> List[str]:
        """Split text by boundary connectors while trimming whitespace and punctuation."""
        parts = self.BOUNDARY_PATTERN.split(text)
        clauses = []
        for p in parts:
            trimmed = p.strip(" ,.;:!?")
            if trimmed:
                clauses.append(trimmed)
        return clauses

    def _parse_clause(self, clause: str, allow_partial: bool) -> Optional[_SegmentCandidate]:
        """Extract structured intent candidate from a single clause."""
        lower = clause.lower().strip()
        words = lower.split()
        if not words:
            return None

        # Strip common conversational politeness prefixes
        for prefix in (
            "can you please ", "could you please ", "would you please ",
            "can you ", "could you ", "would you ", "please ",
            "i want you to ", "i want to ", "help me ", "let's "
        ):
            if lower.startswith(prefix):
                lower = lower[len(prefix):].strip()
                words = lower.split()
                break

        # Check for interruption keywords ("stop", "cancel", "actually stop")
        if lower in {"stop", "cancel", "wait", "hold on"} or lower.startswith("actually stop"):
            return _SegmentCandidate(
                intent_type=IntentType.INTERRUPT,
                parameters={"reason": "User said stop/cancel"},
                source_text=clause,
                confidence=1.0,
                is_complete=True,
            )

        # Check incognito mode
        if "incognito" in lower:
            return _SegmentCandidate(
                intent_type=IntentType.INCOGNITO,
                parameters={"action": "open_incognito"},
                source_text=clause,
                confidence=0.98,
                is_complete=True,
            )

        # Check tab management
        if any(ph in lower for ph in ("new tab", "create a new tab", "open a new tab", "open new tab")):
            return _SegmentCandidate(
                intent_type=IntentType.NEW_TAB,
                parameters={},
                source_text=clause,
                confidence=0.95,
                is_complete=True,
            )

        if any(ph in lower for ph in ("close tab", "close this tab")):
            return _SegmentCandidate(
                intent_type=IntentType.CLOSE_TAB,
                parameters={},
                source_text=clause,
                confidence=0.95,
                is_complete=True,
            )

        # Pattern: CLICK / CHOOSE / SELECT
        click_verbs = ("click", "choose", "select", "pick")
        if words and words[0] in click_verbs:
            target_words = words[1:]
            if target_words and target_words[0] in ("on", "at", "the", "option"):
                target_words = target_words[1:]
            target_raw = " ".join(target_words).strip(" ,.;:?")
            for lead in ("the ", "my ", "an ", "a "):
                if target_raw.lower().startswith(lead):
                    target_raw = target_raw[len(lead):]
            if target_raw:
                return _SegmentCandidate(
                    intent_type=IntentType.CLICK,
                    parameters={"target": target_raw},
                    source_text=clause,
                    confidence=0.90,
                    is_complete=True,
                )

        # Pattern: PRESS_KEY ("press enter", "press tab", "hit enter")
        if words and words[0] in ("press", "hit"):
            key_words = words[1:]
            key_raw = " ".join(key_words).strip(" ,.;:?").lower()
            if key_raw in ("enter", "return", "tab", "esc", "escape", "space", "backspace", "up", "down", "left", "right"):
                return _SegmentCandidate(
                    intent_type=IntentType.PRESS_KEY,
                    parameters={"key": key_raw},
                    source_text=clause,
                    confidence=0.95,
                    is_complete=True,
                )
            elif key_raw:
                target_cand = key_raw
                for lead in ("the ", "option ", "box "):
                    if target_cand.startswith(lead):
                        target_cand = target_cand[len(lead):]
                return _SegmentCandidate(
                    intent_type=IntentType.CLICK,
                    parameters={"target": target_cand},
                    source_text=clause,
                    confidence=0.85,
                    is_complete=True,
                )

        # 1. Pattern: OPEN_APPLICATION ("open <app>", "launch <app>", "start <app>")
        if words and words[0] in {"open", "launch", "start"}:
            if len(words) == 1:
                # Incomplete: "open"
                return _SegmentCandidate(IntentType.OPEN_APPLICATION, {}, clause, 0.3, is_complete=False)

            app_raw = " ".join(words[1:])
            # Filter incomplete trailing words if in partial mode (e.g. "open the")
            if allow_partial and app_raw in self.INCOMPLETE_TRAILERS:
                return _SegmentCandidate(IntentType.OPEN_APPLICATION, {}, clause, 0.4, is_complete=False)

            canonical_app = self._canonicalize_app_name(app_raw)
            is_complete = bool(canonical_app and canonical_app.strip())
            return _SegmentCandidate(
                intent_type=IntentType.OPEN_APPLICATION,
                parameters={"application": canonical_app},
                source_text=clause,
                confidence=0.95 if canonical_app in self.KNOWN_APPS.values() else 0.85,
                is_complete=is_complete,
            )

        # 2. Pattern: NAVIGATE ("go to <destination>", "navigate to <destination>", "go into <destination>")
        nav_prefixes = ("go to ", "navigate to ", "go into ")
        matched_nav = next((p for p in nav_prefixes if lower.startswith(p)), None)
        if matched_nav:
            dest_raw = lower[len(matched_nav):].strip(" ,.;:?")
            for lead in ("the ", "my "):
                if dest_raw.startswith(lead):
                    dest_raw = dest_raw[len(lead):].strip()
            if not dest_raw:
                return _SegmentCandidate(IntentType.NAVIGATE, {}, clause, 0.3, is_complete=False)

            if allow_partial and dest_raw in self.INCOMPLETE_TRAILERS:
                return _SegmentCandidate(IntentType.NAVIGATE, {}, clause, 0.4, is_complete=False)

            dest_title = dest_raw.title()
            return _SegmentCandidate(
                intent_type=IntentType.NAVIGATE,
                parameters={"destination": dest_title},
                source_text=clause,
                confidence=0.90,
                is_complete=bool(dest_raw),
            )

        # 3. Pattern: SEARCH ("search for <query>", "search <query>")
        if words[0] == "search":
            query_words = words[2:] if len(words) > 1 and words[1] in {"for", "about"} else words[1:]
            if not query_words:
                return _SegmentCandidate(IntentType.SEARCH, {}, clause, 0.3, is_complete=False)

            query_raw = " ".join(query_words).strip(" ,.;:")
            if allow_partial and query_raw in self.INCOMPLETE_TRAILERS:
                return _SegmentCandidate(IntentType.SEARCH, {}, clause, 0.4, is_complete=False)

            return _SegmentCandidate(
                intent_type=IntentType.SEARCH,
                parameters={"query": query_raw},
                source_text=clause,
                confidence=0.90,
                is_complete=bool(query_raw),
            )

        # 4. Pattern: TYPE_TEXT ("type <text>", "write <text>")
        if words[0] in {"type", "write"}:
            text_words = words[1:]
            if not text_words:
                return _SegmentCandidate(IntentType.TYPE_TEXT, {}, clause, 0.3, is_complete=False)

            text_raw = " ".join(text_words).strip()
            return _SegmentCandidate(
                intent_type=IntentType.TYPE_TEXT,
                parameters={"text": text_raw},
                source_text=clause,
                confidence=0.85,
                is_complete=bool(text_raw),
            )

        return None

    def _canonicalize_app_name(self, raw_app: str) -> str:
        """Resolve app string to canonical title case or known application name."""
        clean = raw_app.strip(" ,.;:!?").lower()
        for lead in ("the ", "my ", "a "):
            if clean.startswith(lead):
                clean = clean[len(lead):].strip()
        if clean in self.KNOWN_APPS:
            return self.KNOWN_APPS[clean]
        # Return properly title-cased string for generic apps (e.g. "Blender", "Slack")
        return clean.strip().title()

    @staticmethod
    def _intent_signature(intent_type: IntentType, parameters: dict) -> str:
        """Construct a stable string hash signature to prevent duplicate emissions."""
        items = sorted(parameters.items())
        return f"{intent_type.value}:{items}"
