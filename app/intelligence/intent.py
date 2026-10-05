"""
Structured Intent model representing semantic meaning extracted from streaming speech transcripts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class IntentType(str, Enum):
    """Generic intent types recognized by the incremental detector."""

    OPEN_APPLICATION = "OPEN_APPLICATION"
    NAVIGATE = "NAVIGATE"
    SEARCH = "SEARCH"
    TYPE_TEXT = "TYPE_TEXT"
    CLICK = "CLICK"
    PRESS_KEY = "PRESS_KEY"
    NEW_TAB = "NEW_TAB"
    CLOSE_TAB = "CLOSE_TAB"
    INCOGNITO = "INCOGNITO"
    INTERRUPT = "INTERRUPT"


@dataclass(frozen=True)
class Intent:
    """
    Immutable representation of an actionable intent.
    Contains structured parameters, source text, confidence, and creation timestamp.
    """

    intent_type: IntentType
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_text: str = ""
    confidence: float = 1.0
    intent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def application(self) -> Optional[str]:
        """Convenience accessor for application parameter if present."""
        return self.parameters.get("application")

    @property
    def destination(self) -> Optional[str]:
        """Convenience accessor for navigation destination/URL if present."""
        return self.parameters.get("destination") or self.parameters.get("url")

    @property
    def query(self) -> Optional[str]:
        """Convenience accessor for search query if present."""
        return self.parameters.get("query")
