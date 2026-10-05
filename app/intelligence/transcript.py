"""
Transcript data model representing partial and final speech recognition results.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid


@dataclass(frozen=True)
class Transcript:
    """
    Immutable representation of a streaming speech recognition update.
    Distinguishes between interim partial hypotheses and finalized transcript segments.
    """

    text: str
    is_final: bool
    confidence: Optional[float] = None
    segment_id: Optional[str] = None
    transcript_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_empty(self) -> bool:
        """Returns True if the recognized text is empty or whitespace only."""
        return not bool(self.text and self.text.strip())

    @property
    def normalized_text(self) -> str:
        """Returns text with leading and trailing whitespace stripped."""
        return self.text.strip()
