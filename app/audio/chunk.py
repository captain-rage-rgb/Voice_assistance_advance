"""
Audio chunk data model representing discrete slices of streaming audio.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class AudioChunk:
    """
    Immutable representation of an audio chunk captured from an input stream.
    Contains raw 16-bit PCM bytes and complete metadata for downstream consumers.
    """

    data: bytes
    sample_rate: int
    channels: int
    duration_ms: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def sample_count(self) -> int:
        """Returns the total number of audio samples in this chunk."""
        # 16-bit PCM = 2 bytes per sample per channel
        bytes_per_sample = 2 * self.channels
        if bytes_per_sample == 0:
            return 0
        return len(self.data) // bytes_per_sample

    def __len__(self) -> int:
        """Return length in bytes."""
        return len(self.data)
