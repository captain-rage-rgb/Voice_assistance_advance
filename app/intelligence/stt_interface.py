"""
Streaming Speech-to-Text provider protocol abstraction.
Decouples the agent runtime and audio pipeline from any specific STT vendor SDK.
"""

from typing import AsyncIterator, Protocol
from app.audio.chunk import AudioChunk
from app.intelligence.transcript import Transcript


class StreamingSTT(Protocol):
    """Protocol defining streaming speech-to-text providers."""

    async def connect(self) -> None:
        """Establish connection to the streaming STT service."""
        ...

    async def send_audio(self, chunk: AudioChunk) -> None:
        """Send a raw audio chunk to the STT stream."""
        ...

    async def receive_transcripts(self) -> AsyncIterator[Transcript]:
        """Asynchronously yield partial and final transcripts from the STT stream."""
        ...

    async def disconnect(self) -> None:
        """Close connection and clean up resources."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True if connection is actively established."""
        ...
