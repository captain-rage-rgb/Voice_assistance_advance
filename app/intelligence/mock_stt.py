"""
Mock Streaming STT provider for deterministic unit and integration testing.
Allows feeding synthetic partial and final transcript sequences without hitting network or paid APIs.
"""

import asyncio
from typing import AsyncIterator, List, Optional
from app.audio.chunk import AudioChunk
from app.intelligence.stt_interface import StreamingSTT
from app.intelligence.transcript import Transcript


class MockStreamingSTT:
    """
    Mock STT provider that simulates real-time transcript streaming.
    Can be loaded with scripted transcripts or configured to emit on audio chunk reception.
    """

    def __init__(self, scripted_transcripts: Optional[List[Transcript]] = None) -> None:
        self.scripted_transcripts: List[Transcript] = scripted_transcripts or []
        self._connected = False
        self._transcript_queue: asyncio.Queue[Transcript] = asyncio.Queue()
        self.received_chunks: List[AudioChunk] = []

    @property
    def is_connected(self) -> bool:
        """Return connection state."""
        return self._connected

    async def connect(self) -> None:
        """Simulate connection."""
        self._connected = True
        # Preload scripted transcripts into the queue
        for item in self.scripted_transcripts:
            await self._transcript_queue.put(item)

    async def send_audio(self, chunk: AudioChunk) -> None:
        """Record received audio chunk."""
        if not self._connected:
            raise ConnectionError("Cannot send audio while disconnected.")
        self.received_chunks.append(chunk)

    async def push_transcript(self, transcript: Transcript) -> None:
        """Dynamically inject a transcript update during an active test run."""
        await self._transcript_queue.put(transcript)

    async def receive_transcripts(self) -> AsyncIterator[Transcript]:
        """Yield transcripts as they arrive in the queue."""
        while self._connected or not self._transcript_queue.empty():
            try:
                transcript = await asyncio.wait_for(self._transcript_queue.get(), timeout=0.1)
                yield transcript
            except asyncio.TimeoutError:
                if not self._connected:
                    break
                continue

    async def disconnect(self) -> None:
        """Simulate disconnect."""
        self._connected = False
