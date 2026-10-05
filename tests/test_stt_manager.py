"""
Unit tests for STTManager and streaming STT logic using MockStreamingSTT.
Tests partial progression, revisions, duplicate suppression, and finalization.
"""

import asyncio
from app.agent.events import Event, EventBus, EventType
from app.audio.chunk import AudioChunk
from app.intelligence.mock_stt import MockStreamingSTT
from app.intelligence.stt_manager import STTManager
from app.intelligence.transcript import Transcript
from app.state.state_manager import StateManager


def test_stt_manager_streaming_and_deduplication():
    """
    Verify progression:
    partial ('Open') -> duplicate ('Open') -> revision ('Open Chrome') -> final ('Open Chrome')
    Duplicate 'Open' should be suppressed, revision accepted, and final properly dispatched.
    """

    async def _runner():
        audio_queue = asyncio.Queue()
        event_bus = EventBus()
        state_manager = StateManager()

        partial_events = []
        final_events = []

        async def on_partial(evt: Event):
            partial_events.append(evt)

        async def on_final(evt: Event):
            final_events.append(evt)

        event_bus.subscribe(EventType.TRANSCRIPT_PARTIAL, on_partial)
        event_bus.subscribe(EventType.TRANSCRIPT_FINAL, on_final)

        # Scripted transcript stream
        scripted = [
            Transcript(text="Open", is_final=False),
            Transcript(text="Open", is_final=False),  # Duplicate - should be suppressed
            Transcript(text="Open Chrome", is_final=False),  # Revision
            Transcript(text="Open Chrome", is_final=True, confidence=0.98),  # Final
        ]
        mock_provider = MockStreamingSTT(scripted_transcripts=scripted)

        manager = STTManager(
            audio_queue=audio_queue,
            stt_provider=mock_provider,
            event_bus=event_bus,
            state_manager=state_manager,
        )

        await manager.start()

        # Feed an audio chunk
        chunk = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30,
        )
        await audio_queue.put(chunk)

        # Wait for transcripts to process
        await asyncio.sleep(0.1)

        # Assertions
        # 1. Partial events should contain 'Open' and 'Open Chrome' (duplicate filtered)
        assert len(partial_events) == 2
        assert partial_events[0].payload["text"] == "Open"
        assert partial_events[1].payload["text"] == "Open Chrome"

        # 2. Final event
        assert len(final_events) == 1
        assert final_events[0].payload["text"] == "Open Chrome"
        assert final_events[0].payload["confidence"] == 0.98

        # 3. State verification
        assert state_manager.current.latest_final_transcript == "Open Chrome"
        assert state_manager.current.latest_partial_transcript == ""

        # 4. Audio was forwarded to provider
        assert len(mock_provider.received_chunks) == 1

        await manager.stop()
        assert not manager.is_running
        assert state_manager.current.stt_active is False

    asyncio.run(_runner())


def test_stt_manager_connection_failure():
    """Verify clean error event and state update when provider connection fails."""

    class FailingSTT:
        @property
        def is_connected(self):
            return False

        async def connect(self):
            raise ConnectionError("Mock connection failed")

        async def send_audio(self, chunk):
            pass

        async def receive_transcripts(self):
            if False:
                yield

        async def disconnect(self):
            pass

    async def _runner():
        audio_queue = asyncio.Queue()
        event_bus = EventBus()
        state_manager = StateManager()
        error_events = []

        async def on_error(evt: Event):
            error_events.append(evt)

        event_bus.subscribe(EventType.STT_ERROR, on_error)

        manager = STTManager(
            audio_queue=audio_queue,
            stt_provider=FailingSTT(),
            event_bus=event_bus,
            state_manager=state_manager,
        )

        try:
            await manager.start()
        except ConnectionError:
            pass

        assert not manager.is_running
        assert state_manager.current.stt_active is False
        assert state_manager.current.stt_error is not None
        assert len(error_events) == 1

    asyncio.run(_runner())
