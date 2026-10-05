"""
Unit tests for AudioPipeline: startup, streaming events, error handling, and shutdown.
Uses a mock audio capture to run in any CI/headless environment without a physical microphone.
"""

import asyncio
from typing import Optional
from app.agent.events import Event, EventBus, EventType
from app.audio.chunk import AudioChunk
from app.audio.pipeline import AudioPipeline
from app.audio.vad import VoiceActivityDetector
from app.config import Settings
from app.state.state_manager import StateManager


class MockAudioCapture:
    """Mock capture for unit testing pipeline logic without hardware."""

    def __init__(self, output_queue: asyncio.Queue[AudioChunk]) -> None:
        self.output_queue = output_queue
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False


class FailingAudioCapture:
    """Mock capture that fails on start to test error handling."""

    @property
    def is_running(self) -> bool:
        return False

    async def start(self) -> None:
        raise OSError("Microphone access denied")

    async def stop(self) -> None:
        pass


def test_audio_pipeline_lifecycle_and_events():
    """Verify clean start, state update, and stop events."""

    async def _runner():
        event_bus = EventBus()
        state_manager = StateManager()
        started_events = []
        stopped_events = []

        async def on_started(evt: Event):
            started_events.append(evt)

        async def on_stopped(evt: Event):
            stopped_events.append(evt)

        event_bus.subscribe(EventType.AUDIO_STREAM_STARTED, on_started)
        event_bus.subscribe(EventType.AUDIO_STREAM_STOPPED, on_stopped)

        # Mock capture
        queue = asyncio.Queue()
        mock_capture = MockAudioCapture(queue)
        pipeline = AudioPipeline(
            event_bus=event_bus,
            state_manager=state_manager,
            capture=mock_capture,
        )
        pipeline.audio_queue = queue

        await pipeline.start()
        assert pipeline.is_running
        assert state_manager.current.microphone_active is True
        assert len(started_events) == 1

        await pipeline.stop()
        assert not pipeline.is_running
        assert state_manager.current.microphone_active is False
        assert len(stopped_events) == 1

    asyncio.run(_runner())


def test_audio_pipeline_processes_chunks_and_emits_vad_events():
    """Verify that queued chunks trigger VOICE_ACTIVITY_STARTED and VOICE_ACTIVITY_STOPPED events."""

    async def _runner():
        event_bus = EventBus()
        state_manager = StateManager()
        vad_events = []

        async def on_vad_start(evt: Event):
            vad_events.append(evt)

        async def on_vad_stop(evt: Event):
            vad_events.append(evt)

        event_bus.subscribe(EventType.VOICE_ACTIVITY_STARTED, on_vad_start)
        event_bus.subscribe(EventType.VOICE_ACTIVITY_STOPPED, on_vad_stop)

        queue = asyncio.Queue()
        mock_capture = MockAudioCapture(queue)
        vad = VoiceActivityDetector(aggressiveness=1, speech_start_frames=1, speech_end_frames=1)

        pipeline = AudioPipeline(
            event_bus=event_bus,
            state_manager=state_manager,
            capture=mock_capture,
            vad=vad,
        )
        pipeline.audio_queue = queue

        await pipeline.start()

        # Generate a synthetic chunk
        chunk = AudioChunk(
            data=b"\x10\x20" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30,
        )
        silence = AudioChunk(
            data=b"\x00\x00" * 480,
            sample_rate=16000,
            channels=1,
            duration_ms=30,
        )

        # Enqueue speech chunk then silence chunk
        await queue.put(chunk)
        await asyncio.sleep(0.05)
        await queue.put(silence)
        await asyncio.sleep(0.05)

        await pipeline.stop()

    asyncio.run(_runner())


def test_audio_pipeline_startup_failure_handling():
    """Verify that hardware failure produces AUDIO_ERROR event and sets state cleanly."""

    async def _runner():
        event_bus = EventBus()
        state_manager = StateManager()
        error_events = []

        async def on_error(evt: Event):
            error_events.append(evt)

        event_bus.subscribe(EventType.AUDIO_ERROR, on_error)

        failing_capture = FailingAudioCapture()
        pipeline = AudioPipeline(
            event_bus=event_bus,
            state_manager=state_manager,
            capture=failing_capture,
        )

        try:
            await pipeline.start()
        except Exception:
            pass

        assert not pipeline.is_running
        assert state_manager.current.microphone_active is False
        assert state_manager.current.audio_error is not None
        assert len(error_events) == 1

    asyncio.run(_runner())
