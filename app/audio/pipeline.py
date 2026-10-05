"""
Audio Pipeline coordinating asynchronous capture, queue distribution, and VAD processing.
Emits events to EventBus and updates StateManager in real time.
"""

import asyncio
import logging
from typing import Optional
from app.agent.events import Event, EventBus, EventType
from app.audio.capture import AudioCapture, MicrophoneCapture
from app.audio.chunk import AudioChunk
from app.audio.vad import VoiceActivityDetector
from app.config import Settings, settings
from app.state.state_manager import StateManager


class AudioPipeline:
    """
    Coordinates streaming audio capture, chunk queueing, VAD evaluation,
    and event publishing into the agent runtime.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        capture: Optional[AudioCapture] = None,
        vad: Optional[VoiceActivityDetector] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or settings
        self.event_bus = event_bus or EventBus()
        self.state_manager = state_manager or StateManager()
        self.logger = logger or logging.getLogger("voice_agent.audio.pipeline")

        # Bounded chunk queue shared between capture producer and async consumer
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(
            maxsize=self.config.audio_queue_max_size
        )

        self.vad = vad or VoiceActivityDetector(
            aggressiveness=self.config.vad_aggressiveness,
            speech_start_frames=self.config.vad_speech_start_frames,
            speech_end_frames=self.config.vad_speech_end_frames,
            logger=self.logger,
        )

        self.capture = capture or MicrophoneCapture(
            output_queue=self.audio_queue,
            sample_rate=self.config.audio_sample_rate,
            channels=self.config.audio_channels,
            chunk_duration_ms=self.config.audio_chunk_duration_ms,
            device_index=self.config.audio_device_index,
            logger=self.logger,
        )

        self._consumer_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """Return True if pipeline is actively running."""
        return self.capture.is_running and (self._consumer_task is not None and not self._consumer_task.done())

    async def start(self) -> None:
        """Start audio capture and the asynchronous VAD consumer loop."""
        if self.is_running:
            self.logger.warning("Audio pipeline is already running.")
            return

        self.logger.info("AUDIO PIPELINE: Starting...")
        self._stop_event.clear()
        self.vad.reset()

        try:
            # Start microphone capture
            await self.capture.start()
        except Exception as exc:
            error_msg = f"Audio capture startup failed: {exc}"
            self.logger.error("AUDIO_ERROR: %s", error_msg)
            self.state_manager.set_audio_error(error_msg)
            await self.event_bus.publish(Event(EventType.AUDIO_ERROR, {"error": error_msg}))
            raise

        self.state_manager.set_microphone_active(True)
        self.state_manager.set_audio_error(None)

        # Launch consumer coroutine
        self._consumer_task = asyncio.create_task(self._process_audio_stream())

        self.logger.info("AUDIO_STREAM_STARTED: Pipeline active and listening.")
        await self.event_bus.publish(
            Event(
                EventType.AUDIO_STREAM_STARTED,
                {
                    "sample_rate": self.config.audio_sample_rate,
                    "channels": self.config.audio_channels,
                    "chunk_ms": self.config.audio_chunk_duration_ms,
                },
            )
        )

    async def stop(self) -> None:
        """Gracefully stop audio capture and the consumer loop."""
        self.logger.info("AUDIO PIPELINE: Stopping...")
        self._stop_event.set()

        # Stop microphone capture
        try:
            await self.capture.stop()
        except Exception as exc:
            self.logger.warning("Error stopping audio capture: %s", exc)

        # Cancel consumer task
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        # Drain any remaining chunks from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.state_manager.set_microphone_active(False)
        self.state_manager.set_voice_activity_active(False)

        self.logger.info("AUDIO_STREAM_STOPPED: Pipeline stopped.")
        await self.event_bus.publish(Event(EventType.AUDIO_STREAM_STOPPED))

    async def _process_audio_stream(self) -> None:
        """
        Continuous consumer loop processing audio chunks from the queue
        and publishing VAD transition events.
        """
        while not self._stop_event.is_set():
            try:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                is_speech, transition = self.vad.process_chunk(chunk)

                if transition == "SPEECH_STARTED":
                    self.logger.info("VOICE_ACTIVITY_STARTED: User started speaking.")
                    self.state_manager.set_voice_activity_active(True)
                    await self.event_bus.publish(
                        Event(
                            EventType.VOICE_ACTIVITY_STARTED,
                            {"timestamp": chunk.timestamp.isoformat()},
                        )
                    )

                elif transition == "SPEECH_STOPPED":
                    self.logger.info("VOICE_ACTIVITY_STOPPED: User stopped speaking.")
                    self.state_manager.set_voice_activity_active(False)
                    await self.event_bus.publish(
                        Event(
                            EventType.VOICE_ACTIVITY_STOPPED,
                            {"timestamp": chunk.timestamp.isoformat()},
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                error_msg = f"Audio consumer error: {exc}"
                self.logger.error("AUDIO_ERROR: %s", error_msg, exc_info=True)
                self.state_manager.set_audio_error(error_msg)
                await self.event_bus.publish(Event(EventType.AUDIO_ERROR, {"error": error_msg}))
