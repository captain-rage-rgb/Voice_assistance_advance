"""
Streaming STT Manager orchestrating audio forwarding, transcript deduplication,
revision handling, and event emission into EventBus and StateManager.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Optional
from app.agent.events import Event, EventBus, EventType
from app.audio.chunk import AudioChunk
from app.config import Settings, settings
from app.intelligence.deepgram_stt import DeepgramStreamingSTT
from app.intelligence.intent_detector import IncrementalIntentDetector
from app.intelligence.stt_interface import StreamingSTT
from app.intelligence.transcript import Transcript
from app.state.state_manager import StateManager


class STTManager:
    """
    Coordinates streaming audio feeding to an STT provider and receives partial/final transcripts.
    Suppresses consecutive duplicate partials, tracks revisions, measures latencies,
    evaluates incremental intent detection, and publishes TRANSCRIPT_PARTIAL, TRANSCRIPT_FINAL,
    and INTENT_DETECTED events into the EventBus.
    """

    def __init__(
        self,
        audio_queue: asyncio.Queue[AudioChunk],
        stt_provider: Optional[StreamingSTT] = None,
        event_bus: Optional[EventBus] = None,
        state_manager: Optional[StateManager] = None,
        intent_detector: Optional[IncrementalIntentDetector] = None,
        config: Optional[Settings] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.audio_queue = audio_queue
        self.config = config or settings
        self.event_bus = event_bus or EventBus()
        self.state_manager = state_manager or StateManager()
        self.logger = logger or logging.getLogger("voice_agent.intelligence.stt")
        self.intent_detector = intent_detector or IncrementalIntentDetector(logger=self.logger)

        # Provider initialization
        if stt_provider is not None:
            self.provider = stt_provider
        else:
            api_key = self.config.stt_api_key or ""
            self.provider = DeepgramStreamingSTT(
                api_key=api_key,
                model=self.config.stt_model,
                language=self.config.stt_language,
                sample_rate=self.config.audio_sample_rate,
                channels=self.config.audio_channels,
                endpointing_ms=self.config.stt_endpointing_ms,
                logger=self.logger,
            )

        self._audio_forwarder_task: Optional[asyncio.Task] = None
        self._transcript_receiver_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Transcript tracking & deduplication
        self._last_partial_text: str = ""
        self._last_final_text: str = ""
        self._utterance_start_time: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        """Return True if STT tasks are active and provider is connected."""
        return (
            self.provider.is_connected
            and self._audio_forwarder_task is not None
            and not self._audio_forwarder_task.done()
        )

    async def start(self) -> None:
        """Connect STT provider and launch audio forwarding and transcript receiving tasks."""
        if self.is_running:
            self.logger.warning("STTManager is already running.")
            return

        self.logger.info("Starting Streaming STT Manager...")
        self._stop_event.clear()
        self._last_partial_text = ""
        self._last_final_text = ""

        try:
            await self.provider.connect()
        except Exception as exc:
            error_msg = f"Failed to connect STT provider: {exc}"
            self.logger.error("STT_ERROR: %s", error_msg)
            self.state_manager.set_stt_error(error_msg)
            await self.event_bus.publish(Event(EventType.STT_ERROR, {"error": error_msg}))
            raise

        self.state_manager.set_stt_active(True)
        self.state_manager.set_stt_error(None)

        # Launch background async workers
        self._audio_forwarder_task = asyncio.create_task(self._forward_audio())
        self._transcript_receiver_task = asyncio.create_task(self._receive_transcripts())

        self.logger.info("STT_STARTED: Streaming STT active and receiving audio.")
        await self.event_bus.publish(Event(EventType.STT_STARTED))

    async def stop(self) -> None:
        """Stop background tasks, disconnect provider, and emit STT_STOPPED."""
        self.logger.info("Stopping Streaming STT Manager...")
        self._stop_event.set()

        # Cancel audio forwarder
        if self._audio_forwarder_task and not self._audio_forwarder_task.done():
            self._audio_forwarder_task.cancel()
            try:
                await self._audio_forwarder_task
            except asyncio.CancelledError:
                pass
            self._audio_forwarder_task = None

        # Cancel transcript receiver
        if self._transcript_receiver_task and not self._transcript_receiver_task.done():
            self._transcript_receiver_task.cancel()
            try:
                await self._transcript_receiver_task
            except asyncio.CancelledError:
                pass
            self._transcript_receiver_task = None

        # Disconnect provider
        try:
            await self.provider.disconnect()
        except Exception as exc:
            self.logger.warning("Error disconnecting STT provider: %s", exc)

        self.state_manager.set_stt_active(False)
        self.logger.info("STT_STOPPED: Streaming STT stopped.")
        await self.event_bus.publish(Event(EventType.STT_STOPPED))

    async def _forward_audio(self) -> None:
        """Continuously pulls chunks from audio_queue and forwards them to STT provider."""
        while not self._stop_event.is_set():
            try:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if self._utterance_start_time is None:
                    self._utterance_start_time = chunk.timestamp

                await self.provider.send_audio(chunk)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                error_msg = f"Audio forwarding error: {exc}"
                self.logger.error("STT_ERROR in audio forwarder: %s", error_msg)
                self.state_manager.set_stt_error(error_msg)
                await self.event_bus.publish(Event(EventType.STT_ERROR, {"error": error_msg}))
                break

    async def _receive_transcripts(self) -> None:
        """Receives transcripts from STT provider, filters duplicates, and emits events."""
        while not self._stop_event.is_set():
            try:
                async for transcript in self.provider.receive_transcripts():
                    if self._stop_event.is_set():
                        break

                    text = transcript.normalized_text
                    if transcript.is_empty:
                        continue

                    now = datetime.now(timezone.utc)
                    latency_ms = None
                    if self._utterance_start_time:
                        latency_ms = (now - self._utterance_start_time).total_seconds() * 1000.0

                    if transcript.is_final:
                        # Final transcript received
                        self._last_final_text = text
                        self._last_partial_text = ""
                        self._utterance_start_time = None  # Reset for next utterance

                        self.logger.info(
                            "TRANSCRIPT_FINAL: '%s' (confidence: %s, latency: %s ms)",
                            text,
                            f"{transcript.confidence:.2f}" if transcript.confidence else "N/A",
                            f"{latency_ms:.1f}" if latency_ms else "N/A",
                        )
                        self.state_manager.set_final_transcript(text)
                        await self.event_bus.publish(
                            Event(
                                EventType.TRANSCRIPT_FINAL,
                                {
                                    "text": text,
                                    "confidence": transcript.confidence,
                                    "segment_id": transcript.segment_id,
                                    "latency_ms": latency_ms,
                                    "timestamp": transcript.timestamp.isoformat(),
                                },
                            )
                        )
                        # Check for intent completion on final transcript
                        new_intents = self.intent_detector.process_transcript(text, is_final=True)
                        for intent in new_intents:
                            await self.event_bus.publish(
                                Event(
                                    EventType.INTENT_DETECTED,
                                    {
                                        "intent": intent.intent_type.value,
                                        "arguments": intent.parameters,
                                        "source_text": intent.source_text,
                                        "confidence": intent.confidence,
                                    },
                                )
                            )
                    else:
                        # Partial transcript received - filter consecutive identical duplicates
                        if text == self._last_partial_text:
                            continue

                        self._last_partial_text = text
                        self.logger.info(
                            "TRANSCRIPT_PARTIAL: '%s' (latency: %s ms)",
                            text,
                            f"{latency_ms:.1f}" if latency_ms else "N/A",
                        )
                        self.state_manager.set_partial_transcript(text)
                        await self.event_bus.publish(
                            Event(
                                EventType.TRANSCRIPT_PARTIAL,
                                {
                                    "text": text,
                                    "confidence": transcript.confidence,
                                    "latency_ms": latency_ms,
                                    "timestamp": transcript.timestamp.isoformat(),
                                },
                            )
                        )
                        # Incrementally check for completed sub-intents while still speaking
                        new_intents = self.intent_detector.process_transcript(text, is_final=False)
                        for intent in new_intents:
                            await self.event_bus.publish(
                                Event(
                                    EventType.INTENT_DETECTED,
                                    {
                                        "intent": intent.intent_type.value,
                                        "arguments": intent.parameters,
                                        "source_text": intent.source_text,
                                        "confidence": intent.confidence,
                                    },
                                )
                            )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                error_msg = f"Transcript receiver error: {exc}"
                self.logger.error("STT_ERROR in receiver: %s", error_msg)
                self.state_manager.set_stt_error(error_msg)
                await self.event_bus.publish(Event(EventType.STT_ERROR, {"error": error_msg}))
                break
