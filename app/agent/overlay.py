"""
Minimal floating overlay controller and status visualizer for the Real-Time Voice Computer Agent.
Provides minimal state feedback (Listening, Thinking, Executing, Stopped), displays live speech,
and exposes an immediate, non-blocking STOP button tied to the unified cancellation mechanism.
"""

import asyncio
from enum import Enum
import logging
from typing import Callable, Optional
from app.agent.events import Event, EventBus, EventType
from app.agent.runtime import AgentRuntime


class OverlayDisplayState(str, Enum):
    """Minimal visual states of the floating overlay."""

    IDLE = "Idle"
    LISTENING = "Listening"
    THINKING = "Thinking"
    EXECUTING = "Executing"
    VERIFYING = "Verifying"
    RECOVERING = "Recovering"
    STOPPED = "Stopped"


class FloatingOverlay:
    """
    Controller for the minimal top floating overlay.
    Subscribes to live transcript and execution events, displays current spoken words,
    and provides a responsive STOP trigger connected to unified cancellation.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        event_bus: Optional[EventBus] = None,
        on_stop_clicked: Optional[Callable[[], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.runtime = runtime
        self.event_bus = event_bus or runtime.event_bus
        self.on_stop_clicked = on_stop_clicked
        self.logger = logger or logging.getLogger("voice_agent.overlay")

        self.display_state: OverlayDisplayState = OverlayDisplayState.IDLE
        self.live_transcript: str = ""
        self.current_action_label: str = ""
        self.stop_count: int = 0

        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        """Subscribe to agent events to dynamically update overlay state."""
        self.event_bus.subscribe(EventType.AUDIO_STREAM_STARTED, self._on_audio_started)
        self.event_bus.subscribe(EventType.VOICE_ACTIVITY_STARTED, self._on_voice_started)
        self.event_bus.subscribe(EventType.VOICE_ACTIVITY_STOPPED, self._on_voice_stopped)
        self.event_bus.subscribe(EventType.TRANSCRIPT_PARTIAL, self._on_transcript_partial)
        self.event_bus.subscribe(EventType.TRANSCRIPT_FINAL, self._on_transcript_final)
        self.event_bus.subscribe(EventType.ACTION_STARTED, self._on_action_started)
        self.event_bus.subscribe(EventType.ACTION_COMPLETED, self._on_action_ended)
        self.event_bus.subscribe(EventType.ACTION_CANCELLED, self._on_action_ended)
        self.event_bus.subscribe(EventType.VERIFICATION_STARTED, self._on_verification_started)
        self.event_bus.subscribe(EventType.VERIFICATION_SUCCEEDED, self._on_verification_ended)
        self.event_bus.subscribe(EventType.RECOVERY_STARTED, self._on_recovery_started)
        self.event_bus.subscribe(EventType.RECOVERY_COMPLETED, self._on_recovery_ended)
        self.event_bus.subscribe(EventType.RECOVERY_FAILED, self._on_recovery_ended)
        self.event_bus.subscribe(EventType.CLARIFICATION_REQUESTED, self._on_clarification_requested)
        self.event_bus.subscribe(EventType.USER_INTERRUPTED, self._on_interrupted)

    async def _on_audio_started(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.LISTENING

    async def _on_voice_started(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.LISTENING

    async def _on_voice_stopped(self, event: Event) -> None:
        if self.display_state == OverlayDisplayState.LISTENING and self.live_transcript:
            self.display_state = OverlayDisplayState.THINKING

    async def _on_transcript_partial(self, event: Event) -> None:
        self.live_transcript = event.payload.get("text", "")
        self.display_state = OverlayDisplayState.LISTENING

    async def _on_transcript_final(self, event: Event) -> None:
        self.live_transcript = event.payload.get("text", "")

    async def _on_action_started(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.EXECUTING
        self.current_action_label = event.payload.get("action_id", "")

    async def _on_action_ended(self, event: Event) -> None:
        if self.display_state in (OverlayDisplayState.EXECUTING, OverlayDisplayState.VERIFYING, OverlayDisplayState.RECOVERING):
            self.display_state = OverlayDisplayState.LISTENING
            self.current_action_label = ""

    async def _on_verification_started(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.VERIFYING

    async def _on_verification_ended(self, event: Event) -> None:
        if self.display_state == OverlayDisplayState.VERIFYING:
            self.display_state = OverlayDisplayState.LISTENING

    async def _on_recovery_started(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.RECOVERING

    async def _on_recovery_ended(self, event: Event) -> None:
        if self.display_state == OverlayDisplayState.RECOVERING:
            self.display_state = OverlayDisplayState.LISTENING

    async def _on_clarification_requested(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.LISTENING

    async def _on_interrupted(self, event: Event) -> None:
        self.display_state = OverlayDisplayState.STOPPED
        self.current_action_label = ""
        # Return to listening state after a brief visual acknowledgement
        asyncio.create_task(self._reset_to_listening())

    async def _reset_to_listening(self) -> None:
        await asyncio.sleep(0.5)
        if self.display_state == OverlayDisplayState.STOPPED:
            self.display_state = OverlayDisplayState.LISTENING

    async def press_stop(self) -> None:
        """
        User triggers the STOP control on the overlay.
        Immediately updates UI feedback and triggers unified cancellation in the runtime.
        Safe to call rapidly and repeatedly.
        """
        self.stop_count += 1
        self.logger.info("STOP BUTTON PRESSED (count=%d). Triggering immediate cancellation...", self.stop_count)
        self.display_state = OverlayDisplayState.STOPPED
        self.current_action_label = ""

        if self.on_stop_clicked:
            try:
                self.on_stop_clicked()
            except Exception as exc:
                self.logger.warning("Error in on_stop_clicked hook: %s", exc)

        # Trigger runtime interrupt
        await self.runtime.interrupt(reason="User pressed STOP on overlay")
