"""
Generic event model and event bus for decoupled asynchronous event distribution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List
import uuid


class EventType(str, Enum):
    """Core event types recognized across the runtime."""

    # System lifecycle
    SYSTEM_STARTED = "SYSTEM_STARTED"
    SYSTEM_STOPPING = "SYSTEM_STOPPING"
    SYSTEM_STOPPED = "SYSTEM_STOPPED"

    # Audio & Voice Activity Detection lifecycle
    AUDIO_STREAM_STARTED = "AUDIO_STREAM_STARTED"
    AUDIO_STREAM_STOPPED = "AUDIO_STREAM_STOPPED"
    VOICE_ACTIVITY_STARTED = "VOICE_ACTIVITY_STARTED"
    VOICE_ACTIVITY_STOPPED = "VOICE_ACTIVITY_STOPPED"
    AUDIO_ERROR = "AUDIO_ERROR"

    # Streaming Speech-to-Text lifecycle & transcripts
    STT_STARTED = "STT_STARTED"
    STT_STOPPED = "STT_STOPPED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"
    STT_ERROR = "STT_ERROR"

    # Simulated intelligence / intent events
    INTENT_DETECTED = "INTENT_DETECTED"

    # Action lifecycle events
    ACTION_CREATED = "ACTION_CREATED"
    ACTION_QUEUED = "ACTION_QUEUED"
    ACTION_STARTED = "ACTION_STARTED"
    ACTION_COMPLETED = "ACTION_COMPLETED"
    ACTION_FAILED = "ACTION_FAILED"
    ACTION_CANCELLED = "ACTION_CANCELLED"

    # Interruption & errors
    USER_INTERRUPTED = "USER_INTERRUPTED"
    AGENT_ERROR = "AGENT_ERROR"


@dataclass(frozen=True)
class Event:
    """Represents an immutable event occurrence."""

    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Handler type: an async callable accepting an Event
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Lightweight in-memory asynchronous publish-subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = {}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe an async handler to a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe an async handler from an event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribed handlers concurrently.
        Exceptions in individual subscribers are caught to protect the bus.
        """
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                # Logging or error containment will be handled at the runtime level
                pass
