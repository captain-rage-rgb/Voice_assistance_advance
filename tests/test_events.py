"""
Tests for events and the asynchronous EventBus.
"""

import asyncio
from app.agent.events import Event, EventBus, EventType


def test_event_creation():
    """Verify event instantiation, defaults, and payload."""
    event = Event(
        event_type=EventType.INTENT_DETECTED,
        payload={"intent": "open_application", "application": "brave"},
    )
    assert event.event_type == EventType.INTENT_DETECTED
    assert event.payload["application"] == "brave"
    assert event.event_id is not None
    assert event.timestamp is not None


def test_event_bus_subscribe_publish():
    """Verify that event subscribers receive published events."""

    async def _runner():
        bus = EventBus()
        received = []

        async def handler(evt: Event):
            received.append(evt)

        bus.subscribe(EventType.INTENT_DETECTED, handler)

        event = Event(EventType.INTENT_DETECTED, {"app": "brave"})
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload["app"] == "brave"

        # Test unsubscribe
        bus.unsubscribe(EventType.INTENT_DETECTED, handler)
        await bus.publish(event)
        assert len(received) == 1

    asyncio.run(_runner())
